import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from image_io import ImageIO
from image_processing import ImagePreprocessor
from params import PreprocessParams


class BatchWorker(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished_ok = Signal(int, int)

    def __init__(
        self,
        files,
        input_dir,
        output_dir,
        params: PreprocessParams,
        overwrite: bool,
        save_meta: bool,
        image_io: ImageIO | None = None,
        preprocessor: ImagePreprocessor | None = None,
    ):
        super().__init__()
        self.files = files
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.params = params
        self.overwrite = overwrite
        self.save_meta = save_meta
        self.image_io = image_io or ImageIO()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok_count = 0
        fail_count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = self.output_dir / "preprocess_meta"
        if self.save_meta:
            meta_dir.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(self.files, start=1):
            if self._cancel:
                self.log.emit("批次處理已由使用者取消。")
                break

            src_path = Path(src)
            rel = (
                src_path.relative_to(self.input_dir)
                if self.input_dir in src_path.parents or src_path == self.input_dir
                else Path(src_path.name)
            )
            dst_rel = rel.with_suffix("." + self.params.output_format.lower())
            dst_path = self.output_dir / dst_rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            self.progress.emit(idx, len(self.files), src_path.name)
            if dst_path.exists() and not self.overwrite:
                self.log.emit(f"略過既有檔案：{dst_path}")
                continue

            try:
                t0 = time.time()
                img = self.image_io.read(str(src_path))
                if img is None:
                    raise RuntimeError("無法讀取影像。")

                src_h, src_w = img.shape[:2]
                out = self.preprocessor.process(img, self.params)
                out_h, out_w = out.shape[:2]

                encode_params = self._encode_params()
                if not self.image_io.write(str(dst_path), out, encode_params):
                    raise RuntimeError("無法寫入影像。")

                if self.save_meta:
                    self._write_metadata(meta_dir, dst_rel, src_path, dst_path, img, out, src_w, src_h, out_w, out_h, t0)

                ok_count += 1
                self.log.emit(f"已儲存：{dst_path} | 尺寸 {src_w}x{src_h} -> {out_w}x{out_h}")
            except Exception as exc:
                fail_count += 1
                self.log.emit(f"處理失敗：{src_path} | {exc}")

        self.finished_ok.emit(ok_count, fail_count)

    def _encode_params(self):
        ext = self.params.output_format.lower()
        if ext in ["jpg", "jpeg"]:
            return [cv2.IMWRITE_JPEG_QUALITY, int(self.params.jpeg_quality)]
        if ext == "png":
            return [cv2.IMWRITE_PNG_COMPRESSION, int(self.params.png_compression)]
        return []

    def _write_metadata(
        self,
        meta_dir: Path,
        dst_rel: Path,
        src_path: Path,
        dst_path: Path,
        img,
        out,
        src_w: int,
        src_h: int,
        out_w: int,
        out_h: int,
        started_at: float,
    ):
        meta = {
            "source_image": str(src_path),
            "output_image": str(dst_path),
            "source_width": int(src_w),
            "source_height": int(src_h),
            "output_width": int(out_w),
            "output_height": int(out_h),
            "resize": False,
            "display_preview_resize_only": True,
            "source_dtype": str(img.dtype),
            "output_dtype": str(out.dtype),
            "params": asdict(self.params),
            "process_time_sec": round(time.time() - started_at, 4),
        }
        meta_path = meta_dir / dst_rel.with_suffix(".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
