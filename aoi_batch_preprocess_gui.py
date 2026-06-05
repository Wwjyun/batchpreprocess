# -*- coding: utf-8 -*-
"""
AOI Batch Preprocess GUI
- PySide6 GUI
- OpenCV image preprocessing
- Preview before/after
- Batch process without resizing output images
- Save per-image preprocess metadata JSON

Install:
    pip install PySide6 opencv-python numpy

Run:
    python aoi_batch_preprocess_gui.py
"""

import os
# Must be set before importing cv2 for very large images.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit, QProgressBar,
    QMessageBox, QListWidget, QListWidgetItem, QSplitter, QTextEdit
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PreprocessParams:
    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    use_sharpen: bool = False
    sharpen_amount: float = 1.0
    use_gaussian_blur: bool = False
    gaussian_ksize: int = 3
    use_median_blur: bool = False
    median_ksize: int = 3
    use_bilateral_filter: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50
    output_format: str = "png"
    jpeg_quality: int = 95
    png_compression: int = 3


def cv_imread(path: str):
    """Read image with unicode path support."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img


def cv_imwrite(path: str, img: np.ndarray, params=None) -> bool:
    """Write image with unicode path support."""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


def ensure_uint8_for_annotation(img: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 for annotation visibility.
    If source is 16-bit or float, normalize to 0~255.
    """
    if img.dtype == np.uint8:
        return img.copy()
    img_f = img.astype(np.float32)
    min_v = float(np.nanmin(img_f))
    max_v = float(np.nanmax(img_f))
    if max_v <= min_v:
        return np.zeros_like(img_f, dtype=np.uint8)
    img_f = (img_f - min_v) / (max_v - min_v) * 255.0
    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return img
    gamma = max(gamma, 0.01)
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    tile_grid_size = max(2, int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))

    if img.ndim == 2:
        return clahe.apply(img)

    if img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        bgr2 = apply_clahe(bgr, clip_limit, tile_grid_size)
        return cv2.merge([bgr2[:, :, 0], bgr2[:, :, 1], bgr2[:, :, 2], alpha])

    return img


def apply_sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    amount = max(0.0, float(amount))
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def odd_ksize(value: int) -> int:
    value = int(value)
    if value < 1:
        value = 1
    if value % 2 == 0:
        value += 1
    return value


def preprocess_image(img: np.ndarray, p: PreprocessParams) -> np.ndarray:
    """
    Output remains original width/height, but image is converted to uint8 annotation image.
    """
    out = ensure_uint8_for_annotation(img)

    # Brightness / contrast
    out = cv2.convertScaleAbs(out, alpha=float(p.contrast), beta=int(p.brightness))

    # Gamma
    out = apply_gamma(out, float(p.gamma))

    # Local contrast enhancement
    if p.use_clahe:
        out = apply_clahe(out, p.clahe_clip_limit, p.clahe_tile_grid_size)

    # Denoise filters, intentionally light and optional.
    if p.use_gaussian_blur:
        k = odd_ksize(p.gaussian_ksize)
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), 0)

    if p.use_median_blur:
        k = odd_ksize(p.median_ksize)
        if k > 1:
            out = cv2.medianBlur(out, k)

    if p.use_bilateral_filter:
        out = cv2.bilateralFilter(out, int(p.bilateral_d), int(p.bilateral_sigma_color), int(p.bilateral_sigma_space))

    # Sharpen usually after denoise.
    if p.use_sharpen:
        out = apply_sharpen(out, p.sharpen_amount)

    return out


def image_to_qpixmap(img: np.ndarray, max_w: int = 900, max_h: int = 700) -> QPixmap:
    """Convert cv image to QPixmap for preview only; this display scaling does not affect output."""
    if img is None:
        return QPixmap()

    preview = ensure_uint8_for_annotation(img)
    h, w = preview.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if preview.ndim == 2:
        qimg = QImage(preview.data, preview.shape[1], preview.shape[0], preview.strides[0], QImage.Format_Grayscale8)
    elif preview.ndim == 3 and preview.shape[2] == 3:
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    elif preview.ndim == 3 and preview.shape[2] == 4:
        rgba = cv2.cvtColor(preview, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
    else:
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY) if preview.ndim == 3 else preview
        qimg = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8)

    return QPixmap.fromImage(qimg.copy())


class BatchWorker(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished_ok = Signal(int, int)

    def __init__(self, files, input_dir, output_dir, params: PreprocessParams, overwrite: bool, save_meta: bool):
        super().__init__()
        self.files = files
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.params = params
        self.overwrite = overwrite
        self.save_meta = save_meta
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
                self.log.emit("使用者取消批量處理。")
                break

            src_path = Path(src)
            rel = src_path.relative_to(self.input_dir) if self.input_dir in src_path.parents or src_path == self.input_dir else Path(src_path.name)
            dst_rel = rel.with_suffix("." + self.params.output_format.lower())
            dst_path = self.output_dir / dst_rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            self.progress.emit(idx, len(self.files), src_path.name)
            if dst_path.exists() and not self.overwrite:
                self.log.emit(f"略過已存在：{dst_path}")
                continue

            try:
                t0 = time.time()
                img = cv_imread(str(src_path))
                if img is None:
                    raise RuntimeError("讀圖失敗")

                src_h, src_w = img.shape[:2]
                out = preprocess_image(img, self.params)
                out_h, out_w = out.shape[:2]

                encode_params = []
                ext = self.params.output_format.lower()
                if ext in ["jpg", "jpeg"]:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.params.jpeg_quality)]
                elif ext == "png":
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.params.png_compression)]

                if not cv_imwrite(str(dst_path), out, encode_params):
                    raise RuntimeError("寫圖失敗")

                if self.save_meta:
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
                        "process_time_sec": round(time.time() - t0, 4),
                    }
                    meta_name = dst_rel.with_suffix(".json")
                    meta_path = meta_dir / meta_name
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                ok_count += 1
                self.log.emit(f"完成：{dst_path}  原尺寸 {src_w}x{src_h} -> {out_w}x{out_h}")
            except Exception as e:
                fail_count += 1
                self.log.emit(f"失敗：{src_path}，原因：{e}")

        self.finished_ok.emit(ok_count, fail_count)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI 批量影像預處理工具 - 原尺寸輸出")
        self.resize(1500, 950)

        self.input_dir = ""
        self.output_dir = ""
        self.files = []
        self.current_img = None
        self.current_file = None
        self.worker = None

        self.build_ui()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(420)

        path_group = QGroupBox("資料夾")
        path_layout = QGridLayout(path_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        btn_in = QPushButton("選擇輸入")
        btn_out = QPushButton("選擇輸出")
        btn_scan = QPushButton("掃描圖片")
        btn_in.clicked.connect(self.choose_input)
        btn_out.clicked.connect(self.choose_output)
        btn_scan.clicked.connect(self.scan_files)
        path_layout.addWidget(QLabel("輸入資料夾"), 0, 0)
        path_layout.addWidget(self.input_edit, 0, 1)
        path_layout.addWidget(btn_in, 0, 2)
        path_layout.addWidget(QLabel("輸出資料夾"), 1, 0)
        path_layout.addWidget(self.output_edit, 1, 1)
        path_layout.addWidget(btn_out, 1, 2)
        path_layout.addWidget(btn_scan, 2, 0, 1, 3)
        left_layout.addWidget(path_group)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        left_layout.addWidget(QLabel("圖片清單"))
        left_layout.addWidget(self.file_list, stretch=1)

        params_group = QGroupBox("可調預處理參數")
        params_layout = QGridLayout(params_group)
        row = 0

        self.brightness = QSpinBox(); self.brightness.setRange(-255, 255); self.brightness.setValue(0)
        self.contrast = QDoubleSpinBox(); self.contrast.setRange(0.1, 5.0); self.contrast.setSingleStep(0.1); self.contrast.setValue(1.0)
        self.gamma = QDoubleSpinBox(); self.gamma.setRange(0.1, 5.0); self.gamma.setSingleStep(0.1); self.gamma.setValue(1.0)
        for w in [self.brightness, self.contrast, self.gamma]:
            w.valueChanged.connect(self.update_preview)
        params_layout.addWidget(QLabel("亮度 beta"), row, 0); params_layout.addWidget(self.brightness, row, 1); row += 1
        params_layout.addWidget(QLabel("對比 alpha"), row, 0); params_layout.addWidget(self.contrast, row, 1); row += 1
        params_layout.addWidget(QLabel("Gamma"), row, 0); params_layout.addWidget(self.gamma, row, 1); row += 1

        self.use_clahe = QCheckBox("CLAHE 局部對比")
        self.clahe_clip = QDoubleSpinBox(); self.clahe_clip.setRange(0.1, 20.0); self.clahe_clip.setSingleStep(0.1); self.clahe_clip.setValue(2.0)
        self.clahe_tile = QSpinBox(); self.clahe_tile.setRange(2, 64); self.clahe_tile.setValue(8)
        for w in [self.use_clahe, self.clahe_clip, self.clahe_tile]:
            w.stateChanged.connect(self.update_preview) if isinstance(w, QCheckBox) else w.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_clahe, row, 0, 1, 2); row += 1
        params_layout.addWidget(QLabel("CLAHE clipLimit"), row, 0); params_layout.addWidget(self.clahe_clip, row, 1); row += 1
        params_layout.addWidget(QLabel("CLAHE tileGrid"), row, 0); params_layout.addWidget(self.clahe_tile, row, 1); row += 1

        self.use_gaussian = QCheckBox("Gaussian Blur 降噪")
        self.gaussian_k = QSpinBox(); self.gaussian_k.setRange(1, 31); self.gaussian_k.setValue(3)
        self.use_median = QCheckBox("Median Blur 降噪")
        self.median_k = QSpinBox(); self.median_k.setRange(1, 31); self.median_k.setValue(3)
        self.use_bilateral = QCheckBox("Bilateral Filter 保邊降噪")
        self.bilateral_d = QSpinBox(); self.bilateral_d.setRange(1, 31); self.bilateral_d.setValue(5)
        self.bilateral_sc = QSpinBox(); self.bilateral_sc.setRange(1, 200); self.bilateral_sc.setValue(50)
        self.bilateral_ss = QSpinBox(); self.bilateral_ss.setRange(1, 200); self.bilateral_ss.setValue(50)
        for w in [self.use_gaussian, self.use_median, self.use_bilateral]:
            w.stateChanged.connect(self.update_preview)
        for w in [self.gaussian_k, self.median_k, self.bilateral_d, self.bilateral_sc, self.bilateral_ss]:
            w.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_gaussian, row, 0); params_layout.addWidget(self.gaussian_k, row, 1); row += 1
        params_layout.addWidget(self.use_median, row, 0); params_layout.addWidget(self.median_k, row, 1); row += 1
        params_layout.addWidget(self.use_bilateral, row, 0, 1, 2); row += 1
        params_layout.addWidget(QLabel("Bilateral d"), row, 0); params_layout.addWidget(self.bilateral_d, row, 1); row += 1
        params_layout.addWidget(QLabel("Bilateral sigmaColor"), row, 0); params_layout.addWidget(self.bilateral_sc, row, 1); row += 1
        params_layout.addWidget(QLabel("Bilateral sigmaSpace"), row, 0); params_layout.addWidget(self.bilateral_ss, row, 1); row += 1

        self.use_sharpen = QCheckBox("銳化 Sharpen")
        self.sharpen_amount = QDoubleSpinBox(); self.sharpen_amount.setRange(0.0, 5.0); self.sharpen_amount.setSingleStep(0.1); self.sharpen_amount.setValue(1.0)
        self.use_sharpen.stateChanged.connect(self.update_preview)
        self.sharpen_amount.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_sharpen, row, 0); params_layout.addWidget(self.sharpen_amount, row, 1); row += 1

        self.output_format = QComboBox(); self.output_format.addItems(["png", "jpg", "bmp", "tif"])
        self.jpeg_quality = QSpinBox(); self.jpeg_quality.setRange(1, 100); self.jpeg_quality.setValue(95)
        self.png_compression = QSpinBox(); self.png_compression.setRange(0, 9); self.png_compression.setValue(3)
        params_layout.addWidget(QLabel("輸出格式"), row, 0); params_layout.addWidget(self.output_format, row, 1); row += 1
        params_layout.addWidget(QLabel("JPG 品質"), row, 0); params_layout.addWidget(self.jpeg_quality, row, 1); row += 1
        params_layout.addWidget(QLabel("PNG 壓縮"), row, 0); params_layout.addWidget(self.png_compression, row, 1); row += 1

        self.recursive = QCheckBox("包含子資料夾")
        self.overwrite = QCheckBox("覆蓋已存在輸出")
        self.save_meta = QCheckBox("輸出 preprocess_meta JSON")
        self.save_meta.setChecked(True)
        params_layout.addWidget(self.recursive, row, 0, 1, 2); row += 1
        params_layout.addWidget(self.overwrite, row, 0, 1, 2); row += 1
        params_layout.addWidget(self.save_meta, row, 0, 1, 2); row += 1

        left_layout.addWidget(params_group)

        run_group = QGroupBox("批量執行")
        run_layout = QVBoxLayout(run_group)
        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("更新預覽")
        self.btn_start = QPushButton("開始批量")
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_preview.clicked.connect(self.update_preview)
        self.btn_start.clicked.connect(self.start_batch)
        self.btn_cancel.clicked.connect(self.cancel_batch)
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        run_layout.addLayout(btn_row)
        run_layout.addWidget(self.progress)
        left_layout.addWidget(run_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.info_label = QLabel("輸出圖片不縮放；只在 GUI 預覽時縮小顯示。")
        right_layout.addWidget(self.info_label)

        splitter = QSplitter(Qt.Horizontal)
        self.raw_label = QLabel("原圖預覽")
        self.proc_label = QLabel("處理後預覽")
        for lab in [self.raw_label, self.proc_label]:
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("QLabel { background: #222; color: #ddd; border: 1px solid #555; }")
            lab.setMinimumSize(420, 420)
            lab.setScaledContents(False)
        splitter.addWidget(self.raw_label)
        splitter.addWidget(self.proc_label)
        splitter.setSizes([1, 1])
        right_layout.addWidget(splitter, stretch=3)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(220)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_box)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def params(self) -> PreprocessParams:
        return PreprocessParams(
            brightness=self.brightness.value(),
            contrast=self.contrast.value(),
            gamma=self.gamma.value(),
            use_clahe=self.use_clahe.isChecked(),
            clahe_clip_limit=self.clahe_clip.value(),
            clahe_tile_grid_size=self.clahe_tile.value(),
            use_sharpen=self.use_sharpen.isChecked(),
            sharpen_amount=self.sharpen_amount.value(),
            use_gaussian_blur=self.use_gaussian.isChecked(),
            gaussian_ksize=odd_ksize(self.gaussian_k.value()),
            use_median_blur=self.use_median.isChecked(),
            median_ksize=odd_ksize(self.median_k.value()),
            use_bilateral_filter=self.use_bilateral.isChecked(),
            bilateral_d=self.bilateral_d.value(),
            bilateral_sigma_color=self.bilateral_sc.value(),
            bilateral_sigma_space=self.bilateral_ss.value(),
            output_format=self.output_format.currentText(),
            jpeg_quality=self.jpeg_quality.value(),
            png_compression=self.png_compression.value(),
        )

    def log(self, text):
        self.log_box.append(text)

    def choose_input(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if d:
            self.input_dir = d
            self.input_edit.setText(d)

    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if d:
            self.output_dir = d
            self.output_edit.setText(d)

    def scan_files(self):
        self.input_dir = self.input_edit.text().strip()
        if not self.input_dir or not Path(self.input_dir).exists():
            QMessageBox.warning(self, "提醒", "請先選擇有效的輸入資料夾")
            return
        base = Path(self.input_dir)
        pattern = "**/*" if self.recursive.isChecked() else "*"
        self.files = [str(p) for p in base.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        self.files.sort()
        self.file_list.clear()
        for p in self.files:
            item = QListWidgetItem(str(Path(p).relative_to(base)))
            item.setData(Qt.UserRole, p)
            self.file_list.addItem(item)
        self.log(f"掃描完成：{len(self.files)} 張圖片")
        if self.files:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, current, previous):
        if not current:
            return
        path = current.data(Qt.UserRole)
        self.current_file = path
        img = cv_imread(path)
        if img is None:
            self.log(f"讀圖失敗：{path}")
            return
        self.current_img = img
        h, w = img.shape[:2]
        self.info_label.setText(f"目前圖片：{Path(path).name} | 原尺寸：{w} × {h} | dtype={img.dtype} | 輸出不縮圖")
        self.raw_label.setPixmap(image_to_qpixmap(img, self.raw_label.width() - 20, self.raw_label.height() - 20))
        self.update_preview()

    def update_preview(self):
        if self.current_img is None:
            return
        try:
            out = preprocess_image(self.current_img, self.params())
            h, w = out.shape[:2]
            self.proc_label.setPixmap(image_to_qpixmap(out, self.proc_label.width() - 20, self.proc_label.height() - 20))
            self.info_label.setText(
                f"目前圖片：{Path(self.current_file).name if self.current_file else ''} | 原尺寸輸出：{w} × {h} | GUI 預覽縮小不影響輸出"
            )
        except Exception as e:
            self.log(f"預覽失敗：{e}")

    def start_batch(self):
        self.input_dir = self.input_edit.text().strip()
        self.output_dir = self.output_edit.text().strip()
        if not self.files:
            QMessageBox.warning(self, "提醒", "請先掃描圖片")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "提醒", "請先選擇輸出資料夾")
            return
        if Path(self.output_dir).resolve() == Path(self.input_dir).resolve():
            QMessageBox.warning(self, "提醒", "輸出資料夾不要跟輸入資料夾相同，避免覆蓋原圖")
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.worker = BatchWorker(
            self.files,
            self.input_dir,
            self.output_dir,
            self.params(),
            overwrite=self.overwrite.isChecked(),
            save_meta=self.save_meta.isChecked(),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()
        self.log("開始批量處理，輸出保持原始寬高。")

    def cancel_batch(self):
        if self.worker:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    def on_progress(self, idx, total, name):
        self.progress.setMaximum(total)
        self.progress.setValue(idx)
        self.progress.setFormat(f"{idx}/{total} - {name}")

    def on_finished(self, ok, fail):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log(f"批量完成：成功 {ok}，失敗 {fail}")
        QMessageBox.information(self, "完成", f"批量完成\n成功：{ok}\n失敗：{fail}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_img is not None:
            self.raw_label.setPixmap(image_to_qpixmap(self.current_img, self.raw_label.width() - 20, self.raw_label.height() - 20))
            self.update_preview()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
