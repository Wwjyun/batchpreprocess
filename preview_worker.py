import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from image_processing import ImagePreprocessor, ensure_uint8_for_annotation
from params import PreprocessParams


class PreviewWorker(QThread):
    result = Signal(int, object, str)

    def __init__(self, request_id: int, img: np.ndarray, params: PreprocessParams, max_w: int, max_h: int):
        super().__init__()
        self.request_id = request_id
        self.img = img
        self.params = params
        self.max_w = max(1, int(max_w))
        self.max_h = max(1, int(max_h))
        self.preprocessor = ImagePreprocessor()

    def run(self):
        try:
            preview = self._scaled_preview_image(self.img)
            out = self.preprocessor.process(preview, self.params)
            self.result.emit(self.request_id, out, "")
        except Exception as exc:
            self.result.emit(self.request_id, None, str(exc))

    def _scaled_preview_image(self, img: np.ndarray) -> np.ndarray:
        preview = ensure_uint8_for_annotation(img)
        h, w = preview.shape[:2]
        scale = min(self.max_w / max(w, 1), self.max_h / max(h, 1), 1.0)
        if scale >= 1.0:
            return preview
        return cv2.resize(
            preview,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
