import os
from pathlib import Path

# Must be set before importing cv2 for very large images.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from image_processing import ensure_uint8_for_annotation


class ImageIO:
    """OpenCV image IO helpers with Windows Unicode path support."""

    def read(self, path: str):
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    def write(self, path: str, img: np.ndarray, params=None) -> bool:
        ext = Path(path).suffix
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(path)
        return True


class PixmapConverter:
    """Converts OpenCV images to scaled preview pixmaps."""

    def to_qpixmap(self, img: np.ndarray, max_w: int = 900, max_h: int = 700) -> QPixmap:
        if img is None:
            return QPixmap()

        preview = ensure_uint8_for_annotation(img)
        h, w = preview.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            preview = cv2.resize(
                preview,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        if preview.ndim == 2:
            qimg = QImage(
                preview.data,
                preview.shape[1],
                preview.shape[0],
                preview.strides[0],
                QImage.Format_Grayscale8,
            )
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


_DEFAULT_IO = ImageIO()
_DEFAULT_CONVERTER = PixmapConverter()


def cv_imread(path: str):
    return _DEFAULT_IO.read(path)


def cv_imwrite(path: str, img: np.ndarray, params=None) -> bool:
    return _DEFAULT_IO.write(path, img, params)


def image_to_qpixmap(img: np.ndarray, max_w: int = 900, max_h: int = 700) -> QPixmap:
    return _DEFAULT_CONVERTER.to_qpixmap(img, max_w, max_h)
