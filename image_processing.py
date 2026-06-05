import os

# Must be set before importing cv2 for very large images.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import cv2
import numpy as np

from params import PreprocessParams


class ImagePreprocessor:
    """Applies the configured preprocessing pipeline without resizing output."""

    def process(self, img: np.ndarray, params: PreprocessParams) -> np.ndarray:
        out = self.ensure_uint8_for_annotation(img)
        out = cv2.convertScaleAbs(out, alpha=float(params.contrast), beta=int(params.brightness))
        out = self.apply_gamma(out, float(params.gamma))

        if params.use_clahe:
            out = self.apply_clahe(out, params.clahe_clip_limit, params.clahe_tile_grid_size)

        if params.use_gaussian_blur:
            k = self.odd_ksize(params.gaussian_ksize)
            if k > 1:
                out = cv2.GaussianBlur(out, (k, k), 0)

        if params.use_median_blur:
            k = self.odd_ksize(params.median_ksize)
            if k > 1:
                out = cv2.medianBlur(out, k)

        if params.use_bilateral_filter:
            out = cv2.bilateralFilter(
                out,
                int(params.bilateral_d),
                int(params.bilateral_sigma_color),
                int(params.bilateral_sigma_space),
            )

        if params.use_sharpen:
            out = self.apply_sharpen(out, params.sharpen_amount)

        return out

    @staticmethod
    def ensure_uint8_for_annotation(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.copy()

        img_f = img.astype(np.float32)
        min_v = float(np.nanmin(img_f))
        max_v = float(np.nanmax(img_f))
        if max_v <= min_v:
            return np.zeros_like(img_f, dtype=np.uint8)

        img_f = (img_f - min_v) / (max_v - min_v) * 255.0
        return np.clip(img_f, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        if abs(gamma - 1.0) < 1e-6:
            return img
        gamma = max(gamma, 0.01)
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    def apply_clahe(self, img: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
        tile_grid_size = max(2, int(tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(tile_grid_size, tile_grid_size),
        )

        if img.ndim == 2:
            return clahe.apply(img)

        if img.ndim == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            lab2 = cv2.merge([clahe.apply(l), a, b])
            return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            bgr2 = self.apply_clahe(bgr, clip_limit, tile_grid_size)
            return cv2.merge([bgr2[:, :, 0], bgr2[:, :, 1], bgr2[:, :, 2], alpha])

        return img

    @staticmethod
    def apply_sharpen(img: np.ndarray, amount: float) -> np.ndarray:
        amount = max(0.0, float(amount))
        if amount <= 0:
            return img
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    @staticmethod
    def odd_ksize(value: int) -> int:
        value = int(value)
        if value < 1:
            value = 1
        if value % 2 == 0:
            value += 1
        return value


_DEFAULT_PREPROCESSOR = ImagePreprocessor()


def preprocess_image(img: np.ndarray, params: PreprocessParams) -> np.ndarray:
    return _DEFAULT_PREPROCESSOR.process(img, params)


def ensure_uint8_for_annotation(img: np.ndarray) -> np.ndarray:
    return _DEFAULT_PREPROCESSOR.ensure_uint8_for_annotation(img)


def odd_ksize(value: int) -> int:
    return _DEFAULT_PREPROCESSOR.odd_ksize(value)
