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
        if params.processing_mode == "xnview":
            out = self.apply_xnview_brightness_contrast(
                out,
                int(params.xnview_brightness),
                int(params.xnview_contrast),
            )
        elif params.processing_mode == "auto":
            out = self.apply_auto_levels(out, float(params.auto_clip_percent))
        else:
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

        binary = None
        if params.use_threshold:
            binary = self.apply_threshold(out, params)
            out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        edges = None
        if params.use_edge_detection:
            source = binary if binary is not None else self.to_gray(out)
            edges = self.apply_canny(source, params)
            out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if params.use_shape_detection:
            source = edges if edges is not None else binary
            if source is None:
                source = self.apply_threshold(out, params)
            out = self.draw_detected_shapes(out, source, params)

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

    @staticmethod
    def apply_xnview_brightness_contrast(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
        brightness = int(np.clip(brightness, -255, 255))
        contrast = int(np.clip(contrast, -255, 255))
        factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))

        base = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        adjusted = factor * (base.astype(np.float32) - 128.0) + 128.0 + brightness
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.merge([adjusted[:, :, 0], adjusted[:, :, 1], adjusted[:, :, 2], img[:, :, 3]])
        return adjusted

    @staticmethod
    def apply_auto_levels(img: np.ndarray, clip_percent: float) -> np.ndarray:
        clip_percent = float(np.clip(clip_percent, 0.0, 20.0))
        base = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img

        if base.ndim == 2:
            gray = base
        elif base.ndim == 3 and base.shape[2] >= 3:
            gray = cv2.cvtColor(base[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = base

        low = float(np.percentile(gray, clip_percent))
        high = float(np.percentile(gray, 100.0 - clip_percent))
        if high <= low:
            return img.copy()

        scale = 255.0 / (high - low)
        adjusted = (base.astype(np.float32) - low) * scale
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.merge([adjusted[:, :, 0], adjusted[:, :, 1], adjusted[:, :, 2], img[:, :, 3]])
        return adjusted

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
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self, img: np.ndarray, params: PreprocessParams) -> np.ndarray:
        gray = self.to_gray(self.ensure_uint8_for_annotation(img))
        mode = params.threshold_mode
        threshold_type = cv2.THRESH_BINARY_INV if params.threshold_invert else cv2.THRESH_BINARY

        if mode == "adaptive":
            block_size = self.odd_ksize(params.adaptive_block_size)
            block_size = max(3, block_size)
            method = cv2.ADAPTIVE_THRESH_MEAN_C
            if params.adaptive_method == "gaussian":
                method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            return cv2.adaptiveThreshold(
                gray,
                255,
                method,
                threshold_type,
                block_size,
                int(params.adaptive_c),
            )

        if mode == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
            return binary

        value = int(np.clip(params.threshold_value, 0, 255))
        _, binary = cv2.threshold(gray, value, 255, threshold_type)
        return binary

    def apply_canny(self, img: np.ndarray, params: PreprocessParams) -> np.ndarray:
        gray = self.to_gray(self.ensure_uint8_for_annotation(img))
        aperture = self.odd_ksize(params.canny_aperture_size)
        aperture = int(np.clip(aperture, 3, 7))
        low = int(np.clip(params.canny_low, 0, 255))
        high = int(np.clip(params.canny_high, low + 1, 255))
        return cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=True)

    def draw_detected_shapes(self, img: np.ndarray, source: np.ndarray, params: PreprocessParams) -> np.ndarray:
        if img.ndim == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            canvas = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            canvas = img.copy()

        contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = self.to_gray(canvas)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < params.contour_min_area or area > params.contour_max_area:
                continue

            contour = self.refine_contour_subpixel(contour, gray, params)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, perimeter * float(params.approx_epsilon_percent) / 100.0, True)

            if params.detect_rectangles and self.is_rectangle(contour, params):
                box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
                cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)

            if params.detect_circles:
                circle = self.match_circle(contour, area, params)
                if circle is not None:
                    (x, y), radius = circle
                    cv2.circle(canvas, (int(round(x)), int(round(y))), int(round(radius)), (255, 0, 0), 2)

            if params.detect_polygons and self.is_polygon(approx, params):
                cv2.polylines(canvas, [approx.astype(np.int32)], True, (0, 255, 255), 2)

        return canvas

    @staticmethod
    def refine_contour_subpixel(contour: np.ndarray, gray: np.ndarray, params: PreprocessParams) -> np.ndarray:
        if not params.use_subpixel_refine or len(contour) < 3:
            return contour
        points = contour.astype(np.float32)
        win = max(1, int(params.subpixel_window))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        try:
            cv2.cornerSubPix(gray, points, (win, win), (-1, -1), criteria)
            return points
        except cv2.error:
            return contour

    @staticmethod
    def is_rectangle(contour: np.ndarray, params: PreprocessParams) -> bool:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        width, height = float(width), float(height)
        if width <= 0 or height <= 0:
            return False
        short, long = sorted((width, height))
        aspect = long / short
        return (
            params.rect_min_width <= width <= params.rect_max_width
            and params.rect_min_height <= height <= params.rect_max_height
            and params.rect_min_aspect <= aspect <= params.rect_max_aspect
        )

    @staticmethod
    def match_circle(contour: np.ndarray, area: float, params: PreprocessParams):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            return None
        circularity = area / (np.pi * radius * radius)
        if (
            params.circle_min_radius <= radius <= params.circle_max_radius
            and circularity >= params.circle_min_circularity
        ):
            return (x, y), radius
        return None

    @staticmethod
    def is_polygon(approx: np.ndarray, params: PreprocessParams) -> bool:
        vertices = len(approx)
        return params.polygon_min_vertices <= vertices <= params.polygon_max_vertices

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
