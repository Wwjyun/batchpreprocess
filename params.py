from dataclasses import dataclass


@dataclass
class PreprocessParams:
    processing_mode: str = "opencv"
    brightness: int = 0
    contrast: float = 1.0
    xnview_brightness: int = 0
    xnview_contrast: int = 0
    auto_clip_percent: float = 1.0
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
