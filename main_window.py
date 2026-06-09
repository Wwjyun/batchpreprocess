from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from batch_worker import BatchWorker
from image_io import ImageIO, PixmapConverter
from image_processing import ImagePreprocessor
from params import PreprocessParams
from preview_worker import PreviewWorker

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class MainWindow(QMainWindow):
    def __init__(
        self,
        image_io: ImageIO | None = None,
        preprocessor: ImagePreprocessor | None = None,
        pixmap_converter: PixmapConverter | None = None,
    ):
        super().__init__()
        self.setWindowTitle("AOI 批次影像預處理")
        self.resize(1500, 950)

        self.image_io = image_io or ImageIO()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.pixmap_converter = pixmap_converter or PixmapConverter()

        self.input_dir = ""
        self.output_dir = ""
        self.files = []
        self.current_img = None
        self.current_file = None
        self.worker = None
        self.preview_worker = None
        self.preview_request_id = 0
        self.preview_pending = False

        self.build_ui()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(460)

        path_group = QGroupBox("資料夾")
        path_layout = QGridLayout(path_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        btn_in = QPushButton("選擇輸入")
        btn_out = QPushButton("選擇輸出")
        btn_scan = QPushButton("掃描影像")
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
        left_layout.addWidget(QLabel("影像清單"))
        left_layout.addWidget(self.file_list, stretch=1)

        params_group = QGroupBox("預處理參數")
        params_layout = QGridLayout(params_group)
        row = 0

        self.processing_mode = QComboBox()
        self.processing_mode.addItem("一般手動", "opencv")
        self.processing_mode.addItem("XnView 手動", "xnview")
        self.processing_mode.addItem("自動亮度/對比校正", "auto")
        self.processing_mode.currentIndexChanged.connect(self.on_processing_mode_changed)
        params_layout.addWidget(QLabel("校正模式"), row, 0)
        params_layout.addWidget(self.processing_mode, row, 1, 1, 2)
        row += 1

        self.brightness = QSpinBox()
        self.brightness.setRange(-255, 255)
        self.brightness.setValue(0)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 5.0)
        self.contrast.setDecimals(2)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 500)
        self.contrast_slider.setSingleStep(10)
        self.contrast_slider.setPageStep(25)
        self.contrast_slider.setValue(100)
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        self.brightness.valueChanged.connect(self.update_preview)
        self.contrast.valueChanged.connect(self.on_contrast_spin_changed)
        self.contrast_slider.valueChanged.connect(self.on_contrast_slider_changed)
        self.gamma.valueChanged.connect(self.update_preview)
        params_layout.addWidget(QLabel("亮度"), row, 0)
        params_layout.addWidget(self.brightness, row, 1, 1, 2)
        row += 1
        params_layout.addWidget(QLabel("對比度"), row, 0)
        params_layout.addWidget(self.contrast_slider, row, 1)
        params_layout.addWidget(self.contrast, row, 2)
        row += 1
        params_layout.addWidget(QLabel("伽瑪值"), row, 0)
        params_layout.addWidget(self.gamma, row, 1, 1, 2)
        row += 1

        self.xnview_brightness = QSpinBox()
        self.xnview_brightness.setRange(-255, 255)
        self.xnview_brightness.setValue(0)
        self.xnview_contrast = QSpinBox()
        self.xnview_contrast.setRange(-255, 255)
        self.xnview_contrast.setValue(0)
        self.xnview_contrast_slider = QSlider(Qt.Horizontal)
        self.xnview_contrast_slider.setRange(-255, 255)
        self.xnview_contrast_slider.setSingleStep(1)
        self.xnview_contrast_slider.setPageStep(10)
        self.xnview_contrast_slider.setValue(0)
        self.xnview_brightness.valueChanged.connect(self.update_preview)
        self.xnview_contrast.valueChanged.connect(self.on_xnview_contrast_spin_changed)
        self.xnview_contrast_slider.valueChanged.connect(self.on_xnview_contrast_slider_changed)
        params_layout.addWidget(QLabel("XnView 亮度"), row, 0)
        params_layout.addWidget(self.xnview_brightness, row, 1, 1, 2)
        row += 1
        params_layout.addWidget(QLabel("XnView 對比度"), row, 0)
        params_layout.addWidget(self.xnview_contrast_slider, row, 1)
        params_layout.addWidget(self.xnview_contrast, row, 2)
        row += 1

        self.auto_clip_percent = QDoubleSpinBox()
        self.auto_clip_percent.setRange(0.0, 20.0)
        self.auto_clip_percent.setDecimals(2)
        self.auto_clip_percent.setSingleStep(0.25)
        self.auto_clip_percent.setValue(1.0)
        self.auto_clip_percent.valueChanged.connect(self.update_preview)
        params_layout.addWidget(QLabel("自動裁切百分比"), row, 0)
        params_layout.addWidget(self.auto_clip_percent, row, 1, 1, 2)
        row += 1

        self.use_clahe = QCheckBox("啟用 CLAHE 局部對比增強")
        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(0.1, 20.0)
        self.clahe_clip.setSingleStep(0.1)
        self.clahe_clip.setValue(2.0)
        self.clahe_tile = QSpinBox()
        self.clahe_tile.setRange(2, 64)
        self.clahe_tile.setValue(8)
        self.use_clahe.stateChanged.connect(self.update_preview)
        self.clahe_clip.valueChanged.connect(self.update_preview)
        self.clahe_tile.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_clahe, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(QLabel("CLAHE 裁切限制"), row, 0)
        params_layout.addWidget(self.clahe_clip, row, 1)
        row += 1
        params_layout.addWidget(QLabel("CLAHE 網格大小"), row, 0)
        params_layout.addWidget(self.clahe_tile, row, 1)
        row += 1

        self.use_gaussian = QCheckBox("高斯模糊")
        self.gaussian_k = QSpinBox()
        self.gaussian_k.setRange(1, 31)
        self.gaussian_k.setValue(3)
        self.use_median = QCheckBox("中值模糊")
        self.median_k = QSpinBox()
        self.median_k.setRange(1, 31)
        self.median_k.setValue(3)
        self.use_bilateral = QCheckBox("雙邊濾波")
        self.bilateral_d = QSpinBox()
        self.bilateral_d.setRange(1, 31)
        self.bilateral_d.setValue(5)
        self.bilateral_sc = QSpinBox()
        self.bilateral_sc.setRange(1, 200)
        self.bilateral_sc.setValue(50)
        self.bilateral_ss = QSpinBox()
        self.bilateral_ss.setRange(1, 200)
        self.bilateral_ss.setValue(50)
        for widget in [self.use_gaussian, self.use_median, self.use_bilateral]:
            widget.stateChanged.connect(self.update_preview)
        for widget in [self.gaussian_k, self.median_k, self.bilateral_d, self.bilateral_sc, self.bilateral_ss]:
            widget.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_gaussian, row, 0)
        params_layout.addWidget(self.gaussian_k, row, 1)
        row += 1
        params_layout.addWidget(self.use_median, row, 0)
        params_layout.addWidget(self.median_k, row, 1)
        row += 1
        params_layout.addWidget(self.use_bilateral, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(QLabel("雙邊濾波直徑"), row, 0)
        params_layout.addWidget(self.bilateral_d, row, 1)
        row += 1
        params_layout.addWidget(QLabel("雙邊濾波色彩標準差"), row, 0)
        params_layout.addWidget(self.bilateral_sc, row, 1)
        row += 1
        params_layout.addWidget(QLabel("雙邊濾波空間標準差"), row, 0)
        params_layout.addWidget(self.bilateral_ss, row, 1)
        row += 1

        self.use_sharpen = QCheckBox("銳化")
        self.sharpen_amount = QDoubleSpinBox()
        self.sharpen_amount.setRange(0.0, 5.0)
        self.sharpen_amount.setSingleStep(0.1)
        self.sharpen_amount.setValue(1.0)
        self.use_sharpen.stateChanged.connect(self.update_preview)
        self.sharpen_amount.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_sharpen, row, 0)
        params_layout.addWidget(self.sharpen_amount, row, 1)
        row += 1

        threshold_group = QGroupBox("二值化")
        threshold_layout = QGridLayout(threshold_group)
        trow = 0
        self.use_threshold = QCheckBox("啟用二值化")
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItem("固定閾值", "fixed")
        self.threshold_mode.addItem("OTSU", "otsu")
        self.threshold_mode.addItem("自適應", "adaptive")
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(128)
        self.threshold_invert = QCheckBox("反相")
        self.adaptive_method = QComboBox()
        self.adaptive_method.addItem("Gaussian", "gaussian")
        self.adaptive_method.addItem("Mean", "mean")
        self.adaptive_block = QSpinBox()
        self.adaptive_block.setRange(3, 255)
        self.adaptive_block.setSingleStep(2)
        self.adaptive_block.setValue(31)
        self.adaptive_c = QSpinBox()
        self.adaptive_c.setRange(-50, 50)
        self.adaptive_c.setValue(5)
        threshold_layout.addWidget(self.use_threshold, trow, 0, 1, 2)
        trow += 1
        threshold_layout.addWidget(QLabel("模式"), trow, 0)
        threshold_layout.addWidget(self.threshold_mode, trow, 1)
        trow += 1
        threshold_layout.addWidget(QLabel("固定閾值"), trow, 0)
        threshold_layout.addWidget(self.threshold_value, trow, 1)
        trow += 1
        threshold_layout.addWidget(QLabel("自適應方法"), trow, 0)
        threshold_layout.addWidget(self.adaptive_method, trow, 1)
        trow += 1
        threshold_layout.addWidget(QLabel("區塊大小"), trow, 0)
        threshold_layout.addWidget(self.adaptive_block, trow, 1)
        trow += 1
        threshold_layout.addWidget(QLabel("C 值"), trow, 0)
        threshold_layout.addWidget(self.adaptive_c, trow, 1)
        trow += 1
        threshold_layout.addWidget(self.threshold_invert, trow, 0, 1, 2)
        params_layout.addWidget(threshold_group, row, 0, 1, 3)
        row += 1

        edge_group = QGroupBox("抓邊界")
        edge_layout = QGridLayout(edge_group)
        erow = 0
        self.use_edge_detection = QCheckBox("啟用 Canny 邊界")
        self.canny_low = QSpinBox()
        self.canny_low.setRange(0, 255)
        self.canny_low.setValue(50)
        self.canny_high = QSpinBox()
        self.canny_high.setRange(1, 255)
        self.canny_high.setValue(150)
        self.canny_aperture = QSpinBox()
        self.canny_aperture.setRange(3, 7)
        self.canny_aperture.setSingleStep(2)
        self.canny_aperture.setValue(3)
        edge_layout.addWidget(self.use_edge_detection, erow, 0, 1, 2)
        erow += 1
        edge_layout.addWidget(QLabel("低閾值"), erow, 0)
        edge_layout.addWidget(self.canny_low, erow, 1)
        erow += 1
        edge_layout.addWidget(QLabel("高閾值"), erow, 0)
        edge_layout.addWidget(self.canny_high, erow, 1)
        erow += 1
        edge_layout.addWidget(QLabel("孔徑"), erow, 0)
        edge_layout.addWidget(self.canny_aperture, erow, 1)
        params_layout.addWidget(edge_group, row, 0, 1, 3)
        row += 1

        shape_group = QGroupBox("形狀偵測")
        shape_layout = QGridLayout(shape_group)
        srow = 0
        self.use_shape_detection = QCheckBox("啟用輪廓形狀偵測")
        self.contour_min_area = QSpinBox()
        self.contour_min_area.setRange(0, 100000000)
        self.contour_min_area.setValue(50)
        self.contour_max_area = QSpinBox()
        self.contour_max_area.setRange(1, 100000000)
        self.contour_max_area.setValue(1000000)
        self.approx_epsilon = QDoubleSpinBox()
        self.approx_epsilon.setRange(0.1, 20.0)
        self.approx_epsilon.setSingleStep(0.1)
        self.approx_epsilon.setValue(2.0)
        self.use_subpixel = QCheckBox("啟用亞像素精度")
        self.subpixel_window = QSpinBox()
        self.subpixel_window.setRange(1, 31)
        self.subpixel_window.setValue(5)
        shape_layout.addWidget(self.use_shape_detection, srow, 0, 1, 3)
        srow += 1
        shape_layout.addWidget(QLabel("最小面積"), srow, 0)
        shape_layout.addWidget(self.contour_min_area, srow, 1)
        shape_layout.addWidget(QLabel("最大面積"), srow, 2)
        shape_layout.addWidget(self.contour_max_area, srow, 3)
        srow += 1
        shape_layout.addWidget(QLabel("近似誤差 %"), srow, 0)
        shape_layout.addWidget(self.approx_epsilon, srow, 1)
        shape_layout.addWidget(self.use_subpixel, srow, 2)
        shape_layout.addWidget(self.subpixel_window, srow, 3)
        srow += 1

        self.detect_rectangles = QCheckBox("矩形")
        self.detect_rectangles.setChecked(True)
        self.rect_min_width = QDoubleSpinBox()
        self.rect_min_width.setRange(0.0, 100000.0)
        self.rect_min_width.setValue(5.0)
        self.rect_max_width = QDoubleSpinBox()
        self.rect_max_width.setRange(0.0, 100000.0)
        self.rect_max_width.setValue(100000.0)
        self.rect_min_height = QDoubleSpinBox()
        self.rect_min_height.setRange(0.0, 100000.0)
        self.rect_min_height.setValue(5.0)
        self.rect_max_height = QDoubleSpinBox()
        self.rect_max_height.setRange(0.0, 100000.0)
        self.rect_max_height.setValue(100000.0)
        self.rect_min_aspect = QDoubleSpinBox()
        self.rect_min_aspect.setRange(0.01, 100.0)
        self.rect_min_aspect.setValue(0.1)
        self.rect_max_aspect = QDoubleSpinBox()
        self.rect_max_aspect.setRange(0.01, 100.0)
        self.rect_max_aspect.setValue(10.0)
        shape_layout.addWidget(self.detect_rectangles, srow, 0)
        shape_layout.addWidget(QLabel("寬"), srow, 1)
        shape_layout.addWidget(self.rect_min_width, srow, 2)
        shape_layout.addWidget(self.rect_max_width, srow, 3)
        srow += 1
        shape_layout.addWidget(QLabel("矩形高"), srow, 1)
        shape_layout.addWidget(self.rect_min_height, srow, 2)
        shape_layout.addWidget(self.rect_max_height, srow, 3)
        srow += 1
        shape_layout.addWidget(QLabel("長寬比"), srow, 1)
        shape_layout.addWidget(self.rect_min_aspect, srow, 2)
        shape_layout.addWidget(self.rect_max_aspect, srow, 3)
        srow += 1

        self.detect_circles = QCheckBox("圓形")
        self.detect_circles.setChecked(True)
        self.circle_min_radius = QDoubleSpinBox()
        self.circle_min_radius.setRange(0.0, 100000.0)
        self.circle_min_radius.setValue(3.0)
        self.circle_max_radius = QDoubleSpinBox()
        self.circle_max_radius.setRange(0.0, 100000.0)
        self.circle_max_radius.setValue(100000.0)
        self.circle_min_circularity = QDoubleSpinBox()
        self.circle_min_circularity.setRange(0.01, 1.0)
        self.circle_min_circularity.setSingleStep(0.05)
        self.circle_min_circularity.setValue(0.75)
        shape_layout.addWidget(self.detect_circles, srow, 0)
        shape_layout.addWidget(QLabel("半徑"), srow, 1)
        shape_layout.addWidget(self.circle_min_radius, srow, 2)
        shape_layout.addWidget(self.circle_max_radius, srow, 3)
        srow += 1
        shape_layout.addWidget(QLabel("圓度下限"), srow, 1)
        shape_layout.addWidget(self.circle_min_circularity, srow, 2)
        srow += 1

        self.detect_polygons = QCheckBox("多邊形")
        self.detect_polygons.setChecked(True)
        self.polygon_min_vertices = QSpinBox()
        self.polygon_min_vertices.setRange(3, 100)
        self.polygon_min_vertices.setValue(3)
        self.polygon_max_vertices = QSpinBox()
        self.polygon_max_vertices.setRange(3, 100)
        self.polygon_max_vertices.setValue(12)
        shape_layout.addWidget(self.detect_polygons, srow, 0)
        shape_layout.addWidget(QLabel("頂點數"), srow, 1)
        shape_layout.addWidget(self.polygon_min_vertices, srow, 2)
        shape_layout.addWidget(self.polygon_max_vertices, srow, 3)
        params_layout.addWidget(shape_group, row, 0, 1, 3)
        row += 1

        for widget in [
            self.use_threshold,
            self.threshold_mode,
            self.threshold_value,
            self.threshold_invert,
            self.adaptive_method,
            self.adaptive_block,
            self.adaptive_c,
            self.use_edge_detection,
            self.canny_low,
            self.canny_high,
            self.canny_aperture,
            self.use_shape_detection,
            self.contour_min_area,
            self.contour_max_area,
            self.approx_epsilon,
            self.use_subpixel,
            self.subpixel_window,
            self.detect_rectangles,
            self.rect_min_width,
            self.rect_max_width,
            self.rect_min_height,
            self.rect_max_height,
            self.rect_min_aspect,
            self.rect_max_aspect,
            self.detect_circles,
            self.circle_min_radius,
            self.circle_max_radius,
            self.circle_min_circularity,
            self.detect_polygons,
            self.polygon_min_vertices,
            self.polygon_max_vertices,
        ]:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.update_preview)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.update_preview)
            else:
                widget.valueChanged.connect(self.update_preview)

        self.output_format = QComboBox()
        self.output_format.addItems(["png", "jpg", "bmp", "tif"])
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(95)
        self.png_compression = QSpinBox()
        self.png_compression.setRange(0, 9)
        self.png_compression.setValue(3)
        params_layout.addWidget(QLabel("輸出格式"), row, 0)
        params_layout.addWidget(self.output_format, row, 1)
        row += 1
        params_layout.addWidget(QLabel("JPG 品質"), row, 0)
        params_layout.addWidget(self.jpeg_quality, row, 1)
        row += 1
        params_layout.addWidget(QLabel("PNG 壓縮等級"), row, 0)
        params_layout.addWidget(self.png_compression, row, 1)
        row += 1

        self.recursive = QCheckBox("包含子資料夾")
        self.overwrite = QCheckBox("覆寫既有檔案")
        self.save_meta = QCheckBox("儲存預處理中繼資料 JSON")
        self.save_meta.setChecked(True)
        params_layout.addWidget(self.recursive, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(self.overwrite, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(self.save_meta, row, 0, 1, 2)
        row += 1

        left_layout.addWidget(params_group)

        run_group = QGroupBox("批次處理")
        run_layout = QVBoxLayout(run_group)
        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("預覽")
        self.btn_start = QPushButton("開始")
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
        self.info_label = QLabel("請選擇影像以預覽預處理結果。輸出尺寸不會被改變。")
        right_layout.addWidget(self.info_label)

        splitter = QSplitter(Qt.Horizontal)
        self.raw_label = QLabel("原始影像預覽")
        self.proc_label = QLabel("處理後預覽")
        for label in [self.raw_label, self.proc_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("QLabel { background: #222; color: #ddd; border: 1px solid #555; }")
            label.setMinimumSize(420, 420)
            label.setScaledContents(False)
        splitter.addWidget(self.raw_label)
        splitter.addWidget(self.proc_label)
        splitter.setSizes([1, 1])
        right_layout.addWidget(splitter, stretch=3)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(220)
        right_layout.addWidget(QLabel("紀錄"))
        right_layout.addWidget(self.log_box)

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, stretch=1)

    def params(self) -> PreprocessParams:
        return PreprocessParams(
            processing_mode=self.processing_mode.currentData(),
            brightness=self.brightness.value(),
            contrast=self.contrast.value(),
            xnview_brightness=self.xnview_brightness.value(),
            xnview_contrast=self.xnview_contrast.value(),
            auto_clip_percent=self.auto_clip_percent.value(),
            gamma=self.gamma.value(),
            use_clahe=self.use_clahe.isChecked(),
            clahe_clip_limit=self.clahe_clip.value(),
            clahe_tile_grid_size=self.clahe_tile.value(),
            use_sharpen=self.use_sharpen.isChecked(),
            sharpen_amount=self.sharpen_amount.value(),
            use_gaussian_blur=self.use_gaussian.isChecked(),
            gaussian_ksize=ImagePreprocessor.odd_ksize(self.gaussian_k.value()),
            use_median_blur=self.use_median.isChecked(),
            median_ksize=ImagePreprocessor.odd_ksize(self.median_k.value()),
            use_bilateral_filter=self.use_bilateral.isChecked(),
            bilateral_d=self.bilateral_d.value(),
            bilateral_sigma_color=self.bilateral_sc.value(),
            bilateral_sigma_space=self.bilateral_ss.value(),
            use_threshold=self.use_threshold.isChecked(),
            threshold_mode=self.threshold_mode.currentData(),
            threshold_value=self.threshold_value.value(),
            threshold_invert=self.threshold_invert.isChecked(),
            adaptive_method=self.adaptive_method.currentData(),
            adaptive_block_size=ImagePreprocessor.odd_ksize(self.adaptive_block.value()),
            adaptive_c=self.adaptive_c.value(),
            use_edge_detection=self.use_edge_detection.isChecked(),
            canny_low=self.canny_low.value(),
            canny_high=self.canny_high.value(),
            canny_aperture_size=ImagePreprocessor.odd_ksize(self.canny_aperture.value()),
            use_shape_detection=self.use_shape_detection.isChecked(),
            contour_min_area=self.contour_min_area.value(),
            contour_max_area=self.contour_max_area.value(),
            approx_epsilon_percent=self.approx_epsilon.value(),
            use_subpixel_refine=self.use_subpixel.isChecked(),
            subpixel_window=self.subpixel_window.value(),
            detect_rectangles=self.detect_rectangles.isChecked(),
            rect_min_width=self.rect_min_width.value(),
            rect_max_width=self.rect_max_width.value(),
            rect_min_height=self.rect_min_height.value(),
            rect_max_height=self.rect_max_height.value(),
            rect_min_aspect=self.rect_min_aspect.value(),
            rect_max_aspect=self.rect_max_aspect.value(),
            detect_circles=self.detect_circles.isChecked(),
            circle_min_radius=self.circle_min_radius.value(),
            circle_max_radius=self.circle_max_radius.value(),
            circle_min_circularity=self.circle_min_circularity.value(),
            detect_polygons=self.detect_polygons.isChecked(),
            polygon_min_vertices=self.polygon_min_vertices.value(),
            polygon_max_vertices=self.polygon_max_vertices.value(),
            output_format=self.output_format.currentText(),
            jpeg_quality=self.jpeg_quality.value(),
            png_compression=self.png_compression.value(),
        )

    def log(self, text):
        self.log_box.append(text)

    def on_contrast_spin_changed(self, value):
        slider_value = int(round(float(value) * 100))
        if self.contrast_slider.value() != slider_value:
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(slider_value)
            self.contrast_slider.blockSignals(False)
        self.update_preview()

    def on_contrast_slider_changed(self, value):
        spin_value = round(value / 100.0, 2)
        if abs(self.contrast.value() - spin_value) > 1e-6:
            self.contrast.blockSignals(True)
            self.contrast.setValue(spin_value)
            self.contrast.blockSignals(False)
        self.update_preview()

    def on_xnview_contrast_spin_changed(self, value):
        if self.xnview_contrast_slider.value() != value:
            self.xnview_contrast_slider.blockSignals(True)
            self.xnview_contrast_slider.setValue(value)
            self.xnview_contrast_slider.blockSignals(False)
        self.update_preview()

    def on_xnview_contrast_slider_changed(self, value):
        if self.xnview_contrast.value() != value:
            self.xnview_contrast.blockSignals(True)
            self.xnview_contrast.setValue(value)
            self.xnview_contrast.blockSignals(False)
        self.update_preview()

    def on_processing_mode_changed(self):
        self.update_preview()

    def choose_input(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if folder:
            self.input_dir = folder
            self.input_edit.setText(folder)

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if folder:
            self.output_dir = folder
            self.output_edit.setText(folder)

    def scan_files(self):
        self.input_dir = self.input_edit.text().strip()
        if not self.input_dir or not Path(self.input_dir).exists():
            QMessageBox.warning(self, "缺少輸入", "請先選擇一個存在的輸入資料夾。")
            return

        base = Path(self.input_dir)
        pattern = "**/*" if self.recursive.isChecked() else "*"
        self.files = [str(p) for p in base.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        self.files.sort()

        self.file_list.clear()
        for path in self.files:
            item = QListWidgetItem(str(Path(path).relative_to(base)))
            item.setData(Qt.UserRole, path)
            self.file_list.addItem(item)

        self.log(f"掃描完成：共 {len(self.files)} 張影像。")
        if self.files:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, current, previous):
        if not current:
            return

        path = current.data(Qt.UserRole)
        self.current_file = path
        img = self.image_io.read(path)
        if img is None:
            self.log(f"無法讀取：{path}")
            return

        self.current_img = img
        h, w = img.shape[:2]
        self.info_label.setText(f"影像：{Path(path).name} | 尺寸={w}x{h} | 資料型別={img.dtype} | 輸出不縮放")
        self.raw_label.setPixmap(
            self.pixmap_converter.to_qpixmap(img, self.raw_label.width() - 20, self.raw_label.height() - 20)
        )
        self.update_preview()

    def update_preview(self):
        if self.current_img is None:
            return
        self.preview_request_id += 1
        self.start_preview_worker()

    def start_preview_worker(self):
        if self.current_img is None:
            return

        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_pending = True
            return

        self.preview_pending = False
        self.preview_worker = PreviewWorker(
            self.preview_request_id,
            self.current_img,
            self.params(),
            self.proc_label.width() - 20,
            self.proc_label.height() - 20,
        )
        self.preview_worker.result.connect(self.on_preview_ready)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_ready(self, request_id, out, error):
        if request_id != self.preview_request_id:
            if self.preview_pending:
                self.start_preview_worker()
            return

        if error:
            self.log(f"預覽失敗：{error}")
        elif out is not None:
            self.proc_label.setPixmap(
                self.pixmap_converter.to_qpixmap(out, self.proc_label.width() - 20, self.proc_label.height() - 20)
            )
            name = Path(self.current_file).name if self.current_file else ""
            h, w = self.current_img.shape[:2]
            self.info_label.setText(f"影像：{name} | 處理後尺寸={w}x{h} | 僅預覽縮放")

        if self.preview_pending:
            self.start_preview_worker()

    def on_preview_finished(self):
        worker = self.sender()
        if worker is self.preview_worker:
            self.preview_worker = None
        if worker is not None:
            worker.deleteLater()
        if self.preview_pending:
            self.start_preview_worker()

    def start_batch(self):
        self.input_dir = self.input_edit.text().strip()
        self.output_dir = self.output_edit.text().strip()
        if not self.files:
            QMessageBox.warning(self, "沒有檔案", "開始前請先掃描影像。")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "缺少輸出", "請先選擇輸出資料夾。")
            return
        if Path(self.output_dir).resolve() == Path(self.input_dir).resolve():
            QMessageBox.warning(self, "輸出無效", "輸出資料夾必須和輸入資料夾不同。")
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
            image_io=self.image_io,
            preprocessor=self.preprocessor,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()
        self.log("批次處理已開始。")

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
        self.log(f"批次處理完成：成功={ok}，失敗={fail}")
        QMessageBox.information(self, "批次處理完成", f"成功：{ok}\n失敗：{fail}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_img is not None:
            self.raw_label.setPixmap(
                self.pixmap_converter.to_qpixmap(
                    self.current_img,
                    self.raw_label.width() - 20,
                    self.raw_label.height() - 20,
                )
            )
            self.update_preview()


def run_app(argv=None):
    import sys

    app = QApplication(sys.argv if argv is None else argv)
    win = MainWindow()
    win.show()
    return app.exec()
