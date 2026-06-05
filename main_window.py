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

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class MainWindow(QMainWindow):
    def __init__(
        self,
        image_io: ImageIO | None = None,
        preprocessor: ImagePreprocessor | None = None,
        pixmap_converter: PixmapConverter | None = None,
    ):
        super().__init__()
        self.setWindowTitle("AOI Batch Preprocess")
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

        self.build_ui()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(420)

        path_group = QGroupBox("Folders")
        path_layout = QGridLayout(path_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        btn_in = QPushButton("Browse input")
        btn_out = QPushButton("Browse output")
        btn_scan = QPushButton("Scan images")
        btn_in.clicked.connect(self.choose_input)
        btn_out.clicked.connect(self.choose_output)
        btn_scan.clicked.connect(self.scan_files)
        path_layout.addWidget(QLabel("Input folder"), 0, 0)
        path_layout.addWidget(self.input_edit, 0, 1)
        path_layout.addWidget(btn_in, 0, 2)
        path_layout.addWidget(QLabel("Output folder"), 1, 0)
        path_layout.addWidget(self.output_edit, 1, 1)
        path_layout.addWidget(btn_out, 1, 2)
        path_layout.addWidget(btn_scan, 2, 0, 1, 3)
        left_layout.addWidget(path_group)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        left_layout.addWidget(QLabel("Images"))
        left_layout.addWidget(self.file_list, stretch=1)

        params_group = QGroupBox("Preprocess Parameters")
        params_layout = QGridLayout(params_group)
        row = 0

        self.brightness = QSpinBox()
        self.brightness.setRange(-255, 255)
        self.brightness.setValue(0)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 5.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        for widget in [self.brightness, self.contrast, self.gamma]:
            widget.valueChanged.connect(self.update_preview)
        params_layout.addWidget(QLabel("Brightness beta"), row, 0)
        params_layout.addWidget(self.brightness, row, 1)
        row += 1
        params_layout.addWidget(QLabel("Contrast alpha"), row, 0)
        params_layout.addWidget(self.contrast, row, 1)
        row += 1
        params_layout.addWidget(QLabel("Gamma"), row, 0)
        params_layout.addWidget(self.gamma, row, 1)
        row += 1

        self.use_clahe = QCheckBox("CLAHE local contrast")
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
        params_layout.addWidget(QLabel("CLAHE clipLimit"), row, 0)
        params_layout.addWidget(self.clahe_clip, row, 1)
        row += 1
        params_layout.addWidget(QLabel("CLAHE tileGrid"), row, 0)
        params_layout.addWidget(self.clahe_tile, row, 1)
        row += 1

        self.use_gaussian = QCheckBox("Gaussian Blur")
        self.gaussian_k = QSpinBox()
        self.gaussian_k.setRange(1, 31)
        self.gaussian_k.setValue(3)
        self.use_median = QCheckBox("Median Blur")
        self.median_k = QSpinBox()
        self.median_k.setRange(1, 31)
        self.median_k.setValue(3)
        self.use_bilateral = QCheckBox("Bilateral Filter")
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
        params_layout.addWidget(QLabel("Bilateral d"), row, 0)
        params_layout.addWidget(self.bilateral_d, row, 1)
        row += 1
        params_layout.addWidget(QLabel("Bilateral sigmaColor"), row, 0)
        params_layout.addWidget(self.bilateral_sc, row, 1)
        row += 1
        params_layout.addWidget(QLabel("Bilateral sigmaSpace"), row, 0)
        params_layout.addWidget(self.bilateral_ss, row, 1)
        row += 1

        self.use_sharpen = QCheckBox("Sharpen")
        self.sharpen_amount = QDoubleSpinBox()
        self.sharpen_amount.setRange(0.0, 5.0)
        self.sharpen_amount.setSingleStep(0.1)
        self.sharpen_amount.setValue(1.0)
        self.use_sharpen.stateChanged.connect(self.update_preview)
        self.sharpen_amount.valueChanged.connect(self.update_preview)
        params_layout.addWidget(self.use_sharpen, row, 0)
        params_layout.addWidget(self.sharpen_amount, row, 1)
        row += 1

        self.output_format = QComboBox()
        self.output_format.addItems(["png", "jpg", "bmp", "tif"])
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(95)
        self.png_compression = QSpinBox()
        self.png_compression.setRange(0, 9)
        self.png_compression.setValue(3)
        params_layout.addWidget(QLabel("Output format"), row, 0)
        params_layout.addWidget(self.output_format, row, 1)
        row += 1
        params_layout.addWidget(QLabel("JPG quality"), row, 0)
        params_layout.addWidget(self.jpeg_quality, row, 1)
        row += 1
        params_layout.addWidget(QLabel("PNG compression"), row, 0)
        params_layout.addWidget(self.png_compression, row, 1)
        row += 1

        self.recursive = QCheckBox("Scan recursively")
        self.overwrite = QCheckBox("Overwrite existing files")
        self.save_meta = QCheckBox("Save preprocess_meta JSON")
        self.save_meta.setChecked(True)
        params_layout.addWidget(self.recursive, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(self.overwrite, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(self.save_meta, row, 0, 1, 2)
        row += 1

        left_layout.addWidget(params_group)

        run_group = QGroupBox("Batch")
        run_layout = QVBoxLayout(run_group)
        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_start = QPushButton("Start")
        self.btn_cancel = QPushButton("Cancel")
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
        self.info_label = QLabel("Select an image to preview preprocessing. Output size is unchanged.")
        right_layout.addWidget(self.info_label)

        splitter = QSplitter(Qt.Horizontal)
        self.raw_label = QLabel("Original preview")
        self.proc_label = QLabel("Processed preview")
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
            gaussian_ksize=ImagePreprocessor.odd_ksize(self.gaussian_k.value()),
            use_median_blur=self.use_median.isChecked(),
            median_ksize=ImagePreprocessor.odd_ksize(self.median_k.value()),
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
        folder = QFileDialog.getExistingDirectory(self, "Choose input folder")
        if folder:
            self.input_dir = folder
            self.input_edit.setText(folder)

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.output_dir = folder
            self.output_edit.setText(folder)

    def scan_files(self):
        self.input_dir = self.input_edit.text().strip()
        if not self.input_dir or not Path(self.input_dir).exists():
            QMessageBox.warning(self, "Missing input", "Choose an existing input folder first.")
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

        self.log(f"Scan complete: {len(self.files)} images.")
        if self.files:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, current, previous):
        if not current:
            return

        path = current.data(Qt.UserRole)
        self.current_file = path
        img = self.image_io.read(path)
        if img is None:
            self.log(f"Could not read: {path}")
            return

        self.current_img = img
        h, w = img.shape[:2]
        self.info_label.setText(f"Image: {Path(path).name} | size={w}x{h} | dtype={img.dtype} | output is not resized")
        self.raw_label.setPixmap(
            self.pixmap_converter.to_qpixmap(img, self.raw_label.width() - 20, self.raw_label.height() - 20)
        )
        self.update_preview()

    def update_preview(self):
        if self.current_img is None:
            return
        try:
            out = self.preprocessor.process(self.current_img, self.params())
            h, w = out.shape[:2]
            self.proc_label.setPixmap(
                self.pixmap_converter.to_qpixmap(out, self.proc_label.width() - 20, self.proc_label.height() - 20)
            )
            name = Path(self.current_file).name if self.current_file else ""
            self.info_label.setText(f"Image: {name} | processed size={w}x{h} | preview scaling only")
        except Exception as exc:
            self.log(f"Preview failed: {exc}")

    def start_batch(self):
        self.input_dir = self.input_edit.text().strip()
        self.output_dir = self.output_edit.text().strip()
        if not self.files:
            QMessageBox.warning(self, "No files", "Scan images before starting.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Missing output", "Choose an output folder first.")
            return
        if Path(self.output_dir).resolve() == Path(self.input_dir).resolve():
            QMessageBox.warning(self, "Invalid output", "Output folder must be different from input folder.")
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
        self.log("Batch started.")

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
        self.log(f"Batch finished: ok={ok}, failed={fail}")
        QMessageBox.information(self, "Batch finished", f"Success: {ok}\nFailed: {fail}")

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
