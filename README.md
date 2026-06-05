# AOI Batch Preprocess

PySide6 desktop GUI for batch image preprocessing before annotation or model training.
It previews original and processed images side by side, then batch-exports processed images
without resizing the output dimensions.

## Features

- Select input and output folders.
- Scan images recursively or only the selected folder.
- Preview original and processed images side by side.
- Batch-process supported image files while preserving relative folder structure.
- Keep output width and height unchanged.
- Normalize non-uint8 sources to uint8 for annotation visibility.
- Optional per-image metadata JSON under `preprocess_meta/`.
- Unicode Windows path support through OpenCV buffer-based IO.

Supported input extensions:

```text
.jpg, .jpeg, .png, .bmp, .tif, .tiff
```

## Project Structure

```text
batch_process/
  main.py                       # Preferred application entry point
  aoi_batch_preprocess_gui.py   # Compatibility launcher for the old command
  main_window.py                # PySide6 GUI and user workflow
  batch_worker.py               # QThread batch execution
  image_processing.py           # OOP preprocessing pipeline
  image_io.py                   # Image read/write and QPixmap conversion
  params.py                     # PreprocessParams dataclass
  requirements.txt              # Runtime dependencies
```

## Requirements

- Python 3.10+
- Windows desktop session for the PySide6 GUI

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you are using the shared workspace virtual environment from the parent folder:

```powershell
..\env\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Preferred entry point:

```powershell
python main.py
```

Using the parent workspace virtual environment:

```powershell
..\env\Scripts\python.exe main.py
```

The legacy command still works:

```powershell
python aoi_batch_preprocess_gui.py
```

## Processing Pipeline

The pipeline is implemented by `ImagePreprocessor` in `image_processing.py`:

1. Convert source image to uint8 when needed.
2. Apply brightness and contrast with OpenCV `convertScaleAbs`.
3. Apply gamma correction.
4. Optionally apply CLAHE local contrast enhancement.
5. Optionally apply Gaussian blur, median blur, or bilateral filter.
6. Optionally apply sharpening.

Preview images may be scaled down for display only. Exported images keep their processed
pixel dimensions equal to the source image dimensions.

## Metadata Output

When `Save preprocess_meta JSON` is enabled, the app writes one JSON file per output image
under:

```text
<output folder>/preprocess_meta/
```

Each metadata file records source path, output path, image dimensions, source/output dtype,
preprocess parameters, and processing time.

## Development Checks

Syntax check:

```powershell
python -m py_compile main.py aoi_batch_preprocess_gui.py params.py image_processing.py image_io.py batch_worker.py main_window.py
```

Import smoke test:

```powershell
python -B -c "from main_window import MainWindow; from image_processing import ImagePreprocessor; from params import PreprocessParams; print('imports ok')"
```

Processing smoke test:

```powershell
python -B -c "import numpy as np; from image_processing import ImagePreprocessor; from params import PreprocessParams; img=np.arange(48, dtype=np.uint8).reshape(4,4,3); out=ImagePreprocessor().process(img, PreprocessParams(brightness=5, contrast=1.2)); assert out.shape == img.shape; assert out.dtype == np.uint8; print('smoke ok')"
```
