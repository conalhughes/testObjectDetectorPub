# Configuration Parameters Reference

This document provides a comprehensive reference for all configurable parameters in `config.py`. All of these parameters can be modified either by:
1. **Editing `config.py` directly** - Changes persist across runs
2. **Using CLI arguments** - Override for a single run (see `CLI_ARGUMENTS.md`)

---

## Model Configuration

### `MODEL_NAME`
- **Type:** String
- **Default:** `"ball_detector"`
- **Description:** Base name for your model. Used in saved model filenames.
- **CLI Argument:** `--model-name`

### `MODEL_VERSION`
- **Type:** String
- **Default:** `"v1"`
- **Description:** Version identifier for the model. Useful for tracking different experiments or iterations.
- **CLI Argument:** `--model-version`
- **Example:** `"v1"`, `"v2"`, `"experiment_1"`

### `MODEL_SIZE`
- **Type:** String
- **Default:** `"n"` (nano)
- **Options:** `'n'` (nano), `'s'` (small), `'m'` (medium), `'l'` (large), `'x'` (xlarge)
- **Description:** YOLOv8 model size. Larger models are more accurate but slower and require more memory.
- **CLI Argument:** `--model-size`
- **Recommendations:**
  - `n` (nano) - Fast inference, good for embedded devices
  - `s` (small) - Balance of speed and accuracy
  - `m` (medium) - Better accuracy, moderate speed
  - `l` (large) - High accuracy, slower
  - `x` (xlarge) - Highest accuracy, slowest

---

## Data Configuration

### `INPUT_SIZE`
- **Type:** Tuple (width, height)
- **Default:** `(640, 360)`
- **Description:** Image dimensions for training.
- **CLI Arguments:** `--img-width`, `--img-height`
- **Notes:** 
  - Must be multiples of 32 for YOLOv8
  - Default matches 16:9 aspect ratio (1280x720 / 2)

### `BATCH_SIZE`
- **Type:** Integer
- **Default:** `16`
- **Description:** Number of images processed in each training step.
- **CLI Argument:** `--batch-size`
- **Recommendations:**
  - Increase if you have more GPU memory
  - Decrease if you get out-of-memory errors
  - Typical values: 8, 16, 32, 64

### `NUM_WORKERS`
- **Type:** Integer
- **Default:** `4`
- **Description:** Number of CPU threads for data loading.
- **CLI Argument:** `--num-workers`
- **Recommendations:**
  - Set to number of CPU cores (or slightly less)
  - Too high can cause memory issues
  - Set to 0 to disable multiprocessing

### `NUM_CLASSES`
- **Type:** Integer
- **Default:** `1`
- **Description:** Number of object classes in your dataset.
- **CLI Argument:** `--num-classes`
- **Example:** Single class (ball) = 1, Multiple classes = 2+

### `CLASS_NAMES`
- **Type:** List of strings
- **Default:** `['ball']`
- **Description:** Names of object classes. Length must match `NUM_CLASSES`.
- **Notes:** Not directly configurable via CLI, edit `config.py`

---

## Training Configuration

### `EPOCHS`
- **Type:** Integer
- **Default:** `100`
- **Description:** Maximum number of training epochs.
- **CLI Argument:** `--epochs`
- **Notes:** Training may stop earlier if early stopping is triggered

### `LEARNING_RATE`
- **Type:** Float
- **Default:** `0.01`
- **Description:** Initial learning rate for the optimizer.
- **CLI Argument:** `--learning-rate`
- **Recommendations:**
  - Start with 0.01 for most cases
  - Decrease (e.g., 0.001) if training is unstable
  - Increase (e.g., 0.02) for faster convergence (with caution)

### `PATIENCE`
- **Type:** Integer
- **Default:** `50`
- **Description:** Number of epochs with no improvement before early stopping.
- **CLI Argument:** `--patience`
- **Notes:** Set to a high value (or disable in code) to prevent early stopping

### `DEVICE`
- **Type:** String
- **Default:** `"cuda"`
- **Options:** `"cuda"` or `"cpu"`
- **Description:** Device for training/inference. Automatically falls back to CPU if CUDA unavailable.
- **CLI Argument:** `--device`
- **Notes:** See `DEVICE_FALLBACK.md` for details on automatic fallback

---

## Data Augmentation (Training-Time)

### `AUGMENT`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Enable/disable all data augmentation during training.
- **CLI Argument:** `--augment` / `--no-augment`

### `MOSAIC`
- **Type:** Float (0.0 to 1.0)
- **Default:** `1.0`
- **Description:** Probability of applying mosaic augmentation (combines 4 images).
- **CLI Argument:** `--mosaic`
- **Recommendations:** Keep at 1.0 for better performance with small datasets

### `HSV_H` (Hue)
- **Type:** Float
- **Default:** `0.015`
- **Description:** Range for hue adjustment in HSV color space.
- **CLI Argument:** `--hsv-h`
- **Range:** 0.0 to 1.0

### `HSV_S` (Saturation)
- **Type:** Float
- **Default:** `0.7`
- **Description:** Range for saturation adjustment in HSV color space.
- **CLI Argument:** `--hsv-s`
- **Range:** 0.0 to 1.0

### `HSV_V` (Value/Brightness)
- **Type:** Float
- **Default:** `0.4`
- **Description:** Range for value/brightness adjustment in HSV color space.
- **CLI Argument:** `--hsv-v`
- **Range:** 0.0 to 1.0

### `DEGREES`
- **Type:** Float
- **Default:** `0.0`
- **Description:** Maximum rotation angle in degrees.
- **CLI Argument:** `--degrees`
- **Example:** `10.0` allows ±10° rotation

### `TRANSLATE`
- **Type:** Float
- **Default:** `0.1`
- **Description:** Maximum translation as fraction of image size.
- **CLI Argument:** `--translate`
- **Range:** 0.0 to 1.0

### `SCALE`
- **Type:** Float
- **Default:** `0.5`
- **Description:** Range for random scaling.
- **CLI Argument:** `--scale`
- **Example:** `0.5` allows scaling from 0.5x to 1.5x

### `SHEAR`
- **Type:** Float
- **Default:** `0.0`
- **Description:** Maximum shear angle in degrees.
- **CLI Argument:** `--shear`

### `PERSPECTIVE`
- **Type:** Float
- **Default:** `0.0`
- **Description:** Strength of perspective transformation.
- **CLI Argument:** `--perspective`
- **Range:** 0.0 to 0.001 (small values recommended)

### `FLIPUD`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.0`
- **Description:** Probability of vertical flip.
- **CLI Argument:** `--flipud`
- **Recommendations:** Keep at 0.0 unless objects can appear upside down

### `FLIPLR`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.5`
- **Description:** Probability of horizontal flip.
- **CLI Argument:** `--fliplr`
- **Recommendations:** `0.5` is good for objects that can appear on either side

---

## Data Preprocessing Configuration

### `TRAIN_RATIO`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.7` (70%)
- **Description:** Proportion of data for training set.
- **CLI Argument:** `--train-ratio`
- **Notes:** Must sum to 1.0 with `VAL_RATIO` and `TEST_RATIO`

### `VAL_RATIO`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.2` (20%)
- **Description:** Proportion of data for validation set.
- **CLI Argument:** `--val-ratio`

### `TEST_RATIO`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.1` (10%)
- **Description:** Proportion of data for test set.
- **CLI Argument:** `--test-ratio`

### `RESIZE_IMAGES`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Resize images during preprocessing.
- **Notes:** Not directly configurable via CLI

### `TARGET_SIZE`
- **Type:** Tuple (width, height)
- **Default:** `(544, 448)`
- **Description:** Target size for image resizing during preprocessing.
- **Notes:** Should match `INPUT_SIZE`

### `MAINTAIN_ASPECT_RATIO`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Maintain aspect ratio when resizing (pad with black).
- **Notes:** Not directly configurable via CLI

### `IMAGE_QUALITY`
- **Type:** Integer (1-100)
- **Default:** `95`
- **Description:** JPEG quality for saved images.
- **Notes:** Not directly configurable via CLI

### `REMOVE_EMPTY_LABELS`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Remove images with no annotations during preprocessing.
- **Notes:** Not directly configurable via CLI

### `VERIFY_LABELS`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Verify label format and coordinates are valid.
- **Notes:** Not directly configurable via CLI

### `MIN_BOX_SIZE`
- **Type:** Float
- **Default:** `0.01`
- **Description:** Minimum bounding box size (normalized) to keep during preprocessing.
- **Notes:** Not directly configurable via CLI

### `PREPROCESS_AUGMENT`
- **Type:** Boolean
- **Default:** `False`
- **Description:** Apply augmentation during data preprocessing (separate from training augmentation).
- **Notes:** Not directly configurable via CLI

### `AUGMENT_FACTOR`
- **Type:** Integer
- **Default:** `1`
- **Description:** Number of augmented versions per image if `PREPROCESS_AUGMENT` is enabled.
- **Notes:** Not directly configurable via CLI

---

## Directory Paths

### `RAW_DATA_DIR`
- **Type:** String
- **Default:** `"./raw_data"`
- **Description:** Directory containing original, unprocessed data.

### `RAW_IMAGES`
- **Type:** String
- **Default:** `"./raw_data/images"`
- **Description:** Original images before preprocessing.

### `RAW_LABELS`
- **Type:** String
- **Default:** `"./raw_data/labels"`
- **Description:** Original labels (YOLO format) before preprocessing.

### `BACKUP_DIR`
- **Type:** String
- **Default:** `"./backup"`
- **Description:** Backup location for previous data splits.

### `DATA_DIR`
- **Type:** String
- **Default:** `"./data"`
- **Description:** Root directory for processed, split data.

### `TRAIN_IMAGES` / `TRAIN_LABELS`
- **Type:** String
- **Default:** `"./data/train/images"` / `"./data/train/labels"`
- **Description:** Training set images and labels.

### `VAL_IMAGES` / `VAL_LABELS`
- **Type:** String
- **Default:** `"./data/val/images"` / `"./data/val/labels"`
- **Description:** Validation set images and labels.

### `TEST_IMAGES` / `TEST_LABELS`
- **Type:** String
- **Default:** `"./data/test/images"` / `"./data/test/labels"`
- **Description:** Test set images and labels.

### `STATS_DIR`
- **Type:** String
- **Default:** `"./stats"`
- **Description:** Directory for dataset statistics and visualizations.

### `VERSION_STATS_DIR`
- **Type:** String
- **Default:** `"./stats/{MODEL_VERSION}"`
- **Description:** Version-specific statistics folder.

### `MODELS_DIR`
- **Type:** String
- **Default:** `"./models"`
- **Description:** Directory for saved model weights.

### `RUNS_DIR`
- **Type:** String
- **Default:** `"./runs"`
- **Description:** YOLOv8 training runs output directory.

### `MODEL_SAVE_PATH`
- **Type:** String
- **Default:** `"./models/{MODEL_NAME}_{MODEL_VERSION}.pt"`
- **Description:** Path for final saved model.

### `BEST_MODEL_PATH`
- **Type:** String
- **Default:** `"./models/{MODEL_NAME}_{MODEL_VERSION}_best.pt"`
- **Description:** Path for best model checkpoint.

---

## Validation & Testing Configuration

### `CONF_THRESHOLD`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.25`
- **Description:** Minimum confidence score for a detection to be considered valid.
- **CLI Argument:** `--conf-threshold`
- **Recommendations:**
  - Increase (e.g., 0.5) to reduce false positives
  - Decrease (e.g., 0.1) to catch more detections (more false positives)

### `IOU_THRESHOLD`
- **Type:** Float (0.0 to 1.0)
- **Default:** `0.45`
- **Description:** IoU threshold for Non-Maximum Suppression (NMS).
- **CLI Argument:** `--iou-threshold`
- **Recommendations:**
  - Increase (e.g., 0.7) to allow overlapping detections
  - Decrease (e.g., 0.3) to suppress more overlaps

### `SAVE_PREDICTIONS`
- **Type:** Boolean
- **Default:** `True`
- **Description:** Save visualization images with prediction boxes.
- **Notes:** Not directly configurable via CLI

### `MAX_DET`
- **Type:** Integer
- **Default:** `300`
- **Description:** Maximum number of detections per image.
- **CLI Argument:** `--max-det`
- **Notes:** Increase if you expect many objects in a single image

---

## Plotting Configuration

### `PLOT_DPI`
- **Type:** Integer
- **Default:** `150`
- **Description:** DPI (dots per inch) for saved plots and visualizations.
- **Notes:** Not directly configurable via CLI
- **Recommendations:** Higher values (300) for publication-quality plots

### `PLOT_STYLE`
- **Type:** String
- **Default:** `'seaborn-v0_8-darkgrid'`
- **Description:** Matplotlib style for plots.
- **Notes:** Not directly configurable via CLI

---

## Quick Reference: Most Commonly Modified Parameters

### For Experimentation
- `MODEL_VERSION` - Track different experiments
- `EPOCHS` - Control training duration
- `BATCH_SIZE` - Adjust for your GPU memory
- `LEARNING_RATE` - Fine-tune training speed
- `MODEL_SIZE` - Balance accuracy vs. speed

### For Data Quality
- `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` - Adjust data splits
- `AUGMENT` - Enable/disable augmentation
- `FLIPLR`, `MOSAIC` - Control specific augmentations

### For Inference
- `CONF_THRESHOLD` - Filter low-confidence detections
- `IOU_THRESHOLD` - Control overlap suppression
- `DEVICE` - Choose CPU or GPU

### For Performance
- `NUM_WORKERS` - Speed up data loading
- `BATCH_SIZE` - Optimize GPU utilization
- `INPUT_SIZE` - Balance speed vs. accuracy

---

## How to Modify Parameters

### Method 1: Edit config.py (Persistent)
```python
# config.py
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

### Method 2: CLI Arguments (Single Run)
```bash
./train.sh --epochs 200 --batch-size 32 --learning-rate 0.001
```

### Method 3: Combination
Edit `config.py` for default values, override specific parameters via CLI:
```bash
# Use config.py defaults but change just the learning rate
./train.sh --learning-rate 0.005
```

---

## See Also
- **CLI_ARGUMENTS.md** - Complete CLI arguments reference with examples
- **QUICKSTART.md** - Quick start guide with common parameter combinations
- **INDEX.md** - Documentation navigation hub
- **DEVICE_FALLBACK.md** - GPU/CPU device selection details
