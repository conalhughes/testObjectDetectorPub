# YOLOv8 Object Detection - Ball Detector

A comprehensive YOLOv8-based object detection project with complete data processing, training, validation, and testing pipelines.

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[CLI_ARGUMENTS.md](CLI_ARGUMENTS.md)** - Command-line arguments for training/testing
- **[DATA_MANAGEMENT.md](DATA_MANAGEMENT.md)** - Data preprocessing and split management  
- **[DEVICE_FALLBACK.md](DEVICE_FALLBACK.md)** - GPU/CPU automatic fallback

## 🚀 Quick Start

```bash
# 1. Place your images and labels in raw_data/
# 2. Run training (creates venv, installs deps, trains model)
./train.sh

# 3. Test the model
./test.sh
```

**That's it!** See [QUICKSTART.md](QUICKSTART.md) for details.

---

## Project Structure

```
testBallDetector/
├── config.py                 # Global configuration (model name, version, hyperparameters)
├── preprocess_data.py       # Data preprocessing and splitting script
├── data_utils.py            # Data processing and validation utilities
├── train.py                 # Training pipeline
├── test.py                  # Testing and evaluation pipeline
├── requirements.txt         # Python dependencies
│
├── raw_data/                # Raw dataset (place your data here!)
│   ├── images/             # All your raw images
│   └── labels/             # All your raw labels (YOLO format)
│
├── backup/                  # Automatic backups of raw data
│   └── raw_data_backup_{timestamp}/
│
├── data/                    # Processed and split dataset (auto-generated)
│   ├── data.yaml           # YOLO data configuration (auto-generated)
│   ├── train/
│   │   ├── images/         # Training images (processed)
│   │   └── labels/         # Training labels
│   ├── val/
│   │   ├── images/         # Validation images (processed)
│   │   └── labels/         # Validation labels
│   └── test/
│       ├── images/         # Test images (processed)
│       └── labels/         # Test labels
│
├── models/                  # Saved models
│   ├── {model_name}_{version}.pt
│   └── {model_name}_{version}_best.pt
│
├── runs/                    # Training run outputs
│   └── {model_name}_{version}_{timestamp}/
│       ├── weights/
│       ├── results.csv
│       └── plots/
│
└── stats/                   # Statistics and visualization plots
    └── {version}/           # Version-specific subfolder (e.g., v1/)
        ├── preprocessing_report_{version}_{timestamp}.png
        ├── preprocessing_summary_{version}_{timestamp}.txt
        ├── dataset_statistics_{version}.png
        ├── sample_images_{version}.png
        ├── training_results_{version}.png
        ├── test_predictions_{version}.png
        ├── prediction_statistics_{version}.png
        ├── training_summary_{version}.txt
        ├── test_report_{version}_{timestamp}.txt
        └── {module}_{timestamp}.log  # Log files
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Raw Dataset

Place all your images and labels in the `raw_data/` directory:

- **Images**: `raw_data/images/` - All your images in one folder
- **Labels**: `raw_data/labels/` - Corresponding label files (YOLO format)

**Label Format (YOLO format):**
Each `.txt` file should contain one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all values are normalized (0-1) relative to image dimensions.

**Example** (`image001.txt`):
```
0 0.5 0.5 0.2 0.3
```

**Supported image formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`
**Note**: Label filename must match image filename (except extension)

### 3. Configure Your Project (Optional)

**Default config works out of the box**, but you can customize settings in `config.py` or via command-line arguments.

**Via config file:**
```python
# config.py
EPOCHS = 100
BATCH_SIZE = 16
MODEL_SIZE = "n"  # n, s, m, l, x
```

**Via command-line:**
```bash
./train.sh --epochs 200 --batch-size 32 --model-size s
```

See [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) for all available options.

---

### Quick Start (Recommended)

**One-command training:**
```bash
./train.sh
```

This automated script will:
1. ✅ Create virtual environment
2. ✅ Install dependencies
3. ✅ Preprocess data (or use existing split)
4. ✅ Train the model

**Options:**
- `./train.sh` - Interactive mode (prompts if data exists)
- `./train.sh --reprocess` - Force new random split from raw_data
- `./train.sh --use-existing` - Use existing data split without prompting

See [QUICKSTART.md](QUICKSTART.md) for more details.

---

## Usage

### Quick Start (Recommended)

```bash
./train.sh              # Interactive mode
./train.sh --reprocess  # Force new data split
./train.sh --epochs 200 --batch-size 32  # Override config
```

See [QUICKSTART.md](QUICKSTART.md) for full guide and [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) for all options.

### Testing

```bash
./test.sh                    # Test default model (v1)
./test.sh --model-version v2  # Test specific version
```

---

## Advanced Usage

### Manual Workflow

If you prefer step-by-step control over the automated `./train.sh`:

### Step 1: Preprocess Dataset

```bash
python preprocess_data.py
```

Creates backup, validates labels, resizes images, splits into train/val/test (70/20/10).

📖 **Data Management**: See [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) for:
- When to reprocess vs reuse data splits
- How to create fresh random splits
- Managing multiple experiments with same split

**Outputs**: `data/train/`, `data/val/`, `data/test/`, `backup/`, stats/v1/preprocessing_*.png

---

### Step 2: Validate Data (Optional)

```bash
python data_utils.py
```

Generates dataset statistics and visualizations.

---

### Step 3: Train the Model

```bash
python train.py                                  # Use config.py defaults
python train_cli.py --epochs 200 --batch-size 32 # Override via CLI
```

📖 **CLI Options**: See [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) for all parameters.  
📖 **Device Fallback**: See [DEVICE_FALLBACK.md](DEVICE_FALLBACK.md) for GPU/CPU handling.

**Outputs**: `models/*_best.pt`, `stats/*/training_results_*.png`, `runs/`

---

### Step 4: Test the Model

```bash
python test.py                                # Use config.py defaults
python test_cli.py --model-version v2         # Test specific version
```

**Outputs**: `stats/*/test_predictions_*.png`, `stats/*/test_results_*.json`

---

## Model Sizes

The YOLOv8 model comes in 5 sizes:

| Size | Parameters | Speed | Accuracy | Use Case |
|------|------------|-------|----------|----------|
| `n` (nano) | 3.2M | Fastest | Good | Edge devices, real-time |
| `s` (small) | 11.2M | Very Fast | Better | Mobile devices |
| `m` (medium) | 25.9M | Fast | Good | General purpose |
| `l` (large) | 43.7M | Moderate | Excellent | High accuracy needs |
| `x` (xlarge) | 68.2M | Slower | Best | Maximum accuracy |

Choose based on your speed/accuracy requirements and set in `config.py`.

## Complete Workflow

1. **Prepare raw data** → Place all images and labels in `raw_data/`
2. **Configure settings** → Edit `config.py` with your preferences
3. **Preprocess data** → Run `python preprocess_data.py`
4. **Validate (optional)** → Run `python data_utils.py`
5. **Train model** → Run `python train.py`
6. **Test model** → Run `python test.py`
7. **Review results** → Check `stats/` folder for all reports and plots

## Tips for Best Results

1. **Dataset Quality**
   - Ensure high-quality annotations
   - Balance class distribution when possible
   - Include diverse examples (lighting, angles, sizes)
   - Use at least 100-500 images per class for good results

2. **Training**
   - Start with a small model (`n` or `s`) for faster experimentation
   - Use a larger model (`m`, `l`, or `x`) for final production
   - Monitor validation metrics to detect overfitting
   - Adjust learning rate if training is unstable

3. **Augmentation**
   - Enable augmentation for small datasets
   - Reduce augmentation if training is unstable
   - Adjust augmentation parameters in `config.py`

4. **Hardware**
   - GPU recommended for training (CUDA-enabled)
   - CPU training is slow but possible for small datasets
   - Reduce batch size if running out of memory

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No raw images found | Place images in `raw_data/images/` (supported: jpg, jpeg, png, bmp) |
| Preprocessing fails | Check label format, ensure filenames match, verify split ratios sum to 1.0 |
| CUDA out of memory | Reduce batch size, use smaller model, or reduce image size |
| Model not found | Run `./train.sh` first to train a model |
| Low accuracy | Increase epochs, use larger model, check label quality, add more data |
| Training on CPU slow | Expected - use `--device cuda` on GPU machine for 10-20x speedup |

For detailed help, see documentation files listed at the top.

---

## License & Citation

This project uses Ultralytics YOLOv8 (AGPL-3.0).

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```
