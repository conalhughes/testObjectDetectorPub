# Quick Start Guide - YOLOv8 Ball Detector

## 🚀 One-Command Training

Run the complete pipeline with a single command:

```bash
./train.sh
```

This will:
1. ✅ Create virtual environment (`.venv`)
2. ✅ Install all dependencies
3. ✅ Preprocess your data (resize, split, validate)
4. ✅ Train the model

**Estimated time:** 30-60 minutes with GPU, several hours with CPU

---

## Training Options

### Pipeline Options

**Interactive mode (default):**
```bash
./train.sh
```
Prompts if processed data exists - choose to reuse or re-split.

**Force re-processing:**
```bash
./train.sh --reprocess
```
Clears `data/` and creates fresh random split from `raw_data/`.

**Use existing data:**
```bash
./train.sh --use-existing
```
Skips preprocessing prompt, uses existing split.

### Training Parameters (CLI Arguments)

Override any config setting via command line:

```bash
# Change epochs and batch size
./train.sh --epochs 200 --batch-size 32

# Try different model size
./train.sh --model-size s --learning-rate 0.001

# Train new version with custom settings
./train.sh --model-version v2 --epochs 150 --model-size m

# Combine pipeline and training options
./train.sh --reprocess --model-version exp1 --batch-size 8
```

📖 **See [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) for complete list of arguments.**

Common options:
- `--epochs N` - Training epochs (default: 100)
- `--batch-size N` - Batch size (default: 16)  
- `--learning-rate LR` - Learning rate (default: 0.01)
- `--model-size SIZE` - Model size: n, s, m, l, x (default: n)
- `--model-version VER` - Version name (default: v1)
- `--device DEVICE` - cuda or cpu (default: cuda with auto-fallback)
- `--img-width W --img-height H` - Image size (default: 640x360)

---

## 🔄 When to Re-Process Data

**Use existing split when:**
- ✅ Comparing different model sizes or hyperparameters
- ✅ Resuming training after interruption
- ✅ You want consistent train/val/test splits

**Re-process when:**
- ⚠️ You've added/removed images from `raw_data/`
- ⚠️ You've updated labels
- ⚠️ You want different random split for validation
- ⚠️ Testing data augmentation variations

**Note:** Each preprocessing creates a new random 70/20/10 split. For reproducible experiments, use the same split (option 1).

---

## 📊 Testing the Model

After training completes, test the model:

```bash
./test.sh
```

This evaluates the model on the test set and generates:
- Prediction visualizations
- Performance metrics (mAP, precision, recall)
- Detailed test report

---

## 📁 What Gets Created

### During Training:
```
.venv/                          # Virtual environment
backup/                         # Backup of raw data
data/                          # Processed dataset
  ├── train/                   # Training data (70%)
  ├── val/                     # Validation data (20%)
  └── test/                    # Test data (10%)
models/
  └── ball_detector_v1_best.pt # Best trained model
stats/v1/                      # All plots and reports
  ├── preprocessing_report_v1_*.png
  ├── dataset_statistics_v1.png
  ├── training_results_v1.png
  └── *.log                    # Log files
runs/                          # Full training details
```

---

## 🔧 Manual Commands (Alternative)

If you prefer to run steps manually:

### 1. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
source .venv/bin/activate
python preprocess_data.py
```

### 3. Train Model
```bash
source .venv/bin/activate
python train.py
```

### 4. Test Model
```bash
source .venv/bin/activate
python test.py
```

---

## ⚙️ Customization

Before running `./train.sh`, you can edit `config.py` to:

- Change model size: `MODEL_SIZE = 's'` (or 'm', 'l', 'x')
- Adjust epochs: `EPOCHS = 200`
- Modify batch size: `BATCH_SIZE = 32`
- Change split ratios: `TRAIN_RATIO = 0.8`

---

## 🐛 Troubleshooting

**Virtual environment issues:**
```bash
rm -rf .venv
./train.sh  # Will recreate venv
```

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in `config.py` (try 8 or 4)
- Or use smaller model: `MODEL_SIZE = 'n'`

**Check GPU:**
```bash
nvidia-smi
```

**View logs:**
```bash
cat stats/v1/*.log
```

---

## 📊 Dataset Info

Your dataset after preprocessing:
- **Total images:** 2,570
- **Total annotations:** 3,022 balls
- **Training:** ~1,799 images
- **Validation:** ~514 images
- **Test:** ~257 images

All class IDs have been converted to 0 for single-class detection.

---

## ✅ Ready to Go!

Just run:
```bash
./train.sh
```

Sit back and let it train! 🚀
