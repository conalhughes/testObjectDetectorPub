# Kaggle Setup Guide

This guide explains how to use your ball detector on Kaggle with free GPU access.

## Why Use Kaggle?

- ✅ **Free GPU Access** - Tesla P100 GPUs (30 hours/week)
- ✅ **No Local Storage** - Host dataset on Kaggle, save disk space
- ✅ **Cloud Training** - Train anywhere, download models when done
- ✅ **Same Codebase** - Use identical shell scripts as locally

## Prerequisites

1. **Kaggle Account** - Sign up at [kaggle.com](https://www.kaggle.com)
2. **Your Dataset** - Images and labels ready to upload
3. **This Repository** - Pushed to GitHub

---

## Step-by-Step Setup

### 1. Prepare Your Dataset Locally

Make sure your local `raw_data/` is organized correctly:

```bash
raw_data/
├── images/
│   ├── ball_001.jpg
│   ├── ball_002.jpg
│   └── ...
└── labels/
    ├── ball_001.txt
    ├── ball_002.txt
    └── ...
```

### 2. Create Kaggle Dataset

**Option A: Upload via Kaggle Website**

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your `raw_data/images/` folder
4. Upload your `raw_data/labels/` folder
5. Set Title: "Ball Detector Training Data" (or your preference)
6. Set visibility (Private recommended for your data)
7. Click "Create"

**Option B: Upload via Kaggle API**

```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
# Download kaggle.json from your Kaggle account settings
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "ball-detector-training-data",
  "id": "yourusername/ball-detector-training-data",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Create dataset
cd raw_data
kaggle datasets create -p . -r zip
```

### 3. Push Your Code to GitHub

```bash
# Make sure your code is pushed
git add .
git commit -m "Add Kaggle support"
git push origin main
```

### 4. Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Settings (right sidebar):
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Language:** Python
   - **Environment:** Latest available

4. Add your dataset:
   - Click "+ Add Data" (right sidebar)
   - Search for your dataset
   - Click "Add"

5. Copy the notebook template:
   - Upload `KAGGLE_SETUP.ipynb` OR
   - Copy-paste cells from the template below

### 5. Run Training on Kaggle

Follow the notebook cells in order:

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/RE-ObjectDetector-NSFR.git
%cd RE-ObjectDetector-NSFR

# 2. Check GPU
!nvidia-smi

# 3. Setup dataset (replace 'your-dataset-name')
!python setup_kaggle_data.py /kaggle/input/ball-detector-training-data

# 4. Train!
!./train.sh --device cuda --epochs 100
```

### 6. Download Your Trained Model

After training completes:
1. Check the "Output" tab in the right sidebar
2. Navigate to `models/` folder
3. Download `ball_detector_v1_best.pt`
4. Use it locally or on your robot!

---

## Usage Examples

### Quick Training (30 minutes)
```python
!./train.sh --epochs 30 --batch-size 32 --patience 10 --device cuda
```

### Full Training (2-3 hours)
```python
!./train.sh --epochs 100 --batch-size 16 --device cuda
```

### Experiment with Different Models
```python
# Try nano model (fastest)
!./train.sh --model-size n --epochs 50 --device cuda

# Try small model (better accuracy)
!./train.sh --model-size s --epochs 50 --device cuda
```

### Custom Configuration
```python
!./train.sh \\
  --model-version kaggle-v1 \\
  --epochs 100 \\
  --batch-size 32 \\
  --learning-rate 0.01 \\
  --img-width 640 \\
  --img-height 360 \\
  --device cuda
```

---

## Key Differences: Local vs Kaggle

| Feature | Local | Kaggle |
|---------|-------|--------|
| **Data Location** | `raw_data/` folder | Kaggle dataset (symlinked) |
| **GPU** | Your hardware | Free Tesla P100 (30h/week) |
| **Storage** | Your disk | ~30GB notebook space |
| **Setup** | `./train.sh` | `setup_kaggle_data.py` first |
| **Results** | Saved locally | Download from output |
| **Time Limit** | Unlimited | 12 hour session max |

---

## Tips & Tricks

### 💡 Save GPU Hours
- Use `--patience 20` to stop early
- Start with `--epochs 30` to test
- Use `--model-size n` for faster experiments

### 💾 Manage Disk Space
- Kaggle limits: ~30GB
- `setup_kaggle_data.py` uses symlinks (no copy)
- Delete `runs/` between experiments if needed:
  ```python
  !rm -rf runs/detect
  ```

### 📊 View Results in Notebook
```python
from IPython.display import Image, display
display(Image(filename='stats/v1/training_results_v1.png'))
```

### 🔄 Continue Training
If your session times out, you can resume:
```python
# Your model is saved in /kaggle/working/models/
# Download it, then re-upload and continue training
```

### 📦 Download Multiple Files
```python
# Create a zip of all results
!zip -r results.zip models/ stats/
# Download results.zip from Output tab
```

---

## Troubleshooting

### "Dataset not found"
- Check dataset is added: "+ Add Data" button
- Verify path: `/kaggle/input/YOUR-DATASET-NAME/`
- List datasets: `!ls /kaggle/input/`

### "GPU not available"
- Check Settings → Accelerator → GPU T4 x2
- Verify: `!nvidia-smi`
- Fallback to CPU (slower): `--device cpu`

### "Out of disk space"
- Delete runs: `!rm -rf runs/`
- Use smaller batch size: `--batch-size 8`
- Reduce epochs: `--epochs 30`

### "Session timeout"
- Kaggle limit: 12 hours per session
- Save frequently: Click "Save Version"
- Download models before timeout

### Shell scripts not executable
```python
!chmod +x *.sh
```

---

## Best Practices

### 🎯 For Experimentation
1. Upload dataset to Kaggle (one time)
2. Create notebook from template
3. Run quick experiments with different configs
4. Download best model
5. Use locally for inference

### 🚀 For Production Training
1. Test locally with `--epochs 5` first
2. Verify on Kaggle with `--epochs 30`
3. Full training: `--epochs 100 --device cuda`
4. Download and version control the best model

### 💰 For Free Tier Limits (30h/week)
- Each 100-epoch training ≈ 2-3 hours
- Run ~10 full trainings per week
- Or many quick 30-epoch experiments
- Use `--patience` to save time

---

## Example Workflow

```python
# 1. Setup (runs once per notebook)
!git clone https://github.com/YOUR_USERNAME/RE-ObjectDetector-NSFR.git
%cd RE-ObjectDetector-NSFR
!python setup_kaggle_data.py

# 2. Quick test (5 minutes)
!./train.sh --epochs 5 --device cuda

# 3. Experiment (30 minutes each)
!./train.sh --model-version exp1 --model-size n --epochs 30 --device cuda
!./train.sh --model-version exp2 --model-size s --epochs 30 --device cuda

# 4. Best config, full training (2-3 hours)
!./train.sh --model-version final --model-size s --epochs 100 --device cuda

# 5. Test
!./test.sh --model-version final --device cuda

# 6. Download models from Output tab
```

---

## Next Steps

After training on Kaggle:
1. Download your `*_best.pt` model
2. Copy to your local `models/` directory
3. Test locally: `./test.sh --model-version v1`
4. Deploy to your robot!

For more info, see:
- [INDEX.md](INDEX.md) - All documentation
- [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) - Full parameter reference
- [CONFIGURATION_PARAMETERS.md](CONFIGURATION_PARAMETERS.md) - Config options
