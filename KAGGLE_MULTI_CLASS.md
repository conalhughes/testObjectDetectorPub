# Multi-Class Configuration for Kaggle Notebooks

This document explains how the multi-class configuration system works in Kaggle environments.

## How It Works

### 1. Configuration Files (In GitHub Repository)

When you clone the repository in Kaggle, these files are automatically included:

- **`classes_config.yaml`** - Defines all object classes
- **`config.py`** - Automatically reads `classes_config.yaml` and provides:
  - `DATASET_NAME` - Name of your dataset
  - `NUM_CLASSES` - Number of classes
  - `CLASS_NAMES` - List of class names
  - `CLASS_COLORS` - RGB colors for visualization

### 2. Kaggle Notebook Setup

The KAGGLE_SETUP.ipynb notebook handles everything automatically:

#### Cell 1: Clone Repository
```python
!git clone https://github.com/conalhughes/RE-ObjectDetector-NSFR.git
%cd RE-ObjectDetector-NSFR
sys.path.insert(0, '/kaggle/working/RE-ObjectDetector-NSFR')
```

This ensures that:
- Your repository code is available
- Python can import modules like `config.py`
- The working directory is set correctly

#### Cell 2.5: Configure Classes (Optional)
```python
import config as cfg
print(f"Classes: {cfg.CLASS_NAMES}")
print(f"Number of classes: {cfg.NUM_CLASSES}")
```

You can modify `classes_config.yaml` here if needed.

#### Cell 3-onwards: Training
All training scripts automatically use the configured classes.

## Using Multi-Class Configuration in Kaggle

### Single Class (Default)
The default `classes_config.yaml` is configured for single-class ball detection:

```yaml
dataset_name: "Ball Detector"
classes:
  - id: 0
    name: "ball"
    color: [0, 255, 0]
```

This works as-is for ball detection datasets.

### Multi-Class in Kaggle

If you want to train on multiple classes:

1. **Before running training**, edit the configuration cell (2.5):

```python
# Configure object classes
import yaml

with open('classes_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update classes
config['dataset_name'] = 'RoboCup Multi-Object'
config['classes'] = [
    {'id': 0, 'name': 'ball', 'color': [0, 255, 0]},
    {'id': 1, 'name': 'robot', 'color': [255, 0, 0]},
    {'id': 2, 'name': 'goal', 'color': [0, 0, 255]},
]

# Save updated configuration
with open('classes_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Reload config module to apply changes
import importlib
import config as cfg
importlib.reload(cfg)
print(f"✓ Classes: {cfg.CLASS_NAMES}")
```

2. **Ensure your dataset labels use the correct class IDs** (0, 1, 2, etc.)

3. **Run training** - the rest works automatically

## Verifying Configuration in Kaggle

After loading the repository, run:

```python
import config as cfg

print(f"Dataset: {cfg.DATASET_NAME}")
print(f"Classes: {cfg.CLASS_NAMES}")
print(f"Number of classes: {cfg.NUM_CLASSES}")
```

Expected output:
```
Dataset: Ball Detector
Classes: ['ball']
Number of classes: 1
```

## Troubleshooting in Kaggle

### "ModuleNotFoundError: No module named 'config'"
- Ensure you ran the clone cell first
- Verify `sys.path` includes `/kaggle/working/RE-ObjectDetector-NSFR`

### Classes not updating
- Make sure to reload the config module after modifying `classes_config.yaml`:
```python
import importlib
import config as cfg
importlib.reload(cfg)
```

### YAML parsing errors
- Check that `classes_config.yaml` is valid YAML
- Make sure indentation is consistent (use spaces, not tabs)
- Verify all class IDs are unique

## Label Format in Kaggle

Your dataset labels must use the correct class IDs. For multi-class example:

**Label file format (YOLO):**
```
class_id x_center y_center width height
```

**Example multi-class labels:**
```
0 0.5 0.5 0.1 0.1    # ball (class 0)
1 0.3 0.7 0.15 0.2   # robot (class 1)
2 0.8 0.2 0.25 0.3   # goal (class 2)
```

## Uploading Multi-Class Datasets to Kaggle

1. **Create your dataset** with the correct directory structure:
```
my-robocup-dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt  (contains class IDs: 0, 1, 2, etc.)
    ├── image2.txt
    └── ...
```

2. **Upload as Kaggle dataset** - use Kaggle website or CLI

3. **In KAGGLE_SETUP.ipynb**, update the data setup cell:
```python
!python setup_kaggle_data.py /kaggle/input/my-robocup-dataset

# Or if auto-detection works:
!python setup_kaggle_data.py
```

4. **Configure classes** in cell 2.5 to match your dataset

5. **Run training** - rest is automatic

## Supported Configurations

The system supports:
- ✅ Single-class datasets (default)
- ✅ Multi-class datasets (2-10+ classes)
- ✅ Any YOLO-format labels
- ✅ Kaggle datasets
- ✅ Custom datasets

## Performance Notes

- GPU training in Kaggle is typically 2-3x faster than CPU
- Free tier: 30 hours/week GPU time
- Sessions timeout after 12 hours
- Use `--epochs 50` for quick experiments
- Use `--epochs 100+` for production models

## Next Steps

1. Prepare your dataset with correct class IDs
2. Upload to Kaggle as a dataset
3. Use KAGGLE_SETUP.ipynb to train
4. Configure classes as needed
5. Download trained model (`*_best.pt`)
