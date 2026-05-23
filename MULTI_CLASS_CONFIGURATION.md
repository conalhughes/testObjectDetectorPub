# Multi-Class Configuration Guide

This project now supports **multiple object detection classes** for any dataset. Easily configure which objects you want to detect.

## Quick Start

### Option 1: Edit Configuration Notebook (Recommended)
The easiest way to configure classes:

```bash
# Open the configuration notebook in Jupyter
jupyter notebook CLASS_CONFIGURATION.ipynb
```

Then:
1. Update the **Dataset Name** 
2. Edit the **classes** list in Cell 2 with your object classes
3. Run **Save Configuration** to generate `classes_config.yaml`
4. Verify your configuration in the **Verification** cell

### Option 2: Edit YAML Configuration
Directly edit `classes_config.yaml`:

```yaml
dataset_name: "RoboCup Multi-Object Detector"
classes:
  - id: 0
    name: "ball"
    color: [0, 255, 0]      # Green
  - id: 1
    name: "robot"
    color: [255, 0, 0]      # Red
  - id: 2
    name: "goal"
    color: [0, 0, 255]      # Blue
```

## Configuration File Format

Each class requires:
- **id**: Unique integer identifier (0, 1, 2, ...) - must match your label files
- **name**: String name of the class (e.g., "ball", "robot")
- **color**: RGB color values [R, G, B] (0-255) for visualization

## Example Configurations

### Single Class (Ball Detection)
```yaml
dataset_name: "Ball Detector"
classes:
  - id: 0
    name: "ball"
    color: [0, 255, 0]
```

### Multi-Class (RoboCup)
```yaml
dataset_name: "RoboCup Multi-Object Detector"
classes:
  - id: 0
    name: "ball"
    color: [0, 255, 0]
  - id: 1
    name: "robot_blue"
    color: [255, 0, 0]
  - id: 2
    name: "robot_yellow"
    color: [0, 0, 255]
  - id: 3
    name: "goal"
    color: [255, 255, 0]
```

### COCO-like Dataset
```yaml
dataset_name: "General Object Detector"
classes:
  - id: 0
    name: "person"
    color: [0, 255, 0]
  - id: 1
    name: "car"
    color: [255, 0, 0]
  - id: 2
    name: "dog"
    color: [0, 0, 255]
  - id: 3
    name: "cat"
    color: [255, 255, 0]
  - id: 4
    name: "bicycle"
    color: [0, 255, 255]
```

## How It Works

1. **classes_config.yaml** - Defines your classes
2. **config.py** - Automatically reads `classes_config.yaml` and sets:
   - `NUM_CLASSES` - Number of classes
   - `CLASS_NAMES` - List of class names
   - `CLASS_COLORS` - RGB colors for visualization
3. **Training Scripts** - Use `NUM_CLASSES` and `CLASS_NAMES` from config
4. **Label Files** - Must use class IDs (0, 1, 2, ...) matching your configuration

## Label File Format

Your YOLO label files in `raw_data/labels/` must use the correct class IDs:

```
# For 2-class detection (ball=0, robot=1)
# image.txt:
0 0.5 0.5 0.1 0.1    # ball at center
1 0.3 0.7 0.15 0.2   # robot lower-left
```

## Workflow

1. **Configure Classes**
   ```bash
   jupyter notebook CLASS_CONFIGURATION.ipynb
   ```

2. **Prepare Your Data**
   - Place images in `raw_data/images/`
   - Place YOLO-format labels in `raw_data/labels/`
   - Ensure class IDs in labels match your configuration

3. **Train Model**
   ```bash
   ./train.sh --epochs 100 --batch-size 16
   ```

4. **Test Model**
   ```bash
   ./test.sh
   ```

## Verifying Configuration

Run this to check that your configuration loads correctly:

```bash
python -c "import config; print(f'Classes: {config.CLASS_NAMES}')"
```

Or in a Jupyter cell:
```python
import config
print(f"Dataset: {config.DATASET_NAME}")
print(f"Classes: {config.CLASS_NAMES}")
print(f"Number of Classes: {config.NUM_CLASSES}")
```

## Troubleshooting

### "Module config not found" error
- Make sure you're in the project directory
- Run `pip install -r requirements.txt`

### Class IDs mismatch
- Verify your label files use IDs that match `classes_config.yaml`
- Run `python test_cli.py` to check label validation

### Model not detecting all classes
- Ensure all class IDs appear in training data
- Check that preprocessing doesn't filter out minority classes
- Review training plots in `stats/`

## Support for Different Datasets

This system works with:
- ✅ Custom single-class datasets
- ✅ Multi-class datasets (RoboCup, COCO, etc.)
- ✅ Roboflow-exported datasets
- ✅ Kaggle datasets
- ✅ Any YOLO-format dataset

Just configure your classes and you're ready to train!
