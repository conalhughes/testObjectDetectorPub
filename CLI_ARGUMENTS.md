# CLI Arguments Guide

## Overview

You can now override any `config.py` setting via command-line arguments without editing files. This is perfect for:
- **Hyperparameter tuning** - Try different learning rates, batch sizes, etc.
- **Quick experiments** - Test different model sizes or augmentation settings
- **Automated runs** - Script multiple training runs with different configurations
- **Version management** - Train multiple model versions simultaneously

## Training Arguments

### Basic Usage

```bash
# Use config.py defaults
./train.sh

# Override specific parameters
./train.sh --epochs 200 --batch-size 32 --learning-rate 0.001

# Combine with pipeline options
./train.sh --reprocess --epochs 50 --model-size s
```

### Complete Training Options

#### Model Configuration
```bash
--model-name NAME          # Model name (default: ball_detector)
--model-version VER        # Model version (default: v1)
--model-size SIZE          # n, s, m, l, x (default: n)
```

#### Training Parameters
```bash
--epochs N                 # Number of epochs (default: 100)
--batch-size N             # Batch size (default: 16)
--learning-rate LR         # Learning rate (default: 0.01)
--lr                       # Short form of --learning-rate
--patience N               # Early stopping patience (default: 50)
--device DEVICE            # cuda or cpu (default: cuda)
--workers N                # Data loading workers (default: 4)
```

#### Image Size
```bash
--img-width W              # Image width for training (default: 544 from config.INPUT_SIZE)
--img-height H             # Image height for training (default: 448 from config.INPUT_SIZE)
```

#### Data Augmentation
```bash
--mosaic P                 # Mosaic probability 0.0-1.0 (default: 1.0)
--hsv-h V                  # HSV-Hue augmentation (default: 0.015)
--hsv-s V                  # HSV-Saturation (default: 0.7)
--hsv-v V                  # HSV-Value (default: 0.4)
--degrees D                # Rotation degrees (default: 0.0)
--translate T              # Translation (default: 0.1)
--scale S                  # Scale (default: 0.5)
--shear S                  # Shear degrees (default: 0.0)
--perspective P            # Perspective (default: 0.0)
--flipud P                 # Vertical flip prob 0.0-1.0 (default: 0.0)
--fliplr P                 # Horizontal flip prob 0.0-1.0 (default: 0.5)
```

#### Validation/Testing
```bash
--conf-threshold T         # Confidence threshold (default: 0.25)
--iou-threshold T          # IoU threshold for NMS (default: 0.45)
```

## Testing Arguments

### Basic Usage

```bash
# Use config.py defaults
./test.sh

# Test specific model version
./test.sh --model-version v2

# Override confidence threshold
./test.sh --conf-threshold 0.5 --iou-threshold 0.6
```

### Complete Testing Options

```bash
--model-name NAME          # Model name (default: ball_detector)
--model-version VER        # Model version (default: v1)
--img-width W              # Image width for training (default: 544 from config.INPUT_SIZE)
--img-height H             # Image height for training (default: 448 from config.INPUT_SIZE)
--batch-size N             # Batch size (default: 16)
--device DEVICE            # cuda or cpu (default: cuda)
--conf-threshold T         # Confidence threshold (default: 0.25)
--iou-threshold T          # IoU threshold (default: 0.45)
--max-det N                # Max detections per image (default: 300)
```

## Common Use Cases

### 1. Quick Test with Different Model Size
```bash
# Try small model instead of nano
./train.sh --model-size s --epochs 50
```

### 2. Train Multiple Versions
```bash
# Train version 1 with nano model
./train.sh --model-version v1 --model-size n --epochs 100

# Train version 2 with small model
./train.sh --model-version v2 --model-size s --epochs 100

# Train version 3 with different learning rate
./train.sh --model-version v3 --learning-rate 0.001 --epochs 150
```

### 3. Hyperparameter Tuning
```bash
# Try different batch sizes
./train.sh --use-existing --model-version v1_bs8 --batch-size 8
./train.sh --use-existing --model-version v1_bs32 --batch-size 32

# Try different learning rates
./train.sh --use-existing --model-version v1_lr001 --lr 0.001
./train.sh --use-existing --model-version v1_lr01 --lr 0.01
```

### 4. Augmentation Experiments
```bash
# No augmentation
./train.sh --model-version v1_no_aug --mosaic 0.0 --fliplr 0.0

# Heavy augmentation
./train.sh --model-version v1_heavy_aug \
    --mosaic 1.0 \
    --fliplr 0.5 \
    --flipud 0.2 \
    --degrees 15 \
    --translate 0.2 \
    --scale 0.7
```

### 5. Quick CPU Test
```bash
# Test on CPU with fewer epochs
./train.sh --device cpu --epochs 10 --batch-size 8
```

### 6. High-Resolution Training
```bash
# Train with larger images
./train.sh --img-width 1280 --img-height 720 --batch-size 8
```

### 7. Test Different Models
```bash
# Test the best v1 model
./test.sh --model-version v1

# Test v2 with higher confidence threshold
./test.sh --model-version v2 --conf-threshold 0.5

# Test on CPU
./test.sh --device cpu
```

### 8. Automated Training Script
```bash
#!/bin/bash
# train_experiments.sh

# Baseline
./train.sh --model-version baseline --epochs 100

# Experiment 1: Larger model
./train.sh --model-version exp1_small --model-size s --epochs 100

# Experiment 2: Different learning rate
./train.sh --model-version exp2_lr --learning-rate 0.001 --epochs 150

# Experiment 3: More augmentation
./train.sh --model-version exp3_aug \
    --fliplr 0.7 \
    --degrees 10 \
    --scale 0.6 \
    --epochs 100
```

## Pipeline Options

These control the shell script behavior (separate from training parameters):

```bash
--reprocess                # Delete data/ and re-split from raw_data
--use-existing             # Use existing data split without prompt
```

**Combine pipeline and training options:**
```bash
# Reprocess data and train with new settings
./train.sh --reprocess --model-version v2 --batch-size 32

# Use existing split and change only training params
./train.sh --use-existing --epochs 200 --lr 0.005
```

## Help Commands

```bash
# Show all options
./train.sh --help
./test.sh --help

# Or directly from Python
source .venv/bin/activate
python train_cli.py --help
python test_cli.py --help
```

## Configuration Priority

Arguments override config.py values:

1. **Command-line arguments** (highest priority)
2. **config.py defaults** (used if no argument provided)

Example:
```python
# config.py
EPOCHS = 100
BATCH_SIZE = 16
```

```bash
# This will use EPOCHS=200, BATCH_SIZE=16
./train.sh --epochs 200

# This will use EPOCHS=100, BATCH_SIZE=32
./train.sh --batch-size 32

# This will use EPOCHS=200, BATCH_SIZE=32
./train.sh --epochs 200 --batch-size 32
```

## Output Organization

Each model version gets its own directory:

```
models/
  ├── ball_detector_v1_best.pt
  ├── ball_detector_v2_best.pt
  └── ball_detector_exp1_small_best.pt

stats/
  ├── v1/
  │   ├── train_*.log
  │   └── training_results_v1.png
  ├── v2/
  │   ├── train_*.log
  │   └── training_results_v2.png
  └── exp1_small/
      ├── train_*.log
      └── training_results_exp1_small.png
```

## Best Practices

### ✅ Use Version Names Descriptively
```bash
# Good - descriptive version names
./train.sh --model-version baseline_100ep
./train.sh --model-version small_lr001
./train.sh --model-version heavy_aug_200ep

# Avoid - cryptic names
./train.sh --model-version v1
./train.sh --model-version test123
```

### ✅ Keep Experiments Organized
```bash
# Create a log of what each version is
echo "v1: baseline, 100 epochs, batch 16" >> experiments.log
./train.sh --model-version v1

echo "v2: small model, 150 epochs, lr 0.001" >> experiments.log
./train.sh --model-version v2 --model-size s --epochs 150 --lr 0.001
```

### ✅ Use Same Split for Fair Comparison
```bash
# First run creates the split
./train.sh --model-version baseline

# Subsequent runs use same split
./train.sh --use-existing --model-version exp1
./train.sh --use-existing --model-version exp2
```

### ✅ Document Your Results
After each experiment, note the results:
```bash
./train.sh --model-version v1 --epochs 100
# mAP@0.5: 0.85

./train.sh --use-existing --model-version v2 --model-size s --epochs 100
# mAP@0.5: 0.89 ✓ Better!
```

## Troubleshooting

**Q: I changed model-version but it's using old data**  
A: Use `--reprocess` to create fresh split, or `--use-existing` to explicitly use old split.

**Q: Arguments not working**  
A: Make sure arguments come AFTER `--reprocess` or `--use-existing`:
```bash
# Correct
./train.sh --reprocess --epochs 200

# Wrong (--epochs treated as unknown option)
./train.sh --epochs 200 --reprocess
```

**Q: How to see all available arguments?**  
A: Run `./train.sh --help` or `python train_cli.py --help`

**Q: Can I mix config.py and CLI arguments?**  
A: Yes! CLI arguments override config.py. Unspecified arguments use config.py defaults.
