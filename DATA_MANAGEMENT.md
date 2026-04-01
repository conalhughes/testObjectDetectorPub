# Data Management Guide

## Overview

This project supports multiple training runs with flexible data management. You can either:
- **Reuse the same data split** for consistent comparisons across runs
- **Re-process raw data** to get a fresh random split

## Data Flow

```
raw_data/               (Your original images and labels)
    ├── images/         → Never modified
    └── labels/         → Never modified
           ↓
    preprocessing      (Resize, validate, split)
           ↓
data/                   (Processed and split dataset)
    ├── train/          → 70% of samples
    ├── val/            → 20% of samples
    └── test/           → 10% of samples
```

## When Data Gets Re-Split

### Automatic Re-splitting
The `preprocess_data.py` script will **always create a new random split** when it runs.

### Persistence Between Runs
- Once `data/` folder exists, `train.sh` will **ask** if you want to reuse it or re-process
- This allows consistent splits across multiple training runs

## Usage Scenarios

### Scenario 1: First Training Run
```bash
./train.sh
```
- No `data/` exists yet
- Automatically preprocesses raw_data/
- Creates train/val/test split
- Trains model

### Scenario 2: Second Training Run (Same Split)
```bash
./train.sh
```
- Detects existing `data/`
- Prompts: "Use existing split?"
- Select option 1 → Uses same split
- Good for comparing hyperparameters, model sizes, etc.

### Scenario 3: Second Training Run (New Split)
```bash
./train.sh --reprocess
```
OR
```bash
./train.sh
# Then select option 2 when prompted
```
- Deletes `data/` and `backup/`
- Re-processes from `raw_data/`
- Creates **new random split**
- Good for validating model isn't overfitting to specific split

### Scenario 4: Just Re-process Data
```bash
./reprocess.sh
```
- Standalone script to re-create data splits
- Asks for confirmation
- Clears and re-processes
- Then run `./train.sh --use-existing`

### Scenario 5: Automated/CI Runs
```bash
# Always reprocess
./train.sh --reprocess

# Or always use existing
./train.sh --use-existing
```

## Command Reference

### Interactive Training (Default)
```bash
./train.sh
```
Prompts if data exists. Options:
1. Use existing split (recommended for consistency)
2. Re-process and create new split
3. Skip check and continue

### Force Re-processing
```bash
./train.sh --reprocess
```
- No prompts
- Deletes `data/` and `backup/`
- Creates fresh split from `raw_data/`

### Use Existing Data
```bash
./train.sh --use-existing
```
- No prompts
- Uses existing `data/` split
- Good for automated runs

### Manual Re-processing
```bash
./reprocess.sh
```
- Interactive confirmation
- Clears `data/` and `backup/`
- Re-processes from `raw_data/`
- Follow up with `./train.sh --use-existing`

## Best Practices

### ✅ For Reproducible Experiments
1. Run preprocessing once: `./reprocess.sh`
2. Train with different configs: `./train.sh --use-existing`
3. All runs use identical train/val/test split
4. Results are directly comparable

### ✅ For Robust Model Validation
1. Train on split 1: `./train.sh`
2. Train on split 2: `./train.sh --reprocess`
3. Train on split 3: `./train.sh --reprocess`
4. Average performance across different splits
5. Ensures model generalizes beyond specific train/test split

### ✅ After Updating Raw Data
1. Add/remove images in `raw_data/images/`
2. Update labels in `raw_data/labels/`
3. Re-process: `./train.sh --reprocess`
4. New split will include all changes

### ⚠️ Important Notes

- **Random seed:** Preprocessing uses `random.seed(42)` for reproducibility
- **Same split:** As long as raw_data hasn't changed, you can recreate the exact same split by re-running preprocessing (due to fixed seed)
- **Version folders:** Each model version (v1, v2, etc.) can have different data splits
- **Backup:** First preprocessing creates `backup/` of raw_data (only once)

## File Locations

### Input (Never Modified)
- `raw_data/images/` - Original images
- `raw_data/labels/` - Original YOLO format labels

### Generated (Safe to Delete)
- `data/` - Processed dataset (train/val/test)
- `backup/` - Backup of raw_data (created once)
- `models/` - Trained model weights
- `stats/` - Plots and reports
- `runs/` - Detailed training logs
- `.venv/` - Python virtual environment

### Logs
- `stats/v1/preprocessing_*.log` - Preprocessing details
- `stats/v1/train_*.log` - Training logs
- `stats/v1/test_*.log` - Testing logs

## Example Workflow

```bash
# 1. Initial setup and first training
./train.sh
# Creates split: Train=1799, Val=514, Test=257

# 2. Try different learning rate (same split)
# Edit config.py: LEARNING_RATE = 0.001
./train.sh --use-existing
# Uses same split for fair comparison

# 3. Try different model size (same split)
# Edit config.py: MODEL_SIZE = "s"
./train.sh --use-existing
# Still uses same split

# 4. Validate on different split
./train.sh --reprocess
# Creates new split: Train=1812, Val=502, Test=256
# Different samples in each set

# 5. Add more training data
# Copy new images to raw_data/images/
# Add new labels to raw_data/labels/
./train.sh --reprocess
# Creates new split with all data: Train=2100, Val=600, Test=300
```

## Troubleshooting

**Q: My train/val/test counts changed between runs**
- A: You ran preprocessing again (or selected option 2). New random split was created.

**Q: I want the exact same split for all experiments**
- A: Always use `./train.sh --use-existing` after first run. Never delete `data/` folder.

**Q: How do I know which split was used for a model?**
- A: Check `stats/v1/preprocessing_*.log` - shows exact files in each split.

**Q: Can I manually choose which images go in train/val/test?**
- A: Not currently automated. You would need to manually copy files to `data/train/`, `data/val/`, `data/test/` and skip preprocessing.

**Q: Does the random seed guarantee identical splits?**
- A: Yes, IF raw_data hasn't changed. Same images + same seed = same split.
