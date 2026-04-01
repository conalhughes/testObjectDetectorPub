# Documentation Index

## Quick Navigation

### 🚀 Getting Started
- **[README.md](README.md)** - Main documentation, project overview
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes

### 📖 Detailed Guides
- **[CONFIGURATION_PARAMETERS.md](CONFIGURATION_PARAMETERS.md)** - All config.py parameters explained
- **[CLI_ARGUMENTS.md](CLI_ARGUMENTS.md)** - All command-line arguments for training/testing
- **[DATA_MANAGEMENT.md](DATA_MANAGEMENT.md)** - Data preprocessing and split management
- **[DEVICE_FALLBACK.md](DEVICE_FALLBACK.md)** - GPU/CPU automatic fallback
- **[KAGGLE.md](KAGGLE.md)** - Train on Kaggle with free GPUs

---

## Documentation Summary

### README.md
**What it covers:**
- Project structure and setup
- Quick start (automated workflow)
- Manual workflow (step-by-step)
- Model sizes and selection
- Tips for best results
- Troubleshooting

**When to use:**
- First time setup
- Understanding project structure  
- General troubleshooting

### QUICKSTART.md
**What it covers:**
- One-command training
- Pipeline options (--reprocess, --use-existing)
- CLI arguments quick reference
- When to re-process data
- Testing the model
- What files get created

**When to use:**
- Want to start training immediately
- Need quick reference for common commands
- Don't want to read full documentation

### CONFIGURATION_PARAMETERS.md
**What it covers:**
- All config.py parameters with descriptions
- Data types and default values
- CLI argument mappings
- Recommendations for common use cases
- Quick reference for frequently modified parameters
- Three methods to modify parameters

**When to use:**
- Want to understand what each config parameter does
- Need to know default values and valid options
- Looking for parameter recommendations
- Want to compare config.py vs CLI arguments

### CLI_ARGUMENTS.md
**What it covers:**
- Complete list of all training arguments
- Complete list of all testing arguments
- Usage examples for every argument
- Common use cases (hyperparameter tuning, version management, etc.)
- Automated training scripts
- Configuration priority

**When to use:**
- Trying different hyperparameters
- Training multiple model versions
- Automating training runs
- Need detailed argument documentation

### DATA_MANAGEMENT.md
**What it covers:**
- Data flow from raw_data to train/val/test
- When data gets re-split
- When to reuse vs re-process
- Usage scenarios (same split for comparison, new split for validation)
- Command reference for data operations
- Best practices for reproducible experiments

**When to use:**
- Managing multiple training runs
- Want consistent train/val/test splits
- Need fresh random splits
- Adding/removing data from raw_data
- Understanding data preprocessing

### DEVICE_FALLBACK.md
**What it covers:**
- How automatic GPU/CPU fallback works
- Configuration (DEVICE setting in config.py)
- Log output examples
- Performance comparison
- Troubleshooting GPU issues

**When to use:**
- Moving code between GPU and CPU machines
- GPU not being detected
- Want to use CPU for testing
- Experiencing GPU errors
- Optimizing for different hardware

### KAGGLE.md
**What it covers:**
- Why use Kaggle for training
- Step-by-step setup guide
- Upload dataset to Kaggle
- Create and run notebook
- Download trained models
- Tips for GPU limits and disk space
- Troubleshooting common issues
- Differences between local and Kaggle

**When to use:**
- Want free GPU access for training
- Save local disk space by hosting dataset on Kaggle
- Train models in the cloud
- Need faster training than CPU
- Experimenting with different configurations
- Running out of local GPU resources

---

## Typical User Journeys

### First-Time User
1. Read [README.md](README.md) - Understand project
2. Follow [QUICKSTART.md](QUICKSTART.md) - Get running quickly
3. Reference [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) - Experiment with settings

### Hyperparameter Tuning
1. Quick reference: [QUICKSTART.md](QUICKSTART.md)
2. Detailed arguments: [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md)
3. Data management: [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) - Keep same split

### Production Deployment
1. Model sizes: [README.md](README.md)
2. Device handling: [DEVICE_FALLBACK.md](DEVICE_FALLBACK.md)
3. CLI arguments: [CLI_ARGUMENTS.md](CLI_ARGUMENTS.md) - Script automation

### Troubleshooting
1. Common issues: [README.md](README.md) - Troubleshooting section
2. Data issues: [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md)
3. Device issues: [DEVICE_FALLBACK.md](DEVICE_FALLBACK.md)

---

## Quick Command Reference

```bash
# Basic training
./train.sh

# Training with options
./train.sh --reprocess --epochs 200 --model-size s

# Testing
./test.sh
./test.sh --model-version v2

# Manual steps
python preprocess_data.py
python train_cli.py --help
python test_cli.py --help
```

---

## File Organization

**Core Scripts:**
- `train.sh`, `test.sh` - Automated pipelines
- `train_cli.py`, `test_cli.py` - CLI wrappers
- `train.py`, `test.py` - Core training/testing
- `preprocess_data.py`, `data_utils.py` - Data handling
- `config.py` - Configuration
- `logger_utils.py` - Logging

**Documentation:**
- `README.md` - Main docs
- `QUICKSTART.md` - Quick start
- `CLI_ARGUMENTS.md` - Arguments reference
- `DATA_MANAGEMENT.md` - Data guide
- `DEVICE_FALLBACK.md` - Device handling
- `INDEX.md` - This file

**Generated:**
- `data/` - Processed datasets
- `models/` - Trained models
- `stats/` - Plots and logs
- `runs/` - Full training details
- `backup/` - Data backups
- `.venv/` - Virtual environment
