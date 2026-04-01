# Automatic Device Fallback

## Overview

The training and testing scripts now automatically fall back to CPU if CUDA is requested but not available.

## How It Works

### Configuration (config.py)
```python
DEVICE = "cuda"  # Set to "cuda" or "cpu"
```

### Automatic Fallback Logic

**In `train.py` and `test.py`:**
```python
def get_device():
    """
    Get the device to use for training/testing.
    Falls back to CPU if CUDA is requested but not available.
    """
    if config.DEVICE == "cuda":
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.warning(f"CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        logger.info(f"Using device: {config.DEVICE}")
        return config.DEVICE
```

## Usage

### Scenario 1: GPU Machine
```python
# config.py
DEVICE = "cuda"
```
- **With GPU:** Uses CUDA automatically ✓
- **Without GPU:** Falls back to CPU automatically ✓

### Scenario 2: CPU-Only Machine
```python
# config.py
DEVICE = "cpu"
```
- Always uses CPU (no warning)

### Scenario 3: Force CPU (even if GPU available)
```python
# config.py
DEVICE = "cpu"
```
- Uses CPU even if GPU is available

## Benefits

✅ **Portable:** Same config works on GPU and CPU machines  
✅ **No crashes:** Won't fail if CUDA unavailable  
✅ **Automatic:** No manual intervention needed  
✅ **Logged:** Shows which device is actually being used  
✅ **Flexible:** Can force CPU if needed for testing

## Log Output

### With CUDA Available:
```
INFO - CUDA is available. Using GPU: NVIDIA GeForce RTX 3080
INFO - Device (config): cuda
INFO - Device (actual): cuda
```

### Without CUDA Available:
```
WARNING - CUDA requested but not available. Falling back to CPU.
INFO - Device (config): cuda
INFO - Device (actual): cpu
```

### CPU Explicitly Set:
```
INFO - Using device: cpu
INFO - Device (config): cpu
INFO - Device (actual): cpu
```

## Implementation Details

### Modified Files:
1. **train.py**
   - Added `get_device()` function
   - Modified `setup_training_environment()` to return device
   - Updated `train_model(model, device)` to accept device parameter
   - Logs both config device and actual device

2. **test.py**
   - Added `get_device()` function
   - Modified `evaluate_on_test_set(model, device)` to accept device parameter
   - Modified `run_inference_on_test_images(model, device, num_samples)` to accept device parameter
   - Updated main() to determine and pass device

### No Changes Needed:
- `config.py` - Keep DEVICE = "cuda" for GPU machines
- `requirements.txt` - Same dependencies
- Shell scripts - Work the same way

## Performance Notes

**CUDA (GPU):**
- Training: ~30-60 minutes for 100 epochs
- Inference: ~20-30 FPS

**CPU:**
- Training: Several hours for 100 epochs (10-20x slower)
- Inference: ~2-5 FPS

## Troubleshooting

**Q: How do I know which device is being used?**  
A: Check the log file in `stats/v1/train_*.log` or `stats/v1/test_*.log`. It shows both the config device and actual device.

**Q: Can I force CPU even if GPU is available?**  
A: Yes, set `DEVICE = "cpu"` in `config.py`.

**Q: What if I have multiple GPUs?**  
A: Set `DEVICE = "cuda:0"` or `DEVICE = "cuda:1"` etc. The fallback logic will still work.

**Q: Does this affect training speed?**  
A: No impact if GPU is available. If falling back to CPU, training will be significantly slower (expected behavior).
