"""
Training script for YOLOv8 Object Detection
Handles model training with validation and visualization
"""

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import yaml

import config
import data_utils
from logger_utils import setup_logger

# Configure logging
logger = setup_logger(__name__, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Determine device (auto-fallback to CPU if CUDA not available)
def get_device():
    """
    Get the device to use for training.
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


def setup_training_environment():
    """
    Set up the training environment by creating necessary directories
    and validating the dataset.
    """
    logger.info("=" * 70)
    logger.info("SETTING UP TRAINING ENVIRONMENT")
    logger.info("=" * 70)
    
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    
    # Determine actual device to use
    device = get_device()
    
    # Log configuration
    logger.info("Model Configuration:")
    logger.info(f"  Name: {config.MODEL_NAME}")
    logger.info(f"  Version: {config.MODEL_VERSION}")
    logger.info(f"  Size: {config.MODEL_SIZE}")
    logger.info(f"  Input Size: {config.INPUT_SIZE}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Device (config): {config.DEVICE}")
    logger.info(f"  Device (actual): {device}")
    logger.info(f"  Number of Classes: {config.NUM_CLASSES}")
    logger.info(f"  Classes: {config.CLASS_NAMES}")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    if not data_utils.prepare_dataset():
        logger.error("Dataset preparation failed!")
        return False
    
    return True, device


def initialize_model():
    """
    Initialize the YOLOv8 model with the specified configuration.
    
    Returns:
        YOLO model instance
    """
    logger.info("=" * 70)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 70)
    
    # Load pretrained model based on size
    model_name = f"yolov8{config.MODEL_SIZE}.pt"
    logger.info(f"Loading base model: {model_name}")
    
    try:
        model = YOLO(model_name)
        logger.info("Model loaded successfully")
        
        # Log model summary
        logger.info("Model Summary:")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"  Trainable Parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def train_model(model, device):
    """
    Train the YOLOv8 model.
    
    Args:
        model: YOLO model instance
        device: Device to use for training ('cuda' or 'cpu')
        
    Returns:
        Training results
    """
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 70)
    
    # Path to data.yaml
    data_yaml_path = os.path.join(config.DATA_DIR, 'data.yaml')
    
    # Training timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.MODEL_NAME}_{config.MODEL_VERSION}_{timestamp}"
    
    # Training parameters
    train_params = {
        'data': data_yaml_path,
        'epochs': config.EPOCHS,
        'imgsz': config.INPUT_SIZE,
        'batch': config.BATCH_SIZE,
        'name': run_name,
        'project': config.RUNS_DIR,
        'device': device,  # Use the actual device (with fallback)
        'workers': config.NUM_WORKERS,
        'patience': config.PATIENCE,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': False,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': config.NUM_CLASSES == 1,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        
        # Hyperparameters
        'lr0': config.LEARNING_RATE,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Augmentation parameters
        'hsv_h': config.HSV_H,
        'hsv_s': config.HSV_S,
        'hsv_v': config.HSV_V,
        'degrees': config.DEGREES,
        'translate': config.TRANSLATE,
        'scale': config.SCALE,
        'shear': config.SHEAR,
        'perspective': config.PERSPECTIVE,
        'flipud': config.FLIPUD,
        'fliplr': config.FLIPLR,
        'mosaic': config.MOSAIC,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    logger.info(f"Training Run: {run_name}")
    logger.info(f"Run Directory: {os.path.join(config.RUNS_DIR, run_name)}")
    logger.info("Starting training...")
    logger.info("-" * 70)
    
    try:
        # Train the model
        results = model.train(**train_params)
        
        logger.info("-" * 70)
        logger.info("Training completed successfully!")
        
        # Save final model
        model.save(config.MODEL_SAVE_PATH)
        logger.info(f"Model saved to: {config.MODEL_SAVE_PATH}")
        
        # Copy best model - YOLOv8 creates nested structure: runs/detect/runs/{run_name}
        import shutil
        
        # Try multiple possible locations for best.pt
        possible_paths = [
            os.path.join(config.RUNS_DIR, run_name, 'weights', 'best.pt'),  # Expected path
            os.path.join(config.RUNS_DIR, 'detect', 'runs', run_name, 'weights', 'best.pt'),  # Actual YOLOv8 path
        ]
        
        best_weights = None
        for path in possible_paths:
            if os.path.exists(path):
                best_weights = path
                break
        
        if best_weights:
            shutil.copy(best_weights, config.BEST_MODEL_PATH)
            logger.info(f"Best model copied from: {best_weights}")
            logger.info(f"Best model saved to: {config.BEST_MODEL_PATH}")
        else:
            logger.warning(f"Best model not found. Searched locations:")
            for path in possible_paths:
                logger.warning(f"  - {path}")
        
        return results, run_name
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_training_results(run_name):
    """
    Create comprehensive training visualization plots.
    
    Args:
        run_name: Name of the training run
    """
    logger.info("=" * 70)
    logger.info("GENERATING TRAINING PLOTS")
    logger.info("=" * 70)
    
    # YOLOv8 creates nested structure: runs/detect/runs/{run_name}
    # Try multiple possible paths
    possible_run_dirs = [
        os.path.join(config.RUNS_DIR, 'detect', 'runs', run_name),  # Actual YOLOv8 path
        os.path.join(config.RUNS_DIR, run_name),  # Alternative path
    ]
    
    run_dir = None
    for path in possible_run_dirs:
        if os.path.exists(path):
            run_dir = path
            break
    
    if not run_dir:
        logger.error(f"Run directory not found for: {run_name}")
        logger.warning(f"Checked locations:")
        for path in possible_run_dirs:
            logger.warning(f"  - {path}")
        logger.warning("Skipping training plots generation. Training completed successfully.")
        return
    
    # YOLOv8 saves results.csv in the run directory
    results_csv = os.path.join(run_dir, 'results.csv')
    
    # Check if results.csv exists
    if not os.path.exists(results_csv):
        logger.warning(f"Results file not found at: {results_csv}")
        logger.warning(f"Checking directory contents...")
        logger.info(f"Files in {run_dir}:")
        try:
            for item in os.listdir(run_dir):
                logger.info(f"  - {item}")
        except Exception as e:
            logger.error(f"Could not list directory: {e}")
        logger.warning("Skipping training plots generation. Training completed successfully.")
        return
    
    # Read results
    try:
        df = pd.read_csv(results_csv)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error reading results: {e}")
        return
    
    # Set plot style
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Training and Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
    if 'train/dfl_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Validation Losses
    ax2 = plt.subplot(2, 3, 2)
    if 'val/box_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, color='orange')
    if 'val/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2, color='red')
    if 'val/dfl_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='purple')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Losses')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Precision and Recall
    ax3 = plt.subplot(2, 3, 3)
    if 'metrics/precision(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
    if 'metrics/recall(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='blue')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision and Recall')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: mAP scores
    ax4 = plt.subplot(2, 3, 4)
    if 'metrics/mAP50(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP')
    ax4.set_title('Mean Average Precision')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    # Plot 5: Learning Rate
    ax5 = plt.subplot(2, 3, 5)
    if 'lr/pg0' in df.columns:
        ax5.plot(df['epoch'], df['lr/pg0'], label='LR pg0', linewidth=2)
    if 'lr/pg1' in df.columns:
        ax5.plot(df['epoch'], df['lr/pg1'], label='LR pg1', linewidth=2)
    if 'lr/pg2' in df.columns:
        ax5.plot(df['epoch'], df['lr/pg2'], label='LR pg2', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Get best metrics
    best_epoch = df['epoch'].iloc[-1]
    table_data = [['Metric', 'Value']]
    
    if 'metrics/mAP50(B)' in df.columns:
        best_map50 = df['metrics/mAP50(B)'].max()
        best_map50_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
        table_data.append(['Best mAP@0.5', f'{best_map50:.4f} (Epoch {int(best_map50_epoch)})'])
    
    if 'metrics/mAP50-95(B)' in df.columns:
        best_map5095 = df['metrics/mAP50-95(B)'].max()
        best_map5095_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']
        table_data.append(['Best mAP@0.5:0.95', f'{best_map5095:.4f} (Epoch {int(best_map5095_epoch)})'])
    
    if 'metrics/precision(B)' in df.columns:
        final_precision = df['metrics/precision(B)'].iloc[-1]
        table_data.append(['Final Precision', f'{final_precision:.4f}'])
    
    if 'metrics/recall(B)' in df.columns:
        final_recall = df['metrics/recall(B)'].iloc[-1]
        table_data.append(['Final Recall', f'{final_recall:.4f}'])
    
    table_data.append(['Total Epochs', int(best_epoch)])
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Training Results - {config.MODEL_NAME} {config.MODEL_VERSION}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'training_results_{config.MODEL_VERSION}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"Training results plot saved to: {plot_path}")
    plt.close()
    
    # Save summary to text file
    summary_path = os.path.join(config.VERSION_STATS_DIR, f'training_summary_{config.MODEL_VERSION}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary - {config.MODEL_NAME} {config.MODEL_VERSION}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model Size: {config.MODEL_SIZE}\n")
        f.write(f"  Input Size: {config.INPUT_SIZE}\n")
        f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"  Epochs: {config.EPOCHS}\n")
        f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"  Device: {config.DEVICE}\n\n")
        f.write("Results:\n")
        for row in table_data[1:]:
            f.write(f"  {row[0]}: {row[1]}\n")
    
    logger.info(f"Training summary saved to: {summary_path}")


def main():
    """
    Main training pipeline.
    """
    # Ensure version-specific stats directory exists for log file
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("YOLOv8 OBJECT DETECTION TRAINING")
    logger.info(f"Model: {config.MODEL_NAME} {config.MODEL_VERSION}")
    logger.info("=" * 70)
    
    # Step 1: Setup environment
    result = setup_training_environment()
    if isinstance(result, tuple):
        success, device = result
        if not success:
            logger.error("Environment setup failed. Exiting.")
            return
    else:
        logger.error("Environment setup failed. Exiting.")
        return
    
    # Step 2: Initialize model
    model = initialize_model()
    if model is None:
        logger.error("Model initialization failed. Exiting.")
        return
    
    # Step 3: Train model
    results, run_name = train_model(model, device)
    if results is None:
        logger.error("Training failed. Exiting.")
        return
    
    # Step 4: Plot results
    plot_training_results(run_name)
    
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model saved at: {config.BEST_MODEL_PATH}")
    logger.info(f"Training plots saved in: {config.VERSION_STATS_DIR}")
    logger.info(f"Full training results in: {os.path.join(config.RUNS_DIR, run_name)}")


if __name__ == "__main__":
    main()
