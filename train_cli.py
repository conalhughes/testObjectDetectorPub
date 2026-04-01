#!/usr/bin/env python
"""
Train with CLI arguments
Allows overriding config.py values via command line
"""

import argparse
import sys
import os

# Import config first
import config

def parse_args():
    """Parse command line arguments and override config values."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 Object Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default=config.MODEL_NAME,
                        help='Model name')
    parser.add_argument('--model-version', type=str, default=config.MODEL_VERSION,
                        help='Model version')
    parser.add_argument('--model-size', type=str, default=config.MODEL_SIZE,
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', '--lr', type=float, default=config.LEARNING_RATE,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=config.PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--workers', type=int, default=config.NUM_WORKERS,
                        help='Number of data loading workers')
    
    # Image size
    parser.add_argument('--img-width', type=int, default=config.INPUT_SIZE[0],
                        help='Input image width')
    parser.add_argument('--img-height', type=int, default=config.INPUT_SIZE[1],
                        help='Input image height')
    
    # Augmentation
    parser.add_argument('--mosaic', type=float, default=config.MOSAIC,
                        help='Mosaic augmentation probability (0.0-1.0)')
    parser.add_argument('--hsv-h', type=float, default=config.HSV_H,
                        help='HSV-Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=config.HSV_S,
                        help='HSV-Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=config.HSV_V,
                        help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=config.DEGREES,
                        help='Rotation augmentation (degrees)')
    parser.add_argument('--translate', type=float, default=config.TRANSLATE,
                        help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=config.SCALE,
                        help='Scale augmentation')
    parser.add_argument('--shear', type=float, default=config.SHEAR,
                        help='Shear augmentation (degrees)')
    parser.add_argument('--perspective', type=float, default=config.PERSPECTIVE,
                        help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=config.FLIPUD,
                        help='Vertical flip probability (0.0-1.0)')
    parser.add_argument('--fliplr', type=float, default=config.FLIPLR,
                        help='Horizontal flip probability (0.0-1.0)')
    
    # Testing/Validation
    parser.add_argument('--conf-threshold', type=float, default=config.CONF_THRESHOLD,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=config.IOU_THRESHOLD,
                        help='IoU threshold for NMS')
    
    args = parser.parse_args()
    return args


def update_config(args):
    """Update config module with parsed arguments."""
    # Model configuration
    config.MODEL_NAME = args.model_name
    config.MODEL_VERSION = args.model_version
    config.MODEL_SIZE = args.model_size
    
    # Training configuration
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.PATIENCE = args.patience
    config.DEVICE = args.device
    config.NUM_WORKERS = args.workers
    
    # Image size
    config.INPUT_SIZE = (args.img_width, args.img_height)
    
    # Augmentation
    config.MOSAIC = args.mosaic
    config.HSV_H = args.hsv_h
    config.HSV_S = args.hsv_s
    config.HSV_V = args.hsv_v
    config.DEGREES = args.degrees
    config.TRANSLATE = args.translate
    config.SCALE = args.scale
    config.SHEAR = args.shear
    config.PERSPECTIVE = args.perspective
    config.FLIPUD = args.flipud
    config.FLIPLR = args.fliplr
    
    # Testing/Validation
    config.CONF_THRESHOLD = args.conf_threshold
    config.IOU_THRESHOLD = args.iou_threshold
    
    # Update version-specific paths
    config.VERSION_STATS_DIR = f"{config.STATS_DIR}/{config.MODEL_VERSION}"
    config.MODEL_SAVE_PATH = f"{config.MODELS_DIR}/{config.MODEL_NAME}_{config.MODEL_VERSION}.pt"
    config.BEST_MODEL_PATH = f"{config.MODELS_DIR}/{config.MODEL_NAME}_{config.MODEL_VERSION}_best.pt"


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Update config
    update_config(args)
    
    # Import and run training
    import train
    train.main()


if __name__ == "__main__":
    main()
