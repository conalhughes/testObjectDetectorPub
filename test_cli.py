#!/usr/bin/env python
"""
Test with CLI arguments
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
        description='Test YOLOv8 Object Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default=config.MODEL_NAME,
                        help='Model name')
    parser.add_argument('--model-version', type=str, default=config.MODEL_VERSION,
                        help='Model version')
    
    # Image size
    parser.add_argument('--img-width', type=int, default=config.INPUT_SIZE[0],
                        help='Input image width')
    parser.add_argument('--img-height', type=int, default=config.INPUT_SIZE[1],
                        help='Input image height')
    
    # Testing configuration
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        choices=['cuda', 'cpu'],
                        help='Device to use for testing')
    parser.add_argument('--conf-threshold', type=float, default=config.CONF_THRESHOLD,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=config.IOU_THRESHOLD,
                        help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=config.MAX_DET,
                        help='Maximum detections per image')
    
    args = parser.parse_args()
    return args


def update_config(args):
    """Update config module with parsed arguments."""
    # Model configuration
    config.MODEL_NAME = args.model_name
    config.MODEL_VERSION = args.model_version
    
    # Image size
    config.INPUT_SIZE = (args.img_width, args.img_height)
    
    # Testing configuration
    config.BATCH_SIZE = args.batch_size
    config.DEVICE = args.device
    config.CONF_THRESHOLD = args.conf_threshold
    config.IOU_THRESHOLD = args.iou_threshold
    config.MAX_DET = args.max_det
    
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
    
    # Import and run testing
    import test
    test.main()


if __name__ == "__main__":
    main()
