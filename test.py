"""
Testing and Evaluation script for YOLOv8 Object Detection
Evaluates the trained model on test set and generates comprehensive metrics
"""

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict

import config
from logger_utils import setup_logger

# Configure logging
logger = setup_logger(__name__, f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Determine device (auto-fallback to CPU if CUDA not available)
def get_device():
    """
    Get the device to use for testing.
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


def load_trained_model():
    """
    Load the trained YOLOv8 model.
    
    Returns:
        YOLO model instance or None if loading fails
    """
    logger.info("=" * 70)
    logger.info("LOADING TRAINED MODEL")
    logger.info("=" * 70)
    
    model_path = config.BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        logger.error("Please train the model first using train.py")
        return None
    
    try:
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def evaluate_on_test_set(model, device):
    """
    Evaluate the model on the test dataset using official YOLO validation.
    
    Args:
        model: YOLO model instance
        device: Device to use for evaluation ('cuda' or 'cpu')
        
    Returns:
        Validation metrics
    """
    logger.info("=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)
    
    data_yaml_path = os.path.join(config.DATA_DIR, 'data.yaml')
    
    try:
        # Run validation on test split
        # Note: YOLO uses 'val' split by default, but we can specify test images
        metrics = model.val(
            data=data_yaml_path,
            split='test',
            imgsz=config.INPUT_SIZE,
            batch=config.BATCH_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=device,  # Use the actual device (with fallback)
            plots=True,
            save_json=True,
            save_hybrid=False,
        )
        
        logger.info("-" * 70)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("-" * 70)
        
        # Log metrics
        logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        if hasattr(metrics.box, 'maps'):
            logger.info("Per-class mAP@0.5:")
            for i, map_val in enumerate(metrics.box.maps):
                class_name = config.CLASS_NAMES[i] if i < len(config.CLASS_NAMES) else f"class_{i}"
                logger.info(f"  {class_name}: {map_val:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_inference_on_test_images(model, device, num_samples=12):
    """
    Run inference on sample test images and visualize predictions.
    
    Args:
        model: YOLO model instance
        device: Device to use for inference ('cuda' or 'cpu')
        num_samples: Number of sample images to process
    """
    logger.info("=" * 70)
    logger.info("RUNNING INFERENCE ON TEST IMAGES")
    logger.info("=" * 70)
    
    # Get test image files
    img_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for pattern in img_patterns:
        img_files.extend(glob.glob(os.path.join(config.TEST_IMAGES, pattern)))
    
    if len(img_files) == 0:
        logger.error(f"No test images found in {config.TEST_IMAGES}")
        return None
    
    logger.info(f"Found {len(img_files)} test images")
    
    # Randomly select samples
    np.random.seed(42)
    sample_files = np.random.choice(img_files, min(num_samples, len(img_files)), replace=False)
    
    # Run inference
    results_list = []
    predictions = []
    
    for img_path in sample_files:
        result = model.predict(
            img_path,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INPUT_SIZE,
            device=device,  # Use the actual device (with fallback)
            verbose=False
        )[0]
        
        results_list.append(result)
        
        # Extract predictions
        boxes = result.boxes
        img_predictions = {
            'image': os.path.basename(img_path),
            'num_detections': len(boxes),
            'boxes': boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([]),
            'confidences': boxes.conf.cpu().numpy() if len(boxes) > 0 else np.array([]),
            'classes': boxes.cls.cpu().numpy() if len(boxes) > 0 else np.array([]),
        }
        predictions.append(img_predictions)
    
    logger.info(f"Inference completed on {len(sample_files)} images")
    
    return results_list, predictions, sample_files


def visualize_predictions(results_list, sample_files):
    """
    Create visualization of model predictions on test images.
    
    Args:
        results_list: List of YOLO result objects
        sample_files: List of image file paths
    """
    logger.info("=" * 70)
    logger.info("VISUALIZING PREDICTIONS")
    logger.info("=" * 70)
    
    num_samples = len(results_list)
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (result, img_path) in enumerate(zip(results_list, sample_files)):
        # Get annotated image
        annotated_img = result.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(annotated_img)
        
        # Add title with detection count
        num_detections = len(result.boxes)
        axes[idx].set_title(f"{os.path.basename(img_path)}\nDetections: {num_detections}")
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Test Set Predictions - {config.MODEL_NAME} {config.MODEL_VERSION}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'test_predictions_{config.MODEL_VERSION}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"Predictions visualization saved to: {plot_path}")
    plt.close()


def analyze_prediction_statistics(predictions):
    """
    Analyze and visualize statistics from predictions.
    
    Args:
        predictions: List of prediction dictionaries
    """
    logger.info("=" * 70)
    logger.info("ANALYZING PREDICTION STATISTICS")
    logger.info("=" * 70)
    
    # Collect statistics
    num_detections = [p['num_detections'] for p in predictions]
    all_confidences = []
    all_classes = []
    all_box_areas = []
    
    for pred in predictions:
        all_confidences.extend(pred['confidences'])
        all_classes.extend(pred['classes'])
        
        # Calculate box areas
        for box in pred['boxes']:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            all_box_areas.append(area)
    
    # Log statistics
    logger.info(f"Total test images: {len(predictions)}")
    logger.info(f"Total detections: {sum(num_detections)}")
    logger.info(f"Average detections per image: {np.mean(num_detections):.2f}")
    logger.info(f"Max detections in single image: {max(num_detections) if num_detections else 0}")
    logger.info(f"Images with no detections: {sum(1 for n in num_detections if n == 0)}")
    
    if len(all_confidences) > 0:
        logger.info("Confidence Statistics:")
        logger.info(f"  Mean: {np.mean(all_confidences):.4f}")
        logger.info(f"  Median: {np.median(all_confidences):.4f}")
        logger.info(f"  Min: {np.min(all_confidences):.4f}")
        logger.info(f"  Max: {np.max(all_confidences):.4f}")
        logger.info(f"  Std: {np.std(all_confidences):.4f}")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Detections per image
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(num_detections, bins=max(15, max(num_detections) if num_detections else 1), 
            color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Detections')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Detections per Image Distribution')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Confidence distribution
    ax2 = plt.subplot(2, 3, 2)
    if len(all_confidences) > 0:
        ax2.hist(all_confidences, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.axvline(config.CONF_THRESHOLD, color='red', linestyle='--', 
                   label=f'Threshold: {config.CONF_THRESHOLD}')
        ax2.legend()
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Confidence Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Class distribution
    ax3 = plt.subplot(2, 3, 3)
    if len(all_classes) > 0:
        class_counts = defaultdict(int)
        for cls in all_classes:
            class_counts[int(cls)] += 1
        
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        class_labels = [config.CLASS_NAMES[c] if c < len(config.CLASS_NAMES) else f"C{c}" 
                       for c in classes]
        
        ax3.bar(class_labels, counts, color='#e74c3c', alpha=0.7)
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Number of Detections')
        ax3.set_title('Class Distribution in Predictions')
        ax3.grid(axis='y', alpha=0.3)
        if len(class_labels) > 5:
            ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Box area distribution
    ax4 = plt.subplot(2, 3, 4)
    if len(all_box_areas) > 0:
        ax4.hist(all_box_areas, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Bounding Box Area (pixels²)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bounding Box Area Distribution')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Confidence vs Detection count scatter
    ax5 = plt.subplot(2, 3, 5)
    img_avg_conf = []
    for pred in predictions:
        if pred['num_detections'] > 0:
            avg_conf = np.mean(pred['confidences'])
            img_avg_conf.append((pred['num_detections'], avg_conf))
    
    if img_avg_conf:
        det_counts, avg_confs = zip(*img_avg_conf)
        ax5.scatter(det_counts, avg_confs, alpha=0.6, s=50, color='#e67e22')
    ax5.set_xlabel('Number of Detections')
    ax5.set_ylabel('Average Confidence')
    ax5.set_title('Detections vs Average Confidence')
    ax5.grid(alpha=0.3)
    
    # Plot 6: Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Total Images', len(predictions)],
        ['Total Detections', sum(num_detections)],
        ['Avg Detections/Image', f'{np.mean(num_detections):.2f}'],
        ['Images with 0 Detections', sum(1 for n in num_detections if n == 0)],
    ]
    
    if len(all_confidences) > 0:
        table_data.append(['Mean Confidence', f'{np.mean(all_confidences):.4f}'])
        table_data.append(['Median Confidence', f'{np.median(all_confidences):.4f}'])
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Prediction Statistics - {config.MODEL_NAME} {config.MODEL_VERSION}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'prediction_statistics_{config.MODEL_VERSION}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"Prediction statistics plot saved to: {plot_path}")
    plt.close()


def save_test_results(metrics, predictions):
    """
    Save test results to JSON and text files.
    
    Args:
        metrics: Validation metrics object
        predictions: List of prediction dictionaries
    """
    logger.info("=" * 70)
    logger.info("SAVING TEST RESULTS")
    logger.info("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to JSON
    metrics_dict = {
        'model_name': config.MODEL_NAME,
        'model_version': config.MODEL_VERSION,
        'timestamp': timestamp,
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'input_size': config.INPUT_SIZE,
            'conf_threshold': config.CONF_THRESHOLD,
            'iou_threshold': config.IOU_THRESHOLD,
            'num_classes': config.NUM_CLASSES,
            'class_names': config.CLASS_NAMES,
        },
        'metrics': {
            'mAP_50': float(metrics.box.map50),
            'mAP_50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
    }
    
    # Add per-class metrics if available
    if hasattr(metrics.box, 'maps'):
        metrics_dict['per_class_mAP'] = {
            config.CLASS_NAMES[i] if i < len(config.CLASS_NAMES) else f"class_{i}": float(map_val)
            for i, map_val in enumerate(metrics.box.maps)
        }
    
    json_path = os.path.join(config.VERSION_STATS_DIR, f'test_results_{config.MODEL_VERSION}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Metrics saved to: {json_path}")
    
    # Save detailed text report
    report_path = os.path.join(config.VERSION_STATS_DIR, f'test_report_{config.MODEL_VERSION}_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"TEST EVALUATION REPORT\n")
        f.write(f"Model: {config.MODEL_NAME} {config.MODEL_VERSION}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Input Size: {config.INPUT_SIZE}\n")
        f.write(f"Confidence Threshold: {config.CONF_THRESHOLD}\n")
        f.write(f"IoU Threshold: {config.IOU_THRESHOLD}\n")
        f.write(f"Number of Classes: {config.NUM_CLASSES}\n")
        f.write(f"Classes: {', '.join(config.CLASS_NAMES)}\n\n")
        
        f.write("TEST SET METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        if hasattr(metrics.box, 'maps'):
            f.write("PER-CLASS mAP@0.5\n")
            f.write("-" * 70 + "\n")
            for i, map_val in enumerate(metrics.box.maps):
                class_name = config.CLASS_NAMES[i] if i < len(config.CLASS_NAMES) else f"class_{i}"
                f.write(f"{class_name}: {map_val:.4f}\n")
            f.write("\n")
        
        f.write("PREDICTION STATISTICS\n")
        f.write("-" * 70 + "\n")
        num_detections = [p['num_detections'] for p in predictions]
        all_confidences = [c for p in predictions for c in p['confidences']]
        
        f.write(f"Total test images: {len(predictions)}\n")
        f.write(f"Total detections: {sum(num_detections)}\n")
        f.write(f"Average detections per image: {np.mean(num_detections):.2f}\n")
        f.write(f"Images with no detections: {sum(1 for n in num_detections if n == 0)}\n")
        
        if len(all_confidences) > 0:
            f.write(f"\nConfidence Statistics:\n")
            f.write(f"  Mean: {np.mean(all_confidences):.4f}\n")
            f.write(f"  Median: {np.median(all_confidences):.4f}\n")
            f.write(f"  Min: {np.min(all_confidences):.4f}\n")
            f.write(f"  Max: {np.max(all_confidences):.4f}\n")
            f.write(f"  Std: {np.std(all_confidences):.4f}\n")
    
    logger.info(f"Detailed report saved to: {report_path}")


def main():
    """
    Main testing pipeline.
    """
    # Ensure version-specific stats directory exists for log file
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("YOLOv8 OBJECT DETECTION TESTING")
    logger.info(f"Model: {config.MODEL_NAME} {config.MODEL_VERSION}")
    logger.info("=" * 70)
    
    # Determine device
    device = get_device()
    
    # Step 1: Load model
    model = load_trained_model()
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Step 2: Evaluate on test set
    metrics = evaluate_on_test_set(model, device)
    if metrics is None:
        logger.error("Evaluation failed. Exiting.")
        return
    
    # Step 3: Run inference on sample images
    results_list, predictions, sample_files = run_inference_on_test_images(model, device)
    
    # Step 4: Visualize predictions
    if results_list:
        visualize_predictions(results_list, sample_files)
    
    # Step 5: Analyze prediction statistics
    if predictions:
        analyze_prediction_statistics(predictions)
    
    # Step 6: Save results
    if predictions:
        save_test_results(metrics, predictions)
    
    logger.info("=" * 70)
    logger.info("TESTING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Test results saved in: {config.VERSION_STATS_DIR}")
    logger.info("")
    logger.info("Key Metrics:")
    logger.info(f"  mAP@0.5: {metrics.box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    logger.info(f"  Precision: {metrics.box.mp:.4f}")
    logger.info(f"  Recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
