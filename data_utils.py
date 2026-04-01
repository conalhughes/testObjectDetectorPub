"""
Data processing utilities for YOLOv8 Object Detection
Handles data validation, preprocessing, and augmentation
"""

import os
import glob
import yaml
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from datetime import datetime

import config
from logger_utils import setup_logger

# Configure logging
logger = setup_logger(__name__, f'data_utils_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')


def validate_dataset_structure():
    """
    Validate that the dataset directory structure exists and is correct.
    
    Expected structure:
    data/
        train/
            images/
            labels/
        val/
            images/
            labels/
        test/
            images/
            labels/
    """
    logger.info("=" * 70)
    logger.info("VALIDATING DATASET STRUCTURE")
    logger.info("=" * 70)
    
    required_dirs = [
        config.TRAIN_IMAGES,
        config.TRAIN_LABELS,
        config.VAL_IMAGES,
        config.VAL_LABELS,
        config.TEST_IMAGES,
        config.TEST_LABELS
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        logger.info(f"{status} {dir_path}")
        all_exist = all_exist and exists
    
    if all_exist:
        logger.info("\n✓ All required directories exist")
    else:
        logger.warning("\n✗ Some directories are missing")
        return False
    
    return True


def count_dataset_samples():
    """
    Count the number of images and labels in each split.
    Returns a dictionary with the counts.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COUNTING DATASET SAMPLES")
    logger.info("=" * 70)
    
    splits = {
        'train': (config.TRAIN_IMAGES, config.TRAIN_LABELS),
        'val': (config.VAL_IMAGES, config.VAL_LABELS),
        'test': (config.TEST_IMAGES, config.TEST_LABELS)
    }
    
    counts = {}
    
    for split_name, (img_dir, lbl_dir) in splits.items():
        # Count images (common formats)
        img_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_files = []
        for pattern in img_patterns:
            img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
        
        # Count label files
        lbl_files = glob.glob(os.path.join(lbl_dir, '*.txt'))
        
        counts[split_name] = {
            'images': len(img_files),
            'labels': len(lbl_files)
        }
        
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Images: {counts[split_name]['images']}")
        logger.info(f"  Labels: {counts[split_name]['labels']}")
        
        # Check for mismatch
        if counts[split_name]['images'] != counts[split_name]['labels']:
            logger.warning(f"  ⚠ WARNING: Mismatch between images and labels!")
    
    return counts


def analyze_label_distribution(split='train'):
    """
    Analyze the distribution of classes and bounding boxes in the dataset.
    
    Args:
        split: Dataset split to analyze ('train', 'val', or 'test')
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"ANALYZING {split.upper()} LABEL DISTRIBUTION")
    logger.info("=" * 70)
    
    # Get label directory
    label_dirs = {
        'train': config.TRAIN_LABELS,
        'val': config.VAL_LABELS,
        'test': config.TEST_LABELS
    }
    label_dir = label_dirs[split]
    
    # Read all label files
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    if len(label_files) == 0:
        logger.warning(f"No label files found in {label_dir}")
        return None
    
    class_counts = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    objects_per_image = []
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        objects_per_image.append(len(lines))
        
        for line in lines:
            # YOLO format: class_id x_center y_center width height (normalized)
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                class_counts.append(class_id)
                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(width * height)
    
    # Statistics
    if len(class_counts) > 0:
        class_distribution = Counter(class_counts)
        
        logger.info(f"\nTotal annotations: {len(class_counts)}")
        logger.info(f"Total images with labels: {len(label_files)}")
        logger.info(f"Average objects per image: {np.mean(objects_per_image):.2f}")
        
        logger.info(f"\nClass distribution:")
        for class_id, count in sorted(class_distribution.items()):
            class_name = config.CLASS_NAMES[class_id] if class_id < len(config.CLASS_NAMES) else f"class_{class_id}"
            logger.info(f"  {class_name} (ID {class_id}): {count} instances ({count/len(class_counts)*100:.1f}%)")
        
        logger.info(f"\nBounding box statistics (normalized):")
        logger.info(f"  Width  - Mean: {np.mean(bbox_widths):.3f}, Std: {np.std(bbox_widths):.3f}")
        logger.info(f"  Height - Mean: {np.mean(bbox_heights):.3f}, Std: {np.std(bbox_heights):.3f}")
        logger.info(f"  Area   - Mean: {np.mean(bbox_areas):.3f}, Std: {np.std(bbox_areas):.3f}")
        
        return {
            'class_counts': class_counts,
            'bbox_widths': bbox_widths,
            'bbox_heights': bbox_heights,
            'bbox_areas': bbox_areas,
            'objects_per_image': objects_per_image
        }
    else:
        logger.warning("No valid annotations found")
        return None


def plot_dataset_statistics():
    """
    Create and save visualization plots for dataset statistics.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING DATASET STATISTICS PLOTS")
    logger.info("=" * 70)
    
    # Set plot style
    try:
        plt.style.use(config.PLOT_STYLE)
    except:
        plt.style.use('default')
    
    # Analyze all splits
    stats = {}
    for split in ['train', 'val', 'test']:
        stats[split] = analyze_label_distribution(split)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Dataset split sizes
    ax1 = plt.subplot(3, 3, 1)
    counts = count_dataset_samples()
    splits = list(counts.keys())
    image_counts = [counts[s]['images'] for s in splits]
    ax1.bar(splits, image_counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Split Sizes')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(image_counts):
        ax1.text(i, v + max(image_counts)*0.01, str(v), ha='center', va='bottom')
    
    # Plot 2: Class distribution across splits
    ax2 = plt.subplot(3, 3, 2)
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split]['class_counts']:
            class_dist = Counter(stats[split]['class_counts'])
            classes = sorted(class_dist.keys())
            class_counts_list = [class_dist[c] for c in classes]
            ax2.bar([f"C{c}" for c in classes], class_counts_list, alpha=0.7, label=split)
    ax2.set_xlabel('Class ID')
    ax2.set_ylabel('Number of Instances')
    ax2.set_title('Class Distribution Across Splits')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Objects per image distribution (train)
    ax3 = plt.subplot(3, 3, 3)
    if stats['train'] and stats['train']['objects_per_image']:
        ax3.hist(stats['train']['objects_per_image'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Objects per Image')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Objects per Image Distribution (Train)')
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Bounding box width distribution
    ax4 = plt.subplot(3, 3, 4)
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split]['bbox_widths']:
            ax4.hist(stats[split]['bbox_widths'], bins=30, alpha=0.5, label=split)
    ax4.set_xlabel('Width (normalized)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bounding Box Width Distribution')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Bounding box height distribution
    ax5 = plt.subplot(3, 3, 5)
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split]['bbox_heights']:
            ax5.hist(stats[split]['bbox_heights'], bins=30, alpha=0.5, label=split)
    ax5.set_xlabel('Height (normalized)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Bounding Box Height Distribution')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Bounding box area distribution
    ax6 = plt.subplot(3, 3, 6)
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split]['bbox_areas']:
            ax6.hist(stats[split]['bbox_areas'], bins=30, alpha=0.5, label=split)
    ax6.set_xlabel('Area (normalized)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Bounding Box Area Distribution')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Plot 7: Width vs Height scatter (train)
    ax7 = plt.subplot(3, 3, 7)
    if stats['train'] and stats['train']['bbox_widths']:
        ax7.scatter(stats['train']['bbox_widths'], stats['train']['bbox_heights'], 
                   alpha=0.5, s=10, c='#3498db')
        ax7.set_xlabel('Width (normalized)')
        ax7.set_ylabel('Height (normalized)')
        ax7.set_title('Bounding Box Aspect Ratio (Train)')
        ax7.grid(alpha=0.3)
    
    # Plot 8: Aspect ratio distribution
    ax8 = plt.subplot(3, 3, 8)
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split]['bbox_widths']:
            widths = np.array(stats[split]['bbox_widths'])
            heights = np.array(stats[split]['bbox_heights'])
            aspect_ratios = widths / (heights + 1e-6)
            ax8.hist(aspect_ratios, bins=30, alpha=0.5, label=split)
    ax8.set_xlabel('Aspect Ratio (Width/Height)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Bounding Box Aspect Ratio Distribution')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # Plot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    table_data = [['Metric', 'Train', 'Val', 'Test']]
    table_data.append(['Images', 
                      counts['train']['images'],
                      counts['val']['images'],
                      counts['test']['images']])
    
    for split in ['train', 'val', 'test']:
        if stats[split]:
            total_objects = len(stats[split]['class_counts']) if stats[split]['class_counts'] else 0
            if split == 'train':
                table_data.append(['Total Objects', total_objects, '', ''])
            elif split == 'val':
                table_data[2][2] = total_objects
            else:
                table_data[2][3] = total_objects
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Dataset Statistics - {config.MODEL_NAME} {config.MODEL_VERSION}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'dataset_statistics_{config.MODEL_VERSION}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"\n✓ Dataset statistics plot saved to: {plot_path}")
    plt.close()


def create_data_yaml():
    """
    Create the data.yaml file required by YOLOv8.
    This file specifies dataset paths and class information.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CREATING DATA YAML FILE")
    logger.info("=" * 70)
    
    data_yaml = {
        'path': os.path.abspath(config.DATA_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': config.NUM_CLASSES,
        'names': config.CLASS_NAMES
    }
    
    yaml_path = os.path.join(config.DATA_DIR, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✓ Data YAML file created at: {yaml_path}")
    logger.info("\nContents:")
    logger.info(yaml.dump(data_yaml, default_flow_style=False, sort_keys=False))
    
    return yaml_path


def visualize_sample_images(num_samples=6):
    """
    Visualize sample images with their bounding boxes from the training set.
    
    Args:
        num_samples: Number of sample images to visualize
    """
    logger.info("\n" + "=" * 70)
    logger.info("VISUALIZING SAMPLE IMAGES")
    logger.info("=" * 70)
    
    # Get image files
    img_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for pattern in img_patterns:
        img_files.extend(glob.glob(os.path.join(config.TRAIN_IMAGES, pattern)))
    
    if len(img_files) == 0:
        logger.warning("No images found in training directory")
        return
    
    # Randomly select samples
    np.random.seed(42)
    sample_files = np.random.choice(img_files, min(num_samples, len(img_files)), replace=False)
    
    # Create figure
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for idx, img_path in enumerate(sample_files):
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Get corresponding label file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(config.TRAIN_LABELS, f"{img_name}.txt")
        
        # Draw bounding boxes
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    
                    # Calculate corner coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Draw rectangle
                    color = colors[class_id % len(colors)]
                    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 2)
                    
                    # Draw label
                    class_name = config.CLASS_NAMES[class_id] if class_id < len(config.CLASS_NAMES) else f"C{class_id}"
                    cv2.putText(img, class_name, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
        
        # Display
        axes[idx].imshow(img)
        axes[idx].set_title(os.path.basename(img_path))
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Training Images with Annotations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'sample_images_{config.MODEL_VERSION}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"✓ Sample images plot saved to: {plot_path}")
    plt.close()


def prepare_dataset():
    """
    Main function to prepare and validate the dataset.
    Runs all data processing steps.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATASET PREPARATION")
    logger.info("=" * 70)
    
    # Step 1: Validate structure
    if not validate_dataset_structure():
        logger.warning("\n⚠ WARNING: Dataset structure validation failed!")
        logger.warning("Please ensure the data directory structure is correct.")
        return False
    
    # Step 2: Count samples
    counts = count_dataset_samples()
    
    # Check if dataset is empty
    if counts['train']['images'] == 0:
        logger.warning("\n⚠ WARNING: No training images found!")
        logger.warning("Please add images to the data/train/images directory.")
        return False
    
    # Step 3: Create data.yaml
    create_data_yaml()
    
    # Step 4: Generate statistics plots
    plot_dataset_statistics()
    
    # Step 5: Visualize sample images
    visualize_sample_images()
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ DATASET PREPARATION COMPLETE")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    prepare_dataset()
