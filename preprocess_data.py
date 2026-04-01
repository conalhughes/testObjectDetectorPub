"""
Data Preprocessing Script for YOLOv8 Object Detection
Processes raw dataset, creates backup, and distributes into train/val/test splits

This script:
1. Creates a backup of your raw data
2. Validates and cleans the dataset
3. Optionally resizes images
4. Splits data into train/val/test sets
5. Generates statistics and visualizations
"""

import os
import glob
import shutil
import random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import config
from logger_utils import setup_logger

# Configure logging
logger = setup_logger(__name__, f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')


def create_backup():
    """
    Create a backup of the raw data directory.
    Backs up to backup/ directory with timestamp.
    """
    logger.info("=" * 70)
    logger.info("CREATING BACKUP")
    logger.info("=" * 70)
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(config.BACKUP_DIR, f"raw_data_backup_{timestamp}")
    
    if not os.path.exists(config.RAW_DATA_DIR):
        logger.error(f"Raw data directory not found: {config.RAW_DATA_DIR}")
        return False
    
    # Check if raw data exists
    raw_images = glob.glob(os.path.join(config.RAW_IMAGES, '*.*'))
    raw_labels = glob.glob(os.path.join(config.RAW_LABELS, '*.txt'))
    
    if len(raw_images) == 0:
        logger.error(f"No images found in {config.RAW_IMAGES}")
        return False
    
    logger.info(f"Found {len(raw_images)} images and {len(raw_labels)} label files")
    logger.info(f"Creating backup at: {backup_path}")
    
    try:
        # Copy entire raw_data directory
        shutil.copytree(config.RAW_DATA_DIR, backup_path)
        logger.info("Backup created successfully")
        logger.info(f"Location: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def validate_and_clean_labels(image_path, label_path):
    """
    Validate label file and clean invalid annotations.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the label file
        
    Returns:
        List of valid annotation lines, or None if label is invalid
    """
    if not os.path.exists(label_path):
        return None
    
    # Read image dimensions
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Read and validate labels
    valid_lines = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            
            # Check format
            if len(parts) < 5:
                continue
            
            # Parse values
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Validate coordinates (should be normalized 0-1)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                   0 < width <= 1 and 0 < height <= 1):
                continue
            
            # Check minimum box size
            if width < config.MIN_BOX_SIZE or height < config.MIN_BOX_SIZE:
                continue
            
            # Check class ID is valid
            if class_id < 0 or class_id >= config.NUM_CLASSES:
                continue
            
            valid_lines.append(line.strip())
        
        return valid_lines if len(valid_lines) > 0 else None
        
    except Exception as e:
        logger.debug(f"Failed to validate label {label_path}: {e}")
        return None


def resize_image_and_labels(image_path, labels, target_size, maintain_aspect=True):
    """
    Resize image and adjust label coordinates accordingly.
    
    Args:
        image_path: Path to the image
        labels: List of label strings
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio (pad with black)
        
    Returns:
        Resized image and adjusted labels
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    original_h, original_w = img.shape[:2]
    target_w, target_h = target_size
    
    if maintain_aspect:
        # Calculate scaling factor to fit image in target size
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create black canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        
        # Place resized image on canvas
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        # Adjust label coordinates
        adjusted_labels = []
        for label in labels:
            parts = label.split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert to pixel coordinates
            x_center_px = x_center * original_w
            y_center_px = y_center * original_h
            width_px = width * original_w
            height_px = height * original_h
            
            # Apply scaling
            x_center_px = x_center_px * scale + pad_left
            y_center_px = y_center_px * scale + pad_top
            width_px *= scale
            height_px *= scale
            
            # Convert back to normalized coordinates
            x_center_new = x_center_px / target_w
            y_center_new = y_center_px / target_h
            width_new = width_px / target_w
            height_new = height_px / target_h
            
            adjusted_labels.append(f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}")
        
        return canvas, adjusted_labels
    
    else:
        # Simple resize without maintaining aspect ratio
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Labels don't need adjustment (normalized coordinates scale automatically)
        return resized, labels


def process_and_split_dataset():
    """
    Process raw dataset and split into train/val/test sets.
    """
    logger.info("=" * 70)
    logger.info("PROCESSING AND SPLITTING DATASET")
    logger.info("=" * 70)
    
    # Get all image files
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob.glob(os.path.join(config.RAW_IMAGES, ext)))
    
    logger.info(f"Found {len(all_images)} images in raw data")
    
    if len(all_images) == 0:
        logger.error("No images found. Please add images to raw_data/images/")
        return False
    
    # Process each image
    processed_data = []
    skipped_no_label = 0
    skipped_invalid_label = 0
    skipped_invalid_image = 0
    
    logger.info("Validating and cleaning dataset...")
    for img_path in tqdm(all_images, desc="Processing"):
        # Get corresponding label path
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(config.RAW_LABELS, f"{img_name}.txt")
        
        # Validate labels
        valid_labels = validate_and_clean_labels(img_path, label_path)
        
        if valid_labels is None:
            if not os.path.exists(label_path):
                skipped_no_label += 1
            else:
                skipped_invalid_label += 1
            
            if not config.REMOVE_EMPTY_LABELS:
                # Keep images without labels if configured
                valid_labels = []
            else:
                continue
        
        # Verify image can be read
        test_img = cv2.imread(img_path)
        if test_img is None:
            skipped_invalid_image += 1
            continue
        
        processed_data.append({
            'image_path': img_path,
            'image_name': img_name,
            'labels': valid_labels,
            'extension': os.path.splitext(img_path)[1]
        })
    
    logger.info(f"Processed {len(processed_data)} valid samples")
    logger.info(f"Skipped (no label): {skipped_no_label}")
    logger.info(f"Skipped (invalid label): {skipped_invalid_label}")
    logger.info(f"Skipped (invalid image): {skipped_invalid_image}")
    
    if len(processed_data) == 0:
        logger.error("No valid samples to process")
        return False
    
    # Shuffle data for random split
    random.seed(42)
    random.shuffle(processed_data)
    
    # Calculate split indices
    total = len(processed_data)
    train_end = int(total * config.TRAIN_RATIO)
    val_end = train_end + int(total * config.VAL_RATIO)
    
    splits = {
        'train': processed_data[:train_end],
        'val': processed_data[train_end:val_end],
        'test': processed_data[val_end:]
    }
    
    logger.info("Dataset split:")
    logger.info(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total*100:.1f}%)")
    logger.info(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/total*100:.1f}%)")
    logger.info(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/total*100:.1f}%)")
    
    # Clear existing data directories
    logger.info("Clearing existing data directories...")
    for split_name in ['train', 'val', 'test']:
        img_dir = os.path.join(config.DATA_DIR, split_name, 'images')
        lbl_dir = os.path.join(config.DATA_DIR, split_name, 'labels')
        
        # Remove and recreate directories
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if os.path.exists(lbl_dir):
            shutil.rmtree(lbl_dir)
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
    
    # Process and copy files to respective splits
    logger.info("Copying and processing files...")
    processing_stats = defaultdict(lambda: {'total': 0, 'resized': 0, 'copied': 0})
    
    for split_name, samples in splits.items():
        logger.info(f"Processing {split_name} split...")
        
        img_dir = os.path.join(config.DATA_DIR, split_name, 'images')
        lbl_dir = os.path.join(config.DATA_DIR, split_name, 'labels')
        
        for sample in tqdm(samples, desc=f"  {split_name}"):
            processing_stats[split_name]['total'] += 1
            
            # Determine output filename
            output_name = sample['image_name']
            output_img_path = os.path.join(img_dir, f"{output_name}.jpg")
            output_lbl_path = os.path.join(lbl_dir, f"{output_name}.txt")
            
            # Process image
            if config.RESIZE_IMAGES:
                # Resize image and adjust labels
                resized_img, adjusted_labels = resize_image_and_labels(
                    sample['image_path'],
                    sample['labels'],
                    config.TARGET_SIZE,
                    config.MAINTAIN_ASPECT_RATIO
                )
                
                if resized_img is not None:
                    # Save resized image
                    cv2.imwrite(output_img_path, resized_img, 
                              [cv2.IMWRITE_JPEG_QUALITY, config.IMAGE_QUALITY])
                    labels_to_save = adjusted_labels
                    processing_stats[split_name]['resized'] += 1
                else:
                    # Fallback to copying original
                    shutil.copy2(sample['image_path'], output_img_path)
                    labels_to_save = sample['labels']
                    processing_stats[split_name]['copied'] += 1
            else:
                # Just copy the original image
                shutil.copy2(sample['image_path'], output_img_path)
                labels_to_save = sample['labels']
                processing_stats[split_name]['copied'] += 1
            
            # Save labels
            if len(labels_to_save) > 0:
                with open(output_lbl_path, 'w') as f:
                    f.write('\n'.join(labels_to_save) + '\n')
    
    # Log processing statistics
    logger.info("=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    for split_name, stats in processing_stats.items():
        logger.info(f"{split_name.upper()}:")
        logger.info(f"  Total samples: {stats['total']}")
        if config.RESIZE_IMAGES:
            logger.info(f"  Resized: {stats['resized']}")
            logger.info(f"  Copied (resize failed): {stats['copied']}")
        else:
            logger.info(f"  Copied: {stats['copied']}")
    
    return True


def generate_preprocessing_report():
    """
    Generate a report with preprocessing statistics and visualizations.
    """
    logger.info("=" * 70)
    logger.info("GENERATING PREPROCESSING REPORT")
    logger.info("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect statistics
    stats = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(config.DATA_DIR, split, 'images')
        lbl_dir = os.path.join(config.DATA_DIR, split, 'labels')
        
        images = glob.glob(os.path.join(img_dir, '*.*'))
        labels = glob.glob(os.path.join(lbl_dir, '*.txt'))
        
        # Analyze first few images for size info
        image_sizes = []
        for img_path in images[:min(50, len(images))]:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                image_sizes.append((w, h))
        
        stats[split] = {
            'num_images': len(images),
            'num_labels': len(labels),
            'image_sizes': image_sizes
        }
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Dataset split distribution
    ax1 = plt.subplot(2, 3, 1)
    splits = ['train', 'val', 'test']
    counts = [stats[s]['num_images'] for s in splits]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(splits, counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Split Distribution')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom')
    
    # Plot 2: Image sizes distribution (if resized)
    ax2 = plt.subplot(2, 3, 2)
    if config.RESIZE_IMAGES:
        ax2.text(0.5, 0.5, f'All images resized to:\n{config.TARGET_SIZE[0]}x{config.TARGET_SIZE[1]}',
                ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title('Image Dimensions')
        ax2.axis('off')
    else:
        all_widths = []
        all_heights = []
        for split_stats in stats.values():
            for w, h in split_stats['image_sizes']:
                all_widths.append(w)
                all_heights.append(h)
        
        if all_widths:
            ax2.scatter(all_widths, all_heights, alpha=0.5, s=20)
            ax2.set_xlabel('Width (pixels)')
            ax2.set_ylabel('Height (pixels)')
            ax2.set_title('Image Dimensions Distribution')
            ax2.grid(alpha=0.3)
    
    # Plot 3: Split ratios pie chart
    ax3 = plt.subplot(2, 3, 3)
    ax3.pie(counts, labels=splits, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12})
    ax3.set_title('Split Ratios')
    
    # Plot 4: Configuration table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    config_data = [
        ['Configuration', 'Value'],
        ['Model Name', config.MODEL_NAME],
        ['Model Version', config.MODEL_VERSION],
        ['Resize Images', 'Yes' if config.RESIZE_IMAGES else 'No'],
    ]
    
    if config.RESIZE_IMAGES:
        config_data.append(['Target Size', f'{config.TARGET_SIZE[0]}x{config.TARGET_SIZE[1]}'])
        config_data.append(['Maintain Aspect', 'Yes' if config.MAINTAIN_ASPECT_RATIO else 'No'])
    
    config_data.extend([
        ['Train Ratio', f'{config.TRAIN_RATIO:.1%}'],
        ['Val Ratio', f'{config.VAL_RATIO:.1%}'],
        ['Test Ratio', f'{config.TEST_RATIO:.1%}'],
        ['Num Classes', config.NUM_CLASSES],
    ])
    
    table = ax4.table(cellText=config_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Plot 5: Statistics table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('tight')
    ax5.axis('off')
    
    stats_data = [['Split', 'Images', 'Labels']]
    for split in splits:
        stats_data.append([
            split.capitalize(),
            stats[split]['num_images'],
            stats[split]['num_labels']
        ])
    stats_data.append([
        'Total',
        sum(stats[s]['num_images'] for s in splits),
        sum(stats[s]['num_labels'] for s in splits)
    ])
    
    table2 = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.33, 0.33, 0.33])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    for i in range(3):
        table2[(0, i)].set_facecolor('#3498db')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight total row
    for i in range(3):
        table2[(len(stats_data)-1, i)].set_facecolor('#ecf0f1')
        table2[(len(stats_data)-1, i)].set_text_props(weight='bold')
    
    # Plot 6: Processing info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = f"""
    Preprocessing Complete!
    
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ✓ Backup created
    ✓ Data validated and cleaned
    ✓ Dataset split into train/val/test
    {'✓ Images resized' if config.RESIZE_IMAGES else '✓ Original images preserved'}
    
    Ready for training!
    Run: python train.py
    """
    
    ax6.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle(f'Data Preprocessing Report - {config.MODEL_NAME} {config.MODEL_VERSION}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    plot_path = os.path.join(config.VERSION_STATS_DIR, f'preprocessing_report_{config.MODEL_VERSION}_{timestamp}.png')
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    logger.info(f"Preprocessing report saved to: {plot_path}")
    plt.close()
    
    # Save text report
    report_path = os.path.join(config.VERSION_STATS_DIR, f'preprocessing_summary_{config.MODEL_VERSION}_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA PREPROCESSING SUMMARY\n")
        f.write(f"Model: {config.MODEL_NAME} {config.MODEL_VERSION}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Resize Images: {config.RESIZE_IMAGES}\n")
        if config.RESIZE_IMAGES:
            f.write(f"Target Size: {config.TARGET_SIZE[0]}x{config.TARGET_SIZE[1]}\n")
            f.write(f"Maintain Aspect Ratio: {config.MAINTAIN_ASPECT_RATIO}\n")
            f.write(f"Image Quality: {config.IMAGE_QUALITY}\n")
        f.write(f"Train Ratio: {config.TRAIN_RATIO:.1%}\n")
        f.write(f"Val Ratio: {config.VAL_RATIO:.1%}\n")
        f.write(f"Test Ratio: {config.TEST_RATIO:.1%}\n")
        f.write(f"Remove Empty Labels: {config.REMOVE_EMPTY_LABELS}\n")
        f.write(f"Minimum Box Size: {config.MIN_BOX_SIZE}\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 70 + "\n")
        for split in splits:
            f.write(f"{split.upper()}:\n")
            f.write(f"  Images: {stats[split]['num_images']}\n")
            f.write(f"  Labels: {stats[split]['num_labels']}\n")
        
        total_images = sum(stats[s]['num_images'] for s in splits)
        total_labels = sum(stats[s]['num_labels'] for s in splits)
        f.write(f"\nTOTAL:\n")
        f.write(f"  Images: {total_images}\n")
        f.write(f"  Labels: {total_labels}\n")
    
    logger.info(f"Text summary saved to: {report_path}")


def main():
    """
    Main preprocessing pipeline.
    """
    # Ensure version-specific stats directory exists for log file
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("YOLOv8 DATA PREPROCESSING PIPELINE")
    logger.info(f"Model: {config.MODEL_NAME} {config.MODEL_VERSION}")
    logger.info("=" * 70)
    
    # Verify split ratios
    if abs(config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO - 1.0) > 0.001:
        logger.error("Split ratios must sum to 1.0")
        logger.error(f"Current sum: {config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO}")
        return
    
    # Step 1: Create backup
    logger.info("Step 1/3: Creating backup...")
    if not create_backup():
        response = input("\nBackup failed. Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            logger.warning("Preprocessing cancelled by user.")
            return
    
    # Step 2: Process and split dataset
    logger.info("Step 2/3: Processing and splitting dataset...")
    if not process_and_split_dataset():
        logger.error("Dataset processing failed. Exiting.")
        return
    
    # Step 3: Generate report
    logger.info("Step 3/3: Generating preprocessing report...")
    generate_preprocessing_report()
    
    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed data saved to: {config.DATA_DIR}")
    logger.info(f"Backup saved to: {config.BACKUP_DIR}")
    logger.info(f"Reports saved to: {config.VERSION_STATS_DIR}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review the preprocessing report")
    logger.info("  2. Run: python data_utils.py (optional, for more analysis)")
    logger.info("  3. Run: python train.py (to start training)")


if __name__ == "__main__":
    main()
