#!/usr/bin/env python3
"""
Setup script for Kaggle environment
Downloads dataset from Kaggle and sets up raw_data/ directory

Usage:
    # Auto-detect dataset (searches /kaggle/input recursively)
    python setup_kaggle_data.py
    
    # Specify dataset path explicitly
    python setup_kaggle_data.py /kaggle/input/datasets/conalhughes/testballdataset1

This script:
- Detects if running in Kaggle environment
- Auto-detects dataset location (recursive search for images/labels dirs)
- Creates raw_data/ directory structure
- Symlinks Kaggle dataset to raw_data/ (saves disk space)
- Validates images and labels exist
- Works with any dataset path structure
"""

import os
import shutil
import sys

def setup_kaggle_data(kaggle_dataset_path):
    """
    Setup data from Kaggle dataset.
    
    Args:
        kaggle_dataset_path: Path to Kaggle input directory (e.g., '/kaggle/input/your-dataset')
    """
    print("=" * 70)
    print("KAGGLE DATASET SETUP")
    print("=" * 70)
    print()
    
    # Check if we're in Kaggle environment
    is_kaggle = os.path.exists('/kaggle/input')
    
    if is_kaggle:
        print("✓ Kaggle environment detected")
        print(f"Dataset path: {kaggle_dataset_path}")
    else:
        print("✓ Local environment detected")
        print("Skipping Kaggle dataset setup")
        return
    
    # Check if dataset exists
    if not os.path.exists(kaggle_dataset_path):
        print(f"✗ Error: Dataset not found at {kaggle_dataset_path}")
        print("\nAvailable datasets:")
        if os.path.exists('/kaggle/input'):
            for item in os.listdir('/kaggle/input'):
                print(f"  - /kaggle/input/{item}")
        sys.exit(1)
    
    # Create raw_data directory structure
    print("\nCreating raw_data/ structure...")
    os.makedirs('raw_data/images', exist_ok=True)
    os.makedirs('raw_data/labels', exist_ok=True)
    
    # Copy or symlink data from Kaggle input
    kaggle_images = os.path.join(kaggle_dataset_path, 'images')
    kaggle_labels = os.path.join(kaggle_dataset_path, 'labels')
    
    if os.path.exists(kaggle_images):
        print(f"Linking images from {kaggle_images}")
        # Use symlinks to save space (Kaggle has limited disk)
        if os.path.islink('raw_data/images'):
            os.unlink('raw_data/images')
        elif os.path.exists('raw_data/images'):
            shutil.rmtree('raw_data/images')
        os.symlink(kaggle_images, 'raw_data/images')
        
        image_count = len([f for f in os.listdir(kaggle_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  ✓ {image_count} images linked")
    else:
        print(f"✗ Warning: Images not found at {kaggle_images}")
    
    if os.path.exists(kaggle_labels):
        print(f"Linking labels from {kaggle_labels}")
        if os.path.islink('raw_data/labels'):
            os.unlink('raw_data/labels')
        elif os.path.exists('raw_data/labels'):
            shutil.rmtree('raw_data/labels')
        os.symlink(kaggle_labels, 'raw_data/labels')
        
        label_count = len([f for f in os.listdir(kaggle_labels) if f.endswith('.txt')])
        print(f"  ✓ {label_count} labels linked")
    else:
        print(f"✗ Warning: Labels not found at {kaggle_labels}")
    
    print()
    print("=" * 70)
    print("✓ SETUP COMPLETE")
    print("=" * 70)
    print()
    print("You can now run training:")
    print("  !./train.sh --epochs 100 --device cuda")
    print()


def find_kaggle_dataset(root_path='/kaggle/input', max_depth=5):
    """
    Recursively search for a dataset directory.
    Looks for directories containing 'images' or 'labels' subdirectories.
    
    Args:
        root_path: Root path to search from
        max_depth: Maximum directory depth to search
    
    Returns:
        Path to dataset directory or None if not found
    """
    def search_recursive(path, current_depth):
        if current_depth > max_depth or not os.path.isdir(path):
            return None
        
        # Check if this directory looks like a dataset
        has_images = os.path.isdir(os.path.join(path, 'images'))
        has_labels = os.path.isdir(os.path.join(path, 'labels'))
        
        if has_images or has_labels:
            return path
        
        # Search subdirectories
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    result = search_recursive(item_path, current_depth + 1)
                    if result:
                        return result
        except (PermissionError, OSError):
            pass
        
        return None
    
    if os.path.exists(root_path):
        return search_recursive(root_path, 0)
    return None


def main():
    """Main entry point."""
    # Allow user to specify dataset path as argument
    # Or auto-detect if running in Kaggle environment
    
    if len(sys.argv) > 1:
        kaggle_dataset_path = sys.argv[1]
    else:
        # Try to auto-detect dataset in Kaggle environment
        if os.path.exists('/kaggle/input'):
            print("Auto-detecting dataset in /kaggle/input...")
            kaggle_dataset_path = find_kaggle_dataset('/kaggle/input')
            
            if kaggle_dataset_path:
                print(f"✓ Auto-detected dataset: {kaggle_dataset_path}")
            else:
                print("✗ No dataset found in /kaggle/input")
                print("\nUsage: python setup_kaggle_data.py /path/to/dataset")
                print("\nDataset should have this structure:")
                print("  dataset/")
                print("    ├── images/")
                print("    └── labels/")
                sys.exit(1)
        else:
            print("Not in Kaggle environment. No setup needed.")
            print("Your local raw_data/ directory will be used.")
            return
    
    setup_kaggle_data(kaggle_dataset_path)


if __name__ == "__main__":
    main()
