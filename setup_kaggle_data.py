#!/usr/bin/env python3
"""
Setup script for Kaggle environment
Downloads dataset from Kaggle and sets up raw_data/ directory

Usage:
    # Auto-detect single dataset
    python setup_kaggle_data.py
    
    # Specify dataset path
    python setup_kaggle_data.py /kaggle/input/your-dataset-name
    
This script:
- Detects if running in Kaggle environment
- Creates raw_data/ directory structure
- Symlinks Kaggle dataset to raw_data/ (saves disk space)
- Validates images and labels exist
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


def main():
    """Main entry point."""
    # Default Kaggle dataset path - user should modify this
    # Format: /kaggle/input/{dataset-name}
    
    if len(sys.argv) > 1:
        kaggle_dataset_path = sys.argv[1]
    else:
        # Try to auto-detect dataset
        if os.path.exists('/kaggle/input'):
            datasets = [d for d in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{d}')]
            if len(datasets) == 1:
                kaggle_dataset_path = f'/kaggle/input/datasets/{datasets[0]}'
                print(f"Auto-detected dataset: {kaggle_dataset_path}")
            else:
                print("Multiple datasets found. Please specify which one to use:")
                print("\nUsage: python setup_kaggle_data.py /kaggle/input/your-dataset-name")
                print("\nAvailable datasets:")
                for ds in datasets:
                    print(f"  - /kaggle/input/datasets/{ds}")
                sys.exit(1)
        else:
            print("Not in Kaggle environment. No setup needed.")
            print("Your local raw_data/ directory will be used.")
            return
    
    setup_kaggle_data(kaggle_dataset_path)


if __name__ == "__main__":
    main()
