"""
Fix class IDs in YOLO label files
Converts all class IDs (0-7) to class 0 for single-class detection
"""

import os
import glob
from pathlib import Path

def fix_label_file(label_path):
    """
    Read a label file and convert all class IDs to 0.
    Preserves all bounding box coordinates.
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        modified = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id != 0:
                    # Change class ID to 0, keep rest the same
                    parts[0] = '0'
                    modified = True
                fixed_lines.append(' '.join(parts) + '\n')
            elif line.strip():  # Keep non-empty lines
                fixed_lines.append(line)
        
        # Write back if modified
        if modified:
            with open(label_path, 'w') as f:
                f.writelines(fixed_lines)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False


def main():
    """
    Process all label files in raw_data/labels directory.
    """
    labels_dir = "./raw_data/labels"
    
    if not os.path.exists(labels_dir):
        print(f"Error: Directory {labels_dir} not found!")
        return
    
    # Get all .txt files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    print("Converting all class IDs to 0...")
    
    modified_count = 0
    
    for label_file in label_files:
        if fix_label_file(label_file):
            modified_count += 1
    
    print(f"\n✓ Complete!")
    print(f"  Modified: {modified_count} files")
    print(f"  Unchanged: {len(label_files) - modified_count} files")
    print(f"\nAll class IDs are now 0 for single-class detection.")
    print("You can now run: python preprocess_data.py")


if __name__ == "__main__":
    main()
