#!/usr/bin/env python
"""
Verify that the multi-class configuration system is working correctly.
Run this to ensure classes_config.yaml and config.py are in sync.
"""

import sys
from pathlib import Path

def verify_config():
    """Verify configuration loading and consistency."""
    print("=" * 70)
    print("MULTI-CLASS CONFIGURATION VERIFICATION")
    print("=" * 70)
    
    # Check if classes_config.yaml exists
    print("\n1. Checking classes_config.yaml...")
    if not Path("classes_config.yaml").exists():
        print("   ✗ ERROR: classes_config.yaml not found!")
        return False
    print("   ✓ classes_config.yaml found")
    
    # Try loading YAML
    print("\n2. Loading YAML configuration...")
    try:
        import yaml
        with open("classes_config.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
        print(f"   ✓ YAML loaded successfully")
        print(f"     - Dataset: {yaml_config.get('dataset_name', 'Unknown')}")
        print(f"     - Classes: {len(yaml_config.get('classes', []))} defined")
    except Exception as e:
        print(f"   ✗ ERROR loading YAML: {e}")
        return False
    
    # Try importing config.py
    print("\n3. Importing config.py...")
    try:
        import config
        print("   ✓ config.py imported successfully")
    except Exception as e:
        print(f"   ✗ ERROR importing config.py: {e}")
        return False
    
    # Verify consistency
    print("\n4. Verifying configuration consistency...")
    try:
        yaml_classes = len(yaml_config.get('classes', []))
        config_classes = config.NUM_CLASSES
        
        if yaml_classes != config_classes:
            print(f"   ✗ ERROR: Class count mismatch!")
            print(f"     - YAML: {yaml_classes} classes")
            print(f"     - config.py: {config_classes} classes")
            return False
        
        print(f"   ✓ Class count matches: {config_classes}")
        
        yaml_names = [c['name'] for c in yaml_config['classes']]
        config_names = config.CLASS_NAMES
        
        if yaml_names != config_names:
            print(f"   ✗ ERROR: Class names mismatch!")
            print(f"     - YAML: {yaml_names}")
            print(f"     - config.py: {config_names}")
            return False
        
        print(f"   ✓ Class names match: {config_names}")
        
    except Exception as e:
        print(f"   ✗ ERROR verifying configuration: {e}")
        return False
    
    # Display current configuration
    print("\n" + "=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"\nDataset: {config.DATASET_NAME}")
    print(f"Number of Classes: {config.NUM_CLASSES}")
    print(f"\nClass Details:")
    print("-" * 70)
    
    for cls in yaml_config['classes']:
        cls_id = cls['id']
        cls_name = cls['name']
        color = tuple(cls['color'])
        print(f"  ID {cls_id}: {cls_name:25s} RGB{color}")
    
    print("-" * 70)
    print(f"\n✓ Configuration verified successfully!")
    print(f"\nNext steps:")
    print(f"  1. Ensure raw_data/labels/ use class IDs: {list(range(config.NUM_CLASSES))}")
    print(f"  2. Run: ./train.sh")
    
    return True

if __name__ == "__main__":
    success = verify_config()
    sys.exit(0 if success else 1)
