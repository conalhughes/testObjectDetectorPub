#!/bin/bash
# Quick script to re-process data from raw_data/ with a fresh split
# This clears existing processed data and creates new train/val/test splits

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================================================"
echo "Re-processing Data from raw_data/"
echo "========================================================================"
echo ""

# Check if raw_data exists
if [ ! -d "raw_data/images" ] || [ ! -d "raw_data/labels" ]; then
    echo -e "${RED}✗ Error: raw_data/images or raw_data/labels not found${NC}"
    exit 1
fi

# Count files (use -L to follow symlinks)
IMG_COUNT=$(find -L raw_data/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
LBL_COUNT=$(find -L raw_data/labels -type f -name "*.txt" 2>/dev/null | wc -l)

echo "Found in raw_data/:"
echo "  Images: $IMG_COUNT"
echo "  Labels: $LBL_COUNT"
echo ""

if [ "$IMG_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ Error: No images found in raw_data/images/${NC}"
    exit 1
fi

# Confirm
echo -e "${YELLOW}⚠️  Warning: This will DELETE existing processed data and create NEW random splits${NC}"
echo ""
echo "Current data/ directory will be cleared:"
if [ -d "data/train" ]; then
    echo "  Train: $(ls data/train/images 2>/dev/null | wc -l) images"
fi
if [ -d "data/val" ]; then
    echo "  Val:   $(ls data/val/images 2>/dev/null | wc -l) images"
fi
if [ -d "data/test" ]; then
    echo "  Test:  $(ls data/test/images 2>/dev/null | wc -l) images"
fi
echo ""

read -p "Continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Clearing existing data..."
rm -rf data/ backup/

echo "Running preprocessing..."
echo ""

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}Note: Virtual environment not found. Using system Python.${NC}"
    echo "Run ./train.sh first to create virtual environment."
    echo ""
fi

# Run preprocessing
if ! python preprocess_data.py; then
    echo ""
    echo -e "${RED}✗ Preprocessing failed${NC}"
    echo "Check stats/v1/preprocessing_*.log for details"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Re-processing complete!${NC}"
echo ""
echo "New split created:"
echo "  Train: $(ls data/train/images 2>/dev/null | wc -l) images"
echo "  Val:   $(ls data/val/images 2>/dev/null | wc -l) images"
echo "  Test:  $(ls data/test/images 2>/dev/null | wc -l) images"
echo ""
echo "Ready for training: ./train.sh --use-existing"
