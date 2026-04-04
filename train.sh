#!/bin/bash
# YOLOv8 Training Pipeline Script
# Creates virtual environment, installs dependencies, and runs training pipeline
#
# Usage:
#   ./train.sh [OPTIONS] [TRAIN_OPTIONS]
#
# Pipeline Options:
#   --reprocess         Force re-processing from raw_data (creates new random split)
#   --use-existing      Use existing data split without prompting
#   --help, -h          Show this help message
#
# Training Options (passed to train_cli.py):
#   --epochs N          Number of training epochs (default: 100)
#   --batch-size N      Batch size (default: 16)
#   --learning-rate LR  Learning rate (default: 0.01)
#   --model-size SIZE   Model size: n, s, m, l, x (default: n)
#   --device DEVICE     Device: cuda or cpu (default: cuda)
#   --img-width W       Image width (default: 640)
#   --img-height H      Image height (default: 360)
#   --workers N         Number of data loading workers (default: 4)
#   --patience N        Early stopping patience (default: 50)
#   --conf-threshold T  Confidence threshold (default: 0.25)
#   --iou-threshold T   IoU threshold (default: 0.45)
#   --mosaic P          Mosaic augmentation probability (default: 1.0)
#   --fliplr P          Horizontal flip probability (default: 0.5)
#   --flipud P          Vertical flip probability (default: 0.0)
#   --degrees D         Rotation degrees (default: 0.0)
#   --translate T       Translation (default: 0.1)
#   --scale S           Scale (default: 0.5)
#   And more... (run with --help for full list)
#
# Examples:
#   ./train.sh --epochs 200 --batch-size 32
#   ./train.sh --reprocess --model-size s --learning-rate 0.001
#   ./train.sh --use-existing --epochs 50 --device cpu

set -e  # Exit on any error

# Parse command line arguments
FORCE_REPROCESS=false
USE_EXISTING=false
TRAIN_ARGS=()

for arg in "$@"; do
    case $arg in
        --reprocess)
            FORCE_REPROCESS=true
            shift
            ;;
        --use-existing)
            USE_EXISTING=true
            shift
            ;;
        --help|-h)
            # Show our help first
            echo "Pipeline Usage: $0 [PIPELINE_OPTIONS] [TRAINING_OPTIONS]"
            echo ""
            echo "Pipeline Options:"
            echo "  --reprocess         Force re-processing from raw_data (creates new random split)"
            echo "  --use-existing      Use existing data split without prompting"
            echo "  --help, -h          Show this help message and training options"
            echo ""
            echo "Training Options:"
            # Activate venv if it exists to show training help
            if [ -d ".venv" ]; then
                source .venv/bin/activate 2>/dev/null && python train_cli.py --help || echo "  (install dependencies first to see all training options)"
            else
                echo "  (run ./train.sh first to create venv and see all training options)"
            fi
            exit 0
            ;;
        *)
            # Collect all other arguments for training
            TRAIN_ARGS+=("$arg")
            ;;
    esac
done

echo "========================================================================"
echo "YOLOv8 Ball Detector - Complete Training Pipeline"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print error and exit
error_exit() {
    echo -e "${RED}✗ Error: $1${NC}" >&2
    exit 1
}

# Check if Python is installed
echo "Checking prerequisites..."
if ! command -v python &> /dev/null; then
    error_exit "Python is not installed. Please install Python 3.8 or higher."
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    error_exit "Python 3.8 or higher is required. Found: Python $PYTHON_VERSION"
fi

echo "✓ Python $PYTHON_VERSION found"

# Check if raw data exists
if [ ! -d "raw_data/images" ] || [ ! -d "raw_data/labels" ]; then
    error_exit "raw_data directory not found or incomplete. Please ensure raw_data/images/ and raw_data/labels/ exist."
fi

# Count files in raw_data (use -L to follow symlinks)
IMAGE_COUNT=$(find -L raw_data/images -type f | wc -l)
LABEL_COUNT=$(find -L raw_data/labels -type f -name "*.txt" | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    error_exit "No images found in raw_data/images/. Please add your dataset."
fi

if [ "$LABEL_COUNT" -eq 0 ]; then
    error_exit "No label files found in raw_data/labels/. Please add your label files."
fi

echo "✓ Found $IMAGE_COUNT images and $LABEL_COUNT labels"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    error_exit "requirements.txt not found. Please ensure it exists in the project directory."
fi

echo "✓ All prerequisites checked"
echo ""

# Step 1: Create virtual environment
echo -e "${BLUE}Step 1/4: Setting up virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    if ! python -m venv .venv; then
        error_exit "Failed to create virtual environment. Try: pip install virtualenv"
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Step 2: Activate virtual environment and install dependencies
echo -e "${BLUE}Step 2/4: Installing dependencies...${NC}"
source .venv/bin/activate || error_exit "Failed to activate virtual environment"

# Upgrade pip
echo "Upgrading pip..."
if ! pip install --upgrade pip > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Failed to upgrade pip, continuing with existing version${NC}"
fi

# Install requirements
echo "Installing required packages (this may take a few minutes)..."
if ! pip install -r requirements.txt; then
    error_exit "Failed to install requirements. Check your internet connection and requirements.txt"
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 3: Preprocess data
echo -e "${BLUE}Step 3/4: Preprocessing data...${NC}"

# Check if data already exists
if [ -d "data/train" ] && [ -d "data/val" ] && [ -d "data/test" ]; then
    
    # Handle based on command line flags
    if [ "$FORCE_REPROCESS" = true ]; then
        echo ""
        echo -e "${YELLOW}--reprocess flag set: Clearing existing data and re-processing from raw_data...${NC}"
        rm -rf data/ backup/
        echo "This will:"
        echo "  - Create backup of raw data"
        echo "  - Validate and clean labels"
        echo "  - Resize images to 640x360"
        echo "  - Split into train/val/test (70/20/10)"
        echo "  - Generate preprocessing report"
        echo ""
        if ! python preprocess_data.py; then
            error_exit "Preprocessing failed. Check stats/v1/preprocessing_*.log for details."
        fi
    elif [ "$USE_EXISTING" = true ]; then
        echo ""
        echo -e "${GREEN}--use-existing flag set: Using existing data split${NC}"
        echo "  Train: $(ls data/train/images 2>/dev/null | wc -l) images"
        echo "  Val:   $(ls data/val/images 2>/dev/null | wc -l) images"
        echo "  Test:  $(ls data/test/images 2>/dev/null | wc -l) images"
        echo ""
    else
        # Interactive mode
        echo ""
        echo -e "${YELLOW}⚠️  Processed data already exists in data/ directory${NC}"
        echo ""
        echo "Options:"
        echo "  1) Use existing data split (faster, keeps same train/val/test split)"
        echo "  2) Re-process and create NEW random split from raw_data"
        echo "  3) Skip preprocessing check and continue"
        echo ""
        read -p "Enter choice (1/2/3) [default: 1]: " choice
        choice=${choice:-1}  # Default to 1 if empty
        
        case $choice in
            2)
                echo ""
                echo -e "${YELLOW}Clearing existing data and re-processing from raw_data...${NC}"
                rm -rf data/ backup/
                echo "This will:"
                echo "  - Create backup of raw data"
                echo "  - Validate and clean labels"
                echo "  - Resize images to 640x360"
                echo "  - Split into train/val/test (70/20/10)"
                echo "  - Generate preprocessing report"
                echo ""
                if ! python preprocess_data.py; then
                    error_exit "Preprocessing failed. Check stats/v1/preprocessing_*.log for details."
                fi
                ;;
            3)
                echo "Skipping preprocessing check..."
                ;;
            1|*)
                echo "Using existing data split..."
                echo "  Train: $(ls data/train/images 2>/dev/null | wc -l) images"
                echo "  Val:   $(ls data/val/images 2>/dev/null | wc -l) images"
                echo "  Test:  $(ls data/test/images 2>/dev/null | wc -l) images"
                ;;
        esac
    fi
else
    # No existing data, run preprocessing
    echo "This will:"
    echo "  - Create backup of raw data"
    echo "  - Validate and clean labels"
    echo "  - Resize images to 640x360"
    echo "  - Split into train/val/test (70/20/10)"
    echo "  - Generate preprocessing report"
    echo ""
    if ! python preprocess_data.py; then
        error_exit "Preprocessing failed. Check stats/v1/preprocessing_*.log for details."
    fi
fi

echo -e "${GREEN}✓ Data preprocessing complete${NC}"
echo ""

# Step 4: Train model
echo -e "${BLUE}Step 4/4: Training model...${NC}"

# Display training arguments if any
if [ ${#TRAIN_ARGS[@]} -gt 0 ]; then
    echo "Training arguments: ${TRAIN_ARGS[@]}"
fi
echo ""

# Check for CUDA/GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ GPU detected and available${NC}"
    else
        echo -e "${YELLOW}Warning: nvidia-smi failed. GPU may not be available. Training will use CPU (slower).${NC}"
    fi
else
    echo -e "${YELLOW}Note: No GPU detected. Training will use CPU (this will be slower).${NC}"
fi
echo ""

echo -e "${YELLOW}Note: Training may take 30-60 minutes with GPU, several hours with CPU.${NC}"
echo ""

# Run training with CLI arguments
if ! python train_cli.py "${TRAIN_ARGS[@]}"; then
    error_exit "Training failed. Check stats/*/train_*.log for details."
fi

echo ""
echo -e "${GREEN}========================================================================"
echo -e "✓ TRAINING COMPLETE!"
echo -e "========================================================================${NC}"
echo ""
echo "Results:"
echo "  - Best model saved in: models/"
echo "  - Training plots: stats/"
echo "  - Full results: runs/"
echo ""
echo "Next steps:"
echo "  1. Review training plots in stats/"
echo "  2. Run testing: ./test.sh"
echo ""
