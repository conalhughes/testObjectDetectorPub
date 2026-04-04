#!/bin/bash
# YOLOv8 Testing Script
# Activates venv and runs model testing
#
# Usage:
#   ./test.sh [OPTIONS]
#
# Options:
#   --model-version VER   Model version to test (default: v1)
#   --model-name NAME     Model name (default: ball_detector)
#   --conf-threshold T    Confidence threshold (default: 0.25)
#   --iou-threshold T     IoU threshold (default: 0.45)
#   --device DEVICE       Device: cuda or cpu (default: cuda)
#   --batch-size N        Batch size (default: 16)
#   --help, -h            Show this help message
#
# Examples:
#   ./test.sh
#   ./test.sh --model-version v2
#   ./test.sh --conf-threshold 0.5 --device cpu

set -e

# Parse command line arguments
TEST_ARGS=()

for arg in "$@"; do
    case $arg in
        --help|-h)
            echo "Testing Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            # Show help (works with or without venv)
            if [ -d ".venv" ]; then
                source .venv/bin/activate 2>/dev/null && python test_cli.py --help || echo "  (install dependencies first to see all options)"
            else
                # Try with system python (Kaggle)
                python test_cli.py --help 2>/dev/null || echo "  (install dependencies first to see all options)"
            fi
            exit 0
            ;;
        *)
            # Collect all arguments for testing
            TEST_ARGS+=("$arg")
            ;;
    esac
done

echo "========================================================================"
echo "YOLOv8 Ball Detector - Testing Pipeline"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print error and exit
error_exit() {
    echo -e "${RED}✗ Error: $1${NC}" >&2
    exit 1
}

# Detect if running in Kaggle environment
IS_KAGGLE=false
if [ -d "/kaggle/working" ]; then
    IS_KAGGLE=true
    echo -e "${YELLOW}✓ Kaggle environment detected${NC}"
fi

# Check if venv exists or if in Kaggle
if [ "$IS_KAGGLE" = false ] && [ ! -d ".venv" ]; then
    error_exit "Virtual environment not found! Please run ./train.sh first."
fi

# Check if Python files exist
if [ ! -f "test_cli.py" ]; then
    error_exit "test_cli.py not found! Please ensure you're in the correct directory."
fi

# Activate venv if not in Kaggle
if [ "$IS_KAGGLE" = false ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate || error_exit "Failed to activate virtual environment"
fi

# Check if model exists (check generic pattern since version might be changed via args)
MODEL_COUNT=$(find models -name "*_best.pt" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    error_exit "No trained models found in models/! Please run ./train.sh first."
fi

echo -e "${GREEN}✓ Found $MODEL_COUNT trained model(s):${NC}"
find models -name "*_best.pt" -exec basename {} \; | sed 's/^/  - /'

# Check which model version will be used
MODEL_VERSION="v1"  # Default
for arg in "${TEST_ARGS[@]}"; do
    if [[ "$arg" == "--model-version" ]]; then
        shift_next=true
    elif [[ "$shift_next" == true ]]; then
        MODEL_VERSION="$arg"
        shift_next=false
    fi
done

echo ""
echo -e "${BLUE}Testing model version: ${MODEL_VERSION}${NC}"

# Check if test data exists
if [ ! -d "data/test/images" ]; then
    error_exit "Test data not found! Please run ./train.sh to preprocess data first."
fi

TEST_IMAGE_COUNT=$(find data/test/images -type f 2>/dev/null | wc -l)
if [ "$TEST_IMAGE_COUNT" -eq 0 ]; then
    error_exit "No test images found! Please run ./train.sh to preprocess data first."
fi

echo -e "${GREEN}✓ Found $TEST_IMAGE_COUNT test images${NC}"
echo ""

# Display test arguments if any
if [ ${#TEST_ARGS[@]} -gt 0 ]; then
    echo "Testing arguments: ${TEST_ARGS[@]}"
    echo ""
fi

# Run testing
echo -e "${BLUE}Running model evaluation on test set...${NC}"
echo ""

if ! python test_cli.py "${TEST_ARGS[@]}"; then
    error_exit "Testing failed. Check stats/*/test_*.log for details."
fi

echo ""
echo -e "${GREEN}========================================================================"
echo -e "✓ TESTING COMPLETE!"
echo -e "========================================================================${NC}"
echo ""
echo "Results saved to stats/"
echo ""
