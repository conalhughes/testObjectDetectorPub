"""
Configuration file for YOLOv8 Object Detection
Edit these global variables between runs
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "ball_detector"
MODEL_VERSION = "v1-1"
MODEL_SIZE = "n"  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
INPUT_SIZE = (640, 360)  # Image size for training - matches camera resolution (1280x720 / 2)
BATCH_SIZE = 16
NUM_WORKERS = 4  # Number of workers for data loading
NUM_CLASSES = 1  # Number of object classes (change based on your dataset)
CLASS_NAMES = ['ball']  # List of class names

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
EPOCHS = 10
LEARNING_RATE = 0.01
PATIENCE = 5  # Early stopping patience
DEVICE = "cuda"  # 'cuda' or 'cpu'

# Data augmentation
AUGMENT = True
MOSAIC = 1.0  # Mosaic augmentation probability
HSV_H = 0.015  # HSV-Hue augmentation
HSV_S = 0.7  # HSV-Saturation augmentation
HSV_V = 0.4  # HSV-Value augmentation
DEGREES = 0.0  # Rotation augmentation (degrees)
TRANSLATE = 0.1  # Translation augmentation
SCALE = 0.5  # Scale augmentation
SHEAR = 0.0  # Shear augmentation (degrees)
PERSPECTIVE = 0.0  # Perspective augmentation
FLIPUD = 0.0  # Vertical flip probability
FLIPLR = 0.5  # Horizontal flip probability

# ============================================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================================
# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.1   # 10% for testing

# Image preprocessing
RESIZE_IMAGES = True  # Whether to resize images during preprocessing
TARGET_SIZE = (544, 448)  # Target size for resizing (width, height) - matches 1280x720 aspect ratio
MAINTAIN_ASPECT_RATIO = True  # Maintain aspect ratio when resizing (pad with black)
IMAGE_QUALITY = 95  # JPEG quality for saved images (1-100)

# Data cleaning
REMOVE_EMPTY_LABELS = True  # Remove images with no annotations
VERIFY_LABELS = True  # Verify label format and coordinates
MIN_BOX_SIZE = 0.01  # Minimum bounding box size (normalized) to keep

# Augmentation during preprocessing (optional, separate from training augmentation)
PREPROCESS_AUGMENT = False  # Apply augmentation during data split
AUGMENT_FACTOR = 1  # How many augmented versions per image (if enabled)

# ============================================================================
# PATHS
# ============================================================================
RAW_DATA_DIR = "./raw_data"
RAW_IMAGES = f"{RAW_DATA_DIR}/images"
RAW_LABELS = f"{RAW_DATA_DIR}/labels"
BACKUP_DIR = "./backup"

DATA_DIR = "./data"
TRAIN_IMAGES = f"{DATA_DIR}/train/images"
TRAIN_LABELS = f"{DATA_DIR}/train/labels"
VAL_IMAGES = f"{DATA_DIR}/val/images"
VAL_LABELS = f"{DATA_DIR}/val/labels"
TEST_IMAGES = f"{DATA_DIR}/test/images"
TEST_LABELS = f"{DATA_DIR}/test/labels"

STATS_DIR = "./stats"
VERSION_STATS_DIR = f"{STATS_DIR}/{MODEL_VERSION}"  # Version-specific stats folder
MODELS_DIR = "./models"
RUNS_DIR = "./runs"

# Model save path
MODEL_SAVE_PATH = f"{MODELS_DIR}/{MODEL_NAME}_{MODEL_VERSION}.pt"
BEST_MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}_{MODEL_VERSION}_best.pt"

# ============================================================================
# VALIDATION & TESTING CONFIGURATION
# ============================================================================
CONF_THRESHOLD = 0.25  # Confidence threshold for predictions
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
SAVE_PREDICTIONS = True  # Save prediction visualizations
MAX_DET = 300  # Maximum detections per image

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================
PLOT_DPI = 150
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Matplotlib style
