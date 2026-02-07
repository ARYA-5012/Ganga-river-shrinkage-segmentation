"""
Configuration file for Ganga River Segmentation project.
Contains all paths, hyperparameters, and constants.
"""

import os

# ==================== Paths ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed")
MASKS_DIR = os.path.join(DATA_ROOT, "masks")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "model_checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ==================== Study Areas ====================
CITIES = ['prayagraj', 'varanasi', 'patna']
YEARS = [2014, 2025]
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']

# City boundary rectangles (lon_min, lat_min, lon_max, lat_max)
CITY_BOUNDS = {
    'prayagraj': [81.7, 25.3, 82.0, 25.6],
    'varanasi': [82.9, 25.2, 83.1, 25.4],
    'patna': [85.0, 25.5, 85.3, 25.7]
}

# ==================== Image Processing ====================
PATCH_SIZE = 512
STRIDE = 256
PIXEL_SIZE_M = 30  # Landsat pixel size in meters

# Normalization parameters (optional, can be computed from data)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ==================== Model Configuration ====================
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"  # Use pretrained weights
IN_CHANNELS = 3
NUM_CLASSES = 1  # Binary segmentation

# ==================== Training Hyperparameters ====================
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5

# Optimizer
OPTIMIZER = "adam"
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
SCHEDULER = "cosine"
T_MAX = NUM_EPOCHS  # For CosineAnnealingLR

# ==================== Loss Configuration ====================
LOSS_TYPE = "bce_dice"  # Combined BCE + Dice Loss

# ==================== Data Augmentation ====================
AUGMENTATION_PROB = 0.5
ROTATION_LIMIT = 90
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2

# ==================== Evaluation ====================
THRESHOLD = 0.5  # Binary classification threshold
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'iou']

# ==================== Google Earth Engine ====================
GEE_PROJECT = "your-gee-project-name"  # Update with your GEE project

# Satellite collections
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

# Cloud masking bands
S2_QA_BAND = 'QA60'
L8_QA_PIXEL_BAND = 'QA_PIXEL'

# ==================== Visualization ====================
COLORMAP = 'Blues'
FIGSIZE = (12, 8)
DPI = 300

# ==================== Miscellaneous ====================
RANDOM_SEED = 42
NUM_WORKERS = 4  # For DataLoader
PIN_MEMORY = True
