"""
Experiment Configuration
Configure paths and parameters for experiments here.
"""

# Model Configuration
MODEL_CONFIG_PATH = "config/DINO/DINO_4scale.py"
MODEL_CHECKPOINT_PATH = "path/to/your/checkpoint.pth"

# Dataset Configuration
DATASET_ROOT = "path/to/your/dataset"
ANNOTATION_FILE = "path/to/your/annotations.json"

# Output Configuration
OUTPUT_DIR = "outputs"
VISUALIZATION_DIR = "visualizations"

# Inference Configuration
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
