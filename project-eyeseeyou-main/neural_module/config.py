# neural_module/config.py
import torch
import os

class Config:
    # Data paths
    DATA_DIR = "data/processed"
    TRAIN_SCENES = "data/processed/scenes/CLEVR_subsampled_scenes.json"
    TRAIN_VAL_SPLIT = "data/processed/train_val_split.json"
    
    # YOLO configuration
    MODEL = "yolov8l.pt"  # YOLOv8 Large model
    MODEL_PATH = None     # Set to path of saved model for inference
    YAML_PATH = "data/clevr.yaml"
    
    # Class definitions
    COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    SHAPES = ["cube", "sphere", "cylinder"]
    MATERIALS = ["rubber", "metal"]
    SIZES = ["small", "large"]
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    IMAGE_SIZE = 640
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # Set to 0 to avoid cache warning
    
    # Directories
    MODEL_SAVE_DIR = "models/saved"
    RESULTS_DIR = "results"
    
    # Detection parameters
    CONF_THRESHOLD = 0.01  # Very low threshold to catch all objects
    NMS_THRESHOLD = 0.2    # Lower NMS threshold to avoid merging objects
    MAX_OBJECTS = 20       # Maximum number of objects to detect
    
    # Create necessary directories
    def __init__(self):
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)