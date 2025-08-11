#!/usr/bin/env python3
"""
YOLOv8 Configuration
Model paths and settings for YOLOv8 detector
"""

import os

# ===== MODEL CONFIGURATION =====

# YOLOv8 Model Path
# Options:
# - "yolov8n.pt" (nano - fastest, lower accuracy)
# - "yolov8s.pt" (small - balanced)
# - "yolov8m.pt" (medium - better accuracy)
# - "yolov8l.pt" (large - high accuracy)
# - "yolov8x.pt" (extra large - highest accuracy)
# - "path/to/your/custom_model.pt" (your trained model)

# Try different model sizes for better detection:
# YOLO_MODEL_PATH = "yolov8n.pt"  # Nano (fastest, least accurate)
# YOLO_MODEL_PATH = "yolov8s.pt"  # Small (balanced)
YOLO_MODEL_PATH = "yolov8m.pt"  # Medium (better accuracy) - TRY THIS
# YOLO_MODEL_PATH = "yolov8l.pt"  # Large (high accuracy)

# TODO: Replace with pallet-trained model: "models/pallet_yolov8.pt"

# Detection Settings
CONFIDENCE_THRESHOLD = 0.1  # Lowered from 0.5 to catch more detections
NMS_THRESHOLD = 0.45       # Non-maximum suppression threshold

# Performance Settings
BATCH_SIZE = 4             # Number of frames to process together
MAX_WORKERS = 2            # Number of detection workers

# Device Settings
FORCE_CPU = False          # Set to True to force CPU inference
DEVICE_ID = 0              # GPU device ID (0, 1, 2, etc.)

# ===== CUSTOM MODEL PATHS =====

# If you have a custom trained model, update this path
CUSTOM_MODEL_PATHS = {
    'pallet_model_v1': 'models/pallet_yolov8_v1.pt',
    'pallet_model_v2': 'models/pallet_yolov8_v2.pt',
    'warehouse_model': 'models/warehouse_yolov8.pt',
    'custom_yolo': 'custom_yolo.pt'
}

# Select which custom model to use (or None for default)
SELECTED_CUSTOM_MODEL = 'custom_yolo'  # Using custom_yolo.pt model

# ===== HELPER FUNCTIONS =====

def get_model_path():
    """Get the model path to use"""
    if SELECTED_CUSTOM_MODEL and SELECTED_CUSTOM_MODEL in CUSTOM_MODEL_PATHS:
        custom_path = CUSTOM_MODEL_PATHS[SELECTED_CUSTOM_MODEL]
        if os.path.exists(custom_path):
            return custom_path
        else:
            print(f"⚠️ Custom model not found: {custom_path}, falling back to default")
    
    return YOLO_MODEL_PATH

def get_device():
    """Get the device to use for inference"""
    if FORCE_CPU:
        return "cpu"
    
    import torch
    if torch.cuda.is_available():
        return f"cuda:{DEVICE_ID}"
    else:
        return "cpu"

def get_detection_config():
    """Get complete detection configuration"""
    return {
        'model_path': get_model_path(),
        'device': get_device(),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'nms_threshold': NMS_THRESHOLD,
        'batch_size': BATCH_SIZE,
        'max_workers': MAX_WORKERS
    }

# ===== ROBOFLOW INTEGRATION =====

# If you're using a model trained on Roboflow, update these settings
ROBOFLOW_CONFIG = {
    'workspace': 'your-workspace',
    'project': 'pallet-detection',
    'version': 1,
    'model_format': 'yolov8',
    'api_key': 'your-api-key'  # Keep this secure!
}

def download_roboflow_model():
    """Download model from Roboflow (optional)"""
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=ROBOFLOW_CONFIG['api_key'])
        project = rf.workspace(ROBOFLOW_CONFIG['workspace']).project(ROBOFLOW_CONFIG['project'])
        dataset = project.version(ROBOFLOW_CONFIG['version']).download(ROBOFLOW_CONFIG['model_format'])
        
        return dataset.location
    except ImportError:
        print("⚠️ Roboflow library not installed. Run: pip install roboflow")
        return None
    except Exception as e:
        print(f"⚠️ Failed to download Roboflow model: {e}")
        return None
