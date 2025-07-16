#!/usr/bin/env python3
"""
Setup Training Environment for Custom Grounding DINO Training
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run shell command with error handling"""
    logger.info(f"🔧 {description}")
    logger.info(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"   ❌ Failed: {e}")
        logger.error(f"   Output: {e.stdout}")
        logger.error(f"   Error: {e.stderr}")
        return False

def setup_training_environment():
    """Setup complete training environment"""
    
    logger.info("🚀 Setting up Custom Grounding DINO Training Environment")
    logger.info("=" * 60)
    
    # Step 1: Create conda environment
    commands = [
        {
            "cmd": "conda create -n grounding_dino_training python=3.9 -y",
            "desc": "Creating conda environment"
        },
        {
            "cmd": "conda activate grounding_dino_training",
            "desc": "Activating environment"
        },
        
        # Step 2: Install PyTorch with CUDA
        {
            "cmd": "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y",
            "desc": "Installing PyTorch with CUDA 11.8"
        },
        
        # Step 3: Install Grounding DINO dependencies
        {
            "cmd": "pip install transformers==4.35.0",
            "desc": "Installing Transformers"
        },
        {
            "cmd": "pip install accelerate==0.24.0",
            "desc": "Installing Accelerate for distributed training"
        },
        {
            "cmd": "pip install datasets==2.14.0",
            "desc": "Installing Datasets library"
        },
        {
            "cmd": "pip install wandb",
            "desc": "Installing Weights & Biases for experiment tracking"
        },
        
        # Step 4: Install annotation tools
        {
            "cmd": "pip install labelme",
            "desc": "Installing LabelMe annotation tool"
        },
        {
            "cmd": "pip install roboflow",
            "desc": "Installing Roboflow SDK"
        },
        
        # Step 5: Install additional dependencies
        {
            "cmd": "pip install opencv-python pillow numpy scipy matplotlib seaborn",
            "desc": "Installing image processing libraries"
        },
        {
            "cmd": "pip install tqdm rich typer",
            "desc": "Installing utility libraries"
        },
        
        # Step 6: Clone Grounding DINO repository
        {
            "cmd": "git clone https://github.com/IDEA-Research/GroundingDINO.git",
            "desc": "Cloning Grounding DINO repository"
        },
        {
            "cmd": "cd GroundingDINO && pip install -e .",
            "desc": "Installing Grounding DINO in development mode"
        }
    ]
    
    success_count = 0
    for command_info in commands:
        if run_command(command_info["cmd"], command_info["desc"]):
            success_count += 1
        else:
            logger.warning(f"⚠️ Command failed, continuing...")
    
    logger.info(f"\n📊 Setup Summary: {success_count}/{len(commands)} commands successful")
    
    # Verification
    logger.info("\n🔍 Verifying installation...")
    verification_script = '''
import torch
import transformers
import cv2
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Transformers version: {transformers.__version__}")
print("✅ Environment setup complete!")
'''
    
    with open("verify_setup.py", "w") as f:
        f.write(verification_script)
    
    logger.info("Run 'python verify_setup.py' to verify installation")

def create_project_structure():
    """Create training project directory structure"""
    
    structure = {
        "training": {
            "data": {
                "raw_images": {},
                "annotations": {},
                "processed": {
                    "train": {},
                    "val": {},
                    "test": {}
                }
            },
            "configs": {},
            "scripts": {},
            "models": {
                "checkpoints": {},
                "final": {}
            },
            "logs": {},
            "evaluation": {}
        }
    }
    
    def create_dirs(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            os.makedirs(path, exist_ok=True)
            logger.info(f"📁 Created: {path}")
            
            if isinstance(content, dict):
                create_dirs(path, content)
    
    logger.info("\n📁 Creating project structure...")
    create_dirs(".", structure)
    
    # Create README files
    readme_content = """# Custom Grounding DINO Training

## Directory Structure
- `data/raw_images/`: Original captured images
- `data/annotations/`: Annotation files (JSON/XML)
- `data/processed/`: Train/val/test splits
- `configs/`: Training configuration files
- `scripts/`: Training and evaluation scripts
- `models/`: Saved model checkpoints
- `logs/`: Training logs and metrics
- `evaluation/`: Model evaluation results

## Next Steps
1. Collect and organize training data
2. Annotate images using LabelMe or CVAT
3. Configure training parameters
4. Start training process
"""
    
    with open("training/README.md", "w") as f:
        f.write(readme_content)
    
    logger.info("✅ Project structure created successfully")

if __name__ == "__main__":
    setup_training_environment()
    create_project_structure()
