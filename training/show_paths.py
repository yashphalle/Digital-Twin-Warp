#!/usr/bin/env python3
"""
Show current training data collection paths
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cv.configs.config import Config

def show_training_paths():
    """Display current training data paths"""
    print("=" * 60)
    print("🗂️  CURRENT TRAINING DATA COLLECTION PATHS")
    print("=" * 60)
    
    print(f"📁 Training Data Root:     {Config.TRAINING_DATA_ROOT}")
    print(f"📁 Raw Collection Dir:     {Config.RAW_COLLECTION_DIR}")
    print(f"📁 Processed Training Dir: {Config.PROCESSED_TRAINING_DIR}")
    
    print("\n" + "=" * 60)
    print("⚙️  COLLECTION SETTINGS")
    print("=" * 60)
    
    print(f"🖼️  Raw Image Quality:      {Config.RAW_IMAGE_QUALITY}%")
    print(f"🖼️  Corrected Image Quality: {Config.CORRECTED_IMAGE_QUALITY}%")
    print(f"⏱️  Frame Skip Count:       {Config.COLLECTION_FRAME_SKIP} frames")
    print(f"⏱️  Collection Interval:    ~{Config.COLLECTION_FRAME_SKIP/20:.0f} seconds (at 20fps)")
    
    print("\n" + "=" * 60)
    print("📂 EXPECTED DIRECTORY STRUCTURE")
    print("=" * 60)
    
    print(f"{Config.RAW_COLLECTION_DIR}/")
    for camera_id in range(1, 12):
        print(f"├── camera_{camera_id}/")
        print(f"│   ├── raw_4k/          # Full resolution images")
        print(f"│   └── corrected_2k/    # Fisheye-corrected images")
    
    print("\n" + "=" * 60)
    print("🔧 TO CHANGE PATHS:")
    print("=" * 60)
    print("Edit cv/configs/config.py and modify:")
    print("• Config.TRAINING_DATA_ROOT")
    print("• Config.RAW_COLLECTION_DIR") 
    print("• Config.PROCESSED_TRAINING_DIR")
    print("=" * 60)

if __name__ == "__main__":
    show_training_paths()
