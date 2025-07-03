#!/usr/bin/env python3
"""
🔧 FIX WAREHOUSE CONFIGURATION FILES
Update all warehouse config files to match the corrected camera calibrations
"""

import json
import os

def fix_json_config(file_path: str):
    """Fix a warehouse JSON configuration file"""
    print(f"🔧 Fixing {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Update warehouse dimensions
        config['length_ft'] = 100.0
        
        # Update camera zones to match calibration files
        camera_updates = {
            "1": {"x_start": 0, "x_end": 62, "y_start": 0, "y_end": 25, "active": True},
            "2": {"x_start": 0, "x_end": 62, "y_start": 25, "y_end": 50, "active": True},
            "3": {"x_start": 0, "x_end": 62, "y_start": 50, "y_end": 75, "active": True},
            "4": {"x_start": 0, "x_end": 62, "y_start": 75, "y_end": 100, "active": True},
            "11": {"y_end": 100}  # Update Camera 11 to go to 100ft
        }
        
        # Apply updates
        for cam_id, updates in camera_updates.items():
            if cam_id in config.get('camera_zones', {}):
                for key, value in updates.items():
                    config['camera_zones'][cam_id][key] = value
                print(f"   ✅ Updated Camera {cam_id}")
        
        # Write back to file
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ {file_path} updated successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to update {file_path}: {e}")

def main():
    """Fix all warehouse configuration files"""
    print("🔧 FIXING WAREHOUSE CONFIGURATION FILES")
    print("=" * 60)
    
    # List of config files to fix
    config_files = [
        "../configs/warehouse_configs/warp_main.json",
        "../warehouse_configs/warp_main.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            fix_json_config(config_file)
        else:
            print(f"⚠️ File not found: {config_file}")
    
    print("\n✅ Configuration files updated!")
    print("🔄 Please restart the CV system to load new configurations")

if __name__ == "__main__":
    main()
