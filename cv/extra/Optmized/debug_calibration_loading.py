#!/usr/bin/env python3
"""
🔍 DEBUG CALIBRATION FILE LOADING
Check which calibration file the CV system is actually loading
"""

import os
import json
import sys

def check_calibration_file_loading():
    """Debug which calibration file is being loaded"""
    
    print("🔍 DEBUGGING CALIBRATION FILE LOADING")
    print("=" * 60)
    
    camera_id = 1
    
    # Check current working directory
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Test the path that CV system uses
    cv_system_path = f"../configs/warehouse_calibration_camera_{camera_id}.json"
    print(f"🎯 CV system tries to load: {cv_system_path}")
    
    # Check if file exists
    if os.path.exists(cv_system_path):
        print(f"✅ File exists at: {os.path.abspath(cv_system_path)}")
        
        # Load and check content
        try:
            with open(cv_system_path, 'r') as f:
                data = json.load(f)
            
            corners = data.get('real_world_corners', [])
            print(f"📍 Real world corners: {corners}")
            
            if corners:
                # Check coordinate range
                x_coords = [corner[0] for corner in corners]
                y_coords = [corner[1] for corner in corners]
                x_range = f"{min(x_coords)}-{max(x_coords)}ft"
                y_range = f"{min(y_coords)}-{max(y_coords)}ft"
                print(f"📐 X range: {x_range}")
                print(f"📐 Y range: {y_range}")
                
                # Determine if this is correct for Camera 1
                if min(x_coords) == 0 and max(x_coords) <= 62:
                    print("✅ This looks like CORRECT Camera 1 calibration (0-62ft range)")
                elif min(x_coords) >= 120 and max(x_coords) <= 180:
                    print("❌ This looks like Camera 8 calibration (120-180ft range) - WRONG!")
                else:
                    print(f"⚠️ Unexpected coordinate range for Camera 1")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ File does not exist at: {os.path.abspath(cv_system_path)}")
        
        # Check alternative paths
        alternative_paths = [
            f"configs/warehouse_calibration_camera_{camera_id}.json",
            f"cv/configs/warehouse_calibration_camera_{camera_id}.json",
            f"../cv/configs/warehouse_calibration_camera_{camera_id}.json"
        ]
        
        print("\n🔍 Checking alternative paths:")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found at: {os.path.abspath(alt_path)}")
            else:
                print(f"❌ Not found: {alt_path}")

if __name__ == "__main__":
    check_calibration_file_loading()
