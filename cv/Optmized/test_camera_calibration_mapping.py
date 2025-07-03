#!/usr/bin/env python3
"""
🔍 TEST CAMERA CALIBRATION MAPPING
Test if each camera is loading the correct calibration file
"""

import json
import os
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_camera_calibration_mapping():
    """Test if cameras are loading correct calibration files"""
    
    print("🔍 TESTING CAMERA CALIBRATION MAPPING")
    print("=" * 60)
    
    expected_ranges = {
        1: {"x_min": 0, "x_max": 62, "y_min": 0, "y_max": 25},
        2: {"x_min": 0, "x_max": 62, "y_min": 25, "y_max": 50},
        3: {"x_min": 0, "x_max": 62, "y_min": 50, "y_max": 75},
        4: {"x_min": 0, "x_max": 62, "y_min": 75, "y_max": 100},
        8: {"x_min": 120, "x_max": 180, "y_min": 0, "y_max": 25},
        9: {"x_min": 120, "x_max": 180, "y_min": 25, "y_max": 50},
        10: {"x_min": 120, "x_max": 180, "y_min": 50, "y_max": 75},
        11: {"x_min": 120, "x_max": 180, "y_min": 75, "y_max": 100},
    }
    
    for camera_id in [1, 2, 3, 4, 8, 9, 10, 11]:
        print(f"\n📹 CAMERA {camera_id}:")
        
        # Load calibration file
        calibration_file = f"../configs/warehouse_calibration_camera_{camera_id}.json"
        
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                
                corners = data.get('real_world_corners', [])
                if corners:
                    x_coords = [corner[0] for corner in corners]
                    y_coords = [corner[1] for corner in corners]
                    
                    actual_x_min, actual_x_max = min(x_coords), max(x_coords)
                    actual_y_min, actual_y_max = min(y_coords), max(y_coords)
                    
                    expected = expected_ranges[camera_id]
                    
                    print(f"   📍 Expected: ({expected['x_min']}-{expected['x_max']}ft, {expected['y_min']}-{expected['y_max']}ft)")
                    print(f"   📄 Actual:   ({actual_x_min}-{actual_x_max}ft, {actual_y_min}-{actual_y_max}ft)")
                    
                    # Check if coordinates match expected range
                    x_match = (actual_x_min == expected['x_min'] and actual_x_max == expected['x_max'])
                    y_match = (actual_y_min == expected['y_min'] and actual_y_max == expected['y_max'])
                    
                    if x_match and y_match:
                        print(f"   ✅ CORRECT calibration")
                    else:
                        print(f"   ❌ WRONG calibration!")
                        
                        # Check if it matches another camera's range
                        for other_cam, other_range in expected_ranges.items():
                            if other_cam != camera_id:
                                other_x_match = (actual_x_min == other_range['x_min'] and actual_x_max == other_range['x_max'])
                                other_y_match = (actual_y_min == other_range['y_min'] and actual_y_max == other_range['y_max'])
                                if other_x_match and other_y_match:
                                    print(f"      🔄 This looks like Camera {other_cam}'s calibration!")
                                    break
                
            except Exception as e:
                print(f"   ❌ Error reading calibration: {e}")
        else:
            print(f"   ❌ Calibration file not found")
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("If Camera 1 shows coordinates like 160.67ft, it means:")
    print("1. Camera 1 is using Camera 8's calibration (120-180ft range)")
    print("2. OR there's a camera ID mismatch in the detection system")
    print("3. OR the CV system hasn't restarted to load new calibrations")

if __name__ == "__main__":
    test_camera_calibration_mapping()
