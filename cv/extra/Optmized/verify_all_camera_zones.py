#!/usr/bin/env python3
"""
🗺️ COMPREHENSIVE CAMERA ZONE VERIFICATION
Verify all 11 cameras are properly mapped to the 180x100ft warehouse layout
"""

import json
import os
import sys
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.warehouse_config import get_warehouse_config

def load_calibration_file(camera_id: int) -> Dict:
    """Load calibration file for a camera"""
    try:
        calibration_file = f"../configs/warehouse_calibration_camera_{camera_id}.json"
        with open(calibration_file, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"❌ Failed to load calibration for Camera {camera_id}: {e}")
        return {}

def verify_camera_zone_mapping():
    """Verify all camera zones are properly mapped"""
    
    print("🗺️ COMPREHENSIVE CAMERA ZONE VERIFICATION")
    print("=" * 80)
    print("📐 Warehouse: 180ft × 100ft")
    print("🎯 Origin: Top-Right (0,0)")
    print("📍 From your screen perspective:")
    print("   - Camera 1 should be TOP RIGHT (0,0)")
    print("   - Camera 11 should be BOTTOM LEFT")
    print("=" * 80)
    
    # Load warehouse config
    warehouse_config = get_warehouse_config()
    
    print(f"\n🏭 Warehouse Config: {warehouse_config.name}")
    print(f"📐 Dimensions: {warehouse_config.width_ft}ft × {warehouse_config.length_ft}ft")
    print(f"🧭 Origin: {warehouse_config.origin_position}")
    print(f"📊 X-axis: {warehouse_config.x_axis_direction}")
    print(f"📊 Y-axis: {warehouse_config.y_axis_direction}")
    print(f"📹 Camera zones found: {list(warehouse_config.camera_zones.keys())}")
    
    # Expected layout from user's perspective
    expected_layout = {
        # Column 1 (RIGHT side of screen) - 0-62ft X
        1: {"x_start": 0, "x_end": 62, "y_start": 0, "y_end": 25, "screen_pos": "TOP RIGHT"},
        2: {"x_start": 0, "x_end": 62, "y_start": 25, "y_end": 50, "screen_pos": "RIGHT UPPER-MID"},
        3: {"x_start": 0, "x_end": 62, "y_start": 50, "y_end": 75, "screen_pos": "RIGHT LOWER-MID"},
        4: {"x_start": 0, "x_end": 62, "y_start": 75, "y_end": 100, "screen_pos": "RIGHT BOTTOM"},
        
        # Column 2 (MIDDLE of screen) - 60-120ft X
        5: {"x_start": 60, "x_end": 120, "y_start": 0, "y_end": 22.5, "screen_pos": "MIDDLE TOP"},
        6: {"x_start": 60, "x_end": 120, "y_start": 22.5, "y_end": 45, "screen_pos": "MIDDLE UPPER-MID"},
        7: {"x_start": 60, "x_end": 120, "y_start": 45, "y_end": 67.5, "screen_pos": "MIDDLE LOWER-MID"},
        
        # Column 3 (LEFT side of screen) - 120-180ft X
        8: {"x_start": 120, "x_end": 180, "y_start": 0, "y_end": 25, "screen_pos": "TOP LEFT"},
        9: {"x_start": 120, "x_end": 180, "y_start": 25, "y_end": 50, "screen_pos": "LEFT UPPER-MID"},
        10: {"x_start": 120, "x_end": 180, "y_start": 50, "y_end": 75, "screen_pos": "LEFT LOWER-MID"},
        11: {"x_start": 120, "x_end": 180, "y_start": 75, "y_end": 100, "screen_pos": "BOTTOM LEFT"},
    }
    
    print(f"\n🔍 VERIFYING ALL {len(expected_layout)} CAMERAS:")
    print("=" * 80)
    
    all_correct = True
    
    for camera_id in range(1, 12):
        print(f"\n📹 CAMERA {camera_id}:")
        
        # Check warehouse config (try both string and int keys)
        zone = None
        if str(camera_id) in warehouse_config.camera_zones:
            zone = warehouse_config.camera_zones[str(camera_id)]
        elif camera_id in warehouse_config.camera_zones:
            zone = warehouse_config.camera_zones[camera_id]

        if zone:
            config_coords = {
                "x_start": zone.x_start,
                "x_end": zone.x_end,
                "y_start": zone.y_start,
                "y_end": zone.y_end
            }
        else:
            print(f"   ❌ Not found in warehouse config")
            all_correct = False
            continue
            
        # Check calibration file
        calibration = load_calibration_file(camera_id)
        if calibration and 'real_world_corners' in calibration:
            corners = calibration['real_world_corners']
            # Extract bounds from corners [top-left, top-right, bottom-right, bottom-left]
            cal_coords = {
                "x_start": min(corners[0][0], corners[3][0]),  # Left side
                "x_end": max(corners[1][0], corners[2][0]),    # Right side
                "y_start": min(corners[0][1], corners[1][1]),  # Top
                "y_end": max(corners[2][1], corners[3][1])     # Bottom
            }
        else:
            print(f"   ❌ No calibration file found")
            all_correct = False
            continue
            
        # Expected coordinates
        expected = expected_layout[camera_id]
        
        # Compare all three sources
        print(f"   📍 Expected: ({expected['x_start']}-{expected['x_end']}ft, {expected['y_start']}-{expected['y_end']}ft) - {expected['screen_pos']}")
        print(f"   🏭 Config:   ({config_coords['x_start']}-{config_coords['x_end']}ft, {config_coords['y_start']}-{config_coords['y_end']}ft)")
        print(f"   📄 Calibr:  ({cal_coords['x_start']}-{cal_coords['x_end']}ft, {cal_coords['y_start']}-{cal_coords['y_end']}ft)")
        
        # Check if they match
        config_match = (
            config_coords['x_start'] == expected['x_start'] and
            config_coords['x_end'] == expected['x_end'] and
            config_coords['y_start'] == expected['y_start'] and
            config_coords['y_end'] == expected['y_end']
        )
        
        cal_match = (
            cal_coords['x_start'] == expected['x_start'] and
            cal_coords['x_end'] == expected['x_end'] and
            cal_coords['y_start'] == expected['y_start'] and
            cal_coords['y_end'] == expected['y_end']
        )
        
        if config_match and cal_match:
            print(f"   ✅ PERFECT MATCH")
        else:
            print(f"   ❌ MISMATCH!")
            if not config_match:
                print(f"      🏭 Config mismatch")
            if not cal_match:
                print(f"      📄 Calibration mismatch")
            all_correct = False
    
    print("\n" + "=" * 80)
    if all_correct:
        print("🎯 ✅ ALL CAMERAS CORRECTLY MAPPED!")
        print("📍 Camera 1 is at TOP RIGHT (0,0)")
        print("📍 Camera 11 is at BOTTOM LEFT")
        print("🗺️ Frontend will display cameras in correct positions")
    else:
        print("❌ SOME CAMERAS HAVE MAPPING ISSUES!")
        print("🔧 Please fix the mismatches above")
    
    print("=" * 80)
    
    # Display visual layout
    print("\n🗺️ VISUAL WAREHOUSE LAYOUT (from your screen perspective):")
    print("=" * 80)
    print("📐 180ft × 100ft warehouse")
    print()
    print("LEFT SIDE          MIDDLE           RIGHT SIDE")
    print("(120-180ft)        (60-120ft)       (0-62ft)")
    print("┌─────────────┬─────────────┬─────────────┐")
    print("│   Cam 8     │   Cam 5     │   Cam 1     │ 0-25ft")
    print("│  (TOP LEFT) │ (MID TOP)   │(TOP RIGHT)  │")
    print("├─────────────┼─────────────┼─────────────┤")
    print("│   Cam 9     │   Cam 6     │   Cam 2     │ 25-50ft")
    print("│             │             │             │")
    print("├─────────────┼─────────────┼─────────────┤")
    print("│   Cam 10    │   Cam 7     │   Cam 3     │ 50-75ft")
    print("│             │             │             │")
    print("├─────────────┼─────────────┼─────────────┤")
    print("│   Cam 11    │     ---     │   Cam 4     │ 75-100ft")
    print("│(BOTTOM LEFT)│             │             │")
    print("└─────────────┴─────────────┴─────────────┘")
    print()
    print("🎯 From YOUR viewing perspective:")
    print("   - Camera 1 = TOP RIGHT corner (0,0)")
    print("   - Camera 11 = BOTTOM LEFT corner")

if __name__ == "__main__":
    verify_camera_zone_mapping()
