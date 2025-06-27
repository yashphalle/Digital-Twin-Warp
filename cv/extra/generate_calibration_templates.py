#!/usr/bin/env python3
"""
Generate Calibration Template Files for All 11 Cameras
Creates individual calibration files for each camera with the new coordinate system
"""

import json
import os
from datetime import datetime
from config import Config

def generate_calibration_template(camera_id):
    """Generate a calibration template for a specific camera"""
    
    # Get camera zone info from config
    if camera_id not in Config.CAMERA_COVERAGE_ZONES:
        print(f"❌ Camera {camera_id} not found in CAMERA_COVERAGE_ZONES")
        return None
    
    zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
    camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
    rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "rtsp://admin:wearewarp!@192.168.0.XX:554/Streaming/channels/1")
    
    # Create calibration template
    calibration_data = {
        "description": f"RTSP Camera {camera_id} Warehouse Calibration - NEW COORDINATE SYSTEM",
        "created_date": datetime.now().isoformat(),
        "camera_info": {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "rtsp_url": rtsp_url,
            "frame_width": 3840,
            "frame_height": 2160,
            "fisheye_corrected": True,
            "lens_mm": 2.8
        },
        "warehouse_dimensions": {
            "width_feet": 60.0,
            "length_feet": 22.5,
            "width_meters": 18.288,
            "length_meters": 6.858,
            "coverage_zone": {
                "x_start_ft": zone["x_start"],
                "x_end_ft": zone["x_end"],
                "y_start_ft": zone["y_start"],
                "y_end_ft": zone["y_end"],
                "center_x_ft": zone["center_x"],
                "center_y_ft": zone["center_y"],
                "column": zone["column"],
                "position": "top" if zone["y_start"] == 0 else 
                           "mid-top" if zone["y_start"] == 22.5 else
                           "mid-bottom" if zone["y_start"] == 45 else "bottom"
            }
        },
        "coordinate_system_info": {
            "warehouse_origin": "top-right corner (0,0)",
            "warehouse_dimensions": "180ft x 90ft",
            "x_axis_direction": "right-to-left (0 to 180ft)",
            "y_axis_direction": "top-to-bottom (0 to 90ft)",
            "local_coverage": "60ft x 22.5ft area",
            "global_position": f"Column {zone['column']}, {'top' if zone['y_start'] == 0 else 'mid-top' if zone['y_start'] == 22.5 else 'mid-bottom' if zone['y_start'] == 45 else 'bottom'} section"
        },
        "image_corners": [
            [0, 0],          # Top-left (template - needs calibration)
            [3840, 0],       # Top-right (template)
            [3840, 2160],    # Bottom-right (template)
            [0, 2160]        # Bottom-left (template)
        ],
        "real_world_corners": [
            [0, 0],          # Top-left of camera coverage area
            [60.0, 0],       # Top-right of camera coverage area
            [60.0, 22.5],    # Bottom-right of camera coverage area
            [0, 22.5]        # Bottom-left of camera coverage area
        ],
        "calibration_info": {
            "corner_order": "top-left, top-right, bottom-right, bottom-left",
            "coordinate_system": "local camera coordinates: origin at top-left, x-axis right, y-axis down",
            "units": "feet",
            "calibration_status": "TEMPLATE - Needs manual calibration" if camera_id != 8 else "CALIBRATED",
            "notes": f"Camera {camera_id} covers 60ft x 22.5ft area. Local coordinates (0-60ft, 0-22.5ft) are transformed to warehouse coordinates by the CoordinateMapper class."
        }
    }
    
    return calibration_data

def generate_all_calibration_files():
    """Generate calibration files for all 11 cameras"""
    print("🎯 GENERATING CALIBRATION TEMPLATE FILES")
    print("=" * 50)
    
    success_count = 0
    
    for camera_id in range(1, 12):  # Cameras 1-11
        try:
            # Generate template data
            calibration_data = generate_calibration_template(camera_id)
            
            if calibration_data is None:
                continue
            
            # Create filename
            filename = f"warehouse_calibration_camera_{camera_id}.json"
            
            # Skip Camera 8 if it already exists and is calibrated
            if camera_id == 8 and os.path.exists("warehouse_calibration.json"):
                print(f"📁 Camera 8: Using existing calibration (warehouse_calibration.json)")
                continue
            
            # Write file
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
            print(f"✅ Camera {camera_id}: {filename}")
            print(f"   └─ Zone: ({zone['x_start']}-{zone['x_end']}ft, {zone['y_start']}-{zone['y_end']}ft)")
            print(f"   └─ Column {zone['column']}, Center: ({zone['center_x']}, {zone['center_y']}ft)")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Camera {camera_id}: Error - {e}")
    
    print(f"\n✅ Generated {success_count} calibration template files")
    print("\n📋 NEXT STEPS:")
    print("1. Use rtsp_calibration_tool.py to calibrate each camera individually")
    print("2. Run each camera through the calibration process to get accurate image_corners")
    print("3. Test coordinate accuracy with test_coordinate_mapping.py")
    print("\n🎯 CALIBRATION COMMAND EXAMPLE:")
    print("   python rtsp_calibration_tool.py --camera-id 1")
    print("   python rtsp_calibration_tool.py --camera-id 2")
    print("   # ... etc for all cameras")

def create_calibration_guide():
    """Create a calibration guide file"""
    guide_content = """
# CAMERA CALIBRATION GUIDE - NEW COORDINATE SYSTEM

## Overview
- Warehouse: 180ft × 90ft
- Origin: Top-right corner (0,0)
- 3 Columns × 4 Camera zones each (Column 2 has office space)
- Each camera covers: 60ft × 22.5ft area

## Camera Layout
```
Column 3          Column 2          Column 1
(120-180ft)       (60-120ft)        (0-60ft)
    |               |                 |
Camera 8 -------- Camera 5 -------- Camera 1    (0-22.5ft)
Camera 9 -------- Camera 6 -------- Camera 2    (22.5-45ft)
Camera 10 ------- Camera 7 -------- Camera 3    (45-67.5ft)
Camera 11 ------- Office   -------- Camera 4    (67.5-90ft)
```

## Calibration Steps

### 1. For Each Camera:
```bash
python rtsp_calibration_tool.py --camera-id X
```

### 2. During Calibration:
- Click 4 corners of the camera's coverage area (60ft × 22.5ft)
- Corner order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
- Ensure coverage area matches the physical camera zone

### 3. Validation:
```bash
python test_coordinate_mapping.py
```

## Expected Coverage Zones:
- Camera 1: (0-60ft, 0-22.5ft) - Column 1 Top
- Camera 2: (0-60ft, 22.5-45ft) - Column 1 Mid-Top
- Camera 3: (0-60ft, 45-67.5ft) - Column 1 Mid-Bottom
- Camera 4: (0-60ft, 67.5-90ft) - Column 1 Bottom
- Camera 5: (60-120ft, 0-22.5ft) - Column 2 Top
- Camera 6: (60-120ft, 22.5-45ft) - Column 2 Mid-Top
- Camera 7: (60-120ft, 45-67.5ft) - Column 2 Mid-Bottom
- Camera 8: (120-180ft, 0-22.5ft) - Column 3 Top
- Camera 9: (120-180ft, 22.5-45ft) - Column 3 Mid-Top
- Camera 10: (120-180ft, 45-67.5ft) - Column 3 Mid-Bottom
- Camera 11: (120-180ft, 67.5-90ft) - Column 3 Bottom

## Coordinate Transformation:
1. Camera detects object at pixel (x, y)
2. Homography converts to local coordinates (0-60ft, 0-22.5ft)
3. Zone offset adds camera position
4. Coordinate system flip (origin top-right)
5. Final warehouse coordinates stored in MongoDB

## Target Precision: ±6 inches
"""
    
    with open("CALIBRATION_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("📖 Created CALIBRATION_GUIDE.md")

if __name__ == "__main__":
    generate_all_calibration_files()
    create_calibration_guide()
    
    print(f"\n🎉 SETUP COMPLETE!")
    print(f"Ready to start calibrating cameras with the new coordinate system.") 