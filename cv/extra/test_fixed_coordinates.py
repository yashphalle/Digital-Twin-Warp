#!/usr/bin/env python3
"""
Test Fixed Coordinate Transformation
Test that objects on the left side of screen now map to higher X coordinates
"""

import json
import numpy as np
import cv2
from detector_tracker import CoordinateMapper

def test_fixed_camera_9():
    """Test the fixed Camera 9 coordinate transformation"""
    print("🔧 TESTING FIXED CAMERA 9 COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # Create coordinate mapper with fixed calibration
    mapper = CoordinateMapper(camera_id=9)
    mapper.load_calibration("warehouse_calibration_camera_9.json")
    
    print("📄 Fixed Calibration Test:")
    
    # Test key points
    test_points = [
        {"name": "Left side of image (should be ~180ft)", "pixel": [0, 1080]},
        {"name": "Right side of image (should be ~120ft)", "pixel": [3840, 1080]},
        {"name": "Center of image (should be ~150ft)", "pixel": [1920, 1080]},
        {"name": "Quarter left (should be ~165ft)", "pixel": [960, 1080]},
        {"name": "Quarter right (should be ~135ft)", "pixel": [2880, 1080]},
    ]
    
    for test in test_points:
        pixel_x, pixel_y = test["pixel"]
        real_x, real_y = mapper.pixel_to_real(pixel_x, pixel_y)
        print(f"{test['name']}: ({pixel_x}, {pixel_y}) → ({real_x:.1f}ft, {real_y:.1f}ft)")
    
    print()
    print("🎯 EXPECTED BEHAVIOR:")
    print("• Left side of image → Higher X coordinates (closer to 180ft)")
    print("• Right side of image → Lower X coordinates (closer to 120ft)")
    print("• This matches your observation: left object should be ~175ft")

def test_user_scenario():
    """Test the specific user scenario"""
    print("\n👤 TESTING YOUR SPECIFIC SCENARIO")
    print("=" * 60)
    
    mapper = CoordinateMapper(camera_id=9)
    mapper.load_calibration("warehouse_calibration_camera_9.json")
    
    print("Your scenario:")
    print("• Object 25ft to the LEFT from camera center")
    print("• Should be around 150 + 25 = 175ft (close to 180ft edge)")
    print("• Previously showing 125ft (150 - 25)")
    print()
    
    # Test a point that's 25% from the left edge (representing left side object)
    left_object_pixel = [960, 1080]  # 25% from left edge
    real_x, real_y = mapper.pixel_to_real(left_object_pixel[0], left_object_pixel[1])
    
    print(f"Object on left side of image:")
    print(f"   Pixel: {left_object_pixel}")
    print(f"   Real coordinates: ({real_x:.1f}ft, {real_y:.1f}ft)")
    
    if real_x > 150:
        print(f"   ✅ CORRECT: X={real_x:.1f}ft is > 150ft (camera center)")
        print(f"   ✅ Object is {real_x - 150:.1f}ft to the LEFT of camera center")
    else:
        print(f"   ❌ WRONG: X={real_x:.1f}ft is < 150ft")
        print(f"   ❌ This would put object to the RIGHT of camera center")

def test_all_cameras():
    """Test all camera calibrations"""
    print("\n📹 TESTING ALL CAMERA CALIBRATIONS")
    print("=" * 60)
    
    cameras = [8, 9, 11]
    
    for camera_id in cameras:
        print(f"\nCamera {camera_id}:")
        
        try:
            mapper = CoordinateMapper(camera_id=camera_id)
            mapper.load_calibration(f"warehouse_calibration_camera_{camera_id}.json")
            
            # Test left and right edges
            left_x, _ = mapper.pixel_to_real(0, 1080)
            right_x, _ = mapper.pixel_to_real(3840, 1080)
            
            print(f"   Left edge (pixel 0): {left_x:.1f}ft")
            print(f"   Right edge (pixel 3840): {right_x:.1f}ft")
            
            if left_x > right_x:
                print(f"   ✅ CORRECT: Left side has higher X coordinates")
            else:
                print(f"   ❌ WRONG: Left side has lower X coordinates")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run coordinate transformation tests"""
    print("🚀 TESTING FIXED COORDINATE TRANSFORMATION")
    print("=" * 60)
    print("Testing the fix for X-axis direction")
    print("=" * 60)
    
    test_fixed_camera_9()
    test_user_scenario()
    test_all_cameras()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print("After the fix:")
    print("• Objects on LEFT side of screen → Higher X coordinates (closer to 180ft)")
    print("• Objects on RIGHT side of screen → Lower X coordinates (closer to 120ft)")
    print("• Your 25ft left object should now show ~175ft instead of ~125ft")

if __name__ == "__main__":
    main()
