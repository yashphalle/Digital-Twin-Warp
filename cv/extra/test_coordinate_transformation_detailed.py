#!/usr/bin/env python3
"""
Detailed Coordinate Transformation Test
Test the exact transformation logic to find X-coordinate issues
"""

import json
import numpy as np
import cv2
from detector_tracker import CoordinateMapper

def test_camera_9_transformation():
    """Test Camera 9 coordinate transformation in detail"""
    print("🔍 DETAILED CAMERA 9 COORDINATE TRANSFORMATION TEST")
    print("=" * 70)
    
    # Load Camera 9 calibration
    with open("warehouse_calibration_camera_9.json", 'r') as f:
        calibration_data = json.load(f)
    
    print("📄 Camera 9 Calibration Data:")
    print(f"Image corners: {calibration_data['image_corners']}")
    print(f"Real world corners: {calibration_data['real_world_corners']}")
    print()
    
    # Create coordinate mapper
    mapper = CoordinateMapper(camera_id=9)
    mapper.load_calibration("warehouse_calibration_camera_9.json")
    
    print("🔄 Homography Matrix:")
    print(mapper.homography_matrix)
    print()
    
    # Test specific pixel coordinates that should map to known real-world coordinates
    test_cases = [
        # Corner tests
        {"name": "Top-left corner", "pixel": [0, 0], "expected": [120, 25]},
        {"name": "Top-right corner", "pixel": [3840, 0], "expected": [180, 25]},
        {"name": "Bottom-right corner", "pixel": [3840, 2160], "expected": [180, 50]},
        {"name": "Bottom-left corner", "pixel": [0, 2160], "expected": [120, 50]},
        
        # Center and quarter points
        {"name": "Image center", "pixel": [1920, 1080], "expected": [150, 37.5]},
        {"name": "Left center", "pixel": [0, 1080], "expected": [120, 37.5]},
        {"name": "Right center", "pixel": [3840, 1080], "expected": [180, 37.5]},
        {"name": "Top center", "pixel": [1920, 0], "expected": [150, 25]},
        {"name": "Bottom center", "pixel": [1920, 2160], "expected": [150, 50]},
        
        # Quarter points
        {"name": "Quarter X, center Y", "pixel": [960, 1080], "expected": [135, 37.5]},
        {"name": "Three-quarter X, center Y", "pixel": [2880, 1080], "expected": [165, 37.5]},
    ]
    
    print("🎯 COORDINATE TRANSFORMATION TESTS:")
    print("-" * 70)
    
    for test in test_cases:
        pixel_x, pixel_y = test["pixel"]
        expected_x, expected_y = test["expected"]
        
        # Transform using the coordinate mapper
        real_x, real_y = mapper.pixel_to_real(pixel_x, pixel_y)
        
        # Calculate errors
        error_x = abs(real_x - expected_x) if real_x else float('inf')
        error_y = abs(real_y - expected_y) if real_y else float('inf')
        
        print(f"{test['name']}:")
        print(f"   Pixel: ({pixel_x}, {pixel_y})")
        print(f"   Expected: ({expected_x:.1f}ft, {expected_y:.1f}ft)")
        print(f"   Actual: ({real_x:.1f}ft, {real_y:.1f}ft)")
        print(f"   Error: X={error_x:.1f}ft, Y={error_y:.1f}ft")
        
        if error_x < 0.1 and error_y < 0.1:
            print(f"   ✅ PASS")
        else:
            print(f"   ❌ FAIL")
        print()

def test_actual_detection_coordinates():
    """Test coordinates from actual detections"""
    print("🎯 ACTUAL DETECTION COORDINATE ANALYSIS")
    print("=" * 70)
    
    # Recent database coordinates
    recent_detections = [
        {"camera": 9, "pixel": [320, 914], "real": [130.0, 46.2], "id": 8},
        {"camera": 9, "pixel": [193, 975], "real": [126.0, 47.6], "id": 10},
        {"camera": 9, "pixel": [1163, 422], "real": [156.3, 34.8], "id": 1},
        {"camera": 11, "pixel": [1763, 748], "real": [175.1, 85.4], "id": 13},
        {"camera": 11, "pixel": [769, 885], "real": [144.0, 87.3], "id": 11},
    ]
    
    print("Recent detections from database:")
    print("-" * 70)
    
    for detection in recent_detections:
        camera = detection["camera"]
        pixel_x, pixel_y = detection["pixel"]
        real_x, real_y = detection["real"]
        obj_id = detection["id"]
        
        # Load the appropriate calibration
        calibration_file = f"warehouse_calibration_camera_{camera}.json"
        mapper = CoordinateMapper(camera_id=camera)
        mapper.load_calibration(calibration_file)
        
        # Test the transformation
        calculated_x, calculated_y = mapper.pixel_to_real(pixel_x, pixel_y)
        
        print(f"Object ID {obj_id} (Camera {camera}):")
        print(f"   Pixel: ({pixel_x}, {pixel_y})")
        print(f"   Database real: ({real_x:.1f}ft, {real_y:.1f}ft)")
        print(f"   Calculated real: ({calculated_x:.1f}ft, {calculated_y:.1f}ft)")
        print(f"   Difference: X={abs(calculated_x - real_x):.1f}ft, Y={abs(calculated_y - real_y):.1f}ft")
        
        # Check if it's in expected range
        if camera == 9:
            expected_x_range = (120, 180)
            expected_y_range = (25, 50)
        elif camera == 11:
            expected_x_range = (120, 180)
            expected_y_range = (75, 90)
        
        x_in_range = expected_x_range[0] <= real_x <= expected_x_range[1]
        y_in_range = expected_y_range[0] <= real_y <= expected_y_range[1]
        
        if x_in_range and y_in_range:
            print(f"   ✅ In expected range")
        else:
            print(f"   ❌ Outside expected range")
            if not x_in_range:
                print(f"      X coordinate {real_x:.1f}ft outside {expected_x_range[0]}-{expected_x_range[1]}ft")
            if not y_in_range:
                print(f"      Y coordinate {real_y:.1f}ft outside {expected_y_range[0]}-{expected_y_range[1]}ft")
        print()

def test_coordinate_scaling_issue():
    """Test if there's a scaling issue in the coordinate transformation"""
    print("🔍 COORDINATE SCALING ANALYSIS")
    print("=" * 70)
    
    # Test Camera 9
    mapper = CoordinateMapper(camera_id=9)
    mapper.load_calibration("warehouse_calibration_camera_9.json")
    
    print("Camera 9 Analysis:")
    print(f"Floor width: {mapper.floor_width_ft}ft")
    print(f"Floor length: {mapper.floor_length_ft}ft")
    print()
    
    # Test edge coordinates
    edge_tests = [
        {"name": "Far left edge", "pixel": [0, 1080]},
        {"name": "Far right edge", "pixel": [3840, 1080]},
        {"name": "Top edge", "pixel": [1920, 0]},
        {"name": "Bottom edge", "pixel": [1920, 2160]},
    ]
    
    print("Edge coordinate tests:")
    for test in edge_tests:
        pixel_x, pixel_y = test["pixel"]
        real_x, real_y = mapper.pixel_to_real(pixel_x, pixel_y)
        print(f"   {test['name']}: ({pixel_x}, {pixel_y}) → ({real_x:.1f}ft, {real_y:.1f}ft)")
    
    print()
    print("Expected ranges:")
    print("   X: 120-180ft (60ft wide)")
    print("   Y: 25-50ft (25ft tall)")
    
    # Calculate actual ranges
    left_x, _ = mapper.pixel_to_real(0, 1080)
    right_x, _ = mapper.pixel_to_real(3840, 1080)
    _, top_y = mapper.pixel_to_real(1920, 0)
    _, bottom_y = mapper.pixel_to_real(1920, 2160)
    
    actual_width = abs(right_x - left_x)
    actual_height = abs(bottom_y - top_y)
    
    print(f"Actual ranges:")
    print(f"   X: {min(left_x, right_x):.1f}-{max(left_x, right_x):.1f}ft ({actual_width:.1f}ft wide)")
    print(f"   Y: {min(top_y, bottom_y):.1f}-{max(top_y, bottom_y):.1f}ft ({actual_height:.1f}ft tall)")
    
    if abs(actual_width - 60) < 0.1 and abs(actual_height - 25) < 0.1:
        print("✅ Scaling is correct")
    else:
        print("❌ Scaling issue detected")
        print(f"   Width should be 60ft, got {actual_width:.1f}ft")
        print(f"   Height should be 25ft, got {actual_height:.1f}ft")

def main():
    """Run detailed coordinate transformation tests"""
    print("🚀 DETAILED COORDINATE TRANSFORMATION DEBUG")
    print("=" * 70)
    print("Investigating X-coordinate transformation issues")
    print("=" * 70)
    
    test_camera_9_transformation()
    test_actual_detection_coordinates()
    test_coordinate_scaling_issue()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print("This test will help identify:")
    print("1. If the homography matrix is correct")
    print("2. If there are scaling issues")
    print("3. If the coordinate transformation logic has bugs")
    print("4. If the calibration files have incorrect values")

if __name__ == "__main__":
    main()
