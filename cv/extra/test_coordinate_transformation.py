#!/usr/bin/env python3
"""
Test Coordinate Transformation for Column 3 Cameras
Verifies that pixel coordinates are properly transformed to global warehouse coordinates
"""

import sys
import logging
from detector_tracker import CoordinateMapper
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coordinate_transformation():
    """Test coordinate transformation for all Column 3 cameras"""
    print("🎯 TESTING COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # Test cases: pixel coordinates and expected global ranges
    test_pixels = [
        (112, 254, "Top-left corner"),      # From calibration
        (3766, 281, "Top-right corner"),    # From calibration  
        (3754, 2030, "Bottom-right corner"), # From calibration
        (147, 1949, "Bottom-left corner"),  # From calibration
        (1920, 1080, "Center of image"),    # Center pixel
        (960, 540, "Quarter point"),        # Quarter point
        (2880, 1620, "Three-quarter point") # Three-quarter point
    ]
    
    for camera_id in [8, 9, 10, 11]:
        print(f"\n📹 CAMERA {camera_id} COORDINATE TRANSFORMATION:")
        print("-" * 50)
        
        # Get expected zone
        zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
        expected_x_range = (zone['x_start'], zone['x_end'])
        expected_y_range = (zone['y_start'], zone['y_end'])
        
        print(f"Expected global range: {expected_x_range[0]}-{expected_x_range[1]}ft × {expected_y_range[0]}-{expected_y_range[1]}ft")
        
        # Initialize coordinate mapper
        mapper = CoordinateMapper(camera_id=camera_id)
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        mapper.load_calibration(calibration_file)
        
        if not mapper.is_calibrated:
            print(f"❌ Camera {camera_id} calibration failed")
            continue
            
        print(f"✅ Calibration loaded")
        
        # Test coordinate transformations
        valid_transformations = 0
        total_tests = len(test_pixels)
        
        for pixel_x, pixel_y, description in test_pixels:
            global_x, global_y = mapper.pixel_to_real(pixel_x, pixel_y)
            
            if global_x is not None and global_y is not None:
                # Check if coordinates are in expected range
                x_in_range = expected_x_range[0] <= global_x <= expected_x_range[1]
                y_in_range = expected_y_range[0] <= global_y <= expected_y_range[1]
                
                status = "✅" if (x_in_range and y_in_range) else "❌"
                
                print(f"   {description}:")
                print(f"      Pixel: ({pixel_x}, {pixel_y})")
                print(f"      Global: ({global_x:.1f}ft, {global_y:.1f}ft) {status}")
                
                if x_in_range and y_in_range:
                    valid_transformations += 1
                else:
                    print(f"      ⚠️  Outside expected range!")
            else:
                print(f"   {description}: ❌ Transformation failed")
        
        success_rate = (valid_transformations / total_tests) * 100
        print(f"\n📊 Camera {camera_id} success rate: {valid_transformations}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 70:  # At least 70% should be in range
            print(f"✅ Camera {camera_id} coordinate transformation working correctly")
        else:
            print(f"❌ Camera {camera_id} coordinate transformation needs adjustment")

def test_specific_coordinates():
    """Test specific coordinates that should map to known locations"""
    print("\n🎯 TESTING SPECIFIC COORDINATE MAPPINGS")
    print("=" * 60)
    
    # Test center points for each camera
    center_pixel = (1920, 1080)  # Center of 4K image
    
    expected_centers = {
        8: (150, 12.5),   # Center of 120-180ft × 0-25ft
        9: (150, 37.5),   # Center of 120-180ft × 25-50ft  
        10: (150, 62.5),  # Center of 120-180ft × 50-75ft
        11: (150, 82.5)   # Center of 120-180ft × 75-90ft
    }
    
    for camera_id, expected_center in expected_centers.items():
        print(f"\n📹 Camera {camera_id} center point test:")
        
        mapper = CoordinateMapper(camera_id=camera_id)
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        mapper.load_calibration(calibration_file)
        
        if mapper.is_calibrated:
            global_x, global_y = mapper.pixel_to_real(center_pixel[0], center_pixel[1])
            
            if global_x is not None and global_y is not None:
                expected_x, expected_y = expected_center
                
                # Allow 5ft tolerance
                x_diff = abs(global_x - expected_x)
                y_diff = abs(global_y - expected_y)
                
                print(f"   Expected center: ({expected_x}ft, {expected_y}ft)")
                print(f"   Actual result:   ({global_x:.1f}ft, {global_y:.1f}ft)")
                print(f"   Difference:      ({x_diff:.1f}ft, {y_diff:.1f}ft)")
                
                if x_diff <= 5 and y_diff <= 5:
                    print(f"   ✅ Center mapping accurate (within 5ft tolerance)")
                else:
                    print(f"   ❌ Center mapping inaccurate (outside 5ft tolerance)")
            else:
                print(f"   ❌ Transformation failed")
        else:
            print(f"   ❌ Calibration failed")

def main():
    """Run coordinate transformation tests"""
    print("🚀 COORDINATE TRANSFORMATION TEST SUITE")
    print("=" * 60)
    print("Testing global coordinate mapping for Column 3 cameras")
    print("Expected: Objects should appear in 120-180ft × 0-90ft range")
    print("=" * 60)
    
    # Run tests
    test_coordinate_transformation()
    test_specific_coordinates()
    
    print("\n" + "=" * 60)
    print("🎯 COORDINATE TRANSFORMATION TEST COMPLETE")
    print("=" * 60)
    print("\n🚀 NEXT STEPS:")
    print("1. If tests pass: Start multi-camera system")
    print("2. If tests fail: Check calibration files")
    print("3. Verify objects appear in correct global coordinates")
    print("\n📍 Expected object coordinate ranges:")
    print("   Camera 8:  120-180ft × 0-25ft")
    print("   Camera 9:  120-180ft × 25-50ft") 
    print("   Camera 10: 120-180ft × 50-75ft")
    print("   Camera 11: 120-180ft × 75-90ft")

if __name__ == "__main__":
    main()
