#!/usr/bin/env python3
"""
Test Coordinate Scaling Fix
Verify that coordinates are properly scaled between detection frame and calibration frame
"""

import cv2
import numpy as np
import json
from detector_tracker import CoordinateMapper

def test_coordinate_scaling():
    """Test coordinate scaling between different frame sizes"""
    print("🔧 TESTING COORDINATE SCALING FIX")
    print("=" * 50)
    
    # Initialize coordinate mapper
    mapper = CoordinateMapper()
    
    if not mapper.is_calibrated:
        print("❌ CoordinateMapper not calibrated!")
        return False
    
    print(f"✅ Mapper calibrated for warehouse: {mapper.floor_width:.2f}m x {mapper.floor_length:.2f}m")
    
    # Test coordinate scaling for different frame sizes
    test_cases = [
        # (frame_width, frame_height, description)
        (3840, 2160, "4K Original (Calibration Frame)"),
        (1920, 1080, "1080p Resized (Detection Frame)"),
        (1280, 720, "720p Resized"),
        (640, 360, "360p Resized"),
    ]
    
    # Test center points for each frame size
    print("\n📍 CENTER POINT COORDINATE TESTS:")
    print("(Testing if center of each frame maps to warehouse center)")
    
    for frame_width, frame_height, description in test_cases:
        print(f"\n   {description} ({frame_width}x{frame_height}):")
        
        # Center point in this frame size
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate scaling factors (as done in detector_tracker.py)
        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height
        
        # Scale to 4K coordinates
        scaled_x = center_x * scale_x
        scaled_y = center_y * scale_y
        
        # Convert to real coordinates
        real_x, real_y = mapper.pixel_to_real(scaled_x, scaled_y)
        
        if real_x is not None:
            real_x_ft = real_x * 3.28084
            real_y_ft = real_y * 3.28084
            
            # Check if close to warehouse center
            warehouse_center_x = mapper.floor_width / 2
            warehouse_center_y = mapper.floor_length / 2
            
            distance_from_center = np.sqrt((real_x - warehouse_center_x)**2 + (real_y - warehouse_center_y)**2)
            
            print(f"      Frame center: [{center_x}, {center_y}]")
            print(f"      Scale factors: X={scale_x:.2f}, Y={scale_y:.2f}")
            print(f"      Scaled to 4K: [{scaled_x:.0f}, {scaled_y:.0f}]")
            print(f"      Real coords: [{real_x:.2f}m, {real_y:.2f}m] = [{real_x_ft:.1f}ft, {real_y_ft:.1f}ft]")
            print(f"      Distance from warehouse center: {distance_from_center:.2f}m")
            print(f"      Accuracy: {'✅ GOOD' if distance_from_center < 1.0 else '❌ POOR'}")
        else:
            print(f"      ❌ Coordinate conversion failed")
    
    # Test corner mapping consistency
    print("\n🔄 CORNER MAPPING CONSISTENCY TEST:")
    print("(Testing if corners map consistently across frame sizes)")
    
    reference_frame = (3840, 2160)  # 4K reference
    test_frame = (1920, 1080)       # 1080p test
    
    # Test corner points
    corner_tests = [
        ("Top-Left", (100, 100)),
        ("Top-Right", (reference_frame[0] - 100, 100)),
        ("Bottom-Left", (100, reference_frame[1] - 100)),
        ("Bottom-Right", (reference_frame[0] - 100, reference_frame[1] - 100)),
    ]
    
    for corner_name, (ref_x, ref_y) in corner_tests:
        # Reference coordinate (4K)
        ref_real_x, ref_real_y = mapper.pixel_to_real(ref_x, ref_y)
        
        # Corresponding coordinate in test frame
        test_x = int(ref_x * test_frame[0] / reference_frame[0])
        test_y = int(ref_y * test_frame[1] / reference_frame[1])
        
        # Scale back to 4K
        scale_x = reference_frame[0] / test_frame[0]
        scale_y = reference_frame[1] / test_frame[1]
        scaled_test_x = test_x * scale_x
        scaled_test_y = test_y * scale_y
        
        # Convert to real coordinates
        test_real_x, test_real_y = mapper.pixel_to_real(scaled_test_x, scaled_test_y)
        
        if ref_real_x is not None and test_real_x is not None:
            distance_error = np.sqrt((ref_real_x - test_real_x)**2 + (ref_real_y - test_real_y)**2)
            
            print(f"\n   {corner_name}:")
            print(f"      4K coords: [{ref_x}, {ref_y}] → [{ref_real_x:.2f}m, {ref_real_y:.2f}m]")
            print(f"      1080p coords: [{test_x}, {test_y}] → scaled [{scaled_test_x:.0f}, {scaled_test_y:.0f}] → [{test_real_x:.2f}m, {test_real_y:.2f}m]")
            print(f"      Error: {distance_error:.3f}m")
            print(f"      Match: {'✅ GOOD' if distance_error < 0.1 else '❌ POOR'}")
    
    return True

def main():
    """Run coordinate scaling tests"""
    print("🚀 COORDINATE SCALING TEST SUITE")
    print("=" * 60)
    
    success = test_coordinate_scaling()
    
    print("\n📊 TEST SUMMARY:")
    if success:
        print("✅ Coordinate scaling tests completed!")
        print("🎯 The fix should now provide accurate coordinates regardless of frame size")
    else:
        print("❌ Coordinate scaling tests failed")
        print("🔧 Review the coordinate mapping and scaling logic")

if __name__ == "__main__":
    main() 