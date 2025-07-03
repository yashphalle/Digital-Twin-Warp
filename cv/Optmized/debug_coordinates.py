#!/usr/bin/env python3
"""
🔍 DEBUG COORDINATE MAPPING
Test coordinate mapping for negative values
"""

import sys
import os
import numpy as np
import cv2
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_coordinate_mapping():
    """Test coordinate mapping for Camera 1"""
    
    print("🔍 DEBUGGING COORDINATE MAPPING FOR CAMERA 1")
    print("=" * 60)
    
    # Load calibration data
    calibration_file = "../configs/warehouse_calibration_camera_1.json"
    
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        print(f"✅ Loaded calibration: {calibration_file}")
        print(f"📊 Image corners: {calibration_data['image_corners']}")
        print(f"📊 Real world corners: {calibration_data['real_world_corners']}")
        
        # Extract corners
        image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
        real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(image_corners, real_world_corners)
        
        print(f"\n🔧 Perspective Matrix:")
        print(perspective_matrix)
        
        # Test specific points
        test_points = [
            ([960, 540], "Center of 1920x1080 frame"),
            ([1649, 251], "Sample detection point from logs"),
            ([0, 0], "Top-left corner"),
            ([1920, 0], "Top-right corner"),
            ([0, 1080], "Bottom-left corner"),
            ([1920, 1080], "Bottom-right corner"),
            ([112, 254], "Calibration corner 1"),
            ([3766, 281], "Calibration corner 2"),
            ([3754, 2030], "Calibration corner 3"),
            ([147, 1949], "Calibration corner 4")
        ]
        
        print(f"\n📍 TESTING COORDINATE TRANSFORMATIONS:")
        print("-" * 60)
        
        for point, description in test_points:
            # Convert to homogeneous coordinates
            pixel_point = np.array([[point[0], point[1]]], dtype=np.float32)
            
            # Apply perspective transformation
            real_world_point = cv2.perspectiveTransform(
                pixel_point.reshape(1, 1, 2), 
                perspective_matrix
            ).reshape(2)
            
            x_ft, y_ft = real_world_point[0], real_world_point[1]
            
            # Check if coordinates are in expected range
            in_range = (0 <= x_ft <= 62) and (0 <= y_ft <= 25)
            status = "✅" if in_range else "❌"
            
            print(f"{status} {description}")
            print(f"   Pixel: ({point[0]}, {point[1]}) → Physical: ({x_ft:.2f}, {y_ft:.2f}) ft")
            
            if not in_range:
                if x_ft < 0:
                    print(f"   ⚠️ X coordinate is NEGATIVE: {x_ft:.2f}")
                if y_ft < 0:
                    print(f"   ⚠️ Y coordinate is NEGATIVE: {y_ft:.2f}")
                if x_ft > 62:
                    print(f"   ⚠️ X coordinate is TOO HIGH: {x_ft:.2f} (max: 62)")
                if y_ft > 25:
                    print(f"   ⚠️ Y coordinate is TOO HIGH: {y_ft:.2f} (max: 25)")
            print()
        
        # Check if the calibration corners map correctly
        print(f"\n🎯 CALIBRATION CORNER VERIFICATION:")
        print("-" * 60)
        
        expected_corners = calibration_data['real_world_corners']
        
        for i, (img_corner, expected_real) in enumerate(zip(image_corners, expected_corners)):
            pixel_point = np.array([[img_corner[0], img_corner[1]]], dtype=np.float32)
            real_world_point = cv2.perspectiveTransform(
                pixel_point.reshape(1, 1, 2), 
                perspective_matrix
            ).reshape(2)
            
            actual_x, actual_y = real_world_point[0], real_world_point[1]
            expected_x, expected_y = expected_real[0], expected_real[1]
            
            error_x = abs(actual_x - expected_x)
            error_y = abs(actual_y - expected_y)
            
            print(f"Corner {i+1}:")
            print(f"   Expected: ({expected_x}, {expected_y}) ft")
            print(f"   Actual:   ({actual_x:.2f}, {actual_y:.2f}) ft")
            print(f"   Error:    ({error_x:.2f}, {error_y:.2f}) ft")
            
            if error_x > 1.0 or error_y > 1.0:
                print(f"   ❌ HIGH ERROR!")
            else:
                print(f"   ✅ Good calibration")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing coordinate mapping: {e}")
        return False

def suggest_fixes():
    """Suggest potential fixes for negative coordinates"""
    
    print("\n🔧 POTENTIAL FIXES FOR NEGATIVE COORDINATES:")
    print("=" * 60)
    
    print("1. 📐 Check calibration corner order:")
    print("   - Image corners should be in clockwise order")
    print("   - Real world corners should match the same order")
    print()
    
    print("2. 🔄 Check coordinate system orientation:")
    print("   - X-axis: Left to Right (0 to 62 ft)")
    print("   - Y-axis: Top to Bottom (0 to 25 ft)")
    print()
    
    print("3. 🎯 Verify calibration points:")
    print("   - Corner 1: Top-right of warehouse area")
    print("   - Corner 2: Top-left of warehouse area") 
    print("   - Corner 3: Bottom-left of warehouse area")
    print("   - Corner 4: Bottom-right of warehouse area")
    print()
    
    print("4. 🔍 Check for fisheye distortion:")
    print("   - Ensure fisheye correction is applied before calibration")
    print("   - Verify image corners are from corrected image")

def main():
    """Main debug function"""
    
    print("🔍 COORDINATE MAPPING DEBUG TOOL")
    print("=" * 80)
    
    success = test_coordinate_mapping()
    
    if not success:
        print("❌ Coordinate mapping test failed!")
    
    suggest_fixes()
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("1. Check the test results above")
    print("2. Look for negative coordinates or out-of-range values")
    print("3. If calibration corners have high error, recalibrate")
    print("4. If specific points are negative, adjust calibration")
    print("=" * 80)

if __name__ == "__main__":
    main()
