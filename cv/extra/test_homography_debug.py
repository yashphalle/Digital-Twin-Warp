#!/usr/bin/env python3
"""
Debug Homography Matrix Calculation
Test the coordinate transformation to find the issue
"""

import json
import numpy as np
import cv2

def test_homography_calculation():
    """Test homography calculation for Camera 9"""
    print("🔍 DEBUGGING HOMOGRAPHY CALCULATION")
    print("=" * 60)
    
    # Load Camera 9 calibration
    with open("warehouse_calibration_camera_9.json", 'r') as f:
        calibration_data = json.load(f)
    
    print("📄 Calibration Data:")
    print(f"Image corners: {calibration_data['image_corners']}")
    print(f"Real world corners: {calibration_data['real_world_corners']}")
    print()
    
    # Extract corners
    image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
    real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
    
    print("🔢 Corner Analysis:")
    print("Image corners (pixels):")
    for i, corner in enumerate(image_corners):
        corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        print(f"   {corner_names[i]}: ({corner[0]}, {corner[1]})")
    
    print("\nReal world corners (feet):")
    for i, corner in enumerate(real_world_corners):
        corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        print(f"   {corner_names[i]}: ({corner[0]}, {corner[1]})")
    
    # Calculate homography
    homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
    
    print(f"\n🔄 Homography Matrix:")
    print(homography_matrix)
    
    # Test coordinate transformation
    print(f"\n🎯 COORDINATE TRANSFORMATION TESTS:")
    
    test_points = [
        ("Image center", [1920, 1080]),  # Center of 4K image
        ("Top-left corner", [0, 0]),
        ("Top-right corner", [3840, 0]),
        ("Bottom-right corner", [3840, 2160]),
        ("Bottom-left corner", [0, 2160]),
        ("Quarter point", [960, 540]),
        ("Three-quarter point", [2880, 1620]),
    ]
    
    for name, pixel_point in test_points:
        pixel_array = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        real_point = cv2.perspectiveTransform(pixel_array, homography_matrix)
        real_x = float(real_point[0][0][0])
        real_y = float(real_point[0][0][1])
        
        print(f"{name}: ({pixel_point[0]}, {pixel_point[1]}) → ({real_x:.1f}ft, {real_y:.1f}ft)")
    
    print(f"\n📊 EXPECTED VS ACTUAL:")
    print(f"Expected Camera 9 range: 120-180ft × 25-50ft")
    print(f"Image center should map to: ~150ft × ~37.5ft")
    
    # Calculate what image center actually maps to
    center_pixel = np.array([[[1920, 1080]]], dtype=np.float32)
    center_real = cv2.perspectiveTransform(center_pixel, homography_matrix)
    center_x = float(center_real[0][0][0])
    center_y = float(center_real[0][0][1])
    
    print(f"Image center actually maps to: {center_x:.1f}ft × {center_y:.1f}ft")
    
    if 120 <= center_x <= 180 and 25 <= center_y <= 50:
        print("✅ Coordinate transformation is working correctly!")
    else:
        print("❌ Coordinate transformation is NOT working correctly!")
        print("\n🔧 DIAGNOSIS:")
        print("The homography matrix is mapping the entire 4K image to the calibrated area.")
        print("This suggests the image_corners in the calibration file are wrong.")
        print("They should represent the actual calibrated area corners, not the full image.")

def test_correct_calibration():
    """Test what the correct calibration should look like"""
    print(f"\n🎯 CORRECT CALIBRATION ANALYSIS")
    print("=" * 60)
    
    print("PROBLEM IDENTIFIED:")
    print("The calibration file has image_corners as the full 4K image corners:")
    print("   [0,0], [3840,0], [3840,2160], [0,2160]")
    print()
    print("But real_world_corners are the specific area:")
    print("   [120,25], [180,25], [180,50], [120,50]")
    print()
    print("This creates a homography that maps the ENTIRE 4K image to a 60×25ft area!")
    print()
    print("SOLUTION:")
    print("The image_corners should represent the actual calibrated area in the image,")
    print("not the full image corners. For example:")
    print("   [500,300], [3300,300], [3300,1800], [500,1800]")
    print("   (These would be the pixel coordinates of the actual floor area)")

def main():
    """Run homography debug tests"""
    print("🚀 HOMOGRAPHY DEBUG TEST")
    print("=" * 60)
    print("Debugging coordinate transformation issues")
    print("=" * 60)
    
    test_homography_calculation()
    test_correct_calibration()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    print("The coordinate transformation issue is caused by incorrect image_corners")
    print("in the calibration files. The image_corners should represent the actual")
    print("calibrated floor area in the image, not the full image dimensions.")
    print()
    print("NEXT STEPS:")
    print("1. Re-calibrate cameras with correct image corner coordinates")
    print("2. Or modify the calibration files to use proper image corners")
    print("3. Test the coordinate transformation again")

if __name__ == "__main__":
    main()
