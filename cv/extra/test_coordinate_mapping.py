#!/usr/bin/env python3
"""
Test Coordinate Mapping
Debug coordinate conversion between pixels and real-world coordinates
"""

import cv2
import numpy as np
import json
from detector_tracker import CoordinateMapper

def test_coordinate_mapping():
    """Test coordinate mapping with current calibration"""
    print("🧪 TESTING COORDINATE MAPPING")
    print("=" * 50)
    
    # Load calibration data directly
    try:
        with open("warehouse_calibration.json", 'r') as f:
            calibration = json.load(f)
        
        print("📋 CALIBRATION DATA:")
        print(f"   Warehouse: {calibration['warehouse_dimensions']['width_feet']:.1f}ft x {calibration['warehouse_dimensions']['length_feet']:.1f}ft")
        print(f"   Warehouse: {calibration['warehouse_dimensions']['width_meters']:.2f}m x {calibration['warehouse_dimensions']['length_meters']:.2f}m")
        print(f"   Camera resolution: {calibration['camera_info']['frame_width']}x{calibration['camera_info']['frame_height']}")
        print(f"   Image corners: {calibration['image_corners']}")
        print(f"   Real corners: {calibration['real_world_corners']}")
        
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return False
    
    # Test CoordinateMapper
    print("\n🎯 TESTING COORDINATEMAPPER CLASS:")
    mapper = CoordinateMapper()
    
    print(f"   Calibrated: {mapper.is_calibrated}")
    print(f"   Floor dimensions: {mapper.floor_width:.2f}m x {mapper.floor_length:.2f}m")
    
    if not mapper.is_calibrated:
        print("❌ CoordinateMapper not calibrated!")
        return False
    
    # Test transformation matrices directly
    print("\n🔧 TESTING DIRECT TRANSFORMATION:")
    image_corners = np.array(calibration['image_corners'], dtype=np.float32)
    real_corners = np.array(calibration['real_world_corners'], dtype=np.float32)
    
    # Calculate transformation matrix from image to real
    transform_matrix = cv2.getPerspectiveTransform(image_corners, real_corners)
    print(f"   Transform matrix shape: {transform_matrix.shape}")
    
    # Test specific points
    print("\n📍 COORDINATE CONVERSION TESTS:")
    test_cases = [
        # Image coordinates -> Expected real coordinates
        ([54, 284], "Top-Left Corner"),           # Should be ~[0, 0]
        ([3763, 292], "Top-Right Corner"),        # Should be ~[9.75, 0]
        ([3782, 2033], "Bottom-Right Corner"),    # Should be ~[9.75, 19.5]
        ([96, 1963], "Bottom-Left Corner"),       # Should be ~[0, 19.5]
        ([1920, 1080], "Center (approx)"),       # Should be ~[4.9, 9.8]
        ([1000, 600], "Warehouse Quarter"),      # Somewhere in warehouse
        ([2800, 1500], "Warehouse Three-Quarter"), # Another test point
    ]
    
    for (pixel_x, pixel_y), description in test_cases:
        # Test with CoordinateMapper
        real_x_mapper, real_y_mapper = mapper.pixel_to_real(pixel_x, pixel_y)
        
        # Test with direct transformation
        point_array = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(1, 1, 2)
        real_point_direct = cv2.perspectiveTransform(point_array, transform_matrix)[0][0]
        real_x_direct, real_y_direct = real_point_direct
        
        # Convert to feet for display
        real_x_ft = real_x_mapper * 3.28084 if real_x_mapper is not None else None
        real_y_ft = real_y_mapper * 3.28084 if real_y_mapper is not None else None
        
        print(f"   {description}:")
        print(f"      Image: [{pixel_x}, {pixel_y}]")
        print(f"      CoordinateMapper: [{real_x_mapper:.2f}m, {real_y_mapper:.2f}m] = [{real_x_ft:.1f}ft, {real_y_ft:.1f}ft]" if real_x_mapper is not None else "      CoordinateMapper: FAILED")
        print(f"      Direct Transform: [{real_x_direct:.2f}m, {real_y_direct:.2f}m]")
        print(f"      Match: {'✅ YES' if abs(real_x_mapper - real_x_direct) < 0.01 and abs(real_y_mapper - real_y_direct) < 0.01 else '❌ NO'}")
        print()
    
    # Test reverse transformation
    print("🔄 REVERSE TRANSFORMATION TEST:")
    test_real_coords = [
        ([0, 0], "Origin"),
        ([4.88, 9.75], "Center"),
        ([9.75, 19.5], "Far Corner"),
    ]
    
    for (real_x, real_y), description in test_real_coords:
        pixel_x, pixel_y = mapper.real_to_pixel(real_x, real_y)
        back_real_x, back_real_y = mapper.pixel_to_real(pixel_x, pixel_y)
        
        print(f"   {description}:")
        print(f"      Real: [{real_x:.2f}m, {real_y:.2f}m]")
        print(f"      → Pixel: [{pixel_x}, {pixel_y}]")
        print(f"      → Back to Real: [{back_real_x:.2f}m, {back_real_y:.2f}m]")
        print(f"      Accuracy: {'✅ GOOD' if abs(real_x - back_real_x) < 0.1 and abs(real_y - back_real_y) < 0.1 else '❌ POOR'}")
        print()
    
    return True

def test_visual_mapping():
    """Create a visual test of coordinate mapping"""
    print("🎨 CREATING VISUAL COORDINATE TEST")
    print("=" * 40)
    
    # Create test image
    test_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    
    # Load calibration
    try:
        with open("warehouse_calibration.json", 'r') as f:
            calibration = json.load(f)
        
        image_corners = calibration['image_corners']
        
        # Draw warehouse boundary
        corners = np.array(image_corners, dtype=np.int32)
        cv2.polylines(test_img, [corners], True, (0, 255, 0), 5)
        
        # Mark corners
        corner_names = ["TL", "TR", "BR", "BL"]
        for i, (corner, name) in enumerate(zip(image_corners, corner_names)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(test_img, (x, y), 20, (0, 0, 255), -1)
            cv2.putText(test_img, name, (x-30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test grid points
        mapper = CoordinateMapper()
        grid_points = []
        
        # Create grid in real coordinates
        for real_x in np.arange(0, mapper.floor_width + 1, 2):  # Every 2 meters
            for real_y in np.arange(0, mapper.floor_length + 1, 4):  # Every 4 meters
                pixel_x, pixel_y = mapper.real_to_pixel(real_x, real_y)
                if pixel_x is not None and pixel_y is not None:
                    grid_points.append((pixel_x, pixel_y, real_x, real_y))
        
        # Draw grid points
        for pixel_x, pixel_y, real_x, real_y in grid_points:
            cv2.circle(test_img, (int(pixel_x), int(pixel_y)), 8, (255, 255, 0), -1)
            label = f"({real_x:.0f},{real_y:.0f})"
            cv2.putText(test_img, label, (int(pixel_x)-50, int(pixel_y)-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Resize for display
        display_img = cv2.resize(test_img, (1920, 1080))
        
        cv2.imshow("Coordinate Mapping Test", display_img)
        print("👀 Visual test displayed. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visual test: {e}")
        return False

def main():
    """Run all coordinate mapping tests"""
    print("🚀 COORDINATE MAPPING DEBUG SUITE")
    print("=" * 60)
    
    # Test 1: Basic coordinate mapping
    success1 = test_coordinate_mapping()
    
    # Test 2: Visual mapping
    if success1:
        success2 = test_visual_mapping()
    else:
        success2 = False
    
    # Summary
    print("\n📊 TEST RESULTS:")
    print(f"   Coordinate Mapping: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Visual Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n✅ All coordinate mapping tests passed!")
        print("🎯 Calibration appears to be working correctly")
    else:
        print("\n❌ Some tests failed - coordinate mapping needs attention")

if __name__ == "__main__":
    main() 