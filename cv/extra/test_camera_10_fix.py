#!/usr/bin/env python3
"""
Test Camera 10 X-Axis Fix
Verify that Camera 10 now has correct X-axis direction
"""

from detector_tracker import CoordinateMapper

def test_camera_10_coordinates():
    """Test Camera 10's fixed coordinate transformation"""
    print("🔧 TESTING CAMERA 10 X-AXIS FIX")
    print("=" * 50)
    
    # Create coordinate mapper for Camera 10
    mapper = CoordinateMapper(camera_id=10)
    mapper.load_calibration("warehouse_calibration_camera_10.json")
    
    print("📄 Camera 10 Fixed Calibration Test:")
    print("Coverage area: 120-180ft X, 50-75ft Y")
    print()
    
    # Test key points to verify X-axis direction
    test_points = [
        {"name": "Left side of image (should be ~180ft)", "pixel": [0, 1080]},
        {"name": "Right side of image (should be ~120ft)", "pixel": [3840, 1080]},
        {"name": "Center of image (should be ~150ft)", "pixel": [1920, 1080]},
        {"name": "Quarter left (should be ~165ft)", "pixel": [960, 1080]},
        {"name": "Quarter right (should be ~135ft)", "pixel": [2880, 1080]},
    ]
    
    print("🎯 COORDINATE TRANSFORMATION TESTS:")
    print("-" * 50)
    
    for test in test_points:
        pixel_x, pixel_y = test["pixel"]
        real_x, real_y = mapper.pixel_to_real(pixel_x, pixel_y)
        print(f"{test['name']}: ({pixel_x}, {pixel_y}) → ({real_x:.1f}ft, {real_y:.1f}ft)")
    
    print()
    print("✅ EXPECTED BEHAVIOR:")
    print("• Left side of image → Higher X coordinates (closer to 180ft)")
    print("• Right side of image → Lower X coordinates (closer to 120ft)")
    print("• This should now match the other cameras (8, 9, 11)")
    
    # Test edge coordinates
    left_x, _ = mapper.pixel_to_real(0, 1080)
    right_x, _ = mapper.pixel_to_real(3840, 1080)
    
    print(f"\n📊 VERIFICATION:")
    print(f"Left edge: {left_x:.1f}ft")
    print(f"Right edge: {right_x:.1f}ft")
    
    if left_x > right_x:
        print("✅ CORRECT: Left side has higher X coordinates")
        print("✅ Camera 10 X-axis is now fixed!")
    else:
        print("❌ WRONG: Left side still has lower X coordinates")
        print("❌ X-axis still needs fixing")

def compare_all_cameras():
    """Compare coordinate systems of all active cameras"""
    print("\n" + "=" * 50)
    print("📹 COMPARING ALL CAMERA COORDINATE SYSTEMS")
    print("=" * 50)
    
    cameras = [8, 9, 10, 11]
    
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
    """Main test function"""
    print("🚀 CAMERA 10 X-AXIS FIX VERIFICATION")
    print("=" * 50)
    print("Testing the fix for Camera 10's swapped X-axis")
    print("=" * 50)
    
    test_camera_10_coordinates()
    compare_all_cameras()
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)
    print("After the fix, Camera 10 should now:")
    print("• Show objects on YOUR LEFT at higher X coordinates (closer to 180ft)")
    print("• Match the coordinate system of cameras 8, 9, and 11")
    print("• Display correctly in the unified grid display")

if __name__ == "__main__":
    main()
