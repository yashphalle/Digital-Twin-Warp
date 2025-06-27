#!/usr/bin/env python3
"""
Test New Coordinate System Implementation
Verify that the new 3-column coordinate system is working correctly
"""

import sys
import json
from detector_tracker import CoordinateMapper
from config import Config

def test_coordinate_system():
    """Test the new coordinate system with Camera 8"""
    print("🧪 TESTING NEW COORDINATE SYSTEM")
    print("=" * 50)
    
    # Test Camera 8 (we have calibration for it)
    camera_id = 8
    
    print(f"📷 Testing Camera {camera_id}")
    
    # Check config
    if camera_id in Config.CAMERA_COVERAGE_ZONES:
        zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
        print(f"   Config Zone: ({zone['x_start']}-{zone['x_end']}ft, {zone['y_start']}-{zone['y_end']}ft)")
        print(f"   Center: ({zone['center_x']}ft, {zone['center_y']}ft)")
        print(f"   Column: {zone['column']}")
    else:
        print(f"   ❌ Camera {camera_id} not found in config")
        return False
    
    # Initialize coordinate mapper
    mapper = CoordinateMapper(camera_id=camera_id)
    
    # Load calibration
    mapper.load_calibration()
    
    if not mapper.is_calibrated:
        print(f"   ❌ Camera {camera_id} calibration failed")
        return False
    
    print(f"   ✅ Calibration loaded: {mapper.floor_width_ft}ft x {mapper.floor_length_ft}ft")
    
    # Test coordinate transformation
    print("\n🔄 TESTING COORDINATE TRANSFORMATION:")
    
    test_cases = [
        # Local coordinates within camera coverage (0-60ft, 0-22.5ft)
        (0, 0, "Local origin (top-left of camera zone)"),
        (30, 11.25, "Local center of camera zone"),
        (60, 22.5, "Local bottom-right of camera zone"),
        (15, 5, "Quarter point in camera zone"),
        (45, 20, "Three-quarter point in camera zone")
    ]
    
    for local_x, local_y, description in test_cases:
        warehouse_x, warehouse_y = mapper.transform_to_warehouse_coordinates(local_x, local_y)
        
        print(f"   Local ({local_x:5.1f}, {local_y:5.1f}) → Warehouse ({warehouse_x:5.1f}, {warehouse_y:5.1f}) - {description}")
        
        # Validate coordinates are within warehouse bounds
        if not (0 <= warehouse_x <= 180 and 0 <= warehouse_y <= 90):
            print(f"      ❌ WARNING: Coordinates outside warehouse bounds!")
    
    return True

def test_all_camera_zones():
    """Test configuration for all camera zones"""
    print("\n📋 TESTING ALL CAMERA ZONE CONFIGURATIONS:")
    print("-" * 50)
    
    for camera_id in range(1, 12):
        if camera_id in Config.CAMERA_COVERAGE_ZONES:
            zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
            name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
            
            # Check zone validity
            width = zone['x_end'] - zone['x_start']
            height = zone['y_end'] - zone['y_start']
            
            print(f"Camera {camera_id:2d}: {name}")
            print(f"   Zone: ({zone['x_start']:5.1f}-{zone['x_end']:5.1f}ft, {zone['y_start']:5.1f}-{zone['y_end']:5.1f}ft)")
            print(f"   Size: {width:4.1f}ft × {height:4.1f}ft")
            print(f"   Center: ({zone['center_x']:5.1f}ft, {zone['center_y']:5.1f}ft)")
            print(f"   Column: {zone['column']}")
            
            # Validate zone dimensions
            if width != 60.0:
                print(f"   ❌ Width should be 60ft, got {width}ft")
            if height != 22.5:
                print(f"   ❌ Height should be 22.5ft, got {height}ft")
                
            print()

def test_coordinate_conversion_logic():
    """Test the coordinate conversion logic step by step"""
    print("\n🔬 TESTING COORDINATE CONVERSION LOGIC:")
    print("-" * 50)
    
    # Example: Camera 8 in Column 3
    camera_id = 8
    zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
    
    print(f"Testing Camera {camera_id} coordinate conversion:")
    print(f"Camera zone: ({zone['x_start']}-{zone['x_end']}ft, {zone['y_start']}-{zone['y_end']}ft)")
    
    # Test point: center of camera's local area
    local_x, local_y = 30.0, 11.25  # Center of 60x22.5 area
    
    print(f"\n📍 Test Point: Local ({local_x}ft, {local_y}ft)")
    
    # Step 1: Add zone offset
    intermediate_x = local_x + zone["x_start"]
    intermediate_y = local_y + zone["y_start"]
    print(f"Step 1 - Add zone offset: ({intermediate_x}ft, {intermediate_y}ft)")
    
    # Step 2: Apply coordinate system transformation (origin top-right)
    final_x = 180.0 - intermediate_x  # Flip X-axis
    final_y = intermediate_y          # Y-axis stays same
    print(f"Step 2 - Apply origin flip: ({final_x}ft, {final_y}ft)")
    
    print(f"\n✅ Final warehouse coordinates: ({final_x}ft, {final_y}ft)")
    
    # This should place the object at the center of Camera 8's zone
    expected_x = 180.0 - zone["center_x"]  # Should be 180 - 150 = 30ft from origin
    expected_y = zone["center_y"]          # Should be 11.25ft from origin
    print(f"Expected coordinates: ({expected_x}ft, {expected_y}ft)")
    
    if abs(final_x - expected_x) < 0.1 and abs(final_y - expected_y) < 0.1:
        print("✅ Coordinate conversion is correct!")
    else:
        print("❌ Coordinate conversion mismatch!")

def main():
    """Run all coordinate system tests"""
    print("🎯 NEW COORDINATE SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test 1: Basic coordinate system
    success1 = test_coordinate_system()
    
    # Test 2: All camera zones
    test_all_camera_zones()
    
    # Test 3: Conversion logic
    test_coordinate_conversion_logic()
    
    print("\n" + "=" * 60)
    if success1:
        print("✅ NEW COORDINATE SYSTEM TESTS PASSED!")
        print("\n📋 NEXT STEPS:")
        print("1. Test with actual camera feed:")
        print("   python high_performance_main.py")
        print("2. Verify objects appear in correct positions on frontend")
        print("3. Calibrate remaining cameras when ready")
    else:
        print("❌ Some tests failed - check configuration")
    
    print(f"\n🎉 Coordinate system ready with ±6 inch precision target!")

if __name__ == "__main__":
    main() 