#!/usr/bin/env python3
"""
Comprehensive Coordinate System Test
Tests the entire coordinate system from CV to Frontend to Database
"""

import sys
import logging
import numpy as np
from detector_tracker import CoordinateMapper
from config import Config
from warehouse_config import get_warehouse_config, get_camera_zone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_warehouse_config_system():
    """Test the new warehouse configuration system"""
    print("🏭 TESTING WAREHOUSE CONFIGURATION SYSTEM")
    print("=" * 60)
    
    try:
        # Test warehouse config loading
        warehouse_config = get_warehouse_config()
        print(f"✅ Warehouse config loaded: {warehouse_config.name}")
        print(f"   Dimensions: {warehouse_config.width_ft}ft × {warehouse_config.length_ft}ft")
        print(f"   Origin: {warehouse_config.origin_position}")
        print(f"   X-axis: {warehouse_config.x_axis_direction}")
        print(f"   Y-axis: {warehouse_config.y_axis_direction}")
        print(f"   Active cameras: {warehouse_config.active_cameras}")
        
        # Test camera zones
        print(f"\n📹 Camera Zones:")
        for camera_id in warehouse_config.active_cameras:
            zone = get_camera_zone(camera_id)
            print(f"   Camera {camera_id}: {zone.x_start}-{zone.x_end}ft × {zone.y_start}-{zone.y_end}ft")
            print(f"      Column {zone.column}, Row {zone.row}: {zone.camera_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Warehouse config system failed: {e}")
        return False

def test_coordinate_system_consistency():
    """Test consistency between different coordinate systems"""
    print("\n🎯 TESTING COORDINATE SYSTEM CONSISTENCY")
    print("=" * 60)
    
    # Test each active camera
    for camera_id in [8, 9, 10, 11]:
        print(f"\n📹 Camera {camera_id} Coordinate System Test:")
        print("-" * 50)
        
        # Get expected coordinates from Config
        config_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
        print(f"Config zone: {config_zone.get('x_start')}-{config_zone.get('x_end')}ft × {config_zone.get('y_start')}-{config_zone.get('y_end')}ft")
        
        # Get coordinates from warehouse config
        try:
            warehouse_zone = get_camera_zone(camera_id)
            print(f"Warehouse zone: {warehouse_zone.x_start}-{warehouse_zone.x_end}ft × {warehouse_zone.y_start}-{warehouse_zone.y_end}ft")
            
            # Check consistency
            config_matches = (
                config_zone.get('x_start') == warehouse_zone.x_start and
                config_zone.get('x_end') == warehouse_zone.x_end and
                config_zone.get('y_start') == warehouse_zone.y_start and
                config_zone.get('y_end') == warehouse_zone.y_end
            )
            
            if config_matches:
                print(f"✅ Config and warehouse zones match")
            else:
                print(f"❌ Config and warehouse zones don't match")
                
        except Exception as e:
            print(f"❌ Warehouse zone error: {e}")
        
        # Test coordinate transformation
        mapper = CoordinateMapper(camera_id=camera_id)
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        mapper.load_calibration(calibration_file)
        
        if mapper.is_calibrated:
            # Test center point
            center_x, center_y = 1920, 1080  # Center of 4K image
            global_x, global_y = mapper.pixel_to_real(center_x, center_y)
            
            if global_x and global_y:
                print(f"Center transformation: ({center_x}, {center_y}) → ({global_x:.1f}ft, {global_y:.1f}ft)")
                
                # Check if in expected range
                expected_x_range = (config_zone.get('x_start', 0), config_zone.get('x_end', 0))
                expected_y_range = (config_zone.get('y_start', 0), config_zone.get('y_end', 0))
                
                x_in_range = expected_x_range[0] <= global_x <= expected_x_range[1]
                y_in_range = expected_y_range[0] <= global_y <= expected_y_range[1]
                
                if x_in_range and y_in_range:
                    print(f"✅ Coordinates in expected range")
                else:
                    print(f"❌ Coordinates outside expected range")
                    print(f"   Expected: {expected_x_range[0]}-{expected_x_range[1]}ft × {expected_y_range[0]}-{expected_y_range[1]}ft")
            else:
                print(f"❌ Coordinate transformation failed")
        else:
            print(f"❌ Calibration failed")

def test_physical_layout_understanding():
    """Test understanding of physical camera layout"""
    print("\n🏗️ TESTING PHYSICAL LAYOUT UNDERSTANDING")
    print("=" * 60)
    
    print("Warehouse Layout (Top-Down View):")
    print("Origin (0,0) at TOP-RIGHT corner")
    print("X-axis: RIGHT → LEFT (0 to 180ft)")
    print("Y-axis: TOP → BOTTOM (0 to 90ft)")
    print()
    
    print("Physical Camera Layout:")
    print("   Column 3     |   Column 2     |   Column 1")
    print("   (LEFT)       |   (MIDDLE)     |   (RIGHT)")
    print("  120-180ft     |   60-120ft     |   0-60ft")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   Cam 8 🟢    |    Cam 5      |    Cam 1      ← Top Row")
    print("   Cam 9 🟢    |    Cam 6      |    Cam 2      ← Mid-Top Row")
    print("   Cam 10 🟢   |    Cam 7      |    Cam 3      ← Mid-Bottom Row")
    print("   Cam 11 🟢   |   Office      |    Cam 4      ← Bottom Row")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    print("Expected Object Coordinates:")
    print("Camera 8 objects should appear at: 120-180ft × 0-25ft")
    print("Camera 9 objects should appear at: 120-180ft × 25-50ft")
    print("Camera 10 objects should appear at: 120-180ft × 50-75ft")
    print("Camera 11 objects should appear at: 120-180ft × 75-90ft")
    print()
    
    print("Frontend Display (after mirror fix):")
    print("Camera 8 objects should appear on LEFT side of frontend")
    print("This is correct because Camera 8 covers 120-180ft X-axis")
    print("which is the leftmost area when origin is at top-right")

def test_calibration_files_consistency():
    """Test that calibration files have correct global coordinates"""
    print("\n📄 TESTING CALIBRATION FILES CONSISTENCY")
    print("=" * 60)
    
    import json
    
    for camera_id in [8, 9, 10, 11]:
        print(f"\n📹 Camera {camera_id} Calibration File:")
        
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            
            real_world_corners = calibration_data.get('real_world_corners', [])
            
            if real_world_corners and len(real_world_corners) == 4:
                print(f"   Real world corners:")
                for i, corner in enumerate(real_world_corners):
                    corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
                    print(f"      {corner_names[i]}: ({corner[0]}ft, {corner[1]}ft)")
                
                # Calculate coverage area from corners
                min_x = min(corner[0] for corner in real_world_corners)
                max_x = max(corner[0] for corner in real_world_corners)
                min_y = min(corner[1] for corner in real_world_corners)
                max_y = max(corner[1] for corner in real_world_corners)
                
                print(f"   Coverage area: {min_x}-{max_x}ft × {min_y}-{max_y}ft")
                
                # Compare with expected zone
                expected_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
                expected_x_start = expected_zone.get('x_start', 0)
                expected_x_end = expected_zone.get('x_end', 0)
                expected_y_start = expected_zone.get('y_start', 0)
                expected_y_end = expected_zone.get('y_end', 0)
                
                print(f"   Expected zone: {expected_x_start}-{expected_x_end}ft × {expected_y_start}-{expected_y_end}ft")
                
                if (min_x == expected_x_start and max_x == expected_x_end and 
                    min_y == expected_y_start and max_y == expected_y_end):
                    print(f"   ✅ Calibration matches expected zone")
                else:
                    print(f"   ❌ Calibration doesn't match expected zone")
                    print(f"      Difference: X({min_x}-{max_x} vs {expected_x_start}-{expected_x_end}), Y({min_y}-{max_y} vs {expected_y_start}-{expected_y_end})")
            else:
                print(f"   ❌ Invalid real_world_corners in calibration file")
                
        except Exception as e:
            print(f"   ❌ Error reading calibration file: {e}")

def main():
    """Run comprehensive coordinate system tests"""
    print("🚀 COMPREHENSIVE COORDINATE SYSTEM TEST")
    print("=" * 60)
    print("Testing entire coordinate system: CV → Database → Frontend")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Warehouse Config System", test_warehouse_config_system),
        ("Coordinate System Consistency", test_coordinate_system_consistency),
        ("Physical Layout Understanding", test_physical_layout_understanding),
        ("Calibration Files Consistency", test_calibration_files_consistency)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ ALL COORDINATE SYSTEM TESTS PASSED!")
        print("\n🎯 COORDINATE SYSTEM STATUS:")
        print("✅ Camera 8: 120-180ft × 0-25ft (LEFT side of frontend)")
        print("✅ Camera 9: 120-180ft × 25-50ft (LEFT side of frontend)")
        print("✅ Camera 10: 120-180ft × 50-75ft (LEFT side of frontend)")
        print("✅ Camera 11: 120-180ft × 75-90ft (LEFT side of frontend)")
        print("\n🚀 READY FOR PRODUCTION!")
    else:
        print("❌ SOME COORDINATE SYSTEM TESTS FAILED")
        print("\n🔧 ISSUES TO FIX:")
        print("1. Check calibration file coordinates")
        print("2. Verify warehouse configuration")
        print("3. Ensure coordinate transformation is working")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
