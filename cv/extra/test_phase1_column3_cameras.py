#!/usr/bin/env python3
"""
Phase 1 Test: Column 3 Cameras (8, 9, 10, 11) Integration Test
Tests all 4 cameras in the left column with proper coordinate mapping
"""

import sys
import time
import logging
from datetime import datetime
from config import Config
from detector_tracker import CoordinateMapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera_configuration():
    """Test that all Column 3 cameras are properly configured"""
    print("🔧 TESTING CAMERA CONFIGURATION")
    print("=" * 50)
    
    # Check active cameras
    expected_cameras = [8, 9, 10, 11]
    actual_cameras = Config.ACTIVE_CAMERAS
    
    print(f"Expected active cameras: {expected_cameras}")
    print(f"Actual active cameras: {actual_cameras}")
    
    if set(expected_cameras) == set(actual_cameras):
        print("✅ Camera configuration correct")
        return True
    else:
        print("❌ Camera configuration mismatch")
        return False

def test_coordinate_mapping():
    """Test coordinate mapping for all Column 3 cameras"""
    print("\n🎯 TESTING COORDINATE MAPPING")
    print("=" * 50)
    
    success_count = 0
    
    for camera_id in [8, 9, 10, 11]:
        print(f"\n📹 Testing Camera {camera_id}:")
        
        # Initialize coordinate mapper
        mapper = CoordinateMapper(camera_id=camera_id)
        
        # Load calibration
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        try:
            mapper.load_calibration(calibration_file)
            
            if mapper.is_calibrated:
                print(f"   ✅ Calibration loaded: {mapper.floor_width_ft}ft x {mapper.floor_length_ft}ft")
                
                # Test coordinate transformation
                zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
                print(f"   📍 Coverage zone: {zone['x_start']}-{zone['x_end']}ft × {zone['y_start']}-{zone['y_end']}ft")
                
                # Test center point transformation
                center_pixel_x, center_pixel_y = 1920, 1080  # Center of 4K image
                global_x, global_y = mapper.pixel_to_real(center_pixel_x, center_pixel_y)
                
                if global_x and global_y:
                    print(f"   🎯 Center pixel ({center_pixel_x}, {center_pixel_y}) → Global ({global_x:.1f}ft, {global_y:.1f}ft)")
                    
                    # Verify coordinates are within expected range
                    if (zone['x_start'] <= global_x <= zone['x_end'] and 
                        zone['y_start'] <= global_y <= zone['y_end']):
                        print(f"   ✅ Coordinates within expected range")
                        success_count += 1
                    else:
                        print(f"   ❌ Coordinates outside expected range")
                else:
                    print(f"   ❌ Coordinate transformation failed")
            else:
                print(f"   ❌ Calibration failed to load")
                
        except Exception as e:
            print(f"   ❌ Error loading calibration: {e}")
    
    print(f"\n📊 Coordinate mapping test results: {success_count}/4 cameras successful")
    return success_count == 4

def test_camera_coverage_zones():
    """Test that camera coverage zones are properly defined"""
    print("\n📐 TESTING CAMERA COVERAGE ZONES")
    print("=" * 50)
    
    expected_zones = {
        8: {"x_start": 120, "x_end": 180, "y_start": 0, "y_end": 25},
        9: {"x_start": 120, "x_end": 180, "y_start": 25, "y_end": 50},
        10: {"x_start": 120, "x_end": 180, "y_start": 50, "y_end": 75},
        11: {"x_start": 120, "x_end": 180, "y_start": 75, "y_end": 90}
    }
    
    success_count = 0
    
    for camera_id, expected_zone in expected_zones.items():
        actual_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
        
        print(f"Camera {camera_id}:")
        print(f"   Expected: {expected_zone['x_start']}-{expected_zone['x_end']}ft × {expected_zone['y_start']}-{expected_zone['y_end']}ft")
        
        if actual_zone:
            print(f"   Actual:   {actual_zone['x_start']}-{actual_zone['x_end']}ft × {actual_zone['y_start']}-{actual_zone['y_end']}ft")
            
            if (actual_zone['x_start'] == expected_zone['x_start'] and
                actual_zone['x_end'] == expected_zone['x_end'] and
                actual_zone['y_start'] == expected_zone['y_start'] and
                actual_zone['y_end'] == expected_zone['y_end']):
                print(f"   ✅ Zone configuration correct")
                success_count += 1
            else:
                print(f"   ❌ Zone configuration mismatch")
        else:
            print(f"   ❌ Zone not found in configuration")
    
    print(f"\n📊 Coverage zone test results: {success_count}/4 cameras successful")
    return success_count == 4

def test_rtsp_urls():
    """Test that RTSP URLs are configured for all cameras"""
    print("\n🌐 TESTING RTSP CONFIGURATION")
    print("=" * 50)
    
    expected_urls = {
        8: "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1",
        9: "rtsp://admin:wearewarp!@192.168.0.80:554/Streaming/channels/1",
        10: "rtsp://admin:wearewarp!@192.168.0.81:554/Streaming/channels/1",
        11: "rtsp://admin:wearewarp!@192.168.0.82:554/Streaming/channels/1"
    }
    
    success_count = 0
    
    for camera_id, expected_url in expected_urls.items():
        actual_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
        
        print(f"{camera_name}:")
        print(f"   URL: {actual_url}")
        
        if actual_url == expected_url:
            print(f"   ✅ RTSP URL correct")
            success_count += 1
        else:
            print(f"   ❌ RTSP URL mismatch")
            print(f"   Expected: {expected_url}")
    
    print(f"\n📊 RTSP configuration test results: {success_count}/4 cameras successful")
    return success_count == 4

def main():
    """Run all Phase 1 integration tests"""
    print("🚀 PHASE 1: COLUMN 3 CAMERAS INTEGRATION TEST")
    print("=" * 60)
    print("Testing cameras 8, 9, 10, 11 (Left column)")
    print("Coverage: 120-180ft × 0-90ft")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Camera Configuration", test_camera_configuration),
        ("Coordinate Mapping", test_coordinate_mapping),
        ("Coverage Zones", test_camera_coverage_zones),
        ("RTSP URLs", test_rtsp_urls)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 PHASE 1 INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - PHASE 1 READY!")
        print("\n🚀 NEXT STEPS:")
        print("1. Start the multi-camera system:")
        print("   python multi_camera_tracking_system.py")
        print("2. Start the backend:")
        print("   python backend/live_server.py")
        print("3. Check frontend for all 4 camera zones active")
        print("4. Verify objects appear in correct coordinates")
    else:
        print("❌ SOME TESTS FAILED - CHECK CONFIGURATION")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Verify all calibration files exist")
        print("2. Check camera network connectivity")
        print("3. Ensure coordinate zones are properly defined")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
