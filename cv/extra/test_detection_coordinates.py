#!/usr/bin/env python3
"""
Test that the detection pipeline is using correct coordinate transformation
"""

import cv2
import numpy as np
from detector_tracker import DetectorTracker
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection_coordinate_pipeline():
    """Test that detection pipeline uses correct coordinate transformation"""
    print("🎯 TESTING DETECTION COORDINATE PIPELINE")
    print("=" * 60)
    
    # Test each Column 3 camera
    for camera_id in [8, 9, 10, 11]:
        print(f"\n📹 Testing Camera {camera_id} Detection Pipeline:")
        print("-" * 50)
        
        # Initialize detector tracker
        detector_tracker = DetectorTracker()
        detector_tracker.set_camera_id(camera_id)
        
        # Load calibration
        calibration_file = f"warehouse_calibration_camera_{camera_id}.json"
        detector_tracker.coordinate_mapper.load_calibration(calibration_file)
        
        if not detector_tracker.coordinate_mapper.is_calibrated:
            print(f"❌ Camera {camera_id} calibration failed")
            continue
        
        print(f"✅ Camera {camera_id} calibration loaded")
        
        # Create a test frame (simulate detection at center of image)
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 1080p frame
        
        # Simulate a detection at center of frame
        center_x, center_y = 960, 540  # Center of 1080p frame
        
        # Calculate scaling factor (detection is on 1080p, calibration is on 4K)
        scale_x = 3840 / 1920  # 2.0
        scale_y = 2160 / 1080  # 2.0
        
        # Scale coordinates to calibration frame size
        scaled_center_x = center_x * scale_x  # 1920
        scaled_center_y = center_y * scale_y  # 1080
        
        print(f"   Detection center: ({center_x}, {center_y}) on 1080p frame")
        print(f"   Scaled to 4K: ({scaled_center_x:.0f}, {scaled_center_y:.0f})")
        
        # Get real-world coordinates
        real_x, real_y = detector_tracker.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)
        
        if real_x is not None and real_y is not None:
            print(f"   Global coordinates: ({real_x:.1f}ft, {real_y:.1f}ft)")
            
            # Check if coordinates are in expected range
            zone = Config.CAMERA_COVERAGE_ZONES[camera_id]
            expected_x_range = (zone['x_start'], zone['x_end'])
            expected_y_range = (zone['y_start'], zone['y_end'])
            
            x_in_range = expected_x_range[0] <= real_x <= expected_x_range[1]
            y_in_range = expected_y_range[0] <= real_y <= expected_y_range[1]
            
            print(f"   Expected range: {expected_x_range[0]}-{expected_x_range[1]}ft × {expected_y_range[0]}-{expected_y_range[1]}ft")
            
            if x_in_range and y_in_range:
                print(f"   ✅ Coordinates are in expected range")
            else:
                print(f"   ❌ Coordinates are outside expected range")
                if not x_in_range:
                    print(f"      X coordinate {real_x:.1f}ft outside {expected_x_range[0]}-{expected_x_range[1]}ft")
                if not y_in_range:
                    print(f"      Y coordinate {real_y:.1f}ft outside {expected_y_range[0]}-{expected_y_range[1]}ft")
        else:
            print(f"   ❌ Coordinate transformation failed")

def test_actual_detection_process():
    """Test the actual detection process with coordinate transformation"""
    print("\n🔍 TESTING ACTUAL DETECTION PROCESS")
    print("=" * 60)
    
    # Initialize detector tracker for Camera 8
    detector_tracker = DetectorTracker()
    detector_tracker.set_camera_id(8)
    
    # Load calibration
    calibration_file = "warehouse_calibration_camera_8.json"
    detector_tracker.coordinate_mapper.load_calibration(calibration_file)
    
    if not detector_tracker.coordinate_mapper.is_calibrated:
        print("❌ Camera 8 calibration failed")
        return
    
    print("✅ Camera 8 calibration loaded")
    
    # Create a test frame with a simple object (white rectangle)
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Draw a white rectangle at center (simulate an object)
    cv2.rectangle(test_frame, (900, 500), (1020, 580), (255, 255, 255), -1)
    
    print("🎯 Processing test frame with simulated object...")
    
    # Process the frame (this will use the coordinate transformation)
    try:
        result = detector_tracker.process_frame(test_frame)
        
        if isinstance(result, tuple):
            tracked_objects, statistics = result
        else:
            tracked_objects = result.get('tracked_objects', []) if result else []
            statistics = result.get('statistics', {}) if result else {}
        
        print(f"📊 Detection results:")
        print(f"   Objects detected: {len(tracked_objects)}")
        
        if tracked_objects:
            for i, obj in enumerate(tracked_objects):
                print(f"\n   Object {i+1}:")
                print(f"      Pixel center: {obj.get('center')}")
                print(f"      Real center: {obj.get('real_center')}")
                print(f"      Confidence: {obj.get('confidence', 0):.2f}")
                
                real_center = obj.get('real_center')
                if real_center and len(real_center) >= 2:
                    real_x, real_y = real_center[0], real_center[1]
                    
                    # Check if in expected range for Camera 8
                    if 120 <= real_x <= 180 and 0 <= real_y <= 25:
                        print(f"      ✅ Coordinates in expected range (120-180ft × 0-25ft)")
                    else:
                        print(f"      ❌ Coordinates outside expected range")
                        print(f"         Expected: 120-180ft × 0-25ft")
                        print(f"         Actual: {real_x:.1f}ft × {real_y:.1f}ft")
        else:
            print("   ℹ️  No objects detected (this is normal for a simple test frame)")
            
    except Exception as e:
        print(f"❌ Error processing frame: {e}")

def main():
    """Run detection coordinate tests"""
    print("🚀 DETECTION COORDINATE TRANSFORMATION TEST")
    print("=" * 60)
    print("Testing that detection pipeline uses correct global coordinates")
    print("=" * 60)
    
    # Test coordinate transformation in detection pipeline
    test_detection_coordinate_pipeline()
    
    # Test actual detection process
    test_actual_detection_process()
    
    print("\n" + "=" * 60)
    print("🎯 DETECTION COORDINATE TEST COMPLETE")
    print("=" * 60)
    print("\n💡 NEXT STEPS:")
    print("1. If tests pass: Restart multi-camera system")
    print("2. New detections should use correct global coordinates")
    print("3. Check database for objects with 120-180ft coordinates")

if __name__ == "__main__":
    main()
