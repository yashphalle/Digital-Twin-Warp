#!/usr/bin/env python3
"""
🔍 TEST LIVE COORDINATE MAPPING
Test the actual coordinate mapping being used by the running system
"""

import sys
import os
import time

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from gpu_11camera_configurable import GPUCompleteWarehouseTracker

def test_live_coordinate_mapping():
    """Test the actual coordinate mapping used by the system"""
    
    print("🔍 TESTING LIVE COORDINATE MAPPING")
    print("=" * 60)
    
    # Test Camera 1
    camera_id = 1
    print(f"\n📹 TESTING CAMERA {camera_id} (should be 0-62ft range):")
    
    try:
        # Create the same tracker the system uses
        tracker = GPUCompleteWarehouseTracker(camera_id=camera_id, batch_mode=True)
        
        if tracker.coordinate_mapper_initialized:
            print(f"   ✅ Coordinate mapper initialized")
            
            # Test the same transformation the system does
            # Simulate a detection at center of frame
            fake_detection = {
                'bbox': [860, 440, 1060, 640],  # 200x200 box at center
                'center': [960, 540],  # Center of 1920x1080 frame
                'confidence': 0.5,
                'area': 40000
            }
            
            # Use the same method the system uses
            detections = [fake_detection]
            frame_width, frame_height = 1920, 1080
            
            # This is exactly what the system does
            processed_detections = tracker.translate_to_physical_coordinates_batch_gpu(
                detections, frame_width, frame_height
            )
            
            if processed_detections:
                detection = processed_detections[0]
                phys_x = detection.get('physical_x_ft')
                phys_y = detection.get('physical_y_ft')
                status = detection.get('coordinate_status')
                
                print(f"   📍 Center pixel: {fake_detection['center']}")
                print(f"   📍 Physical coords: ({phys_x}, {phys_y}) ft")
                print(f"   📊 Status: {status}")
                
                if phys_x is not None and phys_y is not None:
                    # Check range
                    in_range = (0 <= phys_x <= 62) and (0 <= phys_y <= 25)
                    print(f"   📐 Expected range: 0-62ft, 0-25ft")
                    print(f"   {'✅' if in_range else '❌'} Coordinates {'IN' if in_range else 'OUT OF'} expected range")
                    
                    if not in_range:
                        if 120 <= phys_x <= 180:
                            print(f"   🔄 This looks like Camera 8's coordinates (120-180ft)!")
                            print(f"   🚨 CAMERA 1 IS USING CAMERA 8'S CALIBRATION!")
                        else:
                            print(f"   ⚠️ Coordinates don't match any known camera range")
                else:
                    print(f"   ❌ Coordinate transformation failed")
            else:
                print(f"   ❌ No detections processed")
        else:
            print(f"   ❌ Coordinate mapper not initialized")
            
    except Exception as e:
        print(f"   ❌ Error testing Camera {camera_id}: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION:")
    print("If Camera 1 returns coordinates like 160ft, then:")
    print("1. Camera 1 is using Camera 8's calibration file")
    print("2. OR there's a camera ID mix-up in the system")
    print("3. OR the system needs to be restarted to load new calibrations")

if __name__ == "__main__":
    test_live_coordinate_mapping()
