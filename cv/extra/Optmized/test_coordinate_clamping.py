#!/usr/bin/env python3
"""
🔍 TEST COORDINATE CLAMPING FIX
Test that negative coordinates are now clamped to valid ranges
"""

import sys
import os
import time

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from gpu_11camera_configurable import GPUCompleteWarehouseTracker

def test_coordinate_clamping():
    """Test the coordinate clamping fix"""
    
    print("🔍 TESTING COORDINATE CLAMPING FIX")
    print("=" * 60)
    
    # Test Camera 1
    camera_id = 1
    print(f"\n📹 TESTING CAMERA {camera_id} COORDINATE CLAMPING:")
    
    try:
        # Create the tracker
        tracker = GPUCompleteWarehouseTracker(camera_id=camera_id, batch_mode=True)
        
        if tracker.coordinate_mapper_initialized:
            print(f"   ✅ Coordinate mapper initialized")
            
            # Test problematic points that were producing negative coordinates
            test_detections = [
                {
                    'bbox': [1649, 251, 1849, 451],  # Point that gave Y: -0.21
                    'center': [1649, 251],
                    'confidence': 0.5,
                    'area': 40000
                },
                {
                    'bbox': [0, 0, 200, 200],  # Top-left corner (was negative Y)
                    'center': [0, 0],
                    'confidence': 0.5,
                    'area': 40000
                },
                {
                    'bbox': [1920, 0, 2120, 200],  # Top-right corner (was negative Y)
                    'center': [1920, 0],
                    'confidence': 0.5,
                    'area': 40000
                },
                {
                    'bbox': [960, 540, 1160, 740],  # Center (should be fine)
                    'center': [960, 540],
                    'confidence': 0.5,
                    'area': 40000
                }
            ]
            
            # Process detections
            frame_width, frame_height = 1920, 1080
            processed_detections = tracker.translate_to_physical_coordinates_batch_gpu(
                test_detections, frame_width, frame_height
            )
            
            print(f"\n📊 COORDINATE CLAMPING RESULTS:")
            print("-" * 50)
            
            all_positive = True
            all_in_range = True
            
            for i, detection in enumerate(processed_detections):
                pixel_center = detection.get('center')
                phys_x = detection.get('physical_x_ft')
                phys_y = detection.get('physical_y_ft')
                status = detection.get('coordinate_status')
                
                print(f"\nTest {i+1}: Pixel {pixel_center}")
                print(f"   Physical: ({phys_x}, {phys_y}) ft")
                print(f"   Status: {status}")
                
                # Check for negative coordinates
                if phys_x is not None and phys_x < 0:
                    print(f"   ❌ X coordinate is still NEGATIVE: {phys_x}")
                    all_positive = False
                elif phys_y is not None and phys_y < 0:
                    print(f"   ❌ Y coordinate is still NEGATIVE: {phys_y}")
                    all_positive = False
                else:
                    print(f"   ✅ Coordinates are positive")
                
                # Check range for Camera 1 (0-62ft, 0-25ft)
                if phys_x is not None and phys_y is not None:
                    in_range = (0 <= phys_x <= 62) and (0 <= phys_y <= 25)
                    if in_range:
                        print(f"   ✅ Coordinates in valid range")
                    else:
                        print(f"   ❌ Coordinates out of range (0-62ft, 0-25ft)")
                        all_in_range = False
                        
                    # Check if clamping was applied
                    if status == 'SUCCESS_CLAMPED':
                        print(f"   🔧 Coordinates were clamped to valid range")
                else:
                    print(f"   ❌ Coordinate transformation failed")
                    all_positive = False
                    all_in_range = False
            
            print(f"\n🎯 SUMMARY:")
            print(f"   All coordinates positive: {'✅ YES' if all_positive else '❌ NO'}")
            print(f"   All coordinates in range: {'✅ YES' if all_in_range else '❌ NO'}")
            
            if all_positive and all_in_range:
                print(f"\n🎉 COORDINATE CLAMPING FIX SUCCESSFUL!")
                print(f"   ✅ No more negative coordinates")
                print(f"   ✅ All coordinates within valid warehouse bounds")
                return True
            else:
                print(f"\n⚠️ COORDINATE CLAMPING NEEDS MORE WORK")
                return False
                
        else:
            print(f"   ❌ Coordinate mapper not initialized")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing Camera {camera_id}: {e}")
        return False

def main():
    """Main test function"""
    
    print("🔧 COORDINATE CLAMPING TEST")
    print("=" * 80)
    
    success = test_coordinate_clamping()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 COORDINATE CLAMPING FIX VERIFIED!")
        print("✅ The CV system will now produce only positive coordinates")
        print("✅ All coordinates will be within valid warehouse bounds")
        print("🚀 Restart the CV system to apply the fix")
    else:
        print("⚠️ COORDINATE CLAMPING FIX NEEDS ADJUSTMENT")
        print("❌ Check the test results above for issues")
    print("=" * 80)

if __name__ == "__main__":
    main()
