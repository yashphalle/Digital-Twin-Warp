#!/usr/bin/env python3
"""
🔍 DEBUG COORDINATE MAPPERS
Check if each camera's coordinate mapper is initialized correctly
"""

import sys
import os
import numpy as np

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from gpu_11camera_configurable import GPUCoordinateMapper

def test_coordinate_mappers():
    """Test coordinate mappers for different cameras"""
    
    print("🔍 TESTING COORDINATE MAPPERS")
    print("=" * 60)
    
    test_cameras = [1, 8]  # Camera 1 (0-62ft) vs Camera 8 (120-180ft)
    
    for camera_id in test_cameras:
        print(f"\n📹 TESTING CAMERA {camera_id}:")
        
        # Create coordinate mapper
        mapper = GPUCoordinateMapper(camera_id=camera_id)
        
        # Load calibration
        calibration_file = f"../configs/warehouse_calibration_camera_{camera_id}.json"
        mapper.load_calibration(calibration_file)
        
        if mapper.is_calibrated:
            print(f"   ✅ Calibration loaded successfully")
            
            # Test center point transformation
            center_pixel = [960, 540]  # Center of 1920x1080 frame
            
            # Scale to 4K (like the system does)
            scale_x = 3840 / 1920
            scale_y = 2160 / 1080
            scaled_pixel = [center_pixel[0] * scale_x, center_pixel[1] * scale_y]
            
            print(f"   📍 Original pixel: {center_pixel}")
            print(f"   📍 Scaled pixel: {scaled_pixel}")
            
            # Transform coordinates
            physical_x, physical_y = mapper.pixel_to_real(scaled_pixel[0], scaled_pixel[1])
            
            if physical_x is not None and physical_y is not None:
                print(f"   📍 Physical coords: ({physical_x:.2f}, {physical_y:.2f}) ft")
                
                # Check if coordinates are in expected range
                if camera_id == 1:
                    expected_range = "0-62ft, 0-25ft"
                    in_range = (0 <= physical_x <= 62) and (0 <= physical_y <= 25)
                elif camera_id == 8:
                    expected_range = "120-180ft, 0-25ft"
                    in_range = (120 <= physical_x <= 180) and (0 <= physical_y <= 25)
                
                print(f"   📐 Expected range: {expected_range}")
                print(f"   {'✅' if in_range else '❌'} Coordinates {'IN' if in_range else 'OUT OF'} expected range")
                
                if not in_range:
                    # Check if it matches the other camera's range
                    if camera_id == 1 and (120 <= physical_x <= 180):
                        print(f"   🔄 This looks like Camera 8's coordinates!")
                    elif camera_id == 8 and (0 <= physical_x <= 62):
                        print(f"   🔄 This looks like Camera 1's coordinates!")
            else:
                print(f"   ❌ Coordinate transformation failed")
        else:
            print(f"   ❌ Calibration failed to load")
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("If Camera 1's mapper returns coordinates like 160ft,")
    print("it means Camera 1 is somehow using Camera 8's calibration file.")
    print("This would explain why detections show wrong coordinates.")

def test_batch_coordinate_transformation():
    """Test batch coordinate transformation like the system does"""
    
    print("\n🔍 TESTING BATCH COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # Simulate what the system does
    camera_id = 1
    mapper = GPUCoordinateMapper(camera_id=camera_id)
    calibration_file = f"../configs/warehouse_calibration_camera_{camera_id}.json"
    mapper.load_calibration(calibration_file)
    
    if mapper.is_calibrated:
        # Simulate detection centers (like the system does)
        detection_centers = np.array([
            [960, 540],   # Center of frame
            [500, 300],   # Top-left area
            [1400, 800]   # Bottom-right area
        ], dtype=np.float32)
        
        # Scale to 4K (like the system does)
        frame_width, frame_height = 1920, 1080
        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height
        scaled_centers = detection_centers * np.array([scale_x, scale_y])
        
        print(f"📍 Original centers: {detection_centers}")
        print(f"📍 Scaled centers: {scaled_centers}")
        
        # Batch transformation (like the system does)
        physical_coords = mapper.pixel_to_real_batch_gpu(scaled_centers)
        
        print(f"📍 Physical coordinates:")
        for i, (phys_x, phys_y) in enumerate(physical_coords):
            if not np.isnan([phys_x, phys_y]).any():
                print(f"   Detection {i+1}: ({phys_x:.2f}, {phys_y:.2f}) ft")
                
                # Check range
                in_range = (0 <= phys_x <= 62) and (0 <= phys_y <= 25)
                print(f"   {'✅' if in_range else '❌'} {'IN' if in_range else 'OUT OF'} Camera 1 range (0-62ft, 0-25ft)")
                
                if not in_range and (120 <= phys_x <= 180):
                    print(f"   🔄 This looks like Camera 8's range!")
            else:
                print(f"   Detection {i+1}: FAILED")

if __name__ == "__main__":
    test_coordinate_mappers()
    test_batch_coordinate_transformation()
