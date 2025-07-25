#!/usr/bin/env python3
"""
🗺️ COORDINATE TRANSFORMATION VERIFICATION SCRIPT
Verify pixel-to-physical coordinate transformation for individual cameras
"""

import cv2
import numpy as np
import json
import os
import sys

class CoordinateVerifier:
    """Simple coordinate transformation verifier"""
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration for specific camera"""
        filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
        
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            # Extract calibration points
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            # Calculate homography matrix
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            self.is_calibrated = True
            
            print(f"✅ Camera {self.camera_id} calibration loaded")
            print(f"📍 Warehouse area: {calibration_data['warehouse_area']['description']}")
            
            return calibration_data
            
        except Exception as e:
            print(f"❌ Failed to load calibration for Camera {self.camera_id}: {e}")
            return None
    
    def pixel_to_physical(self, pixel_x: float, pixel_y: float):
        """Transform single pixel coordinate to physical coordinate"""
        if not self.is_calibrated:
            return None, None
        
        try:
            # Prepare point for transformation
            points = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            
            # Apply homography transformation
            transformed = cv2.perspectiveTransform(points, self.homography_matrix)
            
            # Extract result
            physical_x = float(transformed[0][0][0])
            physical_y = float(transformed[0][0][1])
            
            return physical_x, physical_y
            
        except Exception as e:
            print(f"❌ Transformation failed: {e}")
            return None, None
    
    def verify_calibration_points(self):
        """Verify that calibration points transform correctly"""
        filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
        
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            image_corners = calibration_data['image_corners']
            real_world_corners = calibration_data['real_world_corners']
            
            print(f"\n🔍 VERIFYING CALIBRATION POINTS FOR CAMERA {self.camera_id}")
            print("=" * 70)
            
            for i, (pixel_point, expected_physical) in enumerate(zip(image_corners, real_world_corners)):
                pixel_x, pixel_y = pixel_point
                expected_x, expected_y = expected_physical
                
                # Transform using our method
                actual_x, actual_y = self.pixel_to_physical(pixel_x, pixel_y)
                
                if actual_x is not None and actual_y is not None:
                    error_x = abs(actual_x - expected_x)
                    error_y = abs(actual_y - expected_y)
                    
                    print(f"Point {i+1}:")
                    print(f"  📍 Pixel: ({pixel_x}, {pixel_y})")
                    print(f"  🎯 Expected: ({expected_x:.2f}, {expected_y:.2f}) ft")
                    print(f"  📊 Actual: ({actual_x:.2f}, {actual_y:.2f}) ft")
                    print(f"  📏 Error: ({error_x:.3f}, {error_y:.3f}) ft")
                    
                    if error_x < 0.1 and error_y < 0.1:
                        print(f"  ✅ ACCURATE")
                    else:
                        print(f"  ⚠️ HIGH ERROR")
                else:
                    print(f"Point {i+1}: ❌ TRANSFORMATION FAILED")
                
                print()
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
    
    def test_custom_points(self):
        """Test transformation of custom pixel points"""
        print(f"\n🧪 CUSTOM POINT TESTING FOR CAMERA {self.camera_id}")
        print("=" * 50)
        
        # Test some common points
        test_points = [
            (960, 540),    # Center of 1920x1080 frame
            (0, 0),        # Top-left corner
            (1920, 1080),  # Bottom-right corner
            (500, 300),    # Random point 1
            (1400, 800),   # Random point 2
        ]
        
        for pixel_x, pixel_y in test_points:
            physical_x, physical_y = self.pixel_to_physical(pixel_x, pixel_y)
            
            if physical_x is not None and physical_y is not None:
                print(f"📍 Pixel ({pixel_x}, {pixel_y}) → Physical ({physical_x:.2f}, {physical_y:.2f}) ft")
            else:
                print(f"❌ Pixel ({pixel_x}, {pixel_y}) → TRANSFORMATION FAILED")

def main():
    """Main verification function"""
    print("🗺️ COORDINATE TRANSFORMATION VERIFICATION")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid camera ID. Please provide a number (1-11)")
            return
    else:
        camera_id = 8  # Default to Camera 8
    
    print(f"🎥 Testing Camera {camera_id}")
    
    # Create verifier
    verifier = CoordinateVerifier(camera_id)
    
    if not verifier.is_calibrated:
        print(f"❌ Cannot verify Camera {camera_id} - calibration failed")
        return
    
    # Run verification tests
    verifier.verify_calibration_points()
    verifier.test_custom_points()
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()
