#!/usr/bin/env python3
"""
🎥 SINGLE CAMERA TESTING SCRIPT
Test individual cameras with coordinate verification and live detection display
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import logging
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleCameraTester:
    """Test individual camera with coordinate verification"""
    
    def __init__(self, camera_id: int = 8):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()
        
        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        
        # Camera connection
        self.cap = None
        self.connected = False
        
        # Coordinate mapper
        self.coordinate_mapper = None
        self.setup_coordinate_mapper()
        
        print(f"🎥 Single Camera Tester initialized for {self.camera_name}")
        print(f"📡 RTSP URL: {self.rtsp_url}")
        
    def setup_coordinate_mapper(self):
        """Setup coordinate transformation"""
        try:
            # Load calibration file
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            
            with open(calibration_file, 'r') as file:
                calibration_data = json.load(file)
            
            # Extract calibration points
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            # Calculate homography matrix
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            
            print(f"✅ Coordinate mapper loaded for Camera {self.camera_id}")
            print(f"📍 Warehouse area: {calibration_data['warehouse_area']['description']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load coordinate mapper: {e}")
            return False
    
    def pixel_to_physical(self, pixel_x: float, pixel_y: float):
        """Transform pixel coordinate to physical coordinate"""
        if self.homography_matrix is None:
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
    
    def connect_camera(self):
        """Connect to camera"""
        try:
            print(f"🔌 Connecting to {self.camera_name}...")
            
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to connect to camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                print(f"❌ Failed to capture frame from camera")
                return False
            
            self.connected = True
            print(f"✅ {self.camera_name} connected successfully")
            print(f"📐 Frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Camera connection failed: {e}")
            return False
    
    def test_coordinates_on_click(self):
        """Interactive coordinate testing - click on frame to see coordinates"""
        if not self.connected:
            print("❌ Camera not connected")
            return
        
        print("\n🖱️ INTERACTIVE COORDINATE TESTING")
        print("=" * 50)
        print("📌 Click anywhere on the video frame to see coordinates")
        print("🔄 Press 'r' to reset, 'q' to quit")
        print("=" * 50)
        
        # Mouse callback function
        clicked_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Transform coordinates
                phys_x, phys_y = self.pixel_to_physical(x, y)
                
                if phys_x is not None and phys_y is not None:
                    clicked_points.append((x, y, phys_x, phys_y))
                    print(f"📍 Pixel ({x}, {y}) → Physical ({phys_x:.2f}, {phys_y:.2f}) ft")
                else:
                    print(f"❌ Failed to transform pixel ({x}, {y})")
        
        # Setup window and mouse callback
        window_name = f"Camera {self.camera_id} - Click for Coordinates"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Draw clicked points
            display_frame = frame.copy()
            for i, (px, py, phx, phy) in enumerate(clicked_points):
                # Draw circle at clicked point
                cv2.circle(display_frame, (px, py), 8, (0, 255, 0), -1)
                
                # Draw coordinate text
                coord_text = f"({phx:.1f}, {phy:.1f})ft"
                cv2.putText(display_frame, coord_text, (px + 15, py - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw point number
                cv2.putText(display_frame, str(i + 1), (px - 5, py + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Click for coordinates | 'r' reset | 'q' quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                clicked_points.clear()
                print("🔄 Points cleared")
        
        cv2.destroyAllWindows()
    
    def run_live_test(self):
        """Run live camera test with coordinate display"""
        if not self.connect_camera():
            return
        
        print(f"\n🎥 LIVE CAMERA TEST - {self.camera_name}")
        print("=" * 50)
        
        self.test_coordinates_on_click()
        
        print("✅ Live test completed")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("🎥 SINGLE CAMERA COORDINATE TESTER")
    print("=" * 50)
    
    # Get camera ID from command line or use default
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
            if not (1 <= camera_id <= 11):
                print("❌ Camera ID must be between 1 and 11")
                return
        except ValueError:
            print("❌ Invalid camera ID. Please provide a number (1-11)")
            return
    else:
        camera_id = 8  # Default to Camera 8
    
    print(f"🎯 Testing Camera {camera_id}")
    
    # Create tester
    tester = SingleCameraTester(camera_id)
    
    try:
        # Run live test
        tester.run_live_test()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    finally:
        tester.cleanup()
        print("🧹 Cleanup complete")

if __name__ == "__main__":
    main()
