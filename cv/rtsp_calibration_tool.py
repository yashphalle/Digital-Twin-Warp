"""
RTSP Camera Calibration Tool for Warehouse Coordinate Mapping
Enhanced version that works with RTSP cameras and supports real ground measurements
"""

import cv2
import numpy as np
import json
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import logging
from lense_correct2 import OptimizedFisheyeCorrector

logger = logging.getLogger(__name__)

class RTSPWarehouseCalibrator:
    """Enhanced calibration tool for RTSP cameras with warehouse measurements"""
    
    def __init__(self, rtsp_url: str = None):
        self.rtsp_url = rtsp_url or "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"
        
        # Initialize fisheye corrector FIRST
        self.fisheye_corrector = OptimizedFisheyeCorrector(lens_mm=2.8)
        
        # Calibration state
        self.calibration_frame = None
        self.corners = []
        self.corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.current_corner = 0
        
        # Warehouse dimensions (you can set these directly)
        self.warehouse_width_m = 0.0   # meters
        self.warehouse_length_m = 0.0  # meters
        self.warehouse_width_ft = 0.0  # feet
        self.warehouse_length_ft = 0.0 # feet
        
        # Display properties
        self.point_radius = 8
        self.line_thickness = 2
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        # Calibration complete flag
        self.calibration_complete = False
        
        # Camera properties
        self.cap = None
        self.frame_width = 1920
        self.frame_height = 1080
    
    def set_warehouse_dimensions(self, width_ft: float, length_ft: float):
        """Set warehouse dimensions directly in feet"""
        self.warehouse_width_ft = width_ft
        self.warehouse_length_ft = length_ft
        self.warehouse_width_m = width_ft * 0.3048  # Convert to meters
        self.warehouse_length_m = length_ft * 0.3048
        
        print(f"✅ Warehouse dimensions set:")
        print(f"   Width: {width_ft:.1f} ft ({self.warehouse_width_m:.2f} m)")
        print(f"   Length: {length_ft:.1f} ft ({self.warehouse_length_m:.2f} m)")
    
    def connect_to_camera(self):
        """Connect to RTSP camera"""
        try:
            print(f"🔗 Connecting to RTSP camera: {self.rtsp_url}")
            
            # Create video capture with optimized settings
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set buffer size for real-time performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open RTSP stream")
            
            # Get camera properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Connected to camera: {self.frame_width}x{self.frame_height} @ {fps:.1f}fps")
            return True
            
        except Exception as e:
            print(f"❌ Camera connection failed: {e}")
            return False
    
    def capture_calibration_frame(self):
        """Capture a frame for calibration"""
        if not self.cap or not self.cap.isOpened():
            if not self.connect_to_camera():
                return False
        
        print("\n📷 CAPTURE CALIBRATION FRAME")
        print("=" * 40)
        print("• Position camera to show the warehouse floor")
        print("• Ensure good lighting and clear floor markings")
        print("• Press SPACE to capture calibration frame")
        print("• Press 'q' to quit")
        print("=" * 40)
        
        cv2.namedWindow("RTSP Feed - Press SPACE to capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RTSP Feed - Press SPACE to capture", 1200, 600)
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to read frame, attempting to reconnect...")
                if not self.connect_to_camera():
                    break
                continue
            
            frame_count += 1
            
            # Apply fisheye correction FIRST
            corrected_frame = self.fisheye_corrector.correct(frame)
            
            # Resize for display
            display_frame = cv2.resize(corrected_frame, (1200, 600))
            
            # Add instructions overlay
            cv2.putText(display_frame, "FISHEYE CORRECTED - Position to show warehouse floor", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture calibration frame", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count} (Fisheye Corrected)", 
                       (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("RTSP Feed - Press SPACE to capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Store the FISHEYE CORRECTED frame for calibration
                self.calibration_frame = corrected_frame.copy()
                print("✅ Fisheye-corrected calibration frame captured!")
                break
            elif key == ord('q'):
                print("❌ Calibration cancelled")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyWindow("RTSP Feed - Press SPACE to capture")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_corner < 4:
            # Add corner point
            self.corners.append((x, y))
            print(f"✅ {self.corner_names[self.current_corner]} selected: ({x}, {y})")
            
            self.current_corner += 1
            
            if self.current_corner >= 4:
                print("🎯 All corners selected! Press 's' to save calibration or 'r' to reset")
    
    def draw_calibration_overlay(self, frame):
        """Draw calibration overlay on frame"""
        overlay = frame.copy()
        
        # Draw selected corners
        for i, (x, y) in enumerate(self.corners):
            color = self.colors[i]
            cv2.circle(overlay, (x, y), self.point_radius, color, -1)
            cv2.circle(overlay, (x, y), self.point_radius + 2, (255, 255, 255), 2)
            
            # Add corner label
            label = self.corner_names[i]
            cv2.putText(overlay, label, (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw lines between corners
        if len(self.corners) >= 2:
            for i in range(len(self.corners)):
                if i < len(self.corners) - 1 or len(self.corners) == 4:
                    start_point = self.corners[i]
                    end_point = self.corners[(i + 1) % len(self.corners)]
                    cv2.line(overlay, start_point, end_point, (255, 255, 255), self.line_thickness)
        
        # Add instructions
        if self.current_corner < 4:
            instruction = f"Click {self.corner_names[self.current_corner]} corner ({self.current_corner + 1}/4)"
            cv2.putText(overlay, instruction, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(overlay, "Press 's' to save, 'r' to reset, 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add dimension info if available
        if self.warehouse_width_ft > 0 and self.warehouse_length_ft > 0:
            dim_text = f"Warehouse: {self.warehouse_width_ft:.1f}ft x {self.warehouse_length_ft:.1f}ft"
            cv2.putText(overlay, dim_text, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add fisheye correction indicator
        cv2.putText(overlay, "FISHEYE CORRECTED IMAGE", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return overlay
    
    def run_corner_selection(self):
        """Run interactive corner selection"""
        if self.calibration_frame is None:
            print("❌ No calibration frame available")
            return False
        
        print("\n🎯 CORNER SELECTION ON FISHEYE-CORRECTED IMAGE")
        print("=" * 50)
        print("Instructions:")
        print("1. Click corners in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("2. Click on floor reference points (edges, lines, or corners)")
        print("3. The image is ALREADY fisheye-corrected - straight lines should appear straight")
        print("4. Press 's' to save calibration")
        print("5. Press 'r' to reset corners")
        print("6. Press 'q' to quit")
        print("=" * 50)
        
        # Create display window
        cv2.namedWindow("Warehouse Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Warehouse Calibration", 1200, 800)
        cv2.setMouseCallback("Warehouse Calibration", self.mouse_callback)
        
        # Reset corner selection
        self.corners = []
        self.current_corner = 0
        
        while True:
            # Create display frame
            display_frame = self.draw_calibration_overlay(self.calibration_frame)
            cv2.imshow("Warehouse Calibration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset corners
                self.corners = []
                self.current_corner = 0
                print("🔄 Corners reset")
            elif key == ord('s') and len(self.corners) == 4:
                if self.save_calibration():
                    print("✅ Calibration saved successfully!")
                    break
                else:
                    print("❌ Failed to save calibration")
        
        cv2.destroyAllWindows()
        return self.calibration_complete
    
    def save_calibration(self, filename="warehouse_calibration.json"):
        """Save calibration data"""
        if len(self.corners) != 4:
            print("❌ Need exactly 4 corners to save calibration")
            return False
        
        if self.warehouse_width_m <= 0 or self.warehouse_length_m <= 0:
            print("❌ Invalid warehouse dimensions - set dimensions first")
            return False
        
        # Create calibration data
        calibration_data = {
            "description": "RTSP Camera Warehouse Calibration (on fisheye-corrected image)",
            "created_date": datetime.now().isoformat(),
            "camera_info": {
                "rtsp_url": self.rtsp_url,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "fisheye_corrected": True,
                "lens_mm": 2.8
            },
            "warehouse_dimensions": {
                "width_meters": self.warehouse_width_m,
                "length_meters": self.warehouse_length_m,
                "width_feet": self.warehouse_width_ft,
                "length_feet": self.warehouse_length_ft
            },
            "image_corners": self.corners,
            "real_world_corners": [
                [0, 0],                                      # Top-left
                [self.warehouse_width_m, 0],                 # Top-right  
                [self.warehouse_width_m, self.warehouse_length_m], # Bottom-right
                [0, self.warehouse_length_m]                 # Bottom-left
            ],
            "calibration_info": {
                "corner_order": "top-left, top-right, bottom-right, bottom-left",
                "coordinate_system": "origin at top-left, x-axis right, y-axis down",
                "units": "meters"
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"✅ Calibration saved to: {filename}")
            print(f"📐 Warehouse: {self.warehouse_width_ft:.1f}ft x {self.warehouse_length_ft:.1f}ft")
            print(f"📍 Image corners: {self.corners}")
            print(f"📍 Real corners: {calibration_data['real_world_corners']}")
            
            self.calibration_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def run_full_calibration(self, width_ft: float = None, length_ft: float = None):
        """Run complete calibration process"""
        print("\n🎯 RTSP WAREHOUSE CALIBRATION")
        print("=" * 50)
        
        # Set dimensions if provided
        if width_ft and length_ft:
            self.set_warehouse_dimensions(width_ft, length_ft)
        else:
            # Ask for dimensions
            print("Please provide warehouse dimensions...")
            try:
                width_ft = float(input("Enter warehouse width in feet: "))
                length_ft = float(input("Enter warehouse length in feet: "))
                self.set_warehouse_dimensions(width_ft, length_ft)
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid dimensions or cancelled")
                return False
        
        # Step 1: Connect to camera
        if not self.connect_to_camera():
            return False
        
        # Step 2: Capture calibration frame
        if not self.capture_calibration_frame():
            return False
        
        # Step 3: Select corners
        if not self.run_corner_selection():
            return False
        
        print("✅ RTSP Warehouse Calibration Complete!")
        print(f"📁 Calibration file: warehouse_calibration.json")
        print(f"🚀 Ready for warehouse tracking system")
        
        # Clean up
        if self.cap:
            self.cap.release()
        
        return True

def test_coordinate_mapping():
    """Test the coordinate mapping with saved calibration"""
    try:
        with open("warehouse_calibration.json", 'r') as f:
            calibration = json.load(f)
        
        print("\n🧪 TESTING COORDINATE MAPPING")
        print("=" * 40)
        
        # Get transformation matrix
        image_corners = np.array(calibration['image_corners'], dtype=np.float32)
        real_corners = np.array(calibration['real_world_corners'], dtype=np.float32)
        
        # Calculate perspective transformation
        transform_matrix = cv2.getPerspectiveTransform(image_corners, real_corners)
        
        print(f"Image corners: {image_corners}")
        print(f"Real corners: {real_corners}")
        print(f"Transform matrix shape: {transform_matrix.shape}")
        
        # Test some points
        test_points = [
            [960, 540],   # Center of 1920x1080
            [0, 0],       # Top-left
            [1920, 1080]  # Bottom-right
        ]
        
        for point in test_points:
            # Transform point
            point_array = np.array([[point]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(point_array, transform_matrix)[0][0]
            
            print(f"Image point {point} → Real world [{real_point[0]:.2f}m, {real_point[1]:.2f}m]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing coordinate mapping: {e}")
        return False

def main():
    """Main calibration function"""
    # RTSP camera URL (Camera 8)
    rtsp_url = "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"
    
    # Create calibrator
    calibrator = RTSPWarehouseCalibrator(rtsp_url)
    
    # Example usage - replace with your actual warehouse dimensions
    # warehouse_width_ft = 34.5   # Your warehouse width in feet
    # warehouse_length_ft = 26.2  # Your warehouse length in feet
    
    print("🎯 RTSP Camera Calibration Tool")
    print("Please provide your warehouse dimensions:")
    
    try:
        width_ft = float(input("Warehouse width (feet): "))
        length_ft = float(input("Warehouse length (feet): "))
        
        # Run calibration
        success = calibrator.run_full_calibration(width_ft, length_ft)
        
        if success:
            print("\n🧪 Testing coordinate mapping...")
            test_coordinate_mapping()
        
    except (ValueError, KeyboardInterrupt):
        print("❌ Calibration cancelled")

if __name__ == "__main__":
    main() 