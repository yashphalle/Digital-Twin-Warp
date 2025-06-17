"""
Interactive Calibration Tool for Warehouse Coordinate Mapping
Allows user to select 4 corner points and enter physical dimensions
"""

import cv2
import numpy as np
import json
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading

class WarehouseCalibrationTool:
    """Interactive tool for warehouse floor calibration"""
    
    def __init__(self):
        self.image = None
        self.display_image = None
        self.corners = []
        self.corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.current_corner = 0
        self.warehouse_width = 0.0
        self.warehouse_length = 0.0
        self.calibration_complete = False
        
        # Display settings
        self.point_radius = 8
        self.line_thickness = 2
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Blue, Red, Yellow
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_corner < 4:
            # Add corner point
            self.corners.append((x, y))
            print(f"✅ {self.corner_names[self.current_corner]} selected: ({x}, {y})")
            
            # Update display
            self.update_display()
            
            self.current_corner += 1
            
            if self.current_corner >= 4:
                print("🎯 All corners selected! Press 's' to save calibration or 'r' to reset")
    
    def update_display(self):
        """Update the display with current corners"""
        self.display_image = self.image.copy()
        
        # Draw selected corners
        for i, (x, y) in enumerate(self.corners):
            color = self.colors[i]
            cv2.circle(self.display_image, (x, y), self.point_radius, color, -1)
            cv2.circle(self.display_image, (x, y), self.point_radius + 2, (255, 255, 255), 2)
            
            # Add corner label
            label = self.corner_names[i]
            cv2.putText(self.display_image, label, (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw lines between corners if we have enough points
        if len(self.corners) >= 2:
            for i in range(len(self.corners)):
                start_point = self.corners[i]
                end_point = self.corners[(i + 1) % len(self.corners)]
                if i < len(self.corners) - 1 or len(self.corners) == 4:
                    cv2.line(self.display_image, start_point, end_point, (255, 255, 255), self.line_thickness)
        
        # Add instructions
        if self.current_corner < 4:
            instruction = f"Click {self.corner_names[self.current_corner]} corner ({self.current_corner + 1}/4)"
            cv2.putText(self.display_image, instruction, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(self.display_image, "Press 's' to save, 'r' to reset, 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def get_physical_dimensions(self):
        """Get physical warehouse dimensions from user"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        try:
            # Get warehouse width
            width_str = simpledialog.askstring(
                "Warehouse Width", 
                "Enter warehouse width:\n\nExamples:\n• 10.5 (meters)\n• 10.5m (meters)\n• 34.5ft (feet)\n• 34'6\" (feet and inches)",
                initialvalue="10.0"
            )
            
            if not width_str:
                return False
            
            # Get warehouse length
            length_str = simpledialog.askstring(
                "Warehouse Length", 
                "Enter warehouse length:\n\nExamples:\n• 8.0 (meters)\n• 8.0m (meters)\n• 26.2ft (feet)\n• 26'3\" (feet and inches)",
                initialvalue="8.0"
            )
            
            if not length_str:
                return False
            
            # Parse dimensions
            self.warehouse_width = self.parse_dimension(width_str)
            self.warehouse_length = self.parse_dimension(length_str)
            
            # Confirm dimensions
            confirm_msg = f"Confirm warehouse dimensions:\n\nWidth: {self.warehouse_width:.2f} meters ({self.warehouse_width * 3.28084:.1f} feet)\nLength: {self.warehouse_length:.2f} meters ({self.warehouse_length * 3.28084:.1f} feet)"
            
            confirmed = messagebox.askyesno("Confirm Dimensions", confirm_msg)
            
            if confirmed:
                print(f"✅ Warehouse dimensions set: {self.warehouse_width:.2f}m x {self.warehouse_length:.2f}m")
                return True
            else:
                return self.get_physical_dimensions()  # Ask again
                
        except Exception as e:
            messagebox.showerror("Error", f"Invalid dimension format: {e}")
            return self.get_physical_dimensions()  # Ask again
        finally:
            root.destroy()
    
    def parse_dimension(self, dim_str):
        """Parse dimension string to meters"""
        dim_str = dim_str.strip().lower()
        
        # Handle feet and inches (e.g., "34'6\"" or "34ft 6in")
        if "'" in dim_str or "ft" in dim_str:
            # Convert feet to meters
            if "'" in dim_str:
                parts = dim_str.replace('"', '').split("'")
                feet = float(parts[0])
                inches = float(parts[1]) if len(parts) > 1 and parts[1] else 0
                total_feet = feet + inches / 12
            else:
                # Handle "34.5ft" format
                feet_str = dim_str.replace("ft", "").replace("feet", "")
                total_feet = float(feet_str)
            
            return total_feet * 0.3048  # Convert feet to meters
        
        # Handle meters (default)
        elif "m" in dim_str:
            return float(dim_str.replace("m", "").replace("meters", ""))
        else:
            # Assume meters if no unit specified
            return float(dim_str)
    
    def save_calibration(self, filename="warehouse_calibration.json"):
        """Save calibration to JSON file"""
        if len(self.corners) != 4:
            print("❌ Need exactly 4 corners to save calibration")
            return False
        
        if self.warehouse_width <= 0 or self.warehouse_length <= 0:
            print("❌ Invalid warehouse dimensions")
            return False
        
        calibration_data = {
            "description": "Warehouse floor coordinate mapping calibration",
            "created_date": datetime.now().isoformat(),
            "warehouse_dimensions": {
                "width_meters": self.warehouse_width,
                "length_meters": self.warehouse_length,
                "width_feet": self.warehouse_width * 3.28084,
                "length_feet": self.warehouse_length * 3.28084
            },
            "image_corners": self.corners,
            "real_world_corners": [
                [0, 0],                                    # Top-left
                [self.warehouse_width, 0],                 # Top-right  
                [self.warehouse_width, self.warehouse_length], # Bottom-right
                [0, self.warehouse_length]                 # Bottom-left
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
            print(f"📐 Warehouse: {self.warehouse_width:.2f}m x {self.warehouse_length:.2f}m")
            print(f"📍 Corners: {self.corners}")
            
            self.calibration_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def reset_calibration(self):
        """Reset calibration points"""
        self.corners = []
        self.current_corner = 0
        self.calibration_complete = False
        print("🔄 Calibration reset - select corners again")
        self.update_display()
    
    def run_calibration(self, image_source=None):
        """Run interactive calibration"""
        print("🎯 WAREHOUSE CALIBRATION TOOL")
        print("=" * 50)
        print("Instructions:")
        print("1. Click 4 corners in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("2. Enter physical warehouse dimensions")
        print("3. Press 's' to save, 'r' to reset, 'q' to quit")
        print("=" * 50)
        
        # Get image source
        if image_source is None:
            # Try to get from camera
            cap = cv2.VideoCapture(1)  # Try camera 1 first
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Fallback to camera 0
            
            if not cap.isOpened():
                print("❌ No camera found for calibration")
                return False
            
            print("📷 Using live camera feed - press SPACE to capture calibration image")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, "Press SPACE to capture calibration image", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Capture Calibration Image", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space to capture
                    self.image = frame.copy()
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            cap.release()
            cv2.destroyWindow("Capture Calibration Image")
        else:
            # Use provided image
            if isinstance(image_source, str):
                self.image = cv2.imread(image_source)
            else:
                self.image = image_source.copy()
        
        if self.image is None:
            print("❌ Failed to get calibration image")
            return False
        
        # Start calibration
        self.display_image = self.image.copy()
        cv2.namedWindow("Warehouse Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Warehouse Calibration", self.mouse_callback)
        
        self.update_display()
        
        while True:
            cv2.imshow("Warehouse Calibration", self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_calibration()
            elif key == ord('s') and len(self.corners) == 4:
                # Get physical dimensions
                if self.get_physical_dimensions():
                    if self.save_calibration():
                        break
        
        cv2.destroyAllWindows()
        return self.calibration_complete

def main():
    """Main calibration function"""
    calibrator = WarehouseCalibrationTool()
    
    print("🎯 Starting Warehouse Calibration Tool...")
    success = calibrator.run_calibration()
    
    if success:
        print("✅ Calibration completed successfully!")
        print("📁 Calibration file: warehouse_calibration.json")
        print("🚀 You can now run the tracking system with coordinate mapping")
    else:
        print("❌ Calibration cancelled or failed")

if __name__ == "__main__":
    main()
