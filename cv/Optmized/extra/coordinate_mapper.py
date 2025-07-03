"""
Standalone Coordinate Mapper for Physical Coordinate Translation
Extracted from detector_tracker.py to avoid import dependencies
"""

import cv2
import numpy as np
import json
import logging
import os

logger = logging.getLogger(__name__)

class CoordinateMapper:
    """Maps pixel coordinates to real-world coordinates using homography"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        """Initialize coordinate mapper with floor dimensions in FEET"""
        self.floor_width_ft = floor_width  # Floor width in feet
        self.floor_length_ft = floor_length  # Floor length in feet
        self.camera_id = camera_id  # Add camera ID for offset calculation
        self.homography_matrix = None
        self.is_calibrated = False
        
        # Column 3 cameras (8,9,10,11) use direct global coordinate mapping
        # Calibration files map directly to warehouse coordinates - no offset transformation needed
        # This will be expanded for other columns when they are activated
        
        logger.info(f"Coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")
            logger.info(f"Camera {camera_id}: Using direct global coordinate mapping")

    def load_calibration(self, filename=None):
        """Load calibration from JSON file"""
        if filename is None:
            # Auto-generate filename based on camera ID
            if self.camera_id:
                filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            else:
                filename = "../configs/warehouse_calibration.json"
        
        # Handle relative paths from cv/Optmized directory
        if not os.path.isabs(filename) and not filename.startswith('../'):
            filename = f"../{filename}"
            
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            # Extract warehouse dimensions
            warehouse_dims = calibration_data.get('warehouse_dimensions', {})
            self.floor_width_ft = warehouse_dims.get('width_feet', self.floor_width_ft)
            self.floor_length_ft = warehouse_dims.get('length_feet', self.floor_length_ft)
            
            # Extract corner points
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            # Validate corner points
            if len(image_corners) != 4 or len(real_world_corners) != 4:
                raise ValueError("Calibration must contain exactly 4 corner points")
            
            # Calculate homography from image points to real-world points
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            self.is_calibrated = True
            
            logger.info(f"Coordinate calibration loaded from: {filename}")
            logger.info(f"Camera local area: {self.floor_width_ft:.1f}ft x {self.floor_length_ft:.1f}ft")
            
            # Log the coordinate transformation for Column 3 cameras
            if self.camera_id in [8, 9, 10, 11]:
                logger.info(f"Camera {self.camera_id} coordinate mapping: Direct mapping to global coordinates")
            
            logger.info(f"Coordinate mapper initialized - Calibrated: {self.is_calibrated}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates in FEET with warehouse offset"""
        if not self.is_calibrated or self.homography_matrix is None:
            return None, None
        
        try:
            # Apply homography transformation
            pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
            
            # Extract global coordinates (in feet) - calibration files now use direct global mapping
            global_x = float(real_point[0][0][0])
            global_y = float(real_point[0][0][1])

            # For Column 3 cameras (8,9,10,11), coordinates are already global
            # No offset transformation needed since calibration files map directly to global coordinates
            logger.debug(f"Camera {self.camera_id}: Pixel ({pixel_x}, {pixel_y}) → Global ({global_x:.1f}ft, {global_y:.1f}ft)")

            return global_x, global_y
        
        except Exception as e:
            logger.error(f"Error in pixel_to_real conversion: {e}")
            return None, None

    def real_to_pixel(self, real_x, real_y):
        """Convert real-world coordinates (in feet) to pixel coordinates"""
        if not self.is_calibrated or self.homography_matrix is None:
            return None, None
        
        try:
            # For Column 3 cameras (8,9,10,11), coordinates are already global
            # Apply inverse homography transformation directly (no offset removal needed)
            real_point = np.array([[[real_x, real_y]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(real_point, np.linalg.inv(self.homography_matrix))
            
            pixel_x = int(pixel_point[0][0][0])
            pixel_y = int(pixel_point[0][0][1])
            
            return pixel_x, pixel_y
        
        except Exception as e:
            logger.error(f"Error in real_to_pixel conversion: {e}")
            return None, None
