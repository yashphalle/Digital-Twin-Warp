#!/usr/bin/env python3
"""
Coordinate Mapper Module
Critical component for pixel-to-physical coordinate transformation
Extracted from main.py for modular architecture - HANDLE WITH EXTREME CARE
"""

import cv2
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class CoordinateMapper:
    """CPU-based coordinate mapping (same as combined filtering)"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        
        logger.info(f"CPU coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")

    def load_calibration(self, filename=None):
        """Load calibration from JSON file (same as combined filtering)"""
        if filename is None:
            if self.camera_id:
                filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            else:
                filename = "../configs/warehouse_calibration.json"
        
        if not os.path.isabs(filename) and not filename.startswith('../'):
            filename = f"../{filename}"
            
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            warehouse_dims = calibration_data.get('warehouse_dimensions', {})
            self.floor_width_ft = warehouse_dims.get('width_feet', self.floor_width_ft)
            self.floor_length_ft = warehouse_dims.get('length_feet', self.floor_length_ft)
            
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            if len(image_corners) != 4 or len(real_world_corners) != 4:
                raise ValueError("Calibration must contain exactly 4 corner points")
            
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            self.is_calibrated = True
            logger.info(f"CPU coordinate calibration loaded from: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real(self, pixel_x, pixel_y):
        """Single point coordinate transformation"""
        if not self.is_calibrated:
            return None, None
        
        try:
            points = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
            result = transformed_points.reshape(-1, 2)
            if len(result) > 0:
                return float(result[0][0]), float(result[0][1])
            return None, None
        except Exception as e:
            logger.error(f"CPU coordinate transformation failed: {e}")
            return None, None
