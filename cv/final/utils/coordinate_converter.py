#!/usr/bin/env python3
"""
Coordinate Conversion Utilities
Handles pixel to physical coordinate transformations for warehouse tracking
Supports both center-point and 4-corner transformations
"""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class CoordinateConverter:
    """Utility class for coordinate conversions"""
    
    def __init__(self, coordinate_mapper, camera_id: int = None):
        """
        Initialize coordinate converter
        
        Args:
            coordinate_mapper: Initialized coordinate mapper with calibration loaded
            camera_id: Camera ID for logging purposes
        """
        self.coordinate_mapper = coordinate_mapper
        self.camera_id = camera_id
        self.is_calibrated = coordinate_mapper.is_calibrated if coordinate_mapper else False
        
        if self.is_calibrated:
            logger.info(f"✅ Coordinate converter initialized for Camera {camera_id}")
        else:
            logger.warning(f"⚠️ Coordinate converter not calibrated for Camera {camera_id}")
    
    def translate_to_physical_coordinates(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """
        Translate pixel coordinates to physical warehouse coordinates
        Supports both center points and 4-corner transformations
        
        Args:
            detections: List of detection dictionaries
            frame_width: Current frame width
            frame_height: Current frame height
            
        Returns:
            List of detections with physical coordinates added
        """
        if not self.is_calibrated:
            logger.debug("Coordinate mapper not calibrated, skipping physical coordinate translation")
            return detections
        
        try:
            for detection in detections:
                # Calculate center from bounding box
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Add center to detection if not already present
                if 'center' not in detection:
                    detection['center'] = [int(center_x), int(center_y)]

                # Scale coordinates to calibration frame size (4K) for accurate coordinate mapping
                # Calibration files are based on 3840x2160 resolution
                scale_x = 3840 / frame_width
                scale_y = 2160 / frame_height

                scaled_center_x = center_x * scale_x
                scaled_center_y = center_y * scale_y

                # Get physical coordinates using coordinate mapper with scaled coordinates
                real_x, real_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

                if real_x is not None and real_y is not None:
                    detection['physical_x_ft'] = round(real_x, 2)
                    detection['physical_y_ft'] = round(real_y, 2)
                    detection['coordinate_status'] = 'SUCCESS'
                    logger.debug(f"Camera {self.camera_id}: Pixel ({center_x:.1f}, {center_y:.1f}) → Scaled ({scaled_center_x:.1f}, {scaled_center_y:.1f}) → Physical ({real_x:.2f}ft, {real_y:.2f}ft)")
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'CONVERSION_FAILED'
                    logger.debug(f"Camera {self.camera_id}: Coordinate conversion failed for pixel ({center_x:.1f}, {center_y:.1f})")

                # Translate all 4 corners to physical coordinates if corners exist
                corners = detection.get('corners', [])
                if corners and len(corners) == 4:
                    physical_corners = []

                    for corner in corners:
                        pixel_x, pixel_y = corner

                        # Scale corner coordinates
                        scaled_x = pixel_x * scale_x
                        scaled_y = pixel_y * scale_y

                        # Transform to physical coordinates
                        phys_x, phys_y = self.coordinate_mapper.pixel_to_real(scaled_x, scaled_y)

                        if phys_x is not None and phys_y is not None:
                            physical_corners.append([round(phys_x, 2), round(phys_y, 2)])
                        else:
                            physical_corners.append([None, None])

                    detection['physical_corners'] = physical_corners
                    detection['real_center'] = [real_x, real_y] if real_x is not None and real_y is not None else [None, None]
                    
                    logger.debug(f"Camera {self.camera_id}: Transformed 4 corners to physical coordinates")

            return detections

        except Exception as e:
            logger.error(f"Coordinate translation failed for Camera {self.camera_id}: {e}")
            return detections

def create_coordinate_converter(coordinate_mapper, camera_id: int = None) -> CoordinateConverter:
    """
    Factory function to create coordinate converter
    
    Args:
        coordinate_mapper: Initialized coordinate mapper
        camera_id: Camera ID for logging
        
    Returns:
        CoordinateConverter instance
    """
    return CoordinateConverter(coordinate_mapper, camera_id)
