#!/usr/bin/env python3
"""
YOLOv8 Frame Processor Module
Processes frames using YOLOv8 tracking instead of SIFT feature matching
Drop-in replacement for CPUFrameProcessor with YOLOv8 tracking
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class YOLOv8FrameProcessor:
    """Frame processor using YOLOv8 tracking instead of SIFT"""

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.camera_name = f"Camera {camera_id}"

        # Processing components (injected later) - SAME as CPUFrameProcessor
        self.fisheye_corrector = None
        self.pallet_detector = None
        self.filtering = None
        self.coordinate_mapper = None
        self.coordinate_mapper_initialized = False
        self.yolo_tracking_db = None  # YOLOv8 tracking database
        self.color_extractor = None
        self.db_handler = None
        self.display_manager = None

        # Detection stages (same as CPUFrameProcessor)
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.size_filtered_detections = []
        self.final_tracked_detections = []

        # Performance tracking (same as CPUFrameProcessor)
        self.new_objects = 0
        self.existing_objects = 0

        logger.info(f"✅ YOLOv8 Frame Processor initialized for {self.camera_name}")

    @property
    def global_db(self):
        """Compatibility property - maps to yolo_tracking_db for existing pipeline"""
        return self.yolo_tracking_db

    @global_db.setter
    def global_db(self, value):
        """Compatibility setter - maps to yolo_tracking_db for existing pipeline"""
        self.yolo_tracking_db = value

    def inject_components(self, **components):
        """Inject processing components (same interface as CPUFrameProcessor)"""
        for name, component in components.items():
            if hasattr(self, name):
                setattr(self, name, component)
                logger.debug(f"🔧 Injected {name} into YOLOv8 frame processor")

    def assign_global_ids_yolo_tracking(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Assign global IDs using YOLOv8 track IDs instead of SIFT features
        
        Args:
            detections: List of detection dictionaries with track_ids
            frame: Current frame for color extraction
            
        Returns:
            List of detections with global IDs assigned
        """
        if self.yolo_tracking_db is None:
            logger.error("❌ YOLOv8 tracking database not initialized")
            return detections

        global_detections = []
        active_track_ids = []

        for detection in detections:
            try:
                # Extract image region for color analysis (same as SIFT version)
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                image_region = frame[y1:y2, x1:x2]

                # Extract dominant color from detected object
                if self.color_extractor:
                    color_info = self.color_extractor.extract_dominant_color(image_region)
                    detection.update(color_info)

                # Assign global ID using YOLOv8 track ID (REPLACES SIFT)
                global_id, status, confidence = self.yolo_tracking_db.assign_global_id_from_track_id(detection)

                # Update detection with tracking results
                detection.update({
                    'global_id': global_id,
                    'tracking_status': status,
                    'similarity_score': confidence,
                    'tracking_method': 'yolov8_bytetrack'
                })

                # Track active track IDs for cleanup
                if detection.get('track_id') is not None:
                    active_track_ids.append(detection['track_id'])

                global_detections.append(detection)
                
               #logger.debug(f"🎯 YOLOv8 Tracking: track_id={detection.get('track_id')} -> global_id={global_id} ({status})")

            except Exception as e:
                logger.error(f"❌ YOLOv8 tracking assignment failed: {e}")
                # Add detection with failed tracking
                detection.update({
                    'global_id': -1,
                    'tracking_status': 'failed',
                    'similarity_score': 0.0,
                    'tracking_method': 'yolov8_bytetrack'
                })
                global_detections.append(detection)

        # Cleanup old tracks periodically
        if len(active_track_ids) > 0:
            self.yolo_tracking_db.cleanup_old_tracks(active_track_ids)

        # Save database
        self.yolo_tracking_db.save_database()

        logger.info(f"🎯 YOLOv8 Tracking: Processed {len(global_detections)} detections, {len(active_track_ids)} active tracks")
        return global_detections

    def translate_to_physical_coordinates(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """
        Translate pixel coordinates to physical coordinates (SAME logic as CPUFrameProcessor)
        """
        if not self.coordinate_mapper_initialized:
            return detections

        try:
            for detection in detections:
                center = detection.get('center')
                if center is None:
                    center = self.filtering.calculate_center(detection['bbox'])
                    detection['center'] = center

                center_x, center_y = center

                # Scale coordinates to calibration frame size (4K) for accurate coordinate mapping
                # Calibration files are based on 3840x2160 resolution (SAME as CPUFrameProcessor)
                scale_x = 3840 / frame_width
                scale_y = 2160 / frame_height

                scaled_center_x = center_x * scale_x
                scaled_center_y = center_y * scale_y

                # Transform to physical coordinates using scaled coordinates (SAME as CPUFrameProcessor)
                physical_x, physical_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

                if physical_x is not None and physical_y is not None:
                    detection['physical_x_ft'] = round(physical_x, 2)
                    detection['physical_y_ft'] = round(physical_y, 2)
                    detection['coordinate_status'] = 'SUCCESS'
                    logger.debug(f"Camera {self.camera_id}: Pixel ({center_x}, {center_y}) → Scaled ({scaled_center_x:.1f}, {scaled_center_y:.1f}) → Physical ({physical_x:.2f}ft, {physical_y:.2f}ft)")
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'FAILED'
                    logger.debug(f"Camera {self.camera_id}: Coordinate transformation failed for center ({center_x}, {center_y})")

                # Add 4-corner physical coordinates if needed (for compatibility)
                if detection.get('corners'):
                    corners = detection['corners']
                    physical_corners = []
                    for corner in corners:
                        corner_x, corner_y = corner
                        scaled_corner_x = corner_x * scale_x
                        scaled_corner_y = corner_y * scale_y
                        phys_x, phys_y = self.coordinate_mapper.pixel_to_real(scaled_corner_x, scaled_corner_y)
                        physical_corners.append([phys_x, phys_y])

                    detection['physical_corners'] = physical_corners
                    detection['real_center'] = [physical_x, physical_y] if physical_x is not None and physical_y is not None else [None, None]

            return detections

        except Exception as e:
            logger.error(f"YOLOv8 coordinate translation failed: {e}")
            return detections

    def process_frame_yolo_tracking(self, processed_frame: np.ndarray) -> bool:
        """
        Complete YOLOv8 tracking pipeline (replaces SIFT-based processing)
        
        Args:
            processed_frame: Preprocessed frame ready for detection
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # NOTE: Detection already done in detection_pool with tracking
            # Raw detections should already have track_ids from YOLOv8
            
            # Stage 1: Area filtering (same as original)
            self.area_filtered_detections = self.filtering.apply_area_filter(self.raw_detections)

            # Stage 2: Grid cell filtering (same as original)
            self.grid_filtered_detections = self.filtering.apply_grid_cell_filter(self.area_filtered_detections)

            # Stage 3: Physical coordinate translation (same as original)
            frame_height, frame_width = processed_frame.shape[:2]
            self.grid_filtered_detections = self.translate_to_physical_coordinates(
                self.grid_filtered_detections, frame_width, frame_height
            )

            # Stage 4: Physical size filtering (same as original)
            self.size_filtered_detections = self.filtering.apply_physical_size_filter(self.grid_filtered_detections)

            # Stage 5: YOLOv8 tracking-based global ID assignment (REPLACES SIFT)
            self.final_tracked_detections = self.assign_global_ids_yolo_tracking(self.size_filtered_detections, processed_frame)

            # Stage 6: Save detections to database (same as original)
            if self.db_handler and self.db_handler.is_connected():
                for detection in self.final_tracked_detections:
                    self.db_handler.save_detection_to_db(self.camera_id, detection)

            return True

        except Exception as e:
            logger.error(f"❌ YOLOv8 frame processing failed: {e}")
            return False

    def get_detection_counts(self) -> Dict:
        """Get detection counts for each processing stage (same as CPUFrameProcessor)"""
        return {
            'raw_detections': len(self.raw_detections),
            'area_filtered_detections': len(self.area_filtered_detections),
            'grid_filtered_detections': len(self.grid_filtered_detections),
            'size_filtered_detections': len(self.size_filtered_detections),
            'final_tracked_detections': len(self.final_tracked_detections),
            'new_objects': sum(1 for d in self.final_tracked_detections if d.get('tracking_status') == 'new'),
            'existing_objects': sum(1 for d in self.final_tracked_detections if d.get('tracking_status') == 'existing'),
            'tracking_method': 'yolov8_bytetrack'
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.get_detection_counts()

        if self.yolo_tracking_db:
            tracking_stats = self.yolo_tracking_db.get_statistics()
            stats.update(tracking_stats)

        return stats

    def assign_global_ids(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Compatibility method - redirects to YOLOv8 tracking method
        This ensures existing pipeline code works without changes
        """
        return self.assign_global_ids_yolo_tracking(detections, frame)

    def process_frame(self, processed_frame: np.ndarray) -> bool:
        """
        Compatibility method - redirects to YOLOv8 processing method
        This ensures existing pipeline code works without changes
        """
        return self.process_frame_yolo_tracking(processed_frame)
