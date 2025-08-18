#!/usr/bin/env python3
"""
Frame Processing Pipeline Module
Core processing orchestration for warehouse tracking system
Extracted from main.py for modular architecture
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)

class CPUFrameProcessor:
    """CPU-based frame processing pipeline for warehouse tracking"""
    
    def __init__(self, camera_id: int = 1):
        self.camera_id = camera_id
        
        # Processing components (will be injected)
        self.fisheye_corrector = None
        self.pallet_detector = None
        self.filtering = None
        self.coordinate_mapper = None
        self.coordinate_mapper_initialized = False
        self.global_db = None
        self.color_extractor = None
        self.db_handler = None
        self.display_manager = None
        
        # Detection results storage
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.size_filtered_detections = []
        self.final_tracked_detections = []
        
        # Performance tracking
        self.new_objects = 0
        self.existing_objects = 0
        
        logger.info(f"✅ Frame Processor initialized for Camera {camera_id}")

    def inject_components(self, fisheye_corrector, pallet_detector, filtering, 
                         coordinate_mapper, coordinate_mapper_initialized, global_db, 
                         color_extractor, db_handler, display_manager):
        """Inject all required processing components"""
        self.fisheye_corrector = fisheye_corrector
        self.pallet_detector = pallet_detector
        self.filtering = filtering
        self.coordinate_mapper = coordinate_mapper
        self.coordinate_mapper_initialized = coordinate_mapper_initialized
        self.global_db = global_db
        self.color_extractor = color_extractor
        self.db_handler = db_handler
        self.display_manager = display_manager
        
        logger.info(f"✅ All processing components injected for Camera {self.camera_id}")

    def translate_to_physical_coordinates(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """CPU physical coordinate translation"""
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
                # Calibration files are based on 3840x2160 resolution
                scale_x = 3840 / frame_width
                scale_y = 2160 / frame_height

                scaled_center_x = center_x * scale_x
                scaled_center_y = center_y * scale_y

                # Transform to physical coordinates using scaled coordinates
                physical_x, physical_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

                if physical_x is not None and physical_y is not None:
                    detection['physical_x_ft'] = round(physical_x, 2)
                    detection['physical_y_ft'] = round(physical_y, 2)
                    detection['coordinate_status'] = 'SUCCESS'
                    logger.debug(f"Camera {self.camera_id}: Pixel ({center_x}, {center_y}) → Scaled ({scaled_center_x:.1f}, {scaled_center_y:.1f}) → Physical ({physical_x:.2f}ft, {physical_y:.2f}ft)")
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'CONVERSION_FAILED'
                    logger.debug(f"Camera {self.camera_id}: Coordinate conversion failed for pixel ({center_x}, {center_y})")

                # Transform all 4 corners to physical coordinates if corners exist
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
                    detection['real_center'] = [physical_x, physical_y] if physical_x is not None and physical_y is not None else [None, None]

                    logger.debug(f"Camera {self.camera_id}: Transformed 4 corners to physical coordinates")

            return detections

        except Exception as e:
            logger.error(f"CPU coordinate translation failed: {e}")
            return detections

    def assign_global_ids(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """CPU global ID assignment with SIFT feature matching"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            # Initialize global_id to prevent scope errors
            global_id = None

            try:
                # Extract image region for SIFT features and color analysis
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                image_region = frame[y1:y2, x1:x2]

                # Extract dominant color from detected object
                color_info = self.color_extractor.extract_dominant_color(image_region)
                detection.update(color_info)  # Add color data to detection

                # Assign global ID using CPU SIFT features
                global_id, status, similarity = self.global_db.assign_global_id(
                    image_region, detection
                )

                # Debug: Log color extraction after successful ID assignment
                # logger.info(f"🎨 Color extracted for object {global_id}: {color_info}")
                # logger.info(f"🔍 Detection data after color update: {list(detection.keys())}")

                # Update counters
                if status == 'new':
                    self.new_objects += 1
                elif status == 'existing':
                    self.existing_objects += 1

                # Add global tracking info to detection
                detection['global_id'] = global_id
                detection['tracking_status'] = status
                detection['similarity_score'] = similarity

                if global_id != -1:
                    seen_ids.add(global_id)

            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                # Set default values for failed detection
                detection['global_id'] = -1
                detection['tracking_status'] = 'failed'
                detection['similarity_score'] = 0.0

            global_detections.append(detection)

        # Mark disappeared objects
        self.global_db.mark_disappeared_objects(seen_ids)

        return global_detections

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Complete CPU-based frame processing pipeline"""
        processed_frame = frame.copy()

        # Apply fisheye correction
        processed_frame = self.fisheye_corrector.correct(processed_frame)

        # Resize if too large (for performance)
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Complete CPU detection and tracking pipeline
        try:
            # Stage 1: CPU detection
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)

            # Stage 2: CPU area filtering
            self.area_filtered_detections = self.filtering.apply_area_filter(self.raw_detections)

            # Stage 3: CPU grid cell filtering
            self.grid_filtered_detections = self.filtering.apply_grid_cell_filter(self.area_filtered_detections)

            # Stage 4: CPU physical coordinate translation
            frame_height, frame_width = processed_frame.shape[:2]
            self.grid_filtered_detections = self.translate_to_physical_coordinates(
                self.grid_filtered_detections, frame_width, frame_height
            )

            # Stage 5: CPU physical size filtering (filter out objects >15ft)
            self.size_filtered_detections = self.filtering.apply_physical_size_filter(self.grid_filtered_detections)

            # Stage 6: CPU SIFT feature matching and global ID assignment
            # TEMPORARILY COMMENTED OUT FOR TESTING - Stage 6 & 7
            self.final_tracked_detections = self.assign_global_ids(self.size_filtered_detections, processed_frame)

            # For testing: Use size_filtered_detections directly without SIFT matching
            # self.final_tracked_detections = self.size_filtered_detections

            # Stage 7: Save detections to database - COMMENTED OUT FOR TESTING
            if self.db_handler and self.db_handler.is_connected():
                for detection in self.final_tracked_detections:
                    self.db_handler.save_detection_to_db(self.camera_id, detection)

        except Exception as e:
            logger.error(f"CPU detection pipeline failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.size_filtered_detections = []
            self.final_tracked_detections = []

        # Draw results using GUI display manager
        if self.display_manager:
            processed_frame = self.display_manager.render_frame(processed_frame, self, self.filtering)

        return processed_frame

    def get_detection_counts(self) -> Dict[str, int]:
        """Get current detection counts for statistics"""
        return {
            'raw_detections': len(self.raw_detections),
            'area_filtered_detections': len(self.area_filtered_detections),
            'grid_filtered_detections': len(self.grid_filtered_detections),
            'size_filtered_detections': len(self.size_filtered_detections),
            'final_tracked_detections': len(self.final_tracked_detections),
            'new_objects': self.new_objects,
            'existing_objects': self.existing_objects
        }

    def reset_counters(self):
        """Reset object counters"""
        self.new_objects = 0
        self.existing_objects = 0
