#!/usr/bin/env python3
"""
Simple 3-Camera Warehouse Tracking System
No complex threading - sequential processing for reliability
Cameras: 8, 9, 10 (Column 3 - Bottom, Middle, Top)
"""

import cv2
import numpy as np
import logging
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simple3CameraTracker:
    """Simple 3-camera tracking system without complex threading"""
    
    def __init__(self):
        self.camera_ids = [8, 9, 10]  # Column 3 cameras
        self.camera_names = {
            8: "Camera 8 (Column 3 - Bottom)",
            9: "Camera 9 (Column 3 - Middle)", 
            10: "Camera 10 (Column 3 - Top)"
        }
        
        # Initialize cameras
        self.cameras = {}
        self.detectors = {}
        self.coordinate_mappers = {}
        
        # Performance tracking
        self.frame_counts = {cam_id: 0 for cam_id in self.camera_ids}
        self.detection_counts = {cam_id: 0 for cam_id in self.camera_ids}
        self.start_time = time.time()
        
        logger.info("🎥 SIMPLE 3-CAMERA WAREHOUSE TRACKING SYSTEM")
        logger.info("=" * 80)
        logger.info("CAMERAS:")
        for cam_id in self.camera_ids:
            logger.info(f"🎥 {self.camera_names[cam_id]}")
        logger.info("=" * 80)
        logger.info("FEATURES:")
        logger.info("✅ Sequential Processing - No complex threading")
        logger.info("✅ Real-time Grid Display - All cameras simultaneously")
        logger.info("✅ Object Detection - GPU accelerated")
        logger.info("✅ Physical Coordinates - Warehouse coordinate system")
        logger.info("=" * 80)
        
    def initialize_system(self):
        """Initialize all cameras and detection systems"""
        logger.info("🔧 Initializing 3-camera system...")
        
        # Initialize detector (shared across cameras)
        try:
            sys.path.append('..')
            from pallet_detector_simple import SimplePalletDetector
            
            self.detector = SimplePalletDetector()
            self.detector.confidence_threshold = 0.1
            self.detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet", "pallet with cargo", "loaded pallet"]
            self.detector.current_prompt = "pallet wrapped in plastic"
            logger.info("✅ Shared detector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize detector: {e}")
            return False
            
        # Initialize coordinate mappers
        try:
            for cam_id in self.camera_ids:
                from coordinate_mapper import CoordinateMapper
                self.coordinate_mappers[cam_id] = CoordinateMapper(cam_id)
                logger.info(f"✅ Camera {cam_id} coordinate mapper loaded")
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinate mappers: {e}")
            return False
            
        # Initialize cameras
        for cam_id in self.camera_ids:
            if not self._initialize_camera(cam_id):
                logger.error(f"❌ Failed to initialize Camera {cam_id}")
                return False
                
        logger.info("🚀 All systems initialized successfully!")
        return True
        
    def _initialize_camera(self, cam_id: int) -> bool:
        """Initialize a single camera"""
        try:
            # Get camera URL from config
            config = Config()
            camera_url = config.RTSP_CAMERA_URLS.get(cam_id)
            
            # Create camera capture
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                logger.error(f"❌ Cannot open Camera {cam_id}")
                return False
                
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.cameras[cam_id] = cap
            logger.info(f"✅ Camera {cam_id} initialized: {camera_url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing Camera {cam_id}: {e}")
            return False
    
    def process_frame(self, cam_id: int, frame: np.ndarray) -> Dict:
        """Process a single frame for detection and tracking"""
        try:
            # Detect objects
            detections = self.detector.detect_pallets(frame)
            raw_count = len(detections)
            
            # Apply area filtering
            area_filtered = self._apply_area_filter(detections)
            area_count = len(area_filtered)
            
            # Apply grid filtering  
            grid_filtered = self._apply_grid_filter(area_filtered, frame.shape)
            grid_count = len(grid_filtered)
            
            # Apply coordinate translation
            coordinate_filtered = self._translate_coordinates(grid_filtered, frame.shape, cam_id)
            final_count = len(coordinate_filtered)
            
            # Log processing results
            if raw_count > 0:
                logger.info(f"🔧 Camera {cam_id}: {raw_count}→{area_count}→{grid_count}→{final_count} detections")
            
            # Update counters
            self.frame_counts[cam_id] += 1
            self.detection_counts[cam_id] += final_count
            
            return {
                'camera_id': cam_id,
                'camera_name': self.camera_names[cam_id],
                'frame': frame,
                'detections': coordinate_filtered,
                'raw_count': raw_count,
                'final_count': final_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing Camera {cam_id} frame: {e}")
            return {
                'camera_id': cam_id,
                'camera_name': self.camera_names[cam_id], 
                'frame': frame,
                'detections': [],
                'raw_count': 0,
                'final_count': 0
            }
    
    def _apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections by area"""
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Filter out very large (>1M pixels) and very small (<15K pixels) detections
            if 15000 <= area <= 1000000:
                filtered.append(detection)
                
        return filtered
    
    def _apply_grid_filter(self, detections: List[Dict], frame_shape: Tuple) -> List[Dict]:
        """Apply 3x3 grid-based spatial filtering"""
        if len(detections) <= 1:
            return detections
            
        height, width = frame_shape[:2]
        grid_width = width // 3
        grid_height = height // 3
        
        # Group detections by grid cell
        grid_groups = {}
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            grid_x = min(int(center_x // grid_width), 2)
            grid_y = min(int(center_y // grid_height), 2)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in grid_groups:
                grid_groups[grid_key] = []
            grid_groups[grid_key].append(detection)
        
        # Keep only the highest confidence detection per grid cell
        filtered = []
        for group in grid_groups.values():
            if group:
                best_detection = max(group, key=lambda d: d.get('confidence', 0))
                filtered.append(best_detection)
                
        return filtered
    
    def _translate_coordinates(self, detections: List[Dict], frame_shape: Tuple, cam_id: int) -> List[Dict]:
        """Translate pixel coordinates to physical coordinates"""
        translated = []
        height, width = frame_shape[:2]
        
        for detection in detections:
            try:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Get physical coordinates
                mapper = self.coordinate_mappers[cam_id]
                physical_x, physical_y = mapper.pixel_to_physical(center_x, center_y)
                
                # Add physical coordinates to detection
                detection['physical_x_ft'] = physical_x
                detection['physical_y_ft'] = physical_y
                detection['camera_id'] = cam_id
                
                translated.append(detection)
                
            except Exception as e:
                logger.warning(f"⚠️ Coordinate translation failed for Camera {cam_id}: {e}")
                # Keep detection without physical coordinates
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['camera_id'] = cam_id
                translated.append(detection)

        return translated

    def create_display_frame(self, results: List[Dict]) -> np.ndarray:
        """Create grid display showing all camera feeds with detections"""
        # Create 2x2 grid (3 cameras + 1 info panel)
        display_width, display_height = 1800, 1200
        grid_width, grid_height = display_width // 2, display_height // 2

        # Create black background
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Camera positions in grid
        positions = {
            8: (0, 0),   # Top-left
            9: (1, 0),   # Top-right
            10: (0, 1)   # Bottom-left
        }

        total_detections = 0

        for result in results:
            cam_id = result['camera_id']
            frame = result['frame']
            detections = result['detections']

            if cam_id in positions:
                col, row = positions[cam_id]

                # Resize frame to fit grid cell
                resized_frame = cv2.resize(frame, (grid_width, grid_height))

                # Draw detections on frame
                for detection in detections:
                    bbox = detection['bbox']
                    physical_x = detection.get('physical_x_ft')
                    physical_y = detection.get('physical_y_ft')

                    # Scale bbox to resized frame
                    scale_x = grid_width / frame.shape[1]
                    scale_y = grid_height / frame.shape[0]

                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)

                    # Draw bounding box
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw physical coordinates if available
                    if physical_x is not None and physical_y is not None:
                        coord_text = f"({physical_x:.1f}, {physical_y:.1f})"
                        cv2.putText(resized_frame, coord_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Add camera label
                cv2.putText(resized_frame, f"Camera {cam_id} ({len(detections)} objects)",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Place in grid
                y_start = row * grid_height
                y_end = y_start + grid_height
                x_start = col * grid_width
                x_end = x_start + grid_width

                display_frame[y_start:y_end, x_start:x_end] = resized_frame
                total_detections += len(detections)

        # Add info panel (bottom-right)
        self._draw_info_panel(display_frame, total_detections, grid_width, grid_height)

        return display_frame

    def _draw_info_panel(self, display_frame: np.ndarray, total_detections: int,
                        grid_width: int, grid_height: int):
        """Draw information panel"""
        # Info panel position (bottom-right)
        x_start = grid_width
        y_start = grid_height
        x_end = grid_width * 2
        y_end = grid_height * 2

        # Create info background
        info_panel = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)  # Dark gray background

        # Calculate performance stats
        elapsed_time = time.time() - self.start_time
        total_frames = sum(self.frame_counts.values())
        total_detected = sum(self.detection_counts.values())

        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        # Draw info text
        info_lines = [
            "WAREHOUSE TRACKING",
            "=" * 20,
            f"Active Cameras: {len(self.camera_ids)}",
            f"Total Detections: {total_detections}",
            f"Overall FPS: {fps:.1f}",
            f"Runtime: {elapsed_time:.1f}s",
            "",
            "Per Camera Stats:",
        ]

        for cam_id in self.camera_ids:
            frames = self.frame_counts[cam_id]
            detected = self.detection_counts[cam_id]
            cam_fps = frames / elapsed_time if elapsed_time > 0 else 0
            info_lines.append(f"Cam {cam_id}: {cam_fps:.1f} FPS, {detected} total")

        # Draw text lines
        y_offset = 30
        for line in info_lines:
            cv2.putText(info_panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        # Place info panel
        display_frame[y_start:y_end, x_start:x_end] = info_panel

    def run(self):
        """Main execution loop"""
        if not self.initialize_system():
            logger.error("❌ System initialization failed")
            return

        logger.info("🚀 Starting 3-camera tracking...")
        logger.info("Press 'q' to quit")

        try:
            while True:
                results = []

                # Process each camera sequentially
                for cam_id in self.camera_ids:
                    cap = self.cameras[cam_id]
                    ret, frame = cap.read()

                    if ret:
                        result = self.process_frame(cam_id, frame)
                        results.append(result)
                    else:
                        logger.warning(f"⚠️ Failed to read from Camera {cam_id}")

                # Create and display grid
                if results:
                    display_frame = self.create_display_frame(results)
                    cv2.imshow("3-Camera Warehouse Tracking", display_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")

        for cam_id, cap in self.cameras.items():
            cap.release()
            logger.info(f"✅ Camera {cam_id} released")

        cv2.destroyAllWindows()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    tracker = Simple3CameraTracker()
    tracker.run()
