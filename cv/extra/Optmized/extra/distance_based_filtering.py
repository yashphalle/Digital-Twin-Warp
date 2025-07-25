#!/usr/bin/env python3
"""
Distance-Based Filtering for Pallet Detection
Filters detections based on distance between object centers
"""

import cv2
import numpy as np
import logging
import sys
import os
import math
from typing import List, Dict, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceBasedFilteringDetector:
    """Pallet detection with distance-based filtering"""
    
    def __init__(self, camera_id: int = 11):
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
        self.running = False
        
        # Detection components
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = SimplePalletDetector()
        
        # Set detection parameters
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]
        
        # Update DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.1
            self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
        
        # AREA FILTERING SETTINGS (from previous script)
        self.MIN_AREA = 10000   # Exclude very small noise
        self.MAX_AREA = 100000  # Exclude very large background objects
        
        # DISTANCE FILTERING SETTINGS
        self.MIN_DISTANCE = 50   # Minimum distance between object centers (pixels)
        self.distance_step = 10  # Step size for adjusting distance
        
        # Detection results
        self.raw_detections = []
        self.area_filtered_detections = []
        self.distance_filtered_detections = []
        self.filtered_out_by_distance = []
        
        logger.info(f"Initialized distance-based filtering for {self.camera_name}")
        logger.info(f"Area filter: {self.MIN_AREA} - {self.MAX_AREA} pixels")
        logger.info(f"Distance filter: {self.MIN_DISTANCE} pixels minimum")
        logger.info(f"Prompts: {self.pallet_detector.sample_prompts}")
    
    def connect_camera(self) -> bool:
        """Connect to the camera"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False
        
        logger.info(f"Connecting to {self.camera_name}...")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.RTSP_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                return False
            
            logger.info(f"{self.camera_name} connected successfully")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def calculate_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y
    
    def calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two centers"""
        x1, y1 = center1
        x2, y2 = center2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply area-based filtering"""
        filtered = []
        for detection in detections:
            area = detection.get('area', 0)
            if self.MIN_AREA <= area <= self.MAX_AREA:
                filtered.append(detection)
        return filtered
    
    def apply_distance_filter(self, detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply distance-based filtering to remove overlapping/too close detections"""
        if len(detections) <= 1:
            return detections, []
        
        # Calculate centers for all detections
        for detection in detections:
            center = self.calculate_center(detection['bbox'])
            detection['center'] = center
        
        # Sort by confidence (keep higher confidence detections)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        filtered_out = []
        
        for i, detection in enumerate(sorted_detections):
            center = detection['center']
            too_close = False
            
            # Check distance to all already accepted detections
            for accepted in filtered:
                accepted_center = accepted['center']
                distance = self.calculate_distance(center, accepted_center)
                
                if distance < self.MIN_DISTANCE:
                    too_close = True
                    detection['filter_reason'] = 'TOO_CLOSE'
                    detection['distance_to_nearest'] = distance
                    break
            
            if too_close:
                filtered_out.append(detection)
            else:
                filtered.append(detection)
        
        return filtered, filtered_out
    
    def start_detection(self):
        """Start the detection with distance filtering"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Distance-Based Filtering - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        logger.info("=== DISTANCE-BASED FILTERING ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'n'/'p' - Next/Previous prompt")
        logger.info("  '+'/'-' - Increase/Decrease min distance")
        logger.info("  'c' - Toggle center points display")
        logger.info("  'd' - Toggle distance lines display")
        logger.info("=" * 50)
        
        # Display options
        self.show_centers = True
        self.show_distance_lines = True
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('+') or key == ord('='):  # Increase min distance
                    self.MIN_DISTANCE += self.distance_step
                    logger.info(f"Min distance: {self.MIN_DISTANCE}")
                elif key == ord('-'):  # Decrease min distance
                    self.MIN_DISTANCE = max(10, self.MIN_DISTANCE - self.distance_step)
                    logger.info(f"Min distance: {self.MIN_DISTANCE}")
                elif key == ord('c'):  # Toggle center points
                    self.show_centers = not self.show_centers
                    logger.info(f"Center points: {'ON' if self.show_centers else 'OFF'}")
                elif key == ord('d'):  # Toggle distance lines
                    self.show_distance_lines = not self.show_distance_lines
                    logger.info(f"Distance lines: {'ON' if self.show_distance_lines else 'OFF'}")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
        
        self.stop_detection()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and distance filtering"""
        processed_frame = frame.copy()
        
        # Apply fisheye correction if enabled
        if Config.FISHEYE_CORRECTION_ENABLED:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")
        
        # Resize for display
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        # Run detection
        try:
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)
            self.area_filtered_detections = self.apply_area_filter(self.raw_detections)
            self.distance_filtered_detections, self.filtered_out_by_distance = self.apply_distance_filter(self.area_filtered_detections)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.distance_filtered_detections = []
            self.filtered_out_by_distance = []
        
        # Draw results
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)
        
        return processed_frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results with distance filtering status"""
        result_frame = frame.copy()

        # Draw filtered out by distance (orange)
        for detection in self.filtered_out_by_distance:
            bbox = detection['bbox']
            center = detection['center']
            distance = detection.get('distance_to_nearest', 0)

            x1, y1, x2, y2 = bbox

            # Orange for too close
            color = (0, 165, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 1)

            # Draw center point
            if self.show_centers:
                cv2.circle(result_frame, center, 5, color, -1)

            # Label
            label = f"TOO_CLOSE: {distance:.0f}px"
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw accepted detections (green)
        for detection in self.distance_filtered_detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            area = detection.get('area', 0)

            x1, y1, x2, y2 = bbox

            # Green for accepted
            color = (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            if self.show_centers:
                cv2.circle(result_frame, center, 8, color, -1)
                cv2.circle(result_frame, center, 8, (255, 255, 255), 2)

            # Label
            label = f"PALLET: {area:.0f}"
            conf_label = f"Conf: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, conf_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw distance lines between accepted detections
        if self.show_distance_lines and len(self.distance_filtered_detections) > 1:
            for i, det1 in enumerate(self.distance_filtered_detections):
                for j, det2 in enumerate(self.distance_filtered_detections[i+1:], i+1):
                    center1 = det1['center']
                    center2 = det2['center']
                    distance = self.calculate_distance(center1, center2)

                    # Draw line
                    cv2.line(result_frame, center1, center2, (255, 255, 0), 1)

                    # Draw distance text at midpoint
                    mid_x = int((center1[0] + center2[0]) / 2)
                    mid_y = int((center1[1] + center2[1]) / 2)
                    cv2.putText(result_frame, f"{distance:.0f}px", (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return result_frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw filtering information overlay"""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"Prompt: '{self.pallet_detector.current_prompt}'", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"Area Filter: {self.MIN_AREA} - {self.MAX_AREA} pixels", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Distance Filter: {self.MIN_DISTANCE} pixels minimum", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"After Area Filter: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"Final Accepted (Green): {len(self.distance_filtered_detections)}", (20, y_offset), font, 0.5, (0, 255, 0), 1)

        y_offset += 20
        cv2.putText(frame, f"Too Close (Orange): {len(self.filtered_out_by_distance)}", (20, y_offset), font, 0.5, (0, 165, 255), 1)

        y_offset += 25
        cv2.putText(frame, "Controls: +/-=Distance, c=Centers, d=Lines, n/p=Prompt", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        return frame

    def stop_detection(self):
        """Stop the detection"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped detection for {self.camera_name}")


def main():
    """Main function"""
    print("DISTANCE-BASED FILTERING FOR PALLET DETECTION")
    print("=" * 50)
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("Area Filter: 10,000 - 100,000 pixels")
    print("Distance Filter: 50 pixels minimum between centers")
    print("=" * 50)

    detector = DistanceBasedFilteringDetector(camera_id=11)

    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    except Exception as e:
        logger.error(f"Error running detector: {e}")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()
