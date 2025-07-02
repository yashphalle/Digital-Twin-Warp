#!/usr/bin/env python3
"""
Area-Based Filtering for Pallet Detection
Filters detections based on area size and shows filtered objects
"""

import cv2
import numpy as np
import logging
import sys
import os
from typing import List, Dict
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AreaBasedFilteringDetector:
    """Pallet detection with area-based filtering"""
    
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
        
        # AREA FILTERING SETTINGS
        self.MIN_AREA = 10000   # Exclude very small noise
        self.MAX_AREA = 100000  # Exclude very large background objects
        
        # Detection results
        self.raw_detections = []
        self.filtered_detections = []
        self.filtered_out_detections = []
        
        logger.info(f"Initialized area-based filtering for {self.camera_name}")
        logger.info(f"Area filter: {self.MIN_AREA} - {self.MAX_AREA} pixels")
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
    
    def apply_area_filter(self, detections: List[Dict]) -> tuple:
        """Apply area-based filtering and return filtered and rejected detections"""
        filtered = []
        filtered_out = []
        
        for detection in detections:
            area = detection.get('area', 0)
            
            if self.MIN_AREA <= area <= self.MAX_AREA:
                filtered.append(detection)
            else:
                # Mark why it was filtered out
                if area < self.MIN_AREA:
                    detection['filter_reason'] = 'TOO_SMALL'
                else:
                    detection['filter_reason'] = 'TOO_LARGE'
                filtered_out.append(detection)
        
        return filtered, filtered_out
    
    def start_detection(self):
        """Start the detection with area filtering"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Area-Based Filtering - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        logger.info("=== AREA-BASED FILTERING ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'n'/'p' - Next/Previous prompt")
        logger.info("  '+'/'-' - Increase/Decrease max area")
        logger.info("  'a'/'z' - Increase/Decrease min area")
        logger.info("=" * 50)
        
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
                elif key == ord('+') or key == ord('='):  # Increase max area
                    self.MAX_AREA += 5000
                    logger.info(f"Max area: {self.MAX_AREA}")
                elif key == ord('-'):  # Decrease max area
                    self.MAX_AREA = max(self.MIN_AREA + 5000, self.MAX_AREA - 5000)
                    logger.info(f"Max area: {self.MAX_AREA}")
                elif key == ord('a'):  # Increase min area
                    self.MIN_AREA += 1000
                    if self.MIN_AREA >= self.MAX_AREA:
                        self.MIN_AREA = self.MAX_AREA - 5000
                    logger.info(f"Min area: {self.MIN_AREA}")
                elif key == ord('z'):  # Decrease min area
                    self.MIN_AREA = max(1000, self.MIN_AREA - 1000)
                    logger.info(f"Min area: {self.MIN_AREA}")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
        
        self.stop_detection()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and area filtering"""
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
            self.filtered_detections, self.filtered_out_detections = self.apply_area_filter(self.raw_detections)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.raw_detections = []
            self.filtered_detections = []
            self.filtered_out_detections = []
        
        # Draw results
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)
        
        return processed_frame
    
    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results with filtering status"""
        result_frame = frame.copy()
        
        # Draw filtered out detections (rejected)
        for detection in self.filtered_out_detections:
            bbox = detection['bbox']
            area = detection.get('area', 0)
            confidence = detection['confidence']
            reason = detection.get('filter_reason', 'UNKNOWN')
            
            x1, y1, x2, y2 = bbox
            
            # Color based on filter reason
            if reason == 'TOO_SMALL':
                color = (0, 0, 255)  # Red for too small
                label = f"SMALL: {area:.0f}"
            else:  # TOO_LARGE
                color = (255, 0, 255)  # Magenta for too large
                label = f"LARGE: {area:.0f}"
            
            # Draw thin dashed-like border for filtered out
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 1)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw accepted detections (passed filter)
        for detection in self.filtered_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            area = detection.get('area', 0)
            
            x1, y1, x2, y2 = bbox
            
            # Green for accepted detections
            color = (0, 255, 0)
            
            # Draw thick border for accepted
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with area and confidence
            label = f"PALLET: {area:.0f}"
            conf_label = f"Conf: {confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1-40), (x1+max(label_size[0], 120), y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, conf_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw filtering information overlay"""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2
        
        y_offset = 30
        cv2.putText(frame, f"Prompt: '{self.pallet_detector.current_prompt}'", (20, y_offset), font, 0.5, color, 1)
        
        y_offset += 25
        cv2.putText(frame, f"Area Filter: {self.MIN_AREA} - {self.MAX_AREA} pixels", (20, y_offset), font, font_scale, color, thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Accepted (Green): {len(self.filtered_detections)}", (20, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        y_offset += 20
        too_small = len([d for d in self.filtered_out_detections if d.get('filter_reason') == 'TOO_SMALL'])
        too_large = len([d for d in self.filtered_out_detections if d.get('filter_reason') == 'TOO_LARGE'])
        cv2.putText(frame, f"Too Small (Red): {too_small}", (20, y_offset), font, font_scale, (0, 0, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Too Large (Magenta): {too_large}", (20, y_offset), font, font_scale, (255, 0, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, "Controls: n/p=Prompt, +/-=MaxArea, a/z=MinArea", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
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
    print("AREA-BASED FILTERING FOR PALLET DETECTION")
    print("=" * 50)
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("Area Filter: 10,000 - 100,000 pixels")
    print("=" * 50)
    
    detector = AreaBasedFilteringDetector(camera_id=11)
    
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
