#!/usr/bin/env python3
"""
Pallet Detection with Advanced Filtering
Interactive filtering to reduce false positives
"""

import cv2
import numpy as np
import logging
import time
import sys
import os
from typing import List, Dict, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PalletDetectionWithFiltering:
    """Pallet detection with advanced filtering options"""
    
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
        
        # FILTERING OPTIONS
        # 1. Size-based filtering
        self.size_filtering_enabled = True
        self.min_area = 5000  # Minimum area in pixels
        self.max_area = 200000  # Maximum area in pixels
        self.area_step = 1000  # Step size for adjustment
        
        # 2. Edge detection filtering
        self.edge_filtering_enabled = True
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        self.min_edge_density = 0.1  # Minimum edge density (0.0 to 1.0)
        self.edge_step = 10
        
        # 3. Color-based filtering
        self.color_filtering_enabled = True
        self.brown_tolerance = 30  # HSV tolerance for brown detection
        self.white_tolerance = 30  # HSV tolerance for white detection
        self.min_color_percentage = 0.2  # Minimum percentage of brown/white pixels
        self.color_step = 5
        
        # Detection results
        self.raw_detections = []
        self.filtered_detections = []
        
        logger.info(f"Initialized filtering detector for {self.camera_name}")
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
    
    def apply_size_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply size-based filtering"""
        if not self.size_filtering_enabled:
            return detections
        
        filtered = []
        for detection in detections:
            area = detection.get('area', 0)
            if self.min_area <= area <= self.max_area:
                filtered.append(detection)
        
        return filtered
    
    def apply_edge_filter(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply edge detection filtering"""
        if not self.edge_filtering_enabled:
            return detections
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region of interest
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Apply Canny edge detection
            edges = cv2.Canny(roi, self.edge_threshold_low, self.edge_threshold_high)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = roi.shape[0] * roi.shape[1]
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
            
            if edge_density >= self.min_edge_density:
                detection['edge_density'] = edge_density
                filtered.append(detection)
        
        return filtered
    
    def apply_color_filter(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply color-based filtering for brown and white"""
        if not self.color_filtering_enabled:
            return detections
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define brown and white color ranges in HSV
        # Brown: Hue 10-20, Saturation 50-255, Value 20-200
        brown_lower = np.array([10 - self.brown_tolerance//3, 50, 20])
        brown_upper = np.array([20 + self.brown_tolerance//3, 255, 200])
        
        # White: Low saturation, high value
        white_lower = np.array([0, 0, 255 - self.white_tolerance])
        white_upper = np.array([180, self.white_tolerance, 255])
        
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region of interest
            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                continue
            
            # Create masks for brown and white
            brown_mask = cv2.inRange(roi_hsv, brown_lower, brown_upper)
            white_mask = cv2.inRange(roi_hsv, white_lower, white_upper)
            
            # Calculate color percentages
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            brown_pixels = np.sum(brown_mask > 0)
            white_pixels = np.sum(white_mask > 0)
            
            brown_percentage = brown_pixels / total_pixels if total_pixels > 0 else 0
            white_percentage = white_pixels / total_pixels if total_pixels > 0 else 0
            
            # Check if detection has enough brown or white color
            if (brown_percentage >= self.min_color_percentage or 
                white_percentage >= self.min_color_percentage):
                detection['brown_percentage'] = brown_percentage
                detection['white_percentage'] = white_percentage
                filtered.append(detection)
        
        return filtered
    
    def apply_all_filters(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply all enabled filters in sequence"""
        filtered = detections.copy()
        
        # Apply filters in order
        filtered = self.apply_size_filter(filtered)
        filtered = self.apply_edge_filter(filtered, frame)
        filtered = self.apply_color_filter(filtered, frame)
        
        return filtered
    
    def start_detection(self):
        """Start the detection with filtering"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Pallet Detection with Filtering - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        # Ensure window is focused for key input
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
        
        logger.info("=== PALLET DETECTION WITH FILTERING ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'n'/'p' - Next/Previous prompt")
        logger.info("  '1' - Toggle size filtering")
        logger.info("  '2' - Toggle edge filtering") 
        logger.info("  '3' - Toggle color filtering")
        logger.info("  'a'/'z' - Increase/Decrease min area")
        logger.info("  's'/'x' - Increase/Decrease max area")
        logger.info("  'd'/'c' - Increase/Decrease edge threshold")
        logger.info("  'f'/'v' - Increase/Decrease color tolerance")
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
                key = cv2.waitKey(30) & 0xFF  # Increased wait time for better key detection
                if key != 255:  # Key was pressed
                    logger.info(f"Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'special'})")

                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                elif key == ord('1'):  # Toggle size filtering
                    self.size_filtering_enabled = not self.size_filtering_enabled
                    logger.info(f"Size filtering: {'ON' if self.size_filtering_enabled else 'OFF'}")
                elif key == ord('2'):  # Toggle edge filtering
                    self.edge_filtering_enabled = not self.edge_filtering_enabled
                    logger.info(f"Edge filtering: {'ON' if self.edge_filtering_enabled else 'OFF'}")
                elif key == ord('3'):  # Toggle color filtering
                    self.color_filtering_enabled = not self.color_filtering_enabled
                    logger.info(f"Color filtering: {'ON' if self.color_filtering_enabled else 'OFF'}")
                elif key == ord('a'):  # Increase min area
                    self.min_area += self.area_step
                    logger.info(f"Min area: {self.min_area}")
                elif key == ord('z'):  # Decrease min area
                    self.min_area = max(100, self.min_area - self.area_step)
                    logger.info(f"Min area: {self.min_area}")
                elif key == ord('s'):  # Increase max area
                    self.max_area += self.area_step
                    logger.info(f"Max area: {self.max_area}")
                elif key == ord('x'):  # Decrease max area
                    self.max_area = max(self.min_area + 1000, self.max_area - self.area_step)
                    logger.info(f"Max area: {self.max_area}")
                elif key == ord('d'):  # Increase edge threshold
                    self.edge_threshold_high = min(255, self.edge_threshold_high + self.edge_step)
                    logger.info(f"Edge threshold: {self.edge_threshold_high}")
                elif key == ord('c'):  # Decrease edge threshold
                    self.edge_threshold_high = max(50, self.edge_threshold_high - self.edge_step)
                    logger.info(f"Edge threshold: {self.edge_threshold_high}")
                elif key == ord('f'):  # Increase color tolerance
                    self.brown_tolerance = min(50, self.brown_tolerance + self.color_step)
                    self.white_tolerance = min(50, self.white_tolerance + self.color_step)
                    logger.info(f"Color tolerance: {self.brown_tolerance}")
                elif key == ord('v'):  # Decrease color tolerance
                    self.brown_tolerance = max(10, self.brown_tolerance - self.color_step)
                    self.white_tolerance = max(10, self.white_tolerance - self.color_step)
                    logger.info(f"Color tolerance: {self.brown_tolerance}")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
        
        self.stop_detection()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and filtering"""
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
            self.filtered_detections = self.apply_all_filters(self.raw_detections, processed_frame)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.raw_detections = []
            self.filtered_detections = []
        
        # Draw results
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)
        
        return processed_frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results with filtering status"""
        result_frame = frame.copy()

        # Draw raw detections in red (filtered out)
        for detection in self.raw_detections:
            if detection not in self.filtered_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for filtered out
                cv2.putText(result_frame, "FILTERED", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw filtered detections in green (passed filters)
        for detection in self.filtered_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox

            # Green for passed filters
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence and filter info
            label = f"Pallet: {confidence:.2f}"
            if 'edge_density' in detection:
                label += f" E:{detection['edge_density']:.2f}"
            if 'brown_percentage' in detection:
                label += f" B:{detection['brown_percentage']:.2f}"
            if 'white_percentage' in detection:
                label += f" W:{detection['white_percentage']:.2f}"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result_frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw filtering information overlay"""
        height, width = frame.shape[:2]

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 255)
        thickness = 1

        y_offset = 30
        cv2.putText(frame, f"Prompt: '{self.pallet_detector.current_prompt}'", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 20
        cv2.putText(frame, f"Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, font_scale, (0, 0, 255), thickness)

        y_offset += 20
        cv2.putText(frame, f"Filtered Detections: {len(self.filtered_detections)}", (20, y_offset), font, font_scale, (0, 255, 0), thickness)

        y_offset += 25
        size_status = "ON" if self.size_filtering_enabled else "OFF"
        cv2.putText(frame, f"1. Size Filter: {size_status} ({self.min_area}-{self.max_area})", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 20
        edge_status = "ON" if self.edge_filtering_enabled else "OFF"
        cv2.putText(frame, f"2. Edge Filter: {edge_status} (thresh:{self.edge_threshold_high})", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 20
        color_status = "ON" if self.color_filtering_enabled else "OFF"
        cv2.putText(frame, f"3. Color Filter: {color_status} (tol:{self.brown_tolerance})", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, "Controls: 1/2/3=Toggle filters, a/z=MinArea, s/x=MaxArea", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        y_offset += 15
        cv2.putText(frame, "d/c=EdgeThresh, f/v=ColorTol, n/p=Prompt", (20, y_offset), font, 0.4, (255, 255, 255), 1)

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
    print("PALLET DETECTION WITH ADVANCED FILTERING")
    print("=" * 50)
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("=" * 50)

    detector = PalletDetectionWithFiltering(camera_id=11)

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
