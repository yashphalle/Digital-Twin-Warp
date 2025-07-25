#!/usr/bin/env python3
"""
Optimized Pallet Detection Script
Uses the best settings found through interactive tuning
Specifically optimized for wooden skids/pallets on floor
"""

import cv2
import numpy as np
import logging
import time
import sys
import os
from typing import List, Dict, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedPalletDetector:
    """
    Optimized pallet detector with best settings for wooden skids on floor
    Based on interactive tuning results
    """
    
    def __init__(self, camera_id: int = 1):
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
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Detection components
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = SimplePalletDetector()
        
        # OPTIMIZED SETTINGS FOR WOODEN SKIDS/PALLETS
        self._apply_optimized_settings()
        
        # Detection settings
        self.detection_interval = 1  # Process every frame for better detection
        self.detection_counter = 0
        
        # Display options
        self.use_fisheye_correction = Config.FISHEYE_CORRECTION_ENABLED
        
        logger.info(f"Initialized optimized pallet detector for {self.camera_name}")
        logger.info(f"Optimized for wooden skids/pallets on floor")
    
    def _apply_optimized_settings(self):
        """Apply optimized settings based on tuning results"""
        # Best prompt for wooden skids on floor
        self.pallet_detector.set_custom_prompt("wooden skid on floor")
        
        # Lower confidence threshold to catch 10-15% confidence detections
        self.pallet_detector.confidence_threshold = 0.12
        
        # Update the underlying DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.12
            self.pallet_detector.grounding_dino.prompt = "wooden skid on floor"
        
        logger.info("Applied optimized settings:")
        logger.info(f"  - Prompt: 'wooden skid on floor'")
        logger.info(f"  - Confidence threshold: 0.12")
        logger.info(f"  - Detection interval: every frame")
    
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
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"{self.camera_name} connected successfully")
            logger.info(f"Stream: {width}x{height} @ {fps:.1f}fps")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def start_detection(self):
        """Start the optimized detection"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        logger.info(f"Started optimized pallet detection for {self.camera_name}")
        
        # Create display window
        window_name = f"Optimized Pallet Detection - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  's' - Save detection results")
        logger.info("  'i' - Toggle info display")
        
        show_info = True
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Add info overlay if enabled
                if show_info:
                    processed_frame = self._draw_detection_info(processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save results
                    self._save_detection_results()
                elif key == ord('i'):  # Toggle info
                    show_info = not show_info
                    logger.info(f"Info display: {'ON' if show_info else 'OFF'}")
                
                # Update FPS
                self._update_fps()
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
        
        self.stop_detection()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with optimized detection"""
        processed_frame = frame.copy()
        
        # Apply fisheye correction if enabled
        if self.use_fisheye_correction:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")
        
        # Resize for display (maintain aspect ratio)
        height, width = processed_frame.shape[:2]
        if width > 1600:  # Scale down for display
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        # Run optimized pallet detection
        self.detection_counter += 1
        if self.detection_counter >= self.detection_interval:
            self.detection_counter = 0
            try:
                detections = self.pallet_detector.detect_pallets(processed_frame)
                logger.debug(f"Detected {len(detections)} pallets")
            except Exception as e:
                logger.error(f"Detection failed: {e}")
        
        # Draw detections
        processed_frame = self.pallet_detector.draw_detections(processed_frame)
        
        return processed_frame
    
    def _draw_detection_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection information overlay"""
        height, width = frame.shape[:2]
        
        # Overlay parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)  # Yellow
        thickness = 2
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Detection info
        y_offset = 30
        cv2.putText(frame, f"Optimized for: Wooden Skids on Floor", (20, y_offset), font, 0.5, color, 1)
        
        y_offset += 25
        cv2.putText(frame, f"Confidence Threshold: {self.pallet_detector.confidence_threshold:.2f}", (20, y_offset), font, 0.5, color, 1)
        
        y_offset += 25
        pallet_count = len(self.pallet_detector.last_detections)
        cv2.putText(frame, f"Pallets Detected: {pallet_count}", (20, y_offset), font, 0.5, (0, 255, 0), 1)
        
        y_offset += 25
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset), font, 0.5, color, 1)
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def _save_detection_results(self):
        """Save current detection results"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"optimized_pallet_detection_results_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(f"Optimized Pallet Detection Results\n")
                f.write(f"==================================\n")
                f.write(f"Camera: {self.camera_name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Prompt: {self.pallet_detector.current_prompt}\n")
                f.write(f"Confidence Threshold: {self.pallet_detector.confidence_threshold}\n")
                f.write(f"Detections: {len(self.pallet_detector.last_detections)}\n\n")
                
                for i, detection in enumerate(self.pallet_detector.last_detections):
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Bbox: {detection['bbox']}\n")
                    f.write(f"  Confidence: {detection['confidence']:.3f}\n")
                    f.write(f"  Area: {detection['area']:.0f}\n\n")
            
            logger.info(f"Detection results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
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
    print("OPTIMIZED PALLET DETECTION")
    print("=" * 50)
    print("Optimized for wooden skids/pallets on floor")
    print("Uses best settings from interactive tuning")
    print("=" * 50)
    
    detector = OptimizedPalletDetector(camera_id=1)
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()
