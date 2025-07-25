#!/usr/bin/env python3
"""
Interactive Pallet Detection Tuning
Real-time prompt and threshold experimentation with Grounding DINO
"""

import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractivePalletTuning:
    """Interactive pallet detection tuning interface"""
    
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
        
        # Display settings
        self.window_name = f"Pallet Detection Tuning - {self.camera_name}"
        
        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False
        self.display_thread = None
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Detection components
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = SimplePalletDetector()
        
        # Detection settings
        self.enable_detection = True
        self.detection_interval = 2  # Run detection every N frames
        self.detection_counter = 0
        
        # Display options
        self.show_info_overlay = True
        self.use_fisheye_correction = Config.FISHEYE_CORRECTION_ENABLED
        self.fullscreen = False
        
        logger.info(f"Initialized interactive tuning for {self.camera_name}")
        logger.info(f"RTSP URL: {self.rtsp_url}")
    
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
    
    def start_tuning(self):
        """Start the interactive tuning interface"""
        if self.running:
            logger.warning("Tuning already running")
            return
        
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        self.display_thread = threading.Thread(
            target=self._tuning_loop,
            name=f"Camera{self.camera_id}Tuning",
            daemon=True
        )
        self.display_thread.start()
        
        logger.info(f"Started interactive tuning for {self.camera_name}")
    
    def _tuning_loop(self):
        """Main tuning loop"""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)
        
        logger.info(f"Tuning interface created: {self.window_name}")
        logger.info("=== INTERACTIVE PALLET DETECTION TUNING ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit and save settings")
        logger.info("  'n' - Next prompt")
        logger.info("  'p' - Previous prompt") 
        logger.info("  '+' - Increase confidence threshold")
        logger.info("  '-' - Decrease confidence threshold")
        logger.info("  'd' - Toggle detection on/off")
        logger.info("  'i' - Toggle info overlay")
        logger.info("  'c' - Toggle fisheye correction")
        logger.info("  's' - Save current settings")
        logger.info("=" * 50)
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame, attempting reconnection...")
                    if not self._reconnect_camera():
                        break
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow(self.window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    # Save settings before quitting
                    self.pallet_detector.save_settings()
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                elif key == ord('+') or key == ord('='):  # Increase threshold
                    self.pallet_detector.increase_threshold()
                elif key == ord('-') or key == ord('_'):  # Decrease threshold
                    self.pallet_detector.decrease_threshold()
                elif key == ord('d'):  # Toggle detection
                    self.enable_detection = not self.enable_detection
                    logger.info(f"Detection: {'ON' if self.enable_detection else 'OFF'}")
                elif key == ord('i'):  # Toggle info overlay
                    self.show_info_overlay = not self.show_info_overlay
                    logger.info(f"Info overlay: {'ON' if self.show_info_overlay else 'OFF'}")
                elif key == ord('c'):  # Toggle fisheye correction
                    self.use_fisheye_correction = not self.use_fisheye_correction
                    logger.info(f"Fisheye correction: {'ON' if self.use_fisheye_correction else 'OFF'}")
                elif key == ord('s'):  # Save settings
                    self.pallet_detector.save_settings()
                    logger.info("Settings saved!")
                
                # Update FPS
                self._update_fps()
                
            except Exception as e:
                logger.error(f"Error in tuning loop: {e}")
                break
        
        self.running = False
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and overlays"""
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
        
        # Run pallet detection
        if self.enable_detection:
            self.detection_counter += 1
            if self.detection_counter >= self.detection_interval:
                self.detection_counter = 0
                try:
                    self.pallet_detector.detect_pallets(processed_frame)
                except Exception as e:
                    logger.error(f"Detection failed: {e}")
        
        # Draw detections
        processed_frame = self.pallet_detector.draw_detections(processed_frame)
        
        # Add info overlay
        if self.show_info_overlay:
            processed_frame = self.pallet_detector.draw_info_overlay(processed_frame)
            processed_frame = self._draw_camera_info(processed_frame)
        
        return processed_frame
    
    def _draw_camera_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw camera information"""
        height, width = frame.shape[:2]
        
        # Camera info in bottom right
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width-250, height-80), (width-10, height-10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Text
        cv2.putText(frame, f"Camera: {self.camera_name}", (width-240, height-60), font, font_scale, color, thickness)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width-240, height-40), font, font_scale, color, thickness)
        cv2.putText(frame, f"Frame: {self.frame_count}", (width-240, height-20), font, font_scale, color, thickness)
        
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
    
    def _reconnect_camera(self) -> bool:
        """Attempt to reconnect to camera"""
        logger.info("Attempting camera reconnection...")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        time.sleep(2)
        
        return self.connect_camera()
    
    def stop_tuning(self):
        """Stop the tuning interface"""
        self.running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()
        
        logger.info(f"Stopped tuning for {self.camera_name}")
    
    def is_running(self) -> bool:
        """Check if tuning is running"""
        return self.running


def main():
    """Main function"""
    print("INTERACTIVE PALLET DETECTION TUNING")
    print("=" * 50)
    print("Experiment with prompts and thresholds in real-time")
    print("Find optimal settings for your warehouse environment")
    print("=" * 50)
    
    tuning_interface = InteractivePalletTuning(camera_id=11)
    
    try:
        tuning_interface.start_tuning()
        
        while tuning_interface.is_running():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down tuning interface...")
    finally:
        tuning_interface.stop_tuning()


if __name__ == "__main__":
    main()
