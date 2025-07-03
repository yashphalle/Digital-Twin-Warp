#!/usr/bin/env python3
"""
Single Camera Display GUI
Shows feed from Camera 11 in a dedicated GUI window
"""

import cv2
import numpy as np
import threading
import time
import logging
# Removed unused imports
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleCameraDisplay:
    """Single camera display system for Camera 1"""
    
    def __init__(self, camera_id: int = 11):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()
        
        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            # Fallback to config.py settings
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        
        # Display settings - MAINTAIN ORIGINAL CAMERA QUALITY
        self.window_name = f"WARP - {self.camera_name}"
        # Use original camera resolution (4K) - no quality reduction
        self.display_width = Config.RTSP_FRAME_WIDTH   # 3840 (4K width)
        self.display_height = Config.RTSP_FRAME_HEIGHT # 2160 (4K height)
        self.maintain_original_quality = True
        
        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False
        self.display_thread = None
        
        # Frame processing
        self.current_frame = None
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Fisheye correction
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.use_fisheye_correction = Config.FISHEYE_CORRECTION_ENABLED
        
        # Display options
        self.show_info_overlay = True
        self.show_grid = False
        self.fullscreen = False
        
        logger.info(f"Initialized display for {self.camera_name}")
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

            # Set timeout for connection
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

    def disconnect_camera(self):
        """Disconnect from the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        logger.info(f"Disconnected from {self.camera_name}")
    
    def start_display(self):
        """Start the camera display"""
        if self.running:
            logger.warning("Display already running")
            return

        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name=f"Camera{self.camera_id}Display",
            daemon=True
        )
        self.display_thread.start()

        logger.info(f"Started display for {self.camera_name}")
    
    def stop_display(self):
        """Stop the camera display"""
        self.running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        self.disconnect_camera()
        cv2.destroyAllWindows()
        
        logger.info(f"Stopped display for {self.camera_name}")
    
    def _display_loop(self):
        """Main display loop"""
        # Create window with NORMAL flag to allow resizing while maintaining quality
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Set initial window size (will scale 4K content to fit screen)
        # User can resize window as needed - OpenCV will handle scaling
        initial_width = min(1920, self.display_width)  # Max 1920 for initial display
        initial_height = min(1080, self.display_height)  # Max 1080 for initial display
        cv2.resizeWindow(self.window_name, initial_width, initial_height)
        
        logger.info(f"Display window created: {self.window_name}")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'f' - Toggle fullscreen")
        logger.info("  'i' - Toggle info overlay")
        logger.info("  'g' - Toggle grid")
        logger.info("  'c' - Toggle fisheye correction")
        
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
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    self._toggle_fullscreen()
                elif key == ord('i'):  # Toggle info overlay
                    self.show_info_overlay = not self.show_info_overlay
                    logger.info(f"Info overlay: {'ON' if self.show_info_overlay else 'OFF'}")
                elif key == ord('g'):  # Toggle grid
                    self.show_grid = not self.show_grid
                    logger.info(f"Grid overlay: {'ON' if self.show_grid else 'OFF'}")
                elif key == ord('c'):  # Toggle fisheye correction
                    self.use_fisheye_correction = not self.use_fisheye_correction
                    logger.info(f"Fisheye correction: {'ON' if self.use_fisheye_correction else 'OFF'}")
                
                # Update FPS
                self._update_fps()
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                break
        
        self.running = False

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the camera frame - MAINTAINING ORIGINAL QUALITY"""
        processed_frame = frame.copy()

        # Apply fisheye correction if enabled (maintains original resolution)
        if self.use_fisheye_correction:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")

        # NO RESIZING - Keep original camera quality (4K resolution)
        # The window will automatically scale for display while preserving quality

        # Add grid overlay if enabled
        if self.show_grid:
            processed_frame = self._draw_grid(processed_frame)

        # Add info overlay if enabled
        if self.show_info_overlay:
            processed_frame = self._draw_info_overlay(processed_frame)

        return processed_frame

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw grid overlay on frame"""
        height, width = frame.shape[:2]

        # Grid parameters
        grid_color = (0, 255, 0)  # Green
        line_thickness = 1

        # Draw vertical lines
        for i in range(1, 10):
            x = int(width * i / 10)
            cv2.line(frame, (x, 0), (x, height), grid_color, line_thickness)

        # Draw horizontal lines
        for i in range(1, 10):
            y = int(height * i / 10)
            cv2.line(frame, (0, y), (width, y), grid_color, line_thickness)

        return frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]

        # Overlay parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)  # Yellow
        thickness = 2

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Camera info
        y_offset = 30
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        status = "Connected" if self.connected else "Disconnected"
        status_color = (0, 255, 0) if self.connected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, y_offset), font, font_scale, status_color, thickness)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 200, height - 20), font, 0.5, color, 1)

        return frame

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen

        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            logger.info("Fullscreen mode ON")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            logger.info("Windowed mode ON")

    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time

    def _reconnect_camera(self) -> bool:
        """Attempt to reconnect to camera"""
        logger.info("Attempting camera reconnection...")

        self.disconnect_camera()
        time.sleep(2)  # Wait before reconnecting

        return self.connect_camera()

    def is_running(self) -> bool:
        """Check if display is running"""
        return self.running


def main():
    """Main function to run the single camera display"""
    print("SINGLE CAMERA DISPLAY - CAMERA 1")
    print("=" * 50)
    print("This will show Camera 1 feed in a dedicated window")
    print("Controls:")
    print("  'q' or ESC - Quit")
    print("  'f' - Toggle fullscreen")
    print("  'i' - Toggle info overlay")
    print("  'g' - Toggle grid")
    print("  'c' - Toggle fisheye correction")
    print("=" * 50)

    # Create display system for camera 1
    camera_display = SingleCameraDisplay(camera_id=11)

    try:
        # Start display
        camera_display.start_display()

        # Keep main thread alive
        while camera_display.is_running():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera_display.stop_display()


if __name__ == "__main__":
    main()
