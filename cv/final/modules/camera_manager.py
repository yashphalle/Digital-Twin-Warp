#!/usr/bin/env python3
"""
Camera Manager Module
Hardware interface and camera communication for warehouse tracking system
Extracted from main.py for modular architecture
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class CPUCameraManager:
    """CPU-based camera manager for hardware interface and frame operations"""
    
    def __init__(self, camera_id: int, rtsp_url: str, camera_name: str = None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name or f"Camera {camera_id}"
        
        # Camera connection state
        self.cap = None
        self.connected = False
        self.running = False
        
        # Camera configuration
        self.buffer_size = 1  # Minimal buffer for real-time processing
        self.open_timeout_ms = 5000  # 5 second connection timeout
        self.read_timeout_ms = 5000  # 5 second read timeout
        self.max_frame_reads = 10  # Maximum frames to read for latest frame
        
        # Statistics
        self.total_frames_read = 0
        self.frames_skipped = 0
        self.connection_attempts = 0
        self.last_frame_time = None
        
        logger.info(f"✅ Camera Manager initialized for {self.camera_name}")

    def connect_camera(self) -> bool:
        """Connect to the camera with optimized settings"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name}...")
        self.connection_attempts += 1

        try:
            # Release existing connection if any
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Set aggressive timeout settings to prevent hanging
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False

            # Test frame capture to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                self.cap = None
                return False

            logger.info(f"{self.camera_name} connected successfully")
            self.connected = True
            self.running = True
            return True

        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.connected = False
            return False

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame, skipping buffered frames for real-time processing"""
        if not self.is_connected():
            return False, None
            
        frame = None
        frames_skipped_this_read = 0

        # Read up to max_frame_reads frames to get the latest available
        for _ in range(self.max_frame_reads):
            try:
                ret, latest_frame = self.cap.read()
                if ret and latest_frame is not None:
                    if frame is not None:
                        frames_skipped_this_read += 1
                    frame = latest_frame
                    self.total_frames_read += 1
                else:
                    break  # No more frames available
            except Exception as e:
                logger.warning(f"Error reading frame from {self.camera_name}: {e}")
                break

        if frames_skipped_this_read > 0:
            self.frames_skipped += frames_skipped_this_read
            logger.debug(f"{self.camera_name}: Skipped {frames_skipped_this_read} buffered frames for real-time processing")

        success = frame is not None
        if success:
            import time
            self.last_frame_time = time.time()
            
        return success, frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Simple frame reading without buffer management"""
        if not self.is_connected():
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.total_frames_read += 1
                import time
                self.last_frame_time = time.time()
            return ret, frame
        except Exception as e:
            logger.warning(f"Error reading frame from {self.camera_name}: {e}")
            return False, None

    def is_connected(self) -> bool:
        """Check if camera is connected and operational"""
        return self.connected and self.cap is not None and self.cap.isOpened()

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera properties and capabilities"""
        if not self.is_connected():
            return {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'rtsp_url': self.rtsp_url,
                'connected': False,
                'error': 'Camera not connected'
            }
            
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            buffer_size = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
            
            return {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'rtsp_url': self.rtsp_url,
                'connected': True,
                'resolution': f"{width}x{height}",
                'width': width,
                'height': height,
                'fps': fps,
                'buffer_size': buffer_size,
                'total_frames_read': self.total_frames_read,
                'frames_skipped': self.frames_skipped,
                'connection_attempts': self.connection_attempts,
                'last_frame_time': self.last_frame_time
            }
        except Exception as e:
            logger.error(f"Error getting camera info for {self.camera_name}: {e}")
            return {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'rtsp_url': self.rtsp_url,
                'connected': self.connected,
                'error': str(e)
            }

    def set_camera_properties(self, properties: Dict[str, Any]) -> bool:
        """Configure camera settings"""
        if not self.is_connected():
            logger.warning(f"Cannot set properties for disconnected camera {self.camera_name}")
            return False
            
        try:
            success = True
            for prop_name, value in properties.items():
                if prop_name == 'buffer_size':
                    result = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, value)
                    self.buffer_size = value
                elif prop_name == 'fps':
                    result = self.cap.set(cv2.CAP_PROP_FPS, value)
                elif prop_name == 'width':
                    result = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, value)
                elif prop_name == 'height':
                    result = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
                else:
                    logger.warning(f"Unknown camera property: {prop_name}")
                    continue
                    
                if not result:
                    logger.warning(f"Failed to set {prop_name} to {value} for {self.camera_name}")
                    success = False
                else:
                    logger.debug(f"Set {prop_name} to {value} for {self.camera_name}")
                    
            return success
        except Exception as e:
            logger.error(f"Error setting camera properties for {self.camera_name}: {e}")
            return False

    def reconnect_camera(self) -> bool:
        """Attempt to reconnect the camera"""
        logger.info(f"Attempting to reconnect {self.camera_name}...")
        
        # Cleanup existing connection
        self.cleanup_camera()
        
        # Wait a moment before reconnecting
        import time
        time.sleep(1)
        
        # Attempt reconnection
        return self.connect_camera()

    def cleanup_camera(self):
        """Cleanup camera resources"""
        self.running = False
        self.connected = False
        
        if self.cap:
            try:
                self.cap.release()
                logger.debug(f"Released camera resources for {self.camera_name}")
            except Exception as e:
                logger.warning(f"Error releasing camera {self.camera_name}: {e}")
            finally:
                self.cap = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get camera performance statistics"""
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'connected': self.connected,
            'running': self.running,
            'total_frames_read': self.total_frames_read,
            'frames_skipped': self.frames_skipped,
            'connection_attempts': self.connection_attempts,
            'last_frame_time': self.last_frame_time,
            'skip_ratio': self.frames_skipped / max(self.total_frames_read, 1)
        }

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup_camera()
