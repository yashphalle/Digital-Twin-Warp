# cv/GPU/pipelines/ring_buffer.py

import numpy as np
import threading
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class RingBuffer:
    """
    Lock-free ring buffer for multi-camera frame storage
    Each camera has its own circular buffer to avoid contention
    """
    
    def __init__(self, num_cameras: int = 11, buffer_size: int = 30):
        """
        Initialize ring buffer
        
        Args:
            num_cameras: Number of cameras (default 11)
            buffer_size: Frames per camera buffer (default 30 = 2 sec at 15 FPS)
        """
        self.num_cameras = num_cameras
        self.buffer_size = buffer_size
        
        # Pre-allocate buffers for each camera
        # Using object dtype to store frame references (not copying frame data)
        self.frame_buffers = {}
        self.timestamps = {}
        self.write_indices = {}
        self.frame_numbers = {}
        
        # Initialize for cameras 1-11
        for cam_id in range(1, num_cameras + 1):
            self.frame_buffers[cam_id] = [None] * buffer_size
            self.timestamps[cam_id] = [0.0] * buffer_size
            self.write_indices[cam_id] = 0
            self.frame_numbers[cam_id] = [0] * buffer_size
            
        logger.info(f"RingBuffer initialized: {num_cameras} cameras, {buffer_size} frames each")
        
    def write(self, camera_id: int, frame: np.ndarray, frame_number: int = None) -> bool:
        """
        Write frame to camera's buffer (never blocks)
        
        Args:
            camera_id: Camera ID (1-11)
            frame: Frame data
            frame_number: Optional frame sequence number
            
        Returns:
            True if written successfully
        """
        if camera_id not in self.frame_buffers:
            logger.error(f"Camera {camera_id} not initialized in buffer")
            return False
            
        # Calculate write position
        idx = self.write_indices[camera_id] % self.buffer_size
        
        # Write frame data (overwrites oldest if full)
        self.frame_buffers[camera_id][idx] = frame
        self.timestamps[camera_id][idx] = time.time()
        self.frame_numbers[camera_id][idx] = frame_number or self.write_indices[camera_id]
        
        # Increment write index (atomic in Python)
        self.write_indices[camera_id] += 1
        
        return True
    
    def get_latest(self, camera_id: int) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get latest frame from specific camera
        
        Returns:
            Tuple of (frame, timestamp, frame_number) or None
        """
        if camera_id not in self.frame_buffers:
            return None
            
        write_idx = self.write_indices[camera_id]
        if write_idx == 0:
            return None  # No frames written yet
            
        # Get most recent frame
        idx = (write_idx - 1) % self.buffer_size
        frame = self.frame_buffers[camera_id][idx]
        timestamp = self.timestamps[camera_id][idx]
        frame_num = self.frame_numbers[camera_id][idx]
        
        if frame is None:
            return None
            
        return frame, timestamp, frame_num
    
    def get_batch(self, max_age: float = 0.2) -> Dict[int, np.ndarray]:
        """
        Get latest frame from each camera for batch processing
        
        Args:
            max_age: Maximum age in seconds (frames older are ignored)
            
        Returns:
            Dictionary of camera_id -> frame
        """
        batch = {}
        current_time = time.time()
        
        for cam_id in range(1, self.num_cameras + 1):
            result = self.get_latest(cam_id)
            if result is not None:
                frame, timestamp, _ = result
                
                # Check if frame is recent enough
                age = current_time - timestamp
                if age <= max_age:
                    batch[cam_id] = frame
                else:
                    logger.debug(f"Camera {cam_id} frame too old: {age:.3f}s")
                    
        return batch
    
    def get_buffer_status(self) -> Dict[int, Dict]:
        """Get status of all buffers for monitoring"""
        status = {}
        current_time = time.time()
        
        for cam_id in range(1, self.num_cameras + 1):
            write_idx = self.write_indices[cam_id]
            
            # Get latest frame info
            latest_info = None
            if write_idx > 0:
                idx = (write_idx - 1) % self.buffer_size
                timestamp = self.timestamps[cam_id][idx]
                age = current_time - timestamp
                latest_info = {
                    'timestamp': timestamp,
                    'age': age,
                    'frame_number': self.frame_numbers[cam_id][idx]
                }
            
            status[cam_id] = {
                'frames_written': write_idx,
                'buffer_utilization': min(write_idx, self.buffer_size) / self.buffer_size,
                'latest_frame': latest_info
            }
            
        return status