import numpy as np
import threading
import time
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# --- ORIGINAL RING BUFFER (UNCHANGED) ---

class RingBuffer:
    """
    Lock-free ring buffer for multi-camera frame storage.
    Each camera has its own circular buffer to avoid contention.
    """
    
    def __init__(self, num_cameras: int = 11, buffer_size: int = 30):
        """
        Initialize ring buffer.
        
        Args:
            num_cameras: Number of cameras (default 11)
            buffer_size: Frames per camera buffer (default 30 = 2 sec at 15 FPS)
        """
        self.num_cameras = num_cameras
        self.buffer_size = buffer_size
        
        # Pre-allocate buffers for each camera
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
        Write frame to camera's buffer (never blocks).
        """
        if camera_id not in self.frame_buffers:
            logger.error(f"Camera {camera_id} not initialized in buffer")
            return False
            
        idx = self.write_indices[camera_id] % self.buffer_size
        
        self.frame_buffers[camera_id][idx] = frame
        self.timestamps[camera_id][idx] = time.time()
        self.frame_numbers[camera_id][idx] = frame_number or self.write_indices[camera_id]
        
        self.write_indices[camera_id] += 1
        
        return True
    
    def get_latest(self, camera_id: int) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get latest frame from specific camera.
        """
        if camera_id not in self.frame_buffers:
            return None
            
        write_idx = self.write_indices[camera_id]
        if write_idx == 0:
            return None  # No frames written yet
            
        idx = (write_idx - 1) % self.buffer_size
        frame = self.frame_buffers[camera_id][idx]
        timestamp = self.timestamps[camera_id][idx]
        frame_num = self.frame_numbers[camera_id][idx]
        
        if frame is None:
            return None
            
        return frame, timestamp, frame_num
    
    def get_batch(self, max_age: float = 0.2) -> Dict[int, np.ndarray]:
        """
        Get latest frame from each camera for batch processing.
        """
        batch = {}
        current_time = time.time()
        
        for cam_id in range(1, self.num_cameras + 1):
            result = self.get_latest(cam_id)
            if result is not None:
                frame, timestamp, _ = result
                
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

# --- NEW BATCHED RING BUFFER ---

class BatchedRingBuffer:
    """
    MODIFIED: Lock-free ring buffer that supports collecting batches of frames.
    Each camera has its own circular buffer and a separate read pointer to track
    which frames have been processed.
    """

    def __init__(self, num_cameras: int = 11, buffer_size: int = 60):
        """
        Initialize ring buffer.

        Args:
            num_cameras: Number of cameras (default 11)
            buffer_size: Frames per camera buffer (increased default to 60 for batching)
        """
        self.num_cameras = num_cameras
        self.buffer_size = buffer_size

        # Pre-allocate buffers for each camera
        self.frame_buffers = {}
        self.timestamps = {}
        self.write_indices = {}
        self.frame_numbers = {}
        
        # --- NEW: Add a read index for each camera ---
        self.read_indices = {}
        # --- END NEW ---

        # Initialize for cameras 1-11
        for cam_id in range(1, num_cameras + 1):
            self.frame_buffers[cam_id] = [None] * buffer_size
            self.timestamps[cam_id] = [0.0] * buffer_size
            self.write_indices[cam_id] = 0
            self.frame_numbers[cam_id] = [0] * buffer_size
            self.read_indices[cam_id] = 0  # Start reading from the beginning

        logger.info(f"BatchedRingBuffer initialized: {num_cameras} cameras, {buffer_size} frames each")

    def write(self, camera_id: int, frame: np.ndarray, frame_number: int = None) -> bool:
        """
        Write frame to camera's buffer (never blocks).
        """
        if camera_id not in self.frame_buffers:
            logger.error(f"Camera {camera_id} not initialized in buffer")
            return False
            
        idx = self.write_indices[camera_id] % self.buffer_size
        
        self.frame_buffers[camera_id][idx] = frame
        self.timestamps[camera_id][idx] = time.time()
        self.frame_numbers[camera_id][idx] = frame_number or self.write_indices[camera_id]
        
        self.write_indices[camera_id] += 1
        
        return True

    def get_latest(self, camera_id: int) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get the single latest frame from a specific camera (for GUI or color extraction).
        """
        if camera_id not in self.frame_buffers:
            return None
            
        write_idx = self.write_indices[camera_id]
        if write_idx == 0:
            return None  # No frames written yet
            
        idx = (write_idx - 1) % self.buffer_size
        frame = self.frame_buffers[camera_id][idx]
        timestamp = self.timestamps[camera_id][idx]
        frame_num = self.frame_numbers[camera_id][idx]
        
        if frame is None:
            return None
            
        return frame, timestamp, frame_num

    def get_frame_batch(self, camera_id: int, batch_size: int) -> Optional[List[np.ndarray]]:
        """
        Gets a batch of new, unprocessed frames from a specific camera's buffer.
        This is a non-blocking, atomic read. If a full batch is not available, it returns None.

        Args:
            camera_id: The camera to get a batch from.
            batch_size: The desired number of frames in the batch.

        Returns:
            A list of frames if a full batch is available, otherwise None.
        """
        if camera_id not in self.frame_buffers:
            return None

        write_idx = self.write_indices[camera_id]
        read_idx = self.read_indices[camera_id]

        # Check if there are enough new frames to form a full batch
        if write_idx - read_idx < batch_size:
            return None  # Not enough new frames yet

        # A full batch is available, collect the frames
        batch_frames = []
        for i in range(batch_size):
            current_read_pos = (read_idx + i) % self.buffer_size
            frame = self.frame_buffers[camera_id][current_read_pos]
            if frame is not None:
                batch_frames.append(frame)

        # IMPORTANT: Atomically update the read index to mark these frames as processed
        self.read_indices[camera_id] += batch_size
        
        # Handle cases where the collected batch might be incomplete due to buffer wrap-around
        if len(batch_frames) != batch_size:
             logger.warning(f"Camera {camera_id}: Incomplete batch collected. Expected {batch_size}, got {len(batch_frames)}.")
             return None

        return batch_frames

    def get_latest_batch(self, batch_size: int) -> Dict[int, List[np.ndarray]]:
        """
        Gets a batch of the latest unprocessed frames from each camera.

        Args:
            batch_size: The number of frames to collect from each camera.

        Returns:
            A dictionary of {camera_id: [frame1, frame2, ...], ...} for cameras
            that had a full batch available.
        """
        batch_of_batches = {}
        
        for cam_id in range(1, self.num_cameras + 1):
            frame_batch = self.get_frame_batch(cam_id, batch_size)
            if frame_batch:
                batch_of_batches[cam_id] = frame_batch
        
        return batch_of_batches

    def get_buffer_status(self) -> Dict[int, Dict]:
        """Get status of all buffers for monitoring"""
        status = {}
        current_time = time.time()
        
        for cam_id in range(1, self.num_cameras + 1):
            write_idx = self.write_indices[cam_id]
            read_idx = self.read_indices[cam_id]
            
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
                'frames_read': read_idx,
                'unprocessed_frames': write_idx - read_idx,
                'buffer_utilization': min(write_idx, self.buffer_size) / self.buffer_size,
                'latest_frame': latest_info
            }
            
        return status
