# cv/GPU/pipelines/camera_worker.py

import cv2
import time
import logging
import sys
import os
import threading

# Add parent directories to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # cv directory

from GPU.modules.camera_manager import CPUCameraManager
from GPU.modules.fisheye_corrector import OptimizedFisheyeCorrector
from GPU.configs.warehouse_config import get_warehouse_config
from GPU.configs.config import Config  # Add this import

logger = logging.getLogger(__name__)

class CameraWorker:
    """
    Single camera worker that handles:
    1. RTSP connection (using existing CPUCameraManager)
    2. Frame skipping (15 FPS -> 5 FPS)
    3. Fisheye correction (using existing OptimizedFisheyeCorrector)
    4. Resize (using existing logic)
    """
    
    def __init__(self, camera_id: int, frame_skip: int = 3, debug: bool = False):
        self.camera_id = camera_id
        self.frame_skip = frame_skip  # Process every 3rd frame
        self.debug = debug
        self.frame_counter = 0
        self.processed_frames = 0
        
        # Force REMOTE URLs
        Config.switch_to_remote_cameras()
        
        # Get RTSP URL from config.py instead of warehouse_config
        self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id)
        if not self.rtsp_url:
            raise ValueError(f"No RTSP URL found for camera {camera_id}")
        
        # Get camera name from warehouse config (just for display)
        self.warehouse_config = get_warehouse_config()
        self.camera_zone = self.warehouse_config.camera_zones.get(camera_id)
        camera_name = self.camera_zone.camera_name if self.camera_zone else f"Camera {camera_id}"
        
        # Initialize camera manager with correct RTSP URL
        self.camera_manager = CPUCameraManager(
            camera_id=camera_id,
            rtsp_url=self.rtsp_url,  # Use URL from config.py
            camera_name=camera_name
        )
        
        # Initialize fisheye corrector (reusing existing)
        self.fisheye_corrector = OptimizedFisheyeCorrector(lens_mm=2.8)
        
        logger.info(f"CameraWorker initialized for {camera_name}")
        logger.info(f"Using REMOTE URL: {self.rtsp_url}")
        
    def connect(self) -> bool:
        """Connect to camera using existing camera manager"""
        success = self.camera_manager.connect_camera()
        if success:
            logger.info(f"✅ Connected to camera {self.camera_id}")
        else:
            logger.error(f"❌ Failed to connect to camera {self.camera_id}")
        return success
    
    def process_frame(self, frame):
        """Apply preprocessing pipeline (using existing logic)"""
        # Step 1: Fisheye correction
        corrected_frame = self.fisheye_corrector.correct(frame)
        
        # Step 2: Resize if too large (EXACT logic from your code)
        height, width = corrected_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            corrected_frame = cv2.resize(corrected_frame, (new_width, new_height))
            
        return corrected_frame
    
    def get_next_frame(self):
        """Get next processed frame (with frame skipping)"""
        while True:
            # Read frame
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                return None
                
            self.frame_counter += 1
            
            # Frame skipping logic
            if self.frame_counter % self.frame_skip != 1:
                continue  # Skip this frame
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            self.processed_frames += 1
            
            if self.debug and self.processed_frames % 10 == 0:
                logger.info(f"Camera {self.camera_id}: Processed {self.processed_frames} frames (skipped {self.frame_counter - self.processed_frames})")
            
            return processed_frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera_manager.cleanup_camera()
        logger.info(f"Camera {self.camera_id} worker cleaned up")

class ParallelCameraWorker(threading.Thread):
    """Thread wrapper to run CameraWorker in parallel"""
    
    def __init__(self, camera_id: int, ring_buffer, frame_skip: int = 3):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.ring_buffer = ring_buffer
        self.worker = CameraWorker(camera_id, frame_skip=frame_skip, debug=False)
        self.running = False
        self.frames_written = 0
        
    def connect(self):
        """Connect the camera"""
        return self.worker.connect()
        
    def run(self):
        """Run in parallel - continuously read and write frames"""
        self.running = True
        while self.running:
            frame = self.worker.get_next_frame()
            if frame is not None:
                self.ring_buffer.write(self.camera_id, frame)
                self.frames_written += 1
                
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.worker.cleanup()