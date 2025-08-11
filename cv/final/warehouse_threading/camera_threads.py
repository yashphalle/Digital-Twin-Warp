#!/usr/bin/env python3
"""
Camera Threading Manager
Handles camera preprocessing in separate threads
"""

import threading
import time
import logging
import cv2
from typing import List, Dict
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
grandparent_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from modules.camera_manager import CPUCameraManager
from modules.fisheye_corrector import OptimizedFisheyeCorrector

# Import configs using absolute path
import importlib.util
config_path = os.path.join(grandparent_dir, 'configs', 'config.py')
warehouse_config_path = os.path.join(grandparent_dir, 'configs', 'warehouse_config.py')

spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config

spec2 = importlib.util.spec_from_file_location("warehouse_config", warehouse_config_path)
warehouse_config_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(warehouse_config_module)
get_warehouse_config = warehouse_config_module.get_warehouse_config

from .queue_manager import QueueManager, FrameData

logger = logging.getLogger(__name__)

class CameraThreadManager:
    """Manages camera preprocessing threads"""
    
    def __init__(self, active_cameras: List[int], queue_manager: QueueManager):
        self.active_cameras = active_cameras
        self.queue_manager = queue_manager
        self.camera_threads = {}
        self.camera_managers = {}
        self.fisheye_correctors = {}
        self.running = False
        
        # Initialize camera components
        self._initialize_cameras()
        
        logger.info(f"✅ Camera Thread Manager initialized for {len(active_cameras)} cameras")

    def _initialize_cameras(self):
        """Initialize camera managers and fisheye correctors"""
        warehouse_config = get_warehouse_config()
        
        for camera_id in self.active_cameras:
            try:
                # Get camera configuration
                if str(camera_id) in warehouse_config.camera_zones:
                    camera_zone = warehouse_config.camera_zones[str(camera_id)]
                    camera_name = camera_zone.camera_name
                    rtsp_url = camera_zone.rtsp_url
                else:
                    camera_name = f"Camera {camera_id}"
                    rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

                # Create camera manager (REUSE existing module)
                self.camera_managers[camera_id] = CPUCameraManager(
                    camera_id=camera_id,
                    rtsp_url=rtsp_url,
                    camera_name=camera_name
                )
                
                # Create fisheye corrector (REUSE existing module)
                self.fisheye_correctors[camera_id] = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
                
                logger.info(f"📹 Camera {camera_id} components initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Camera {camera_id}: {e}")

    def start_camera_threads(self):
        """Start all camera preprocessing threads"""
        self.running = True
        
        for camera_id in self.active_cameras:
            if camera_id in self.camera_managers:
                thread = threading.Thread(
                    target=self._camera_worker,
                    args=(camera_id,),
                    name=f"Camera-{camera_id}",
                    daemon=True
                )
                self.camera_threads[camera_id] = thread
                thread.start()
                logger.info(f"🚀 Started camera thread for Camera {camera_id}")

    def _camera_worker(self, camera_id: int):
        """Worker function for individual camera thread"""
        camera_manager = self.camera_managers[camera_id]
        fisheye_corrector = self.fisheye_correctors[camera_id]

        # Connect to camera (SAME method as main.py)
        if not camera_manager.connect_camera():
            logger.error(f"❌ Failed to connect Camera {camera_id}")
            return

        logger.info(f"✅ Camera {camera_id} connected and preprocessing started")

        frame_number = 0
        processed_frames = 0
        FRAME_SKIP = 1  # DISABLED: Process every frame (no frame skipping)

        while self.running:
            try:
                # Read frame from camera (SAME method as main.py)
                ret, frame = camera_manager.read_frame()
                if not ret:
                    logger.warning(f"📹 Camera {camera_id}: Failed to read frame")
                    continue

                frame_number += 1

                # Frame skipping logic - only process every FRAME_SKIP frames
                if frame_number % FRAME_SKIP != 0:
                    continue  # Skip this frame

                processed_frames += 1

                # Apply fisheye correction (SAME method as main.py)
                corrected_frame = fisheye_corrector.correct(frame)

                # Resize if too large (for performance)
                height, width = corrected_frame.shape[:2]
                if width > 1600:
                    scale = 1600 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    corrected_frame = cv2.resize(corrected_frame, (new_width, new_height))

                # Create frame data structure (ONLY NEW PART)
                frame_data = FrameData(
                    camera_id=camera_id,
                    frame=corrected_frame,
                    timestamp=time.time(),
                    frame_number=frame_number,
                    stage="preprocessed",
                    metadata={
                        'original_size': (width, height),
                        'corrected': True,
                        'thread_id': threading.current_thread().ident,
                        'processed_frame_number': processed_frames,
                        'frame_skip': FRAME_SKIP
                    }
                )

                # Queue for detection threads (ONLY NEW PART)
                success = self.queue_manager.put_frame('camera_to_detection', frame_data, timeout=0.1)
                if success:
                    logger.debug(f"📹 Camera {camera_id}: Queued frame {frame_number} (processed #{processed_frames}) for detection")
                else:
                    logger.debug(f"📹 Camera {camera_id}: Frame {frame_number} dropped (queue full)")

            except Exception as e:
                logger.error(f"❌ Camera {camera_id} worker error: {e}")
                time.sleep(0.1)  # Brief pause on error

    def stop_camera_threads(self):
        """Stop all camera threads"""
        self.running = False
        
        # Wait for threads to finish
        for camera_id, thread in self.camera_threads.items():
            thread.join(timeout=2.0)
            logger.info(f"🛑 Camera {camera_id} thread stopped")
        
        # Cleanup camera resources (SAME method as main.py)
        for camera_id, camera_manager in self.camera_managers.items():
            camera_manager.cleanup_camera()
            
        logger.info("🧹 All camera threads stopped and cleaned up")

    def get_camera_stats(self) -> Dict[int, Dict]:
        """Get statistics for all cameras"""
        stats = {}
        
        for camera_id, camera_manager in self.camera_managers.items():
            stats[camera_id] = {
                'connected': camera_manager.is_connected(),
                'thread_alive': camera_id in self.camera_threads and self.camera_threads[camera_id].is_alive(),
                'camera_stats': camera_manager.get_statistics()
            }
            
        return stats
