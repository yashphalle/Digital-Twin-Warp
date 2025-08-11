import cv2
import time
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) 

from GPU.modules.camera_manager import CPUCameraManager
from GPU.modules.fisheye_corrector import OptimizedFisheyeCorrector
from GPU.configs.warehouse_config import get_warehouse_config
from GPU.configs.config import Config 

logger = logging.getLogger(__name__)

class CameraWorker:
    
    def __init__(self, camera_id: int, frame_skip: int = 3, debug: bool = False):
        self.camera_id = camera_id
        self.frame_skip = frame_skip  
        self.debug = debug
        self.frame_counter = 0
        self.processed_frames = 0
        
       
        Config.switch_to_remote_cameras()
        self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id)
        if not self.rtsp_url:
            raise ValueError(f"No RTSP URL found for camera {camera_id}")
        
        self.warehouse_config = get_warehouse_config()
        self.camera_zone = self.warehouse_config.camera_zones.get(camera_id)
        camera_name = self.camera_zone.camera_name if self.camera_zone else f"Camera {camera_id}"
        
        # Initialize camera manager with correct RTSP URL
        self.camera_manager = CPUCameraManager(
            camera_id=camera_id,
            rtsp_url=self.rtsp_url,  
            camera_name=camera_name
        )

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
        corrected_frame = self.fisheye_corrector.correct(frame)
        
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
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                return None
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 1:
                continue  
                
            processed_frame = self.process_frame(frame)
            self.processed_frames += 1
            
            if self.debug and self.processed_frames % 10 == 0:
                logger.info(f"Camera {self.camera_id}: Processed {self.processed_frames} frames (skipped {self.frame_counter - self.processed_frames})")
            
            return processed_frame
    
    def cleanup(self):
        self.camera_manager.cleanup_camera()
        logger.info(f"Camera {self.camera_id} worker cleaned up")