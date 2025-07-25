"""
Multi-Camera RTSP Manager for Warehouse Tracking System
Handles all 11 Lorex RTSP cameras with intelligent processing selection
"""

import cv2
import numpy as np
import threading
import time
import queue
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from configs.config import Config

# Import the existing pipeline and corrector classes
from simplified_lorex_pipeline import SimpleLorexPipeline
from lense_correct2 import OptimizedFisheyeCorrector

# Set up logging
logger = logging.getLogger(__name__)

class MultiCameraRTSPManager:
    """Multi-camera RTSP manager for 11-camera warehouse system"""
    
    def __init__(self):
        logger.info("🏭 Initializing Multi-Camera RTSP Manager for 11-camera warehouse system")
        
        # Camera configuration
        self.all_cameras = list(Config.RTSP_CAMERA_URLS.keys())  # [1,2,3,4,5,6,7,8,9,10,11]
        self.active_cameras = Config.ACTIVE_CAMERAS  # [7] - only process Camera 7
        
        # Camera pipelines and threads
        self.camera_pipelines = {}
        self.camera_threads = {}
        self.fisheye_correctors = {}
        
        # System state
        self.running = False
        self.frame_buffers = {}
        
        # Performance tracking
        self.frame_counts = {}
        self.connection_status = {}
        
        # Initialize all cameras (but only process active ones)
        self._initialize_all_cameras()
        
        logger.info(f"✅ Multi-camera system initialized:")
        logger.info(f"   📹 Total cameras configured: {len(self.all_cameras)}")
        logger.info(f"   🎯 Active processing cameras: {self.active_cameras}")
        logger.info(f"   💤 Standby cameras: {[c for c in self.all_cameras if c not in self.active_cameras]}")
    
    def _initialize_all_cameras(self):
        """Initialize all 11 cameras but only start active ones"""
        logger.info("🔧 Initializing camera pipelines...")
        
        # Initialize fisheye correctors for all cameras
        for camera_id in self.all_cameras:
            self.fisheye_correctors[camera_id] = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
            self.frame_counts[camera_id] = 0
            self.frame_buffers[camera_id] = queue.Queue(maxsize=5)
            
        # Initialize pipelines for all cameras (connection ready but not started)
        for camera_id in self.all_cameras:
            rtsp_url = Config.RTSP_CAMERA_URLS[camera_id]
            camera_name = Config.CAMERA_NAMES[camera_id]
            
            try:
                pipeline = SimpleLorexPipeline(rtsp_url, buffer_size=Config.RTSP_BUFFER_SIZE)
                self.camera_pipelines[camera_id] = pipeline
                self.connection_status[camera_id] = "ready"
                
                if camera_id in self.active_cameras:
                    logger.info(f"✅ {camera_name} - Pipeline ready (ACTIVE)")
                else:
                    logger.info(f"💤 {camera_name} - Pipeline ready (STANDBY)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize {camera_name}: {e}")
                self.camera_pipelines[camera_id] = None
                self.connection_status[camera_id] = "failed"
    
    def start_active_cameras(self):
        """Start only the active cameras for processing"""
        logger.info("🚀 Starting active cameras...")
        self.running = True
        
        for camera_id in self.active_cameras:
            if camera_id in self.camera_pipelines and self.camera_pipelines[camera_id]:
                self._start_camera(camera_id)
            else:
                logger.warning(f"⚠️  Camera {camera_id} pipeline not available")
    
    def _start_camera(self, camera_id: int):
        """Start a specific camera"""
        camera_name = Config.CAMERA_NAMES[camera_id]
        pipeline = self.camera_pipelines[camera_id]
        
        try:
            # Connect to camera
            if pipeline.connect_camera():
                # Start capture thread
                pipeline.running = True
                capture_thread = threading.Thread(
                    target=pipeline.frame_capture_thread,
                    name=f"Camera{camera_id}Capture",
                    daemon=True
                )
                capture_thread.start()
                
                self.camera_threads[camera_id] = capture_thread
                self.connection_status[camera_id] = "active"
                
                logger.info(f"🎥 {camera_name} - Started successfully")
                return True
                
            else:
                logger.error(f"❌ {camera_name} - Failed to connect")
                self.connection_status[camera_id] = "connection_failed"
                return False
                
        except Exception as e:
            logger.error(f"❌ {camera_name} - Start failed: {e}")
            self.connection_status[camera_id] = "start_failed"
            return False
    
    def get_frame(self, camera_id: int) -> Optional[Tuple[np.ndarray, datetime]]:
        """Get latest frame from a specific camera"""
        if camera_id not in self.active_cameras:
            logger.warning(f"⚠️  Camera {camera_id} is not active")
            return None
            
        if camera_id not in self.camera_pipelines or not self.camera_pipelines[camera_id]:
            return None
            
        # Get raw frame from pipeline
        frame_data = self.camera_pipelines[camera_id].get_frame()
        if not frame_data:
            return None
            
        raw_frame, timestamp = frame_data
        
        # Apply fisheye correction
        try:
            corrected_frame = self.fisheye_correctors[camera_id].correct(raw_frame)
            
            # Scale down from 4K to processing resolution
            if corrected_frame.shape[1] > Config.RTSP_PROCESSING_WIDTH:
                scale_factor = Config.RTSP_PROCESSING_WIDTH / corrected_frame.shape[1]
                new_width = Config.RTSP_PROCESSING_WIDTH
                new_height = int(corrected_frame.shape[0] * scale_factor)
                corrected_frame = cv2.resize(corrected_frame, (new_width, new_height))
            
            self.frame_counts[camera_id] += 1
            return corrected_frame, timestamp
            
        except Exception as e:
            logger.error(f"❌ Frame processing error for Camera {camera_id}: {e}")
            return None
    
    def get_camera_stats(self, camera_id: int) -> Dict:
        """Get statistics for a specific camera"""
        stats = {
            'camera_id': camera_id,
            'camera_name': Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}"),
            'status': self.connection_status.get(camera_id, 'unknown'),
            'active': camera_id in self.active_cameras,
            'frame_count': self.frame_counts.get(camera_id, 0)
        }
        
        # Add pipeline stats if available
        if camera_id in self.camera_pipelines and self.camera_pipelines[camera_id]:
            pipeline_stats = self.camera_pipelines[camera_id].get_stats()
            stats.update(pipeline_stats)
        
        return stats
    
    def get_all_camera_stats(self) -> Dict:
        """Get statistics for all cameras"""
        all_stats = {}
        
        for camera_id in self.all_cameras:
            all_stats[camera_id] = self.get_camera_stats(camera_id)
        
        # System summary
        active_count = len([c for c in self.all_cameras if self.connection_status.get(c) == 'active'])
        ready_count = len([c for c in self.all_cameras if self.connection_status.get(c) == 'ready'])
        failed_count = len([c for c in self.all_cameras if 'failed' in self.connection_status.get(c, '')])
        
        summary = {
            'total_cameras': len(self.all_cameras),
            'active_cameras': active_count,
            'ready_cameras': ready_count,
            'failed_cameras': failed_count,
            'processing_cameras': self.active_cameras,
            'system_running': self.running
        }
        
        return {'cameras': all_stats, 'summary': summary}
    
    def enable_camera(self, camera_id: int) -> bool:
        """Enable processing for an additional camera"""
        if camera_id not in self.all_cameras:
            logger.error(f"❌ Camera {camera_id} not in system configuration")
            return False
            
        if camera_id in self.active_cameras:
            logger.info(f"ℹ️  Camera {camera_id} already active")
            return True
            
        logger.info(f"🔄 Enabling Camera {camera_id}...")
        
        # Add to active cameras
        self.active_cameras.append(camera_id)
        
        # Start the camera if system is running
        if self.running:
            return self._start_camera(camera_id)
        else:
            logger.info(f"💤 Camera {camera_id} enabled but system not running")
            return True
    
    def disable_camera(self, camera_id: int) -> bool:
        """Disable processing for a camera (but keep pipeline ready)"""
        if camera_id not in self.active_cameras:
            logger.info(f"ℹ️  Camera {camera_id} already inactive")
            return True
            
        logger.info(f"⏸️  Disabling Camera {camera_id}...")
        
        # Stop camera thread
        if camera_id in self.camera_pipelines and self.camera_pipelines[camera_id]:
            self.camera_pipelines[camera_id].running = False
        
        # Remove from active cameras
        self.active_cameras.remove(camera_id)
        self.connection_status[camera_id] = "ready"
        
        logger.info(f"✅ Camera {camera_id} disabled (pipeline ready for reactivation)")
        return True
    
    def cleanup(self):
        """Cleanup all camera resources"""
        logger.info("🧹 Cleaning up multi-camera system...")
        
        self.running = False
        
        # Stop all camera pipelines
        for camera_id, pipeline in self.camera_pipelines.items():
            if pipeline:
                pipeline.cleanup()
        
        # Clear data structures
        self.camera_pipelines.clear()
        self.camera_threads.clear()
        self.frame_buffers.clear()
        
        logger.info("✅ Multi-camera system cleanup complete")

# Legacy compatibility wrapper
class RTSPCameraManager(MultiCameraRTSPManager):
    """Legacy compatibility wrapper for existing code"""
    
    def __init__(self, camera_urls: Optional[List[str]] = None, lens_mm: float = 2.8):
        # Ignore legacy parameters and use new system
        super().__init__()
        logger.info("🔄 Using legacy compatibility mode - all features available")

# Helper functions for easy camera management
def get_active_camera_frame(camera_id: int = 7) -> Optional[Tuple[np.ndarray, datetime]]:
    """Quick function to get frame from active camera"""
    # This would typically be called with a global camera manager instance
    pass

def get_camera_manager() -> MultiCameraRTSPManager:
    """Get or create camera manager instance"""
    if not hasattr(get_camera_manager, '_instance'):
        get_camera_manager._instance = MultiCameraRTSPManager()
    return get_camera_manager._instance 