"""
Multi-Camera RTSP Tracking System for 11-Camera Warehouse
Processes Camera 7 by default, with capability to handle all 11 cameras
"""

import cv2
import numpy as np
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import modules
from configs.config import Config
from detector_tracker import DetectorTracker
from database_handler import DatabaseHandler
from rtsp_camera_manager import MultiCameraRTSPManager
from multi_camera_grid_display import MultiCameraGridDisplay

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class MultiCameraTrackingSystem:
    """Multi-camera RTSP tracking system for 11-camera warehouse"""
    
    def __init__(self):
        logger.info("🏭 Initializing Multi-Camera Tracking System")
        logger.info(f"📹 Configured for {len(Config.RTSP_CAMERA_URLS)} cameras")
        logger.info(f"🎯 Active processing: {Config.ACTIVE_CAMERAS}")

        # Log GPU status for debugging
        Config.log_gpu_status()

        # 🚀 GPU MEMORY MANAGEMENT
        if Config.SEQUENTIAL_CAMERA_PROCESSING:
            logger.info("🔥 SEQUENTIAL PROCESSING MODE: Processing cameras one at a time to prevent GPU overflow")
            self.camera_processing_queue = Config.ACTIVE_CAMERAS.copy()
            self.current_camera_index = 0

        # Initialize camera manager
        self.camera_manager = MultiCameraRTSPManager()
        
        # Initialize tracking components
        self.detector_tracker = DetectorTracker()
        self.database_handler = DatabaseHandler()
        self.grid_display = MultiCameraGridDisplay()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Processing state
        self.running = False
        self.processing_active_cameras = Config.ACTIVE_CAMERAS.copy()
        
        logger.info("✅ Multi-camera tracking system initialized")
    
    def start_system(self):
        """Start the multi-camera tracking system"""
        logger.info("🚀 Starting multi-camera tracking system...")
        
        # Start active cameras
        self.camera_manager.start_active_cameras()
        
        # Wait a moment for cameras to initialize
        time.sleep(2)
        
        # Check camera status
        stats = self.camera_manager.get_all_camera_stats()
        active_count = stats['summary']['active_cameras']
        
        if active_count == 0:
            logger.error("❌ No cameras started successfully")
            return False
        
        logger.info(f"✅ System started with {active_count} active cameras")

        # Start grid display
        self.grid_display.start_display()
        logger.info("🖥️ Grid display started")

        return True
    
    def run_tracking(self):
        """Main tracking loop"""
        logger.info("🎯 Starting tracking loop...")
        
        if not self.start_system():
            logger.error("❌ Failed to start system")
            return
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # Process each active camera
                for camera_id in self.processing_active_cameras:
                    self._process_camera_frame(camera_id)
                
                # Update FPS
                self._update_fps()
                
                # Print stats periodically
                if self.fps_counter % 100 == 0:
                    self._print_system_stats()
                
                # Control frame rate
                processing_time = time.time() - start_time
                target_delay = 1.0 / Config.RTSP_TARGET_FPS
                if processing_time < target_delay:
                    time.sleep(target_delay - processing_time)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        finally:
            self._cleanup()

    def _process_camera_frame(self, camera_id: int):
        """Process a single frame from a specific camera"""
        try:
            # Get frame from camera
            frame_data = self.camera_manager.get_frame(camera_id)
            if not frame_data:
                return

            frame, timestamp = frame_data
            self.frame_count += 1
            
            # Set camera ID for coordinate mapping
            self.detector_tracker.set_camera_id(camera_id)
            
            # Process frame for object detection and tracking
            result_data = self.detector_tracker.process_frame(frame)
            
            # Handle both tuple and dict return formats
            if isinstance(result_data, tuple):
                tracked_objects_list, statistics_dict = result_data
                result_dict = {
                    'tracked_objects': tracked_objects_list if tracked_objects_list else [],
                    'statistics': statistics_dict
                }
            else:
                result_dict = result_data if result_data else {'tracked_objects': []}
            
            # 🚀 GPU UTILIZATION MONITORING (every 500 frames)
            if self.frame_count % 500 == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated(0) / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        gpu_utilization = (memory_used / memory_total) * 100
                        logger.info(f"🚀 GPU UTILIZATION: {gpu_utilization:.1f}% ({memory_used:.2f}GB / {memory_total:.1f}GB)")

                        # Target: 80-90% utilization
                        if gpu_utilization < 70:
                            logger.info("💡 GPU utilization low - consider reducing SKIP_DETECTION_FRAMES")
                        elif gpu_utilization > 95:
                            logger.warning("⚠️ GPU utilization very high - consider increasing SKIP_DETECTION_FRAMES")
                except Exception as e:
                    pass  # Ignore GPU monitoring errors

            # Store tracked objects in database
            if result_dict and result_dict.get('tracked_objects'):
                self._store_objects_with_camera_info(result_dict['tracked_objects'], camera_id)

            # Display frame with tracking overlay
            self._display_camera_frame(frame, result_dict, camera_id)
            
        except Exception as e:
            logger.error(f"❌ Error processing Camera {camera_id}: {e}")
    
    def _store_objects_with_camera_info(self, tracked_objects: List[Dict], camera_id: int):
        """Store tracked objects with camera source information"""
        try:
            if not tracked_objects:
                return
            
            camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
            coverage_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
            
            # Store each object individually using upsert
            for obj in tracked_objects:
                if isinstance(obj, dict):
                    # Convert to proper database format with camera info
                    db_object = {
                        'persistent_id': obj.get('id'),
                        'center': obj.get('center'),
                        'real_center': obj.get('real_center'),
                        'bbox': obj.get('bbox'),
                        'confidence': obj.get('confidence', 0.0),
                        'age_seconds': obj.get('age', 0),
                        'times_seen': obj.get('times_seen', 1),
                        'frame_number': self.frame_count,
                        'last_seen': datetime.now(),
                        'camera_source': camera_name,
                        'camera_id': camera_id,
                        'coverage_zone': coverage_zone,
                        'processing_mode': 'multi_camera_rtsp'
                    }
                    
                    # Add first_seen for new objects
                    if obj.get('status') == 'new':
                        db_object['first_seen'] = datetime.now()
                    
                    # Use upsert to avoid duplicate key conflicts
                    self.database_handler.upsert_object(db_object)
                    
        except Exception as e:
            logger.error(f"❌ Error storing objects from Camera {camera_id}: {e}")
    
    def _display_camera_frame(self, frame: np.ndarray, result_dict: Dict, camera_id: int):
        """Send frame to grid display with tracking overlay"""
        try:
            display_frame = frame.copy()

            # Draw tracked objects
            if result_dict and result_dict.get('tracked_objects'):
                display_frame = self.detector_tracker.draw_tracked_objects(frame, result_dict['tracked_objects'])

            # Add coordinate system info overlay (smaller for grid view)
            display_frame = self.detector_tracker.draw_coordinate_system_info(display_frame)

            # Update grid display with this camera's frame
            self.grid_display.update_camera_frame(camera_id, display_frame)

        except Exception as e:
            logger.error(f"❌ Display error for Camera {camera_id}: {e}")
            # Mark camera as disconnected in grid display
            self.grid_display.set_camera_disconnected(camera_id)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def _print_system_stats(self):
        """Print comprehensive system statistics"""
        logger.info("📊 MULTI-CAMERA SYSTEM STATS")
        logger.info("=" * 50)
        
        # Camera stats
        camera_stats = self.camera_manager.get_all_camera_stats()
        summary = camera_stats['summary']
        
        logger.info(f"🎥 Total Cameras: {summary['total_cameras']}")
        logger.info(f"🟢 Active: {summary['active_cameras']}")
        logger.info(f"💤 Ready: {summary['ready_cameras']}")
        logger.info(f"❌ Failed: {summary['failed_cameras']}")
        logger.info(f"🎯 Processing: {summary['processing_cameras']}")
        
        # Performance stats
        logger.info(f"📈 System FPS: {self.fps:.1f}")
        logger.info(f"🖼️  Total Frames: {self.frame_count}")
        
        # Individual camera details
        for camera_id in Config.ACTIVE_CAMERAS:
            stats = camera_stats['cameras'].get(camera_id, {})
            logger.info(f"📹 {stats.get('camera_name', f'Camera {camera_id}')}: "
                       f"Status={stats.get('status', 'unknown')}, "
                       f"Frames={stats.get('frame_count', 0)}")
    
    def _show_camera_controls(self):
        """Display camera control options"""
        logger.info("🎮 CAMERA CONTROLS")
        logger.info("-" * 30)
        logger.info("Active cameras: " + str(self.processing_active_cameras))
        logger.info("Available cameras: " + str(list(Config.RTSP_CAMERA_URLS.keys())))
        logger.info("Press 'q' to quit, 's' for stats, 'c' for controls")
    
    def enable_additional_camera(self, camera_id: int) -> bool:
        """Enable processing for an additional camera"""
        if camera_id in self.processing_active_cameras:
            logger.info(f"ℹ️  Camera {camera_id} already processing")
            return True
        
        # Enable in camera manager
        if self.camera_manager.enable_camera(camera_id):
            self.processing_active_cameras.append(camera_id)
            logger.info(f"✅ Camera {camera_id} enabled for processing")
            return True
        else:
            logger.error(f"❌ Failed to enable Camera {camera_id}")
            return False
    
    def disable_camera(self, camera_id: int) -> bool:
        """Disable processing for a camera"""
        if camera_id not in self.processing_active_cameras:
            logger.info(f"ℹ️  Camera {camera_id} not currently processing")
            return True
        
        # Disable in camera manager
        if self.camera_manager.disable_camera(camera_id):
            self.processing_active_cameras.remove(camera_id)
            logger.info(f"✅ Camera {camera_id} disabled from processing")
            return True
        else:
            logger.error(f"❌ Failed to disable Camera {camera_id}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        camera_stats = self.camera_manager.get_all_camera_stats()
        
        return {
            'system_running': self.running,
            'processing_cameras': self.processing_active_cameras,
            'system_fps': self.fps,
            'total_frames': self.frame_count,
            'camera_stats': camera_stats,
            'active_count': camera_stats['summary']['active_cameras'],
            'failed_count': camera_stats['summary']['failed_cameras']
        }
    
    def _cleanup(self):
        """Cleanup system resources"""
        logger.info("🧹 Cleaning up multi-camera tracking system...")

        self.running = False
        self.camera_manager.cleanup()
        self.grid_display.stop_display()
        cv2.destroyAllWindows()

        logger.info("✅ Multi-camera tracking system stopped")

def main():
    """Main function for multi-camera tracking"""
    logger.info("🚀 Starting Multi-Camera Warehouse Tracking System")
    logger.info("=" * 60)
    
    # Print configuration
    logger.info(f"📋 Configuration:")
    logger.info(f"   • Total cameras: {len(Config.RTSP_CAMERA_URLS)}")
    logger.info(f"   • Active cameras: {Config.ACTIVE_CAMERAS}")
    logger.info(f"   • Warehouse size: {Config.FULL_WAREHOUSE_WIDTH_FT}ft x {Config.FULL_WAREHOUSE_LENGTH_FT}ft")
    logger.info(f"   • Target FPS: {Config.RTSP_TARGET_FPS}")
    
    # Create and run tracking system
    tracking_system = MultiCameraTrackingSystem()
    
    try:
        tracking_system.run_tracking()
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        logger.info("🏁 Multi-camera tracking system finished")

if __name__ == "__main__":
    main() 