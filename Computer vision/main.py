"""
Main Application - Warehouse Tracking System
Orchestrates dual camera capture, stitching, detection, tracking, and database operations
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

# Import our modules
from config import Config
from camera_manager import CameraManager
from detector_tracker import DetectorTracker
from database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOG_FILE_PATH) if Config.LOG_TO_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class WarehouseTrackingSystem:
    """Main warehouse tracking system orchestrator"""
    
    def __init__(self):
        """Initialize the complete tracking system"""
        logger.info("Initializing Warehouse Tracking System...")
        
        # System components
        self.camera_manager = None
        self.detector_tracker = None
        self.database_handler = None
        
        # System state
        self.running = False
        self.frame_count = 0
        self.session_start_time = datetime.now()
        
        # Performance tracking
        self.total_frame_times = []
        self.system_stats = {
            'total_frames_processed': 0,
            'total_objects_detected': 0,
            'total_objects_tracked': 0,
            'system_uptime_seconds': 0,
            'avg_fps': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize camera manager
            logger.info("Initializing camera manager...")
            self.camera_manager = CameraManager()
            
            # Initialize detector/tracker
            logger.info("Initializing detection and tracking...")
            self.detector_tracker = DetectorTracker(force_gpu=Config.FORCE_GPU)
            
            # Initialize database handler
            logger.info("Initializing database handler...")
            try:
                self.database_handler = DatabaseHandler()
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                logger.warning("Continuing without database...")
                self.database_handler = None
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def process_single_frame(self) -> Dict[str, Any]:
        """Process a single frame through the complete pipeline"""
        frame_start_time = time.time()
        
        try:
            # Step 1: Capture frames from both cameras
            frame1, frame2, capture_success = self.camera_manager.capture_synchronized_frames()
            
            if not capture_success:
                logger.warning("Frame capture failed")
                return {'success': False, 'error': 'capture_failed'}
            
            # Step 2: Stitch frames together
            stitched_frame = self.camera_manager.stitch_frames(frame1, frame2)
            
            if stitched_frame is None:
                logger.warning("Frame stitching failed")
                return {'success': False, 'error': 'stitching_failed'}
            
            # Step 3: Detect and track objects
            tracked_objects, detection_stats = self.detector_tracker.process_frame(stitched_frame)
            
            # Step 4: Store/update objects in database
            database_success = True
            if self.database_handler and tracked_objects:
                database_success = self._store_tracked_objects(tracked_objects)
            
            # Step 5: Create visualization
            annotated_frame = self.detector_tracker.draw_tracked_objects(stitched_frame, tracked_objects)
            
            # Add grid overlay if enabled
            if Config.SHOW_GRID_OVERLAY:
                annotated_frame = self.camera_manager.create_grid_overlay(annotated_frame)
            
            # Calculate frame processing time
            frame_processing_time = time.time() - frame_start_time
            self.total_frame_times.append(frame_processing_time)
            if len(self.total_frame_times) > Config.FPS_CALCULATION_FRAMES:
                self.total_frame_times.pop(0)
            
            # Update system statistics
            self.frame_count += 1
            self.system_stats['total_frames_processed'] = self.frame_count
            self.system_stats['total_objects_detected'] += len(tracked_objects)
            self.system_stats['avg_fps'] = 1.0 / np.mean(self.total_frame_times) if self.total_frame_times else 0
            
            return {
                'success': True,
                'annotated_frame': annotated_frame,
                'tracked_objects': tracked_objects,
                'frame_processing_time': frame_processing_time,
                'database_success': database_success,
                'detection_stats': detection_stats
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _store_tracked_objects(self, tracked_objects: List[Dict]) -> bool:
        """Store tracked objects to database"""
        if not self.database_handler:
            return False
        
        try:
            success_count = 0
            
            for obj in tracked_objects:
                # Prepare object data for database
                db_object = {
                    'persistent_id': obj['id'],
                    'center': obj['center'],
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence'],
                    'age_seconds': obj['age'],
                    'times_seen': obj['times_seen'],
                    'frame_number': self.frame_count,
                    'session_id': id(self),  # Unique session identifier
                    'last_seen': datetime.now()
                }
                
                # Add first_seen for new objects
                if obj.get('status') == 'new':
                    db_object['first_seen'] = datetime.now()
                
                # Use upsert to handle both new and existing objects
                if self.database_handler.upsert_object(db_object):
                    success_count += 1
            
            logger.debug(f"Successfully stored/updated {success_count}/{len(tracked_objects)} objects")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Database storage error: {e}")
            return False
    
    def create_info_overlay(self, frame: np.ndarray, frame_result: Dict[str, Any]) -> np.ndarray:
        """Create comprehensive information overlay"""
        if not Config.SHOW_INFO_OVERLAY:
            return frame
        
        height, width = frame.shape[:2]
        overlay_height = Config.INFO_OVERLAY_HEIGHT
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        
        # Get performance statistics
        camera_stats = self.camera_manager.get_performance_stats()
        detection_stats = frame_result.get('detection_stats', {})
        system_uptime = (datetime.now() - self.session_start_time).total_seconds()
        
        # System information
        line_height = 20
        y_pos = line_height
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color_white = (255, 255, 255)
        color_green = (0, 255, 0)
        color_yellow = (255, 255, 0)
        color_red = (0, 0, 255)
        
        # Title
        cv2.putText(overlay, "Warehouse Tracking System - Dual Camera SIFT Tracker", 
                   (10, y_pos), font, 0.7, color_white, 2)
        y_pos += line_height + 5
        
        # System metrics
        fps = self.system_stats['avg_fps']
        fps_color = color_green if fps > 15 else color_yellow if fps > 10 else color_red
        
        cv2.putText(overlay, f"System FPS: {fps:.1f} | Frame: {self.frame_count} | Uptime: {system_uptime:.0f}s", 
                   (10, y_pos), font, 0.5, fps_color, 1)
        y_pos += line_height
        
        # Camera performance
        cam_fps = camera_stats.get('capture_fps', 0)
        stitch_fps = camera_stats.get('stitch_fps', 0)
        cv2.putText(overlay, f"Camera: {cam_fps:.1f}fps | Stitching: {stitch_fps:.1f}fps | Calibrated: {camera_stats.get('calibrated', False)}", 
                   (10, y_pos), font, 0.45, color_white, 1)
        y_pos += line_height
        
        # Detection and tracking
        det_fps = detection_stats.get('detection_fps', 0)
        track_fps = detection_stats.get('tracking_fps', 0)
        active_objects = detection_stats.get('active_objects', 0)
        
        cv2.putText(overlay, f"Detection: {det_fps:.1f}fps | Tracking: {track_fps:.1f}fps | Active Objects: {active_objects}", 
                   (10, y_pos), font, 0.45, color_white, 1)
        y_pos += line_height
        
        # Object statistics
        new_objects = detection_stats.get('new_objects_created', 0)
        updated_objects = detection_stats.get('objects_updated', 0)
        removed_objects = detection_stats.get('objects_removed', 0)
        
        cv2.putText(overlay, f"Session: New: {new_objects} | Updated: {updated_objects} | Removed: {removed_objects}", 
                   (10, y_pos), font, 0.45, color_yellow, 1)
        y_pos += line_height
        
        # Database status
        db_status = "Connected" if self.database_handler and self.database_handler.connected else "Disconnected"
        db_color = color_green if self.database_handler and self.database_handler.connected else color_red
        
        cv2.putText(overlay, f"Database: {db_status} | Storage: {'OK' if frame_result.get('database_success', False) else 'Failed'}", 
                   (10, y_pos), font, 0.45, db_color, 1)
        y_pos += line_height
        
        # GPU information (if available)
        if detection_stats.get('gpu_memory_used'):
            gpu_used = detection_stats['gpu_memory_used']
            gpu_total = detection_stats['gpu_memory_total']
            gpu_percent = (gpu_used / gpu_total) * 100
            gpu_color = color_green if gpu_percent < 70 else color_yellow if gpu_percent < 90 else color_red
            
            cv2.putText(overlay, f"GPU: {gpu_used:.1f}/{gpu_total:.1f}GB ({gpu_percent:.1f}%)", 
                       (10, y_pos), font, 0.45, gpu_color, 1)
        else:
            cv2.putText(overlay, "Device: CPU", (10, y_pos), font, 0.45, color_white, 1)
        y_pos += line_height
        
        # Controls
        cv2.putText(overlay, "Controls: 'q'=Quit | 's'=Stats | 'c'=Clear DB | 'd'=Database Stats | 'r'=Reset", 
                   (10, y_pos), font, 0.4, color_yellow, 1)
        
        # Combine overlay with frame
        combined = np.vstack([overlay, frame])
        return combined
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input during runtime"""
        if key == ord('q'):
            logger.info("Quit requested by user")
            return False
        
        elif key == ord('s'):
            self.print_system_statistics()
        
        elif key == ord('c'):
            self.clear_old_database_entries()
        
        elif key == ord('d'):
            self.print_database_statistics()
        
        elif key == ord('r'):
            self.reset_system_statistics()
        
        elif key == ord('h'):
            self.print_help()
        
        return True
    
    def print_system_statistics(self):
        """Print comprehensive system statistics"""
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        
        # System uptime
        uptime = (datetime.now() - self.session_start_time).total_seconds()
        print(f"Session uptime: {uptime:.1f} seconds ({uptime/60:.1f} minutes)")
        print(f"Frames processed: {self.frame_count}")
        print(f"Average FPS: {self.system_stats['avg_fps']:.2f}")
        
        # Camera statistics
        camera_stats = self.camera_manager.get_performance_stats()
        print(f"\nCAMERA PERFORMANCE:")
        print(f"  Capture FPS: {camera_stats.get('capture_fps', 0):.2f}")
        print(f"  Stitching FPS: {camera_stats.get('stitch_fps', 0):.2f}")
        print(f"  Calibrated: {camera_stats.get('calibrated', False)}")
        
        # Detection/tracking statistics
        detection_stats = self.detector_tracker.get_performance_stats()
        print(f"\nDETECTION & TRACKING:")
        print(f"  Detection FPS: {detection_stats.get('detection_fps', 0):.2f}")
        print(f"  Tracking FPS: {detection_stats.get('tracking_fps', 0):.2f}")
        print(f"  Active objects: {detection_stats.get('active_objects', 0)}")
        print(f"  Objects created: {detection_stats.get('new_objects_created', 0)}")
        print(f"  Objects updated: {detection_stats.get('objects_updated', 0)}")
        print(f"  Objects removed: {detection_stats.get('objects_removed', 0)}")
        
        # GPU information
        if detection_stats.get('gpu_memory_used'):
            print(f"  GPU Memory: {detection_stats['gpu_memory_used']:.1f}GB / {detection_stats['gpu_memory_total']:.1f}GB")
        
        print("="*50)
    
    def print_database_statistics(self):
        """Print database statistics"""
        if not self.database_handler:
            print("Database not connected")
            return
        
        try:
            stats = self.database_handler.get_object_statistics()
            print("\n" + "="*40)
            print("DATABASE STATISTICS")
            print("="*40)
            
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
            
            print("="*40)
            
        except Exception as e:
            print(f"Error getting database statistics: {e}")
    
    def clear_old_database_entries(self):
        """Clear old database entries"""
        if not self.database_handler:
            print("Database not connected")
            return
        
        try:
            deleted_count = self.database_handler.clear_old_objects(Config.CLEANUP_OLD_DATA_HOURS)
            print(f"Cleared {deleted_count} old database entries")
        except Exception as e:
            print(f"Error clearing database: {e}")
    
    def reset_system_statistics(self):
        """Reset system statistics"""
        self.system_stats = {
            'total_frames_processed': 0,
            'total_objects_detected': 0,
            'total_objects_tracked': 0,
            'system_uptime_seconds': 0,
            'avg_fps': 0.0
        }
        self.frame_count = 0
        self.session_start_time = datetime.now()
        self.total_frame_times.clear()
        print("System statistics reset")
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*40)
        print("KEYBOARD CONTROLS")
        print("="*40)
        print("q - Quit application")
        print("s - Show system statistics")
        print("c - Clear old database entries")
        print("d - Show database statistics")
        print("r - Reset system statistics")
        print("h - Show this help")
        print("="*40)
    
    def run(self):
        """Main application loop"""
        logger.info("Starting Warehouse Tracking System...")
        print("Warehouse Tracking System Started")
        print("Press 'h' for help, 'q' to quit")
        
        self.running = True
        
        try:
            while self.running:
                # Process single frame
                frame_result = self.process_single_frame()
                
                if not frame_result['success']:
                    logger.warning(f"Frame processing failed: {frame_result.get('error', 'unknown')}")
                    continue
                
                # Create display frame with info overlay
                display_frame = self.create_info_overlay(
                    frame_result['annotated_frame'], 
                    frame_result
                )
                
                # Display frame
                cv2.imshow('Warehouse Tracking System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard_input(key):
                        break
                
                # Performance logging
                if Config.PERFORMANCE_LOG_INTERVAL > 0 and self.frame_count % Config.PERFORMANCE_LOG_INTERVAL == 0:
                    logger.info(f"Processed {self.frame_count} frames, avg FPS: {self.system_stats['avg_fps']:.2f}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            raise
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        logger.info("Cleaning up system resources...")
        
        # Close display windows
        cv2.destroyAllWindows()
        
        # Cleanup components
        if self.camera_manager:
            self.camera_manager.close()
        
        if self.detector_tracker:
            self.detector_tracker.cleanup()
        
        if self.database_handler:
            self.database_handler.close_connection()
        
        # Print final statistics
        self.print_system_statistics()
        
        logger.info("System shutdown complete")
        print("System shutdown complete")

def main():
    """Main entry point"""
    print("WAREHOUSE TRACKING SYSTEM")
    print("Dual ZED 2i Camera with SIFT-based Object Tracking")
    print("="*50)
    
    try:
        # Create and run the tracking system
        system = WarehouseTrackingSystem()
        system.run()
        
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        print(f"Error: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    main()
