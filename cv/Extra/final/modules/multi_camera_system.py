#!/usr/bin/env python3
"""
Multi-Camera System Module
Multi-camera coordination and management for warehouse tracking system
Extracted from main.py for modular architecture
"""

import cv2
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MultiCameraCPUSystem:
    """Multi-camera CPU-based tracking system coordinator"""
    
    def __init__(self, active_cameras: List[int], gui_cameras: List[int], 
                 enable_gui: bool = True, tracker_class=None):
        self.active_cameras = active_cameras
        self.gui_cameras = gui_cameras if enable_gui else []
        self.enable_gui = enable_gui
        self.tracker_class = tracker_class  # Injected tracker class
        self.trackers = {}
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.processed_frame_count = 0
        self.frame_skip = 20  # Process every 20th frame

        # Simple FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        logger.info("🎛️ Multi-Camera CPU System Configuration:")
        logger.info(f"📹 Active Cameras: {self.active_cameras}")
        logger.info(f"🖥️ GUI Cameras: {self.gui_cameras}")
        logger.info(f"🎛️ GUI Mode: {'ENABLED' if self.enable_gui else 'HEADLESS'}")

    def set_tracker_class(self, tracker_class):
        """Set the tracker class to use for camera initialization"""
        self.tracker_class = tracker_class

    def initialize_cameras(self) -> bool:
        """Initialize all active cameras"""
        if not self.tracker_class:
            logger.error("❌ No tracker class provided! Use set_tracker_class() first.")
            return False
            
        logger.info(f"🔧 Initializing {len(self.active_cameras)} cameras...")

        for cam_id in self.active_cameras:
            try:
                logger.info(f"🔧 Initializing Camera {cam_id}...")
                tracker = self.tracker_class(camera_id=cam_id)

                if tracker.connect_camera():
                    self.trackers[cam_id] = tracker
                    logger.info(f"✅ Camera {cam_id} initialized successfully")
                else:
                    logger.error(f"❌ Failed to connect Camera {cam_id}")

            except Exception as e:
                logger.error(f"❌ Error initializing Camera {cam_id}: {e}")

        connected_cameras = len(self.trackers)
        if connected_cameras == 0:
            logger.error("❌ No cameras connected successfully!")
            return False

        logger.info(f"🚀 {connected_cameras} out of {len(self.active_cameras)} cameras initialized successfully!")
        return True

    def process_camera_frame(self, cam_id: int, should_process: bool) -> bool:
        """Process a single camera frame"""
        if cam_id not in self.trackers:
            return False

        tracker = self.trackers[cam_id]
        if not tracker.camera_manager.is_connected():
            return False

        try:
            if should_process:
                # Read and process frame
                ret, frame = tracker.camera_manager.read_frame()
                if ret:
                    # Process frame using frame processor
                    processed_frame = tracker.frame_processor.process_frame(frame)

                    # Show GUI if enabled for this camera
                    if self.enable_gui and cam_id in self.gui_cameras:
                        cv2.imshow(f"CPU Camera {cam_id}", processed_frame)
                    return True
            else:
                # Skip processing but still read frame to prevent buffer buildup
                ret, frame = tracker.camera_manager.read_frame()
                return ret
        except Exception as e:
            logger.warning(f"Error processing Camera {cam_id}: {e}")
            return False

    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        total_detections = sum(
            len(tracker.frame_processor.final_tracked_detections) 
            for tracker in self.trackers.values()
        )
        
        total_db_saved = sum(
            tracker.db_handler.get_detection_count() 
            for tracker in self.trackers.values() 
            if hasattr(tracker, 'db_handler')
        )
        
        camera_stats = {}
        for cam_id, tracker in self.trackers.items():
            camera_stats[cam_id] = {
                'connected': tracker.camera_manager.is_connected(),
                'detections': len(tracker.frame_processor.final_tracked_detections),
                'db_saved': tracker.db_handler.get_detection_count() if hasattr(tracker, 'db_handler') else 0,
                'camera_stats': tracker.camera_manager.get_statistics()
            }
        
        return {
            'frame_count': self.frame_count,
            'processed_frame_count': self.processed_frame_count,
            'total_detections': total_detections,
            'total_db_saved': total_db_saved,
            'connected_cameras': len(self.trackers),
            'active_cameras': len(self.active_cameras),
            'camera_stats': camera_stats,
            'fps': self.current_fps
        }

    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input for GUI mode. Returns True to continue, False to quit"""
        if not self.enable_gui:
            return True
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            logger.info("Quit requested by user")
            return False
        elif key == ord('n'):  # Next prompt (apply to all cameras)
            for tracker in self.trackers.values():
                if hasattr(tracker, 'pallet_detector'):
                    tracker.pallet_detector.next_prompt()
            logger.info("Switched to next detection prompt for all cameras")
        elif key == ord('p'):  # Previous prompt (apply to all cameras)
            for tracker in self.trackers.values():
                if hasattr(tracker, 'pallet_detector'):
                    tracker.pallet_detector.previous_prompt()
            logger.info("Switched to previous detection prompt for all cameras")
        
        return True

    def log_progress(self):
        """Log system progress and statistics"""
        if self.processed_frame_count % 10 == 0 and self.processed_frame_count > 0:
            stats = self.get_system_statistics()
            logger.info(
                f"Multi-camera processed {self.processed_frame_count} frames "
                f"(skipped {self.frame_count - self.processed_frame_count}): "
                f"{stats['total_detections']} total objects tracked, "
                f"{stats['total_db_saved']} saved to DB across "
                f"{stats['connected_cameras']} cameras"
            )

    def run(self):
        """Run multi-camera CPU tracking"""
        if not self.initialize_cameras():
            return

        logger.info("🚀 Starting multi-camera CPU tracking...")
        self.running = True
        self.frame_count = 0
        self.processed_frame_count = 0

        try:
            while self.running:
                self.frame_count += 1

                # Frame skipping logic - process every FRAME_SKIP frames
                should_process = self.frame_count % self.frame_skip == 0

                # Process each active camera
                for cam_id in self.active_cameras:
                    self.process_camera_frame(cam_id, should_process)

                if should_process:
                    self.processed_frame_count += 1
                    self.fps_frame_count += 1

                    # Calculate FPS every 5 seconds
                    current_time = time.time()
                    elapsed_time = current_time - self.fps_start_time

                    if elapsed_time >= 5.0:  # Every 5 seconds
                        self.current_fps = self.fps_frame_count / elapsed_time
                        logger.info(f"FPS: {self.current_fps:.2f}")

                        # Reset for next calculation
                        self.fps_start_time = current_time
                        self.fps_frame_count = 0

                # Handle keyboard input
                if not self.handle_keyboard_input():
                    break

                # Log progress
                self.log_progress()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during multi-camera tracking: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources"""
        self.running = False

        # Get final database statistics before cleanup
        final_stats = self.get_system_statistics()
        
        for tracker in self.trackers.values():
            tracker.cleanup()

        cv2.destroyAllWindows()
        logger.info(f"💾 Total detections saved to database: {final_stats['total_db_saved']}")
        logger.info("✅ Multi-camera CPU system shutdown complete")

    def get_connected_cameras(self) -> List[int]:
        """Get list of successfully connected camera IDs"""
        return list(self.trackers.keys())

    def get_camera_tracker(self, cam_id: int):
        """Get tracker instance for specific camera"""
        return self.trackers.get(cam_id)

    def is_running(self) -> bool:
        """Check if the system is currently running"""
        return self.running

    def stop(self):
        """Stop the multi-camera system"""
        self.running = False
        logger.info("Multi-camera system stop requested")

    def reconnect_camera(self, cam_id: int) -> bool:
        """Attempt to reconnect a specific camera"""
        if cam_id not in self.trackers:
            logger.warning(f"Camera {cam_id} not in tracker list")
            return False
            
        tracker = self.trackers[cam_id]
        logger.info(f"Attempting to reconnect Camera {cam_id}...")
        
        if tracker.camera_manager.reconnect_camera():
            logger.info(f"✅ Camera {cam_id} reconnected successfully")
            return True
        else:
            logger.error(f"❌ Failed to reconnect Camera {cam_id}")
            return False

    def get_system_health(self) -> Dict:
        """Get system health status"""
        connected = 0
        disconnected = 0
        
        for tracker in self.trackers.values():
            if tracker.camera_manager.is_connected():
                connected += 1
            else:
                disconnected += 1
        
        return {
            'total_cameras': len(self.active_cameras),
            'connected_cameras': connected,
            'disconnected_cameras': disconnected,
            'health_percentage': (connected / len(self.active_cameras)) * 100 if self.active_cameras else 0,
            'running': self.running
        }
