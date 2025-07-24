#!/usr/bin/env python3
"""
11-Camera Display System
Simple display-only system for warehouse monitoring
"""

import logging
import sys
import os
import time
import signal

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from per_camera_queue_manager import PerCameraQueueManager
from camera_threads import CameraThreadManager
from multi_camera_display import MultiCameraDisplayManager
from configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ElevenCameraDisplaySystem:
    """
    11-Camera Display System
    Simplified system focused on display only (no detection/database)
    """
    
    def __init__(self):
        self.active_cameras = Config.ACTIVE_CAMERAS  # All 11 cameras
        self.queue_manager = None
        self.camera_thread_manager = None
        self.display_manager = None
        self.running = False
        
        logger.info("🏭 11-Camera Display System Initializing...")
        logger.info(f"📹 Active cameras: {self.active_cameras}")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            # 1. Initialize queue manager for 11 cameras
            self.queue_manager = PerCameraQueueManager(
                max_cameras=11,
                active_cameras=self.active_cameras
            )
            
            # 2. Initialize camera thread manager
            self.camera_thread_manager = CameraThreadManager(
                active_cameras=self.active_cameras,
                queue_manager=self.queue_manager
            )
            
            # 3. Initialize display manager
            self.display_manager = MultiCameraDisplayManager(
                active_cameras=self.active_cameras,
                queue_manager=self.queue_manager
            )
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def start_system(self):
        """Start all system components"""
        try:
            self.running = True
            
            # 1. Start camera threads
            logger.info("🚀 Starting camera threads...")
            self.camera_thread_manager.start_camera_threads()
            
            # Wait a moment for cameras to initialize
            time.sleep(2)
            
            # 2. Start display (this will block until user quits)
            logger.info("🖥️  Starting display system...")
            self.display_manager.start_display()
            
        except Exception as e:
            logger.error(f"❌ System start failed: {e}")
            self.stop_system()
    
    def stop_system(self):
        """Stop all system components"""
        logger.info("🛑 Stopping 11-Camera Display System...")
        
        self.running = False
        
        # Stop display
        if self.display_manager:
            self.display_manager.stop_display()
        
        # Stop camera threads
        if self.camera_thread_manager:
            self.camera_thread_manager.stop_camera_threads()
        
        logger.info("✅ System stopped successfully")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info("🛑 Shutdown signal received")
    sys.exit(0)

def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("🏭 11-CAMERA WAREHOUSE DISPLAY SYSTEM")
    logger.info("=" * 80)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Display configuration
    logger.info("📋 SYSTEM CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"🎥 Active Cameras: {Config.ACTIVE_CAMERAS}")
    logger.info(f"🌐 Camera URLs: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    logger.info(f"🔧 Fisheye Correction: {'ON' if Config.FISHEYE_CORRECTION_ENABLED else 'OFF'}")
    logger.info("=" * 50)
    
    # Initialize and run system
    try:
        system = ElevenCameraDisplaySystem()
        
        if system.initialize_components():
            logger.info("🚀 Starting 11-camera display system...")
            system.start_system()
        else:
            logger.error("❌ Failed to initialize system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 System shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
