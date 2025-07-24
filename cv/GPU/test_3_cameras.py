#!/usr/bin/env python3
"""
Test Script: 3-Camera Display
Test the multi-camera system with just 3 cameras first
"""

import logging
import sys
import os
import time

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

def test_3_cameras():
    """Test with just 3 cameras"""
    logger.info("🧪 TESTING 3-CAMERA DISPLAY SYSTEM")
    logger.info("=" * 50)
    
    # Use only 3 cameras for testing
    test_cameras = [7, 8, 9]  # Start with cameras 7, 8, 9
    
    logger.info(f"📹 Test cameras: {test_cameras}")
    logger.info(f"🌐 Camera URLs: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    
    try:
        # 1. Initialize queue manager
        logger.info("🔧 Initializing queue manager...")
        queue_manager = PerCameraQueueManager(
            max_cameras=11,
            active_cameras=test_cameras
        )
        
        # 2. Initialize camera thread manager
        logger.info("🔧 Initializing camera threads...")
        camera_thread_manager = CameraThreadManager(
            active_cameras=test_cameras,
            queue_manager=queue_manager
        )
        
        # 3. Initialize display manager
        logger.info("🔧 Initializing display manager...")
        display_manager = MultiCameraDisplayManager(
            active_cameras=test_cameras,
            queue_manager=queue_manager
        )
        
        # 4. Start camera threads
        logger.info("🚀 Starting camera threads...")
        camera_thread_manager.start_camera_threads()
        
        # Wait for cameras to initialize
        logger.info("⏳ Waiting for cameras to initialize...")
        time.sleep(3)
        
        # 5. Start display
        logger.info("🖥️  Starting display...")
        logger.info("Press 'q' to quit, 'f' for fullscreen")
        display_manager.start_display()
        
        # 6. Cleanup
        logger.info("🛑 Stopping camera threads...")
        camera_thread_manager.stop_camera_threads()
        
        logger.info("✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3_cameras()
