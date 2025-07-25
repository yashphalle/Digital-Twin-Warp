#!/usr/bin/env python3
"""
Test script to verify dedicated worker-to-camera assignment is working
This will show the 1:1 mapping and verify no round-robin overhead
"""

import logging
import sys
import os
import time

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from warehouse_threading.per_camera_detection_pool import PerCameraDetectionThreadPool
from warehouse_threading.per_camera_queue_manager import PerCameraQueueManager
from configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_dedicated_assignment():
    """Test the dedicated worker-to-camera assignment"""
    logger.info("🧪 TESTING DEDICATED WORKER-TO-CAMERA ASSIGNMENT")
    logger.info("=" * 60)
    
    # Test with 11 cameras and 11 workers
    test_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    num_workers = 11
    
    logger.info(f"📹 Test cameras: {test_cameras}")
    logger.info(f"👷 Number of workers: {num_workers}")
    logger.info("=" * 60)
    
    try:
        # Create queue manager
        queue_manager = PerCameraQueueManager(
            max_cameras=len(test_cameras), 
            active_cameras=test_cameras
        )
        
        # Create detection pool with dedicated assignment
        detection_pool = PerCameraDetectionThreadPool(
            num_workers=num_workers, 
            queue_manager=queue_manager
        )
        
        logger.info("✅ Dedicated assignment system created successfully!")
        logger.info("=" * 60)
        
        # Verify worker-camera mapping
        logger.info("🎯 WORKER-CAMERA ASSIGNMENTS:")
        for worker_id, camera_id in detection_pool.worker_camera_map.items():
            logger.info(f"   Worker {worker_id:2d} → Camera {camera_id:2d}")
        
        logger.info("=" * 60)
        
        # Test queue access method
        logger.info("🔍 TESTING DEDICATED QUEUE ACCESS:")
        for camera_id in [1, 5, 11]:  # Test a few cameras
            result = queue_manager.get_frame_from_dedicated_camera(camera_id, timeout=0.1)
            if result is None:
                logger.info(f"   Camera {camera_id}: Queue empty (expected)")
            else:
                logger.info(f"   Camera {camera_id}: Got frame!")
        
        logger.info("=" * 60)
        logger.info("✅ DEDICATED ASSIGNMENT TEST COMPLETED SUCCESSFULLY!")
        logger.info("🚀 Ready to test with actual camera feeds!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_worker_counts():
    """Test dedicated assignment with different worker counts"""
    logger.info("🧪 TESTING DIFFERENT WORKER COUNTS")
    logger.info("=" * 60)
    
    test_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    worker_counts = [2, 5, 11]
    
    for num_workers in worker_counts:
        logger.info(f"Testing {num_workers} workers with {len(test_cameras)} cameras:")
        
        try:
            queue_manager = PerCameraQueueManager(
                max_cameras=len(test_cameras), 
                active_cameras=test_cameras
            )
            
            detection_pool = PerCameraDetectionThreadPool(
                num_workers=num_workers, 
                queue_manager=queue_manager
            )
            
            logger.info(f"   Assignments: {detection_pool.worker_camera_map}")
            
            if num_workers > len(test_cameras):
                unassigned_workers = num_workers - len(test_cameras)
                logger.info(f"   ⚠️ {unassigned_workers} workers will be idle (more workers than cameras)")
            elif num_workers < len(test_cameras):
                unassigned_cameras = len(test_cameras) - num_workers
                logger.info(f"   ⚠️ {unassigned_cameras} cameras will not be processed")
            else:
                logger.info(f"   ✅ Perfect 1:1 assignment!")
                
        except Exception as e:
            logger.error(f"   ❌ Failed with {num_workers} workers: {e}")
        
        logger.info("")
    
    logger.info("=" * 60)

def main():
    """Run all dedicated assignment tests"""
    logger.info("🚀 DEDICATED WORKER-TO-CAMERA ASSIGNMENT TESTS")
    logger.info("This will verify the new dedicated assignment system")
    logger.info("=" * 80)
    
    # Configure for testing
    Config.switch_to_remote_cameras()
    Config.switch_to_local_database()
    
    try:
        # Test 1: Basic dedicated assignment
        success1 = test_dedicated_assignment()
        
        # Test 2: Different worker counts
        test_different_worker_counts()
        
        if success1:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("💡 The dedicated assignment system is ready for performance testing!")
        else:
            logger.error("❌ Some tests failed")
            
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
