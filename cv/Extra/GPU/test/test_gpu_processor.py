# cv/GPU/pipelines/test_gpu_processor.py

import sys
import os
import time
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker import CameraWorker
from GPU.pipelines.ring_buffer import RingBuffer
from GPU.pipelines.gpu_processor import GPUBatchProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_batch_processing(camera_ids=[8, 9], duration=30):
    """
    Test GPU batch processing with specified cameras
    
    Args:
        camera_ids: List of camera IDs to test
        duration: Test duration in seconds
    """
    logger.info(f"Starting GPU batch processing test with cameras: {camera_ids}")
    
    # Initialize components
    camera_workers = {}
    ring_buffer = RingBuffer(num_cameras=11, buffer_size=30)
    
    # Initialize GPU processor
    gpu_processor = GPUBatchProcessor(
        model_path='custom_yolo.pt',
        device='cuda:0',
        active_cameras=camera_ids,
        confidence=0.5
    )
    
    # Connect cameras
    for cam_id in camera_ids:
        worker = CameraWorker(cam_id, frame_skip=3, debug=True)
        if worker.connect():
            camera_workers[cam_id] = worker
            logger.info(f"✅ Camera {cam_id} connected")
        else:
            logger.error(f"❌ Failed to connect camera {cam_id}")
    
    if not camera_workers:
        logger.error("No cameras connected, exiting")
        return
    
    # Start processing
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    logger.info(f"Processing for {duration} seconds...")
    
    try:
        while time.time() - start_time < duration:
            # Step 1: Get frames from cameras and write to buffer
            for cam_id, worker in camera_workers.items():
                frame = worker.get_next_frame()
                if frame is not None:
                    ring_buffer.write(cam_id, frame, frame_count)
                    frame_count += 1
            
            # Step 2: Process batch through GPU
            detections_by_camera = gpu_processor.process_batch(ring_buffer)
            
            # Step 3: Log results
            for cam_id, detections in detections_by_camera.items():
                detection_count += len(detections)
                if detections:
                    logger.info(f"Camera {cam_id}: {len(detections)} detections")
                    for det in detections[:2]:  # Show first 2 detections
                        logger.info(f"  - {det['class']} (conf={det['confidence']:.2f}, "
                                  f"track_id={det['track_id']}, "
                                  f"bbox={[int(x) for x in det['bbox']]})")
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    # Cleanup
    elapsed_time = time.time() - start_time
    logger.info("\n=== Test Summary ===")
    logger.info(f"Duration: {elapsed_time:.1f} seconds")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Total detections: {detection_count}")
    logger.info(f"Average FPS: {frame_count/elapsed_time:.2f}")
    
    # Get GPU processor stats
    stats = gpu_processor.get_stats()
    logger.info(f"GPU Stats: {stats}")
    
    # Get buffer status
    buffer_status = ring_buffer.get_buffer_status()
    for cam_id in camera_ids:
        if cam_id in buffer_status:
            logger.info(f"Camera {cam_id} buffer: {buffer_status[cam_id]}")
    
    # Cleanup
    for worker in camera_workers.values():
        worker.cleanup()
    gpu_processor.cleanup()
    
    logger.info("Test completed!")

if __name__ == "__main__":
    # Test with 2-3 cameras on laptop
    # For A100, change to: camera_ids=[1,2,3,4,5,6,7,8,9,10,11]
    test_gpu_batch_processing(
        camera_ids=[8, 9],  # Start with 2 cameras
        duration=30  # 30 second test
    )