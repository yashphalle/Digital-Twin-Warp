# cv/GPU/pipelines/test_gpu_processor_gui.py

import sys
import os
import time
import logging
import cv2
import numpy as np
from collections import defaultdict

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

def draw_detections(frame, detections, camera_id):
    """Draw bounding boxes and labels on frame"""
    # Create a copy to avoid modifying original
    display_frame = frame.copy()
    
    # Define colors for different classes
    colors = {
        'Pallet': (0, 255, 0),      # Green
        'pallet': (0, 255, 0),      # Green
        'Forklift': (255, 0, 0),    # Blue
        'forklift': (255, 0, 0),    # Blue
        'default': (0, 255, 255)    # Yellow
    }
    
    for det in detections:
        # Get bbox coordinates
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        
        # Get color based on class
        color = colors.get(det['class'], colors['default'])
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{det['class']} #{det['track_id'] or 'N/A'} ({det['confidence']:.2f})"
        
        # Calculate label position
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
        
        # Draw label background
        cv2.rectangle(display_frame, 
                     (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0], label_y + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(display_frame, label,
                   (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
    
    # Add camera info
    info_text = f"Camera {camera_id} - {len(detections)} detections"
    cv2.putText(display_frame, info_text,
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               1, (255, 255, 255), 2)
    
    # Add FPS info
    return display_frame

def test_gpu_batch_processing_with_gui(camera_ids=[8, 9], duration=300):
    """
    Test GPU batch processing with GUI visualization
    
    Args:
        camera_ids: List of camera IDs to test
        duration: Test duration in seconds
    """
    logger.info(f"Starting GPU batch processing test with GUI for cameras: {camera_ids}")
    
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
        worker = CameraWorker(cam_id, frame_skip=3, debug=False)  # Less debug output
        if worker.connect():
            camera_workers[cam_id] = worker
            logger.info(f"✅ Camera {cam_id} connected")
        else:
            logger.error(f"❌ Failed to connect camera {cam_id}")
    
    if not camera_workers:
        logger.error("No cameras connected, exiting")
        return
    
    # Create windows for each camera
    for cam_id in camera_ids:
        window_name = f"Camera {cam_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)  # Resize to fit screen
    
    # Start processing
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    fps_counter = defaultdict(lambda: {'frames': 0, 'start_time': time.time()})
    
    logger.info(f"Processing for {duration} seconds... Press 'q' to quit")
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Step 1: Get frames from cameras and write to buffer
            current_frames = {}
            for cam_id, worker in camera_workers.items():
                frame = worker.get_next_frame()
                if frame is not None:
                    ring_buffer.write(cam_id, frame, frame_count)
                    current_frames[cam_id] = frame
                    frame_count += 1
                    fps_counter[cam_id]['frames'] += 1
            
            # Step 2: Process batch through GPU
            detections_by_camera = gpu_processor.process_batch(ring_buffer)
            
            # Step 3: Visualize results
            for cam_id, frame in current_frames.items():
                detections = detections_by_camera.get(cam_id, [])
                detection_count += len(detections)
                
                # Draw detections on frame
                display_frame = draw_detections(frame, detections, cam_id)
                
                # Calculate and display FPS
                elapsed = time.time() - fps_counter[cam_id]['start_time']
                if elapsed > 1.0:
                    fps = fps_counter[cam_id]['frames'] / elapsed
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(display_frame, fps_text,
                               (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 255), 2)
                    
                    # Reset FPS counter
                    fps_counter[cam_id]['frames'] = 0
                    fps_counter[cam_id]['start_time'] = time.time()
                
                # Show frame
                cv2.imshow(f"Camera {cam_id}", display_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit key pressed")
                break
            
            # Control loop rate (aim for ~10 FPS display update)
            loop_time = time.time() - loop_start
            if loop_time < 0.1:
                time.sleep(0.1 - loop_time)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    # Cleanup
    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    
    logger.info("\n=== Test Summary ===")
    logger.info(f"Duration: {elapsed_time:.1f} seconds")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Total detections: {detection_count}")
    logger.info(f"Average FPS: {frame_count/elapsed_time:.2f}")
    
    # Get GPU processor stats
    stats = gpu_processor.get_stats()
    logger.info(f"GPU Stats: {stats}")
    
    # Cleanup
    for worker in camera_workers.values():
        worker.cleanup()
    gpu_processor.cleanup()
    
    logger.info("Test completed!")

if __name__ == "__main__":
    # Test with GUI
    test_gpu_batch_processing_with_gui(
        camera_ids=[8, 9],  # 2 cameras for laptop
        duration=300  # 5 minutes
    )