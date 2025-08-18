# cv/GPU/test/test_remote_camera.py

import cv2
import time
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker import CameraWorker
from GPU.configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_camera_8_remote():
    """Test Camera 8 with REMOTE URL"""
    print("\n=== Testing Camera 8 (REMOTE) ===\n")
    
    # Force remote mode
    Config.switch_to_remote_cameras()
    
    # Show URL being used
    camera_id = 7
    url = Config.RTSP_CAMERA_URLS.get(camera_id)
    print(f"Camera {camera_id} REMOTE URL: {url}\n")
    
    try:
        # Create and connect
        worker = CameraWorker(camera_id, frame_skip=3, debug=True)
        
        if not worker.connect():
            print("Failed to connect!")
            return
        
        print("✅ Connected! Processing frames for 10 seconds...\n")
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            frame = worker.get_next_frame()
            
            if frame is not None:
                frame_count += 1
                h, w = frame.shape[:2]
                
                if frame_count == 1:
                    print(f"First frame received: {w}x{h}")
                
                # Display
                cv2.imshow(f"Camera {camera_id} - REMOTE", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        # Results
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n=== Results ===")
        print(f"Processed frames: {frame_count}")
        print(f"Effective FPS: {fps:.2f}")
        print(f"Expected FPS: ~5 (from 15 with skip=3)")
        
        worker.cleanup()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_camera_8_remote()