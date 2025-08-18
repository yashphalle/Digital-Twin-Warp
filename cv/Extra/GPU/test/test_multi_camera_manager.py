# cv/GPU/test/test_multi_camera.py

import cv2
import time
import sys
import os
import threading
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker import CameraWorker
from GPU.configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(name)s - %(message)s'
)

def test_single_camera_thread(camera_id: int, duration: int = 5, results: dict = None):
    """Test single camera in a thread"""
    thread_name = f"Camera-{camera_id}"
    threading.current_thread().name = thread_name
    
    print(f"\n[{thread_name}] Starting test...")
    
    try:
        worker = CameraWorker(camera_id, frame_skip=3, debug=False)
        
        if not worker.connect():
            print(f"[{thread_name}] ❌ Failed to connect")
            if results is not None:
                results[camera_id] = {'status': 'failed', 'frames': 0}
            return
            
        print(f"[{thread_name}] ✅ Connected")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            frame = worker.get_next_frame()
            if frame is not None:
                frame_count += 1
                
                # Show frame from each camera in separate window
                window_name = f"Camera {camera_id}"
                cv2.imshow(window_name, cv2.resize(frame, (640, 360)))
                cv2.waitKey(1)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"[{thread_name}] Processed {frame_count} frames in {elapsed:.1f}s = {fps:.2f} FPS")
        
        worker.cleanup()
        cv2.destroyWindow(f"Camera {camera_id}")
        
        if results is not None:
            results[camera_id] = {
                'status': 'success',
                'frames': frame_count,
                'fps': fps,
                'duration': elapsed
            }
            
    except Exception as e:
        print(f"[{thread_name}] ❌ Error: {e}")
        if results is not None:
            results[camera_id] = {'status': 'error', 'error': str(e)}

def test_multiple_cameras(camera_ids: list, duration: int = 5):
    """Test multiple cameras in parallel"""
    print(f"\n=== Testing {len(camera_ids)} Cameras in Parallel ===")
    print(f"Cameras: {camera_ids}")
    print(f"Duration: {duration} seconds\n")
    
    # Force remote cameras
    Config.switch_to_remote_cameras()
    
    # Results dictionary (thread-safe)
    results = {}
    threads = []
    
    # Start all camera threads
    start_time = time.time()
    
    for cam_id in camera_ids:
        thread = threading.Thread(
            target=test_single_camera_thread,
            args=(cam_id, duration, results)
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay to avoid connection storm
    
    # Wait for all threads
    print(f"\nWaiting for {len(threads)} camera threads...")
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n=== Results (Total time: {total_time:.1f}s) ===")
    
    successful = 0
    total_frames = 0
    
    for cam_id in camera_ids:
        result = results.get(cam_id, {})
        status = result.get('status', 'unknown')
        
        if status == 'success':
            successful += 1
            total_frames += result['frames']
            print(f"Camera {cam_id}: ✅ {result['frames']} frames @ {result['fps']:.2f} FPS")
        elif status == 'failed':
            print(f"Camera {cam_id}: ❌ Connection failed")
        elif status == 'error':
            print(f"Camera {cam_id}: ❌ Error: {result.get('error', 'Unknown')}")
        else:
            print(f"Camera {cam_id}: ❓ Unknown status")
    
    print(f"\nSummary:")
    print(f"  Successful cameras: {successful}/{len(camera_ids)}")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Average FPS per camera: {total_frames/total_time/successful:.2f}" if successful > 0 else "  No successful cameras")

def progressive_camera_test():
    """Progressive testing - start with few, then add more"""
    print("\n=== Progressive Multi-Camera Test ===\n")
    
    # Test configurations
    test_configs = [
        ([7, 8], 5),           # 2 cameras, 5 seconds
        ([5, 6, 7, 8], 5),     # 4 cameras, 5 seconds
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 10)  # All cameras, 10 seconds
    ]
    
    for cameras, duration in test_configs:
        print(f"\n{'='*60}")
        input(f"Press Enter to test {len(cameras)} cameras for {duration}s...")
        
        test_multiple_cameras(cameras, duration)
        
        # Clean up windows
        cv2.destroyAllWindows()
        
        if len(cameras) < 11:
            response = input("\nContinue to next test? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == "__main__":
    # Option 1: Test specific cameras
    print("Multi-Camera Testing Options:")
    print("1. Test 2 cameras (7, 8)")
    print("2. Test 4 cameras (5, 6, 7, 8)")  
    print("3. Test all 11 cameras")
    print("4. Progressive test (2 → 4 → 11)")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        test_multiple_cameras([7, 8], duration=5)
    elif choice == '2':
        test_multiple_cameras([5, 6, 7, 8], duration=5)
    elif choice == '3':
        test_multiple_cameras(list(range(1, 12)), duration=10)
    elif choice == '4':
        progressive_camera_test()
    else:
        print("Invalid choice")
    
    cv2.destroyAllWindows()