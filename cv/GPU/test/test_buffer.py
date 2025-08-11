# cv/GPU/test/test_all_cameras_buffer.py

import time
import sys
import os
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker import CameraWorker
from GPU.pipelines.ring_buffer import RingBuffer
from GPU.configs.config import Config

def test_all_cameras_ring_buffer():
    """Test ALL 11 cameras with ring buffer"""
    print("\n=== Testing ALL 11 Cameras → Ring Buffer ===\n")
    
    Config.switch_to_remote_cameras()
    
    # Full size buffer for 11 cameras
    buffer = RingBuffer(num_cameras=11, buffer_size=30)
    
    # ALL cameras
    all_cameras = list(range(1, 12))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    duration = 30  # 30 seconds test
    
    # Track results
    results = {cam_id: {'status': 'starting', 'frames': 0} for cam_id in all_cameras}
    results_lock = threading.Lock()
    
    def camera_worker(cam_id):
        """Simple worker for each camera"""
        try:
            worker = CameraWorker(cam_id, frame_skip=5, debug=False)
            
            if not worker.connect():
                with results_lock:
                    results[cam_id]['status'] = 'failed'
                print(f"Camera {cam_id}: ❌ Failed to connect")
                return
            
            with results_lock:
                results[cam_id]['status'] = 'connected'
            print(f"Camera {cam_id}: ✅ Connected")
            
            start_time = time.time()
            frames = 0
            
            while time.time() - start_time < duration:
                frame = worker.get_next_frame()
                if frame is not None:
                    buffer.write(cam_id, frame, frames)
                    frames += 1
                    
                    # Update results
                    with results_lock:
                        results[cam_id]['frames'] = frames
            
            worker.cleanup()
            print(f"Camera {cam_id}: Completed - {frames} frames")
            
        except Exception as e:
            with results_lock:
                results[cam_id]['status'] = f'error: {e}'
            print(f"Camera {cam_id}: ❌ Error: {e}")
    
    # Start all camera threads
    threads = []
    print("Starting 11 camera threads...\n")
    
    for cam_id in all_cameras:
        thread = threading.Thread(
            target=camera_worker,
            args=(cam_id,),
            name=f"Camera-{cam_id}"
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.2)  # Stagger starts to avoid connection storm
    
    # Monitor progress
    print("\n=== Monitoring Progress ===\n")
    
    for i in range(6):  # Check every 5 seconds
        time.sleep(5)
        
        # Get current status
        with results_lock:
            current_results = results.copy()
        
        # Count statuses
        connected = sum(1 for r in current_results.values() if r['status'] == 'connected')
        failed = sum(1 for r in current_results.values() if 'failed' in r['status'])
        total_frames = sum(r['frames'] for r in current_results.values())
        
        print(f"\n[{i*5 + 5}s] Connected: {connected}/11, Failed: {failed}, Total frames: {total_frames}")
        
        # Test batch retrieval
        batch = buffer.get_batch(max_age=0.5)
        print(f"Batch ready: {len(batch)} cameras")
        
        if len(batch) == 11:
            print("✅ All 11 cameras synchronized!")
        
        # Show per-camera stats
        if i % 2 == 1:  # Every 10 seconds
            print("\nPer-camera frames:")
            for cam_id in all_cameras:
                frames = current_results[cam_id]['frames']
                status = current_results[cam_id]['status']
                symbol = "✅" if status == 'connected' else "❌"
                print(f"  Camera {cam_id:2d}: {symbol} {frames:3d} frames")
    
    # Wait for all threads
    print("\nWaiting for all threads to complete...")
    for thread in threads:
        thread.join()
    
    # Final results
    print("\n=== Final Results ===\n")
    
    with results_lock:
        final_results = results.copy()
    
    successful = sum(1 for r in final_results.values() if r['frames'] > 0)
    total_frames = sum(r['frames'] for r in final_results.values())
    
    print(f"Successful cameras: {successful}/11")
    print(f"Total frames processed: {total_frames}")
    print(f"Average FPS per camera: {total_frames / (successful * duration):.2f}")
    
    # Memory calculation
    print(f"\n=== Memory Usage ===")
    print(f"Buffer: 11 cameras × 30 frames × 4.1MB = ~1,353 MB")
    print(f"Actual frames in memory: {min(30, max(r['frames'] for r in final_results.values()))} per camera")

if __name__ == "__main__":
    test_all_cameras_ring_buffer()