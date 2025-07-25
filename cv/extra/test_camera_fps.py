#!/usr/bin/env python3
"""
Simple Camera FPS Test Script
Tests real FPS from camera without any processing overhead
"""

import cv2
import time
import sys
import os
from typing import Dict, List

# Add cv directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from cv.configs.config import Config
    CAMERA_URLS = Config.RTSP_CAMERA_URLS
    print("✅ Using camera URLs from config")
except ImportError:
    # Fallback camera URLs
    CAMERA_URLS = {
        8: "rtsp://admin:wearewarp!@104.181.138.58:5568/Streaming/channels/1"
    }
    print("⚠️  Using fallback camera URLs")

def test_camera_fps(camera_id: int, rtsp_url: str, test_duration: int = 30) -> Dict:
    """
    Test real FPS from camera without any processing
    
    Args:
        camera_id: Camera ID number
        rtsp_url: RTSP URL for the camera
        test_duration: How long to test in seconds
        
    Returns:
        Dict with FPS statistics
    """
    print(f"\n🎥 Testing Camera {camera_id} FPS...")
    print(f"   URL: {rtsp_url}")
    print(f"   Duration: {test_duration} seconds")
    
    # Connect to camera
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"❌ Failed to connect to Camera {camera_id}")
        return {"success": False, "error": "Connection failed"}
    
    # Set minimal buffer to avoid frame accumulation
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)  # Shorter timeout for testing
    
    print("✅ Connected to camera")
    
    # Get camera properties
    fps_property = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Camera reports FPS: {fps_property}")
    print(f"   Resolution: {width}x{height}")
    
    # Test actual FPS
    frame_count = 0
    failed_reads = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_measurements = []
    
    print(f"\n📊 Starting FPS measurement...")
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Stop after test duration
            if elapsed >= test_duration:
                break
            
            # Read frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_count += 1
                
                # Calculate FPS every 5 seconds
                if current_time - last_fps_time >= 5.0:
                    interval_fps = frame_count / elapsed
                    fps_measurements.append(interval_fps)
                    print(f"   {elapsed:.1f}s: {frame_count} frames, {interval_fps:.2f} FPS")
                    last_fps_time = current_time
            else:
                failed_reads += 1
                time.sleep(0.01)  # Brief pause on failed read
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    
    finally:
        cap.release()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    
    results = {
        "success": True,
        "camera_id": camera_id,
        "test_duration": total_time,
        "total_frames": frame_count,
        "failed_reads": failed_reads,
        "average_fps": average_fps,
        "reported_fps": fps_property,
        "resolution": (width, height),
        "fps_measurements": fps_measurements
    }
    
    print(f"\n📈 Camera {camera_id} Results:")
    print(f"   Total frames: {frame_count}")
    print(f"   Failed reads: {failed_reads}")
    print(f"   Test duration: {total_time:.2f}s")
    print(f"   Average FPS: {average_fps:.2f}")
    print(f"   Camera reported FPS: {fps_property}")
    
    if fps_measurements:
        min_fps = min(fps_measurements)
        max_fps = max(fps_measurements)
        print(f"   FPS range: {min_fps:.2f} - {max_fps:.2f}")
    
    return results

def test_frame_timing(camera_id: int, rtsp_url: str, num_frames: int = 100) -> Dict:
    """
    Test frame-to-frame timing to detect irregularities
    """
    print(f"\n⏱️  Testing frame timing for Camera {camera_id}...")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return {"success": False, "error": "Connection failed"}
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_times = []
    frame_intervals = []
    
    try:
        for i in range(num_frames):
            start_time = time.time()
            ret, frame = cap.read()
            read_time = time.time()
            
            if ret and frame is not None:
                frame_times.append(read_time - start_time)
                
                if len(frame_times) > 1:
                    # Time between frame reads
                    interval = read_time - (frame_times[-2] + start_time)
                    frame_intervals.append(interval)
                
                if i % 20 == 0:
                    print(f"   Frame {i+1}/{num_frames}")
            else:
                print(f"   Failed to read frame {i+1}")
    
    finally:
        cap.release()
    
    if frame_intervals:
        avg_interval = sum(frame_intervals) / len(frame_intervals)
        expected_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        
        print(f"   Average frame interval: {avg_interval*1000:.2f}ms")
        print(f"   Calculated FPS: {expected_fps:.2f}")
        
        return {
            "success": True,
            "average_interval": avg_interval,
            "calculated_fps": expected_fps,
            "frame_intervals": frame_intervals
        }
    
    return {"success": False, "error": "No valid frames"}

def main():
    """Main test function"""
    print("🎯 Camera FPS Test Script")
    print("=" * 50)
    
    # Test configuration
    test_duration = 30  # seconds
    
    # Test all configured cameras
    results = {}
    
    for camera_id, rtsp_url in CAMERA_URLS.items():
        try:
            # Test FPS
            fps_result = test_camera_fps(camera_id, rtsp_url, test_duration)
            
            # Test frame timing
            timing_result = test_frame_timing(camera_id, rtsp_url, 50)
            
            results[camera_id] = {
                "fps_test": fps_result,
                "timing_test": timing_result
            }
            
        except Exception as e:
            print(f"❌ Error testing Camera {camera_id}: {e}")
            results[camera_id] = {"error": str(e)}
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    
    for camera_id, result in results.items():
        if "error" in result:
            print(f"Camera {camera_id}: ❌ {result['error']}")
        else:
            fps_test = result.get("fps_test", {})
            timing_test = result.get("timing_test", {})
            
            if fps_test.get("success"):
                avg_fps = fps_test["average_fps"]
                reported_fps = fps_test["reported_fps"]
                print(f"Camera {camera_id}: {avg_fps:.2f} FPS (reported: {reported_fps})")
                
                if timing_test.get("success"):
                    calc_fps = timing_test["calculated_fps"]
                    print(f"             Timing-based FPS: {calc_fps:.2f}")
            else:
                print(f"Camera {camera_id}: ❌ Test failed")

if __name__ == "__main__":
    main()
