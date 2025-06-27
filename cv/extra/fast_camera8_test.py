#!/usr/bin/env python3
"""
Fast Camera 8 Test - Optimized for Low Latency
Reduced latency object detection with frame skipping and lighter model
Same coordinate system as simple_camera8_test.py
"""

import cv2
import time
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

def main():
    print("🚀 FAST CAMERA 8 TEST - LOW LATENCY VERSION")
    print("=" * 55)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Optimizations: Frame skipping + Lighter model")
    print("Press 'q' to quit")
    print("=" * 55)
    
    # Initialize tracker for Camera 8 with optimizations
    tracker = DetectorTracker()
    tracker.set_camera_id(8)
    
    # Override model settings for speed (use smaller/faster model)
    # You can experiment with these values:
    tracker.confidence_threshold = 0.3  # Slightly higher threshold for speed
    tracker.nms_threshold = 0.4  # Lower NMS threshold for speed
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    print(f"🎯 Detection optimized for speed (conf: 0.3, nms: 0.4)")
    
    # Connect to Camera 8
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    
    # Optimize camera buffer for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        return
    
    print("✅ Connected to Camera 8 (optimized buffer)")
    print("🎯 Looking for objects...")
    
    frame_count = 0
    detection_count = 0
    skip_frames = 20  # Process every 21st frame (skip 20) - Ultra fast mode
    last_tracked_objects = []  # Store last detection results
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    
    consecutive_failures = 0
    max_failures = 10  # Try 10 times before attempting reconnect
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            consecutive_failures += 1
            print(f"⚠️  Failed to read frame (attempt {consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("🔄 Attempting to reconnect to camera...")
                cap.release()
                time.sleep(2)  # Wait 2 seconds before reconnecting
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Re-apply buffer optimization
                consecutive_failures = 0
                if cap.isOpened():
                    print("✅ Reconnected successfully!")
                else:
                    print("❌ Reconnection failed, retrying in 5 seconds...")
                    time.sleep(5)
            continue
        
        # Reset failure counter on successful frame read
        consecutive_failures = 0
        
        frame_count += 1
        fps_counter += 1
        
        # Apply fisheye correction (always needed for display)
        corrected_frame = fisheye_corrector.correct(raw_frame)
        
        # Frame skipping logic for detection
        if frame_count % (skip_frames + 1) == 0:
            # Process this frame for detection
            detection_count += 1
            tracked_objects, perf_stats = tracker.process_frame(corrected_frame)
            last_tracked_objects = tracked_objects  # Store for next frames
        else:
            # Skip detection, use last results
            tracked_objects = last_tracked_objects
        
        # Always draw (even with old detections for smooth display)
        annotated_frame = tracker.draw_tracked_objects(corrected_frame, tracked_objects)
        
        # Draw calibrated zone overlay
        annotated_frame = tracker.draw_calibrated_zone_overlay(annotated_frame)
        
        # Calculate FPS
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = fps_counter / (current_time - last_fps_time)
            fps_counter = 0
            last_fps_time = current_time
        else:
            fps = 0
        
        # Add status info with FPS and skip info
        status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Skip: {skip_frames}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add optimization indicators
        detection_ratio = detection_count / frame_count if frame_count > 0 else 0
        opt_text = f"FAST MODE - Detection Rate: {detection_ratio:.2f} ({detection_count}/{frame_count})"
        cv2.putText(annotated_frame, opt_text, (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Print coordinates only when we actually detect (less spam)
        if frame_count % (skip_frames + 1) == 0 and tracked_objects:
            print(f"\n📍 Frame {frame_count}: {len(tracked_objects)} objects detected")
            for i, obj in enumerate(tracked_objects):
                real_center = obj.get('real_center')
                confidence = obj.get('confidence', 0)
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Confidence: {confidence:.2f}")
        
        # Resize for display (smaller for better performance)
        height, width = annotated_frame.shape[:2]
        display_width = 960  # Smaller than original for speed
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        
        # Display
        cv2.imshow('Camera 8 - Fast Mode (Low Latency)', display_frame)
        
        # Check for quit and dynamic controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Quit requested by user")
            break
        elif key == ord('+') or key == ord('='):
            # Increase skip frames (less detection, more speed)
            skip_frames = min(skip_frames + 1, 10)
            print(f"🔧 Skip frames increased to {skip_frames}")
        elif key == ord('-'):
            # Decrease skip frames (more detection, less speed)
            skip_frames = max(skip_frames - 1, 0)
            print(f"🔧 Skip frames decreased to {skip_frames}")
        elif key == ord('r'):
            # Reset skip frames
            skip_frames = 20
            print(f"🔧 Skip frames reset to {skip_frames}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    
    # Performance summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    
    print(f"\n✅ Fast test completed")
    print(f"📊 Performance Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Detection frames: {detection_count}")
    print(f"   Detection rate: {detection_rate:.2f}")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Runtime: {total_time:.1f}s")
    print(f"💡 Controls: +/- to adjust skip frames, 'r' to reset, 'q' to quit")

if __name__ == "__main__":
    main() 