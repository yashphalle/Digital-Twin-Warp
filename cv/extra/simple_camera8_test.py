#!/usr/bin/env python3
"""
Simple Camera 8 Test with New Coordinate System and Fisheye Correction
Real-time object detection and coordinate display for Camera 8
"""

import cv2
import time
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector
# from watermark_remover import WatermarkRemover  # Temporarily disabled

def main():
    print("🔧 SIMPLE CAMERA 8 TEST WITH FISHEYE CORRECTION")
    print("=" * 50)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Fisheye Correction: ENABLED")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Initialize tracker for Camera 8
    tracker = DetectorTracker()
    tracker.set_camera_id(8)
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    # watermark_remover = WatermarkRemover(camera_id=8)  # Temporarily disabled
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    print(f"ℹ️  Watermark remover disabled - testing camera hardware changes")
    
    # Connect to Camera 8
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        return
    
    print("✅ Connected to Camera 8")
    print("🎯 Looking for objects...")
    
    frame_count = 0
    
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
        
        # Apply fisheye correction FIRST
        corrected_frame = fisheye_corrector.correct(raw_frame)
        
        # Process corrected frame directly (no watermark removal)
        tracked_objects, perf_stats = tracker.process_frame(corrected_frame)
        
        # Draw detections with coordinates on corrected frame
        annotated_frame = tracker.draw_tracked_objects(corrected_frame, tracked_objects)
        
        # Draw calibrated zone overlay
        annotated_frame = tracker.draw_calibrated_zone_overlay(annotated_frame)
        
        # Add status info
        status_text = f"Objects: {len(tracked_objects)} | Camera 8 - RAW FEED"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add processing indicators
        cv2.putText(annotated_frame, "FISHEYE CORRECTED - TESTING HARDWARE WATERMARKS", (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if tracked_objects:
            print(f"\n📍 {len(tracked_objects)} objects detected")
            for i, obj in enumerate(tracked_objects):
                real_center = obj.get('real_center')
                confidence = obj.get('confidence', 0)
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Confidence: {confidence:.2f}")
                else:
                    print(f"  Object {i+1}: No coordinates | Confidence: {confidence:.2f}")
        
        # Resize for display (4K is too big for most screens)
        height, width = annotated_frame.shape[:2]
        display_width = 1280
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        
        # Display
        cv2.imshow('Camera 8 - Raw Feed (Testing Hardware)', display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    print("\n✅ Test completed")

if __name__ == "__main__":
    main() 