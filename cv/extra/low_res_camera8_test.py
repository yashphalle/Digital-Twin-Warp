#!/usr/bin/env python3
"""
Low Resolution Camera 8 Test - Process smaller frames for speed
Resize frames before detection, then scale coordinates back up
Same coordinate system accuracy but much faster processing
"""

import cv2
import time
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

def main():
    print("📏 LOW RESOLUTION CAMERA 8 TEST - RESIZE FOR SPEED")
    print("=" * 60)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Strategy: Resize to 1920x1080 before detection (50% size)")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize tracker for Camera 8
    tracker = DetectorTracker()
    tracker.set_camera_id(8)
    
    # Optimize for speed
    tracker.confidence_threshold = 0.25
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Connect to Camera 8
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        return
    
    print("✅ Connected to Camera 8")
    print("🎯 Processing at reduced resolution for speed...")
    
    frame_count = 0
    detection_count = 0
    skip_frames = 10  # Process every 11th frame
    last_tracked_objects = []
    
    # Processing resolution (much smaller than 4K)
    PROCESS_WIDTH = 1920  # Half of 4K width
    PROCESS_HEIGHT = 1080  # Half of 4K height
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    processing_times = []
    
    consecutive_failures = 0
    max_failures = 10
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            consecutive_failures += 1
            print(f"⚠️  Failed to read frame (attempt {consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("🔄 Attempting to reconnect to camera...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
                if cap.isOpened():
                    print("✅ Reconnected successfully!")
                else:
                    print("❌ Reconnection failed, retrying in 5 seconds...")
                    time.sleep(5)
            continue
        
        consecutive_failures = 0
        frame_count += 1
        fps_counter += 1
        
        # Apply fisheye correction to full resolution frame
        corrected_frame = fisheye_corrector.correct(raw_frame)
        original_height, original_width = corrected_frame.shape[:2]
        
        # Frame skipping for detection
        if frame_count % (skip_frames + 1) == 0:
            process_start = time.time()
            
            # Resize to smaller resolution for detection
            small_frame = cv2.resize(corrected_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Run detection on smaller frame
            tracked_objects, perf_stats = tracker.process_frame(small_frame)
            
            # Scale detection results back to original frame coordinates
            scale_x = original_width / PROCESS_WIDTH
            scale_y = original_height / PROCESS_HEIGHT
            
            scaled_objects = []
            for obj in tracked_objects:
                scaled_obj = obj.copy()
                
                # Scale bounding box
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    scaled_obj['bbox'] = (
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y)
                    )
                
                # Scale center point
                if 'center' in obj:
                    cx, cy = obj['center']
                    scaled_obj['center'] = (int(cx * scale_x), int(cy * scale_y))
                
                # Real coordinates stay the same (already calculated correctly)
                scaled_objects.append(scaled_obj)
            
            last_tracked_objects = scaled_objects
            detection_count += 1
            
            process_time = time.time() - process_start
            processing_times.append(process_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
        else:
            # Use last detection results
            scaled_objects = last_tracked_objects
        
        # Draw on full resolution frame
        display_frame = corrected_frame.copy()
        
        for obj in scaled_objects:
            if 'bbox' in obj and 'center' in obj:
                x1, y1, x2, y2 = obj['bbox']
                center_x, center_y = obj['center']
                confidence = obj.get('confidence', 0)
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Draw confidence
                cv2.putText(display_frame, f'{confidence:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw coordinates
                real_center = obj.get('real_center')
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    coord_text = f'({x:.1f}, {y:.1f})'
                    cv2.putText(display_frame, coord_text, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Calculate FPS
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = fps_counter / (current_time - last_fps_time)
            fps_counter = 0
            last_fps_time = current_time
        else:
            fps = 0
        
        # Add status info
        avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        status_text = f"Objects: {len(scaled_objects)} | FPS: {fps:.1f} | Process: {avg_process_time*1000:.1f}ms"
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add resolution info
        res_text = f"Detection: {PROCESS_WIDTH}x{PROCESS_HEIGHT} | Display: {original_width}x{original_height}"
        cv2.putText(display_frame, res_text, (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Print coordinates when detecting
        if frame_count % (skip_frames + 1) == 0 and scaled_objects:
            print(f"\n📏 Frame {frame_count}: {len(scaled_objects)} objects detected ({avg_process_time*1000:.1f}ms)")
            for i, obj in enumerate(scaled_objects):
                real_center = obj.get('real_center')
                confidence = obj.get('confidence', 0)
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Conf: {confidence:.2f}")
        
        # Resize for display (for screen)
        display_width = 1280
        display_height = int(original_height * (display_width / original_width))
        display_frame = cv2.resize(display_frame, (display_width, display_height))
        
        # Display
        cv2.imshow('Camera 8 - Low Resolution Processing', display_frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Quit requested by user")
            break
        elif key == ord('+') or key == ord('='):
            skip_frames = min(skip_frames + 1, 20)
            print(f"🔧 Skip frames increased to {skip_frames}")
        elif key == ord('-'):
            skip_frames = max(skip_frames - 1, 0)
            print(f"🔧 Skip frames decreased to {skip_frames}")
        elif key == ord('r'):
            skip_frames = 10
            print(f"🔧 Skip frames reset to {skip_frames}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    
    # Performance summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    print(f"\n📏 Low resolution test completed")
    print(f"📊 Performance Summary:")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Average Process Time: {avg_process_time*1000:.1f}ms")
    print(f"   Detection Resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print(f"   Size Reduction: {((original_width*original_height)/(PROCESS_WIDTH*PROCESS_HEIGHT)):.1f}x smaller")

if __name__ == "__main__":
    main() 