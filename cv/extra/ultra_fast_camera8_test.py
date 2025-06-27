#!/usr/bin/env python3
"""
Ultra Fast Camera 8 Test - YOLOv8 Nano for Minimal Latency
Using YOLOv8n instead of Grounding DINO for 10-20x speed improvement
Same coordinate system as other camera tests
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector
from detector_tracker import CoordinateMapper

def main():
    print("⚡ ULTRA FAST CAMERA 8 TEST - YOLO v8 NANO")
    print("=" * 55)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Model: YOLOv8n (10-20x faster than Grounding DINO)")
    print("Press 'q' to quit")
    print("=" * 55)
    
    # Initialize YOLOv8 nano model (much faster)
    print("🔄 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Nano model - fastest
    model.to('cuda' if Config.FORCE_GPU else 'cpu')
    print("✅ YOLOv8n model loaded")
    
    # Initialize coordinate mapper for Camera 8
    coordinate_mapper = CoordinateMapper(
        floor_width=Config.WAREHOUSE_FLOOR_WIDTH,
        floor_length=Config.WAREHOUSE_FLOOR_LENGTH
    )
    coordinate_mapper.camera_id = 8
    
    # Set camera coverage zone
    if 8 in Config.CAMERA_COVERAGE_ZONES:
        zone_config = Config.CAMERA_COVERAGE_ZONES[8]
        coordinate_mapper.camera_coverage_zone = {
            'x_start': zone_config['x_start'],
            'x_end': zone_config['x_end'], 
            'y_start': zone_config['y_start'],
            'y_end': zone_config['y_end']
        }
        print(f"✅ Camera 8 coverage zone set: {coordinate_mapper.camera_coverage_zone}")
    
    # Load calibration
    coordinate_mapper.load_calibration()
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Connect to Camera 8
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    
    # Ultra-minimal buffer for lowest latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        return
    
    print("✅ Connected to Camera 8 (ultra-low latency buffer)")
    print("🎯 Looking for objects...")
    
    frame_count = 0
    detection_count = 0
    skip_frames = 5  # Process every 6th frame for ultra speed
    last_detections = []
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    detection_times = []
    
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
        
        # Apply fisheye correction
        corrected_frame = fisheye_corrector.correct(raw_frame)
        
        # Frame skipping for detection
        if frame_count % (skip_frames + 1) == 0:
            detection_start = time.time()
            
            # Run YOLOv8 detection (much faster than Grounding DINO)
            results = model(corrected_frame, conf=0.25, verbose=False)
            
            detection_time = time.time() - detection_start
            detection_times.append(detection_time)
            if len(detection_times) > 30:
                detection_times.pop(0)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Filter for relevant classes (0=person, others as needed)
                        # You can adjust this filter based on what you want to detect
                        if conf > 0.25:  # Confidence threshold
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Scale coordinates for calibration (assuming 4K calibration)
                            frame_height, frame_width = corrected_frame.shape[:2]
                            scale_x = 3840 / frame_width
                            scale_y = 2160 / frame_height
                            
                            scaled_center_x = center_x * scale_x
                            scaled_center_y = center_y * scale_y
                            
                            # Get real-world coordinates
                            real_x, real_y = None, None
                            if coordinate_mapper.is_calibrated:
                                real_x, real_y = coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)
                            
                            detection = {
                                'center': (center_x, center_y),
                                'real_center': (real_x, real_y) if real_x is not None else None,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(conf),
                                'class': cls
                            }
                            detections.append(detection)
            
            last_detections = detections
            detection_count += 1
        else:
            # Use last detections for smooth display
            detections = last_detections
        
        # Draw detections
        display_frame = corrected_frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Draw confidence
            cv2.putText(display_frame, f'{conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw coordinates if available
            real_center = detection.get('real_center')
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
        avg_detection_time = np.mean(detection_times) if detection_times else 0
        detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        
        status_text = f"Objects: {len(detections)} | FPS: {fps:.1f} | Det FPS: {detection_fps:.1f}"
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add model info
        model_text = f"YOLOv8n - Skip: {skip_frames} | Avg Det: {avg_detection_time*1000:.1f}ms"
        cv2.putText(display_frame, model_text, (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Print coordinates when detecting
        if frame_count % (skip_frames + 1) == 0 and detections:
            print(f"\n⚡ Frame {frame_count}: {len(detections)} objects detected ({detection_time*1000:.1f}ms)")
            for i, det in enumerate(detections):
                real_center = det.get('real_center')
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Conf: {det['confidence']:.2f}")
        
        # Resize for display (smaller for speed)
        height, width = display_frame.shape[:2]
        display_width = 800  # Even smaller for max speed
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(display_frame, (display_width, display_height))
        
        # Display
        cv2.imshow('Camera 8 - Ultra Fast YOLOv8n', display_frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Quit requested by user")
            break
        elif key == ord('+') or key == ord('='):
            skip_frames = min(skip_frames + 1, 10)
            print(f"🔧 Skip frames increased to {skip_frames}")
        elif key == ord('-'):
            skip_frames = max(skip_frames - 1, 0)
            print(f"🔧 Skip frames decreased to {skip_frames}")
        elif key == ord('r'):
            skip_frames = 5
            print(f"🔧 Skip frames reset to {skip_frames}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Performance summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    
    print(f"\n⚡ Ultra fast test completed")
    print(f"📊 Performance Summary:")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Average Detection Time: {avg_detection_time*1000:.1f}ms")
    print(f"   Detection FPS: {1.0/avg_detection_time:.1f}" if avg_detection_time > 0 else "   Detection FPS: N/A")
    print(f"   Total Runtime: {total_time:.1f}s")

if __name__ == "__main__":
    main() 