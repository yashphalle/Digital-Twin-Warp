#!/usr/bin/env python3
"""
Simple Working GPU Version
Fixes all the detection and display issues
"""

import cv2
import time
import torch
import numpy as np
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

def main():
    print("🔧 SIMPLE WORKING GPU VERSION")
    print("=" * 40)
    print("Fixed: Proper detection and display")
    print("Hardware: RTX 4050 Laptop (6GB)")
    print("Press 'q' to quit")
    print("=" * 40)
    
    # Setup GPU conservatively
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% safely
        torch.cuda.empty_cache()
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory limit: 70% of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Initialize detector
    print("🧠 Loading Grounding DINO...")
    detector = DetectorTracker()
    detector.set_camera_id(8)
    
    # Enable FP16 for memory efficiency
    if torch.cuda.is_available():
        try:
            detector.model.half()
            print("✅ FP16 enabled for memory efficiency")
        except:
            print("⚠️  Using FP32")
    
    # Fisheye corrector
    print("🔧 Setting up fisheye correction...")
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    
    # Camera setup
    print("📷 Connecting to camera...")
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Failed to connect to camera")
        return
    
    print("✅ Camera connected successfully")
    print("🎯 Starting detection loop...")
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    processing_times = []
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame")
                continue
            
            frame_count += 1
            
            # Apply fisheye correction
            try:
                corrected_frame = fisheye_corrector.correct(frame)
            except Exception as e:
                print(f"⚠️  Correction error: {e}")
                corrected_frame = frame  # Use original if correction fails
            
            # Process with detector
            process_start = time.time()
            
            try:
                # Clear GPU cache before processing to avoid OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # Use autocast for memory efficiency
                    with torch.amp.autocast('cuda'):
                        tracked_objects, perf_stats = detector.process_frame(corrected_frame)
                else:
                    tracked_objects, perf_stats = detector.process_frame(corrected_frame)
                
                detection_count += 1
                    
            except torch.cuda.OutOfMemoryError:
                print("⚠️  GPU OOM, skipping frame")
                torch.cuda.empty_cache()
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"❌ Detection error - {str(e)[:100]}")
                tracked_objects = []
                perf_stats = {}
            
            process_time = time.time() - process_start
            processing_times.append(process_time)
            if len(processing_times) > 20:
                processing_times.pop(0)
            
            # Create display frame
            display_frame = corrected_frame.copy()
            
            # Draw tracked objects properly
            if tracked_objects:
                display_frame = detector.draw_tracked_objects(display_frame, tracked_objects)
            
            # Draw calibrated zone if available
            if detector.coordinate_mapper.is_calibrated:
                display_frame = detector.draw_calibrated_zone_overlay(display_frame)
            
            # Add performance stats overlay
            current_time = time.time()
            runtime = current_time - start_time
            detection_rate = detection_count / max(runtime, 1)
            avg_process_time = np.mean(processing_times) if processing_times else 0
            
            # GPU stats
            gpu_text = "GPU: N/A"
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                    gpu_text = f"GPU: {memory_percent:.1f}% ({gpu_memory_used:.1f}GB)"
                except:
                    pass
            
            # Status overlay
            status_text = f"Objects: {len(tracked_objects)} | Rate: {detection_rate:.1f} Hz | Process: {avg_process_time*1000:.0f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, gpu_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"Frame: {frame_count} | Detections: {detection_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            cv2.putText(display_frame, "SIMPLE WORKING VERSION", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print periodic updates
            if frame_count % 30 == 0:
                print(f"\n🔧 Frame {frame_count}")
                print(f"    GPU: {gpu_text}")
                print(f"    Objects: {len(tracked_objects)} | Rate: {detection_rate:.1f} Hz")
                print(f"    Process: {avg_process_time*1000:.0f}ms")
                
                # Show coordinates
                for i, obj in enumerate(tracked_objects):
                    real_center = obj.get('real_center')
                    confidence = obj.get('confidence', 0)
                    if real_center and real_center[0] is not None:
                        x, y = real_center
                        print(f"    📍 Object {i+1}: ({x:.1f}, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display if needed
            height, width = display_frame.shape[:2]
            if width > 1400 or height > 900:
                scale = min(1400/width, 900/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            # Display
            cv2.imshow('Camera 8 - Simple Working GPU', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested")
                break
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final results
        total_time = time.time() - start_time
        
        print(f"\n🔧 Simple Working GPU Results:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Frames Captured: {frame_count}")
        print(f"   Detections: {detection_count}")
        print(f"   Detection Rate: {detection_count / max(total_time, 1):.1f} Hz")
        print(f"   Avg Process Time: {np.mean(processing_times)*1000:.0f}ms" if processing_times else "   Avg Process Time: N/A")
        
        if torch.cuda.is_available():
            try:
                final_memory = torch.cuda.memory_allocated() / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                final_percent = (final_memory / total_memory) * 100
                print(f"   Final GPU Memory: {final_percent:.1f}%")
            except:
                pass
        
        print("✅ Simple working version completed!")

if __name__ == "__main__":
    main() 