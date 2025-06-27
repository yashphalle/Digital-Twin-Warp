#!/usr/bin/env python3
"""
Fixed GPU Test - Reliable Display with Memory Safety
Fixes the black screen issue and GPU out of memory problems
"""

import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue, Empty
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

class FixedGPUTest:
    """Fixed version with reliable display and memory safety"""
    
    def __init__(self):
        # Conservative GPU settings
        self.num_workers = 2  # Safe for 6GB GPU
        
        # Setup GPU memory management
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% safely
            torch.cuda.empty_cache()
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memory limit: 70% of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize detector
        self.detector = DetectorTracker()
        self.detector.set_camera_id(8)
        
        # Enable FP16 for memory efficiency
        if torch.cuda.is_available():
            try:
                self.detector.model.half()
                print("✅ FP16 enabled for memory efficiency")
            except:
                print("⚠️  Using FP32")
        
        # Fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # Simple queues
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=5)
        
        # Worker management
        self.running = False
        self.workers = []
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.processing_times = []
        
    def start_system(self):
        """Start the fixed system"""
        self.running = True
        
        # Camera capture worker
        capture_worker = threading.Thread(target=self._capture_worker, daemon=True)
        capture_worker.start()
        self.workers.append(capture_worker)
        
        # Detection workers
        for i in range(self.num_workers):
            detection_worker = threading.Thread(target=self._detection_worker, args=(i,), daemon=True)
            detection_worker.start()
            self.workers.append(detection_worker)
        
        print(f"✅ Fixed system started with {self.num_workers} workers")
        
    def _capture_worker(self):
        """Reliable camera capture"""
        camera_url = Config.RTSP_CAMERA_URLS[8]
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return
            
        print("📷 Camera capture started")
        
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Apply fisheye correction
            try:
                corrected_frame = self.fisheye_corrector.correct(frame)
            except Exception as e:
                print(f"⚠️  Correction error: {e}")
                corrected_frame = frame  # Use original if correction fails
            
            frame_data = {
                'frame': corrected_frame.copy(),
                'frame_id': frame_id,
                'capture_time': time.time()
            }
            
            # Drop old frames if queue full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
                    
            try:
                self.frame_queue.put_nowait(frame_data)
                frame_id += 1
                self.frame_count += 1
            except:
                pass
                
        cap.release()
        print("📷 Camera capture stopped")
        
    def _detection_worker(self, worker_id):
        """Memory-safe detection worker"""
        print(f"🧠 Detection worker {worker_id} started")
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                start_time = time.time()
                
                try:
                    # Clear GPU cache before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                        # Use autocast for memory efficiency
                        with torch.amp.autocast('cuda'):
                            tracked_objects, processed_frame = self.detector.process_frame(frame_data['frame'])
                    else:
                        tracked_objects, processed_frame = self.detector.process_frame(frame_data['frame'])
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️  Worker {worker_id}: GPU OOM, skipping frame")
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"❌ Worker {worker_id}: Detection error - {str(e)[:100]}")
                    # Create fallback result with original frame
                    tracked_objects = []
                    processed_frame = frame_data['frame']
                
                process_time = time.time() - start_time
                frame_age = time.time() - frame_data['capture_time']
                
                result_data = {
                    'frame_id': frame_data['frame_id'],
                    'tracked_objects': tracked_objects,
                    'processed_frame': processed_frame,
                    'process_time': process_time,
                    'frame_age': frame_age,
                    'worker_id': worker_id,
                    'timestamp': time.time(),
                    'original_frame': frame_data['frame']  # Keep original as backup
                }
                
                # Store result safely
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                        
                self.result_queue.put_nowait(result_data)
                
                # Track performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 20:
                    self.processing_times.pop(0)
                    
                self.detection_count += 1
                
            except Empty:
                continue
            except Exception as e:
                print(f"⚠️  Worker {worker_id}: Unexpected error - {str(e)[:100]}")
                continue
                
    def get_latest_result(self):
        """Get latest detection result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def get_stats(self):
        """Get performance statistics"""
        current_time = time.time()
        runtime = current_time - getattr(self, 'start_time', current_time)
        detection_rate = self.detection_count / max(runtime, 1)
        
        # GPU stats
        gpu_stats = {'available': False}
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                gpu_stats = {
                    'available': True,
                    'memory_gb': gpu_memory_used,
                    'memory_percent': memory_percent
                }
            except Exception as e:
                print(f"⚠️  GPU stats error: {e}")
        
        return {
            'detection_rate': detection_rate,
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frames_captured': self.frame_count,
            'detections_completed': self.detection_count,
            'workers': self.num_workers,
            'gpu': gpu_stats
        }
        
    def stop_system(self):
        """Stop the system"""
        print("🛑 Stopping system...")
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=2)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("✅ System stopped")

def main():
    print("🔧 FIXED GPU TEST")
    print("=" * 40)
    print("Goal: Fix black screen and memory issues")
    print("Hardware: RTX 4050 Laptop (6GB)")
    print("Method: Reliable display + Memory safety")
    print("Press 'q' to quit")
    print("=" * 40)
    
    # Initialize system
    system = FixedGPUTest()
    system.start_time = time.time()
    system.start_system()
    
    print("✅ System started - waiting for camera feed...")
    
    # Display variables
    last_result = None
    frame_counter = 0
    last_display_time = time.time()
    
    try:
        while True:
            # Get latest result
            result = system.get_latest_result()
            if result:
                last_result = result
                frame_counter += 1
                
            # Create display frame
            display_frame = None
            
            if last_result:
                # Try to use processed frame first
                if last_result.get('processed_frame') is not None:
                    try:
                        processed_frame = last_result['processed_frame']
                        
                        # Validate frame format
                        if isinstance(processed_frame, np.ndarray) and len(processed_frame.shape) == 3:
                            if processed_frame.dtype != np.uint8:
                                # Convert to uint8 if needed
                                if processed_frame.max() <= 1.0:
                                    processed_frame = (processed_frame * 255).astype(np.uint8)
                                else:
                                    processed_frame = processed_frame.astype(np.uint8)
                            display_frame = processed_frame.copy()
                        else:
                            print(f"⚠️  Invalid processed frame format: {type(processed_frame)}, shape: {getattr(processed_frame, 'shape', 'unknown')}")
                            # Fall back to original frame
                            if last_result.get('original_frame') is not None:
                                display_frame = last_result['original_frame'].copy()
                                
                    except Exception as e:
                        print(f"⚠️  Frame processing error: {e}")
                        # Fall back to original frame
                        if last_result.get('original_frame') is not None:
                            display_frame = last_result['original_frame'].copy()
                
                # If still no display frame, use original
                if display_frame is None and last_result.get('original_frame') is not None:
                    display_frame = last_result['original_frame'].copy()
            
            # Create status frame if no camera feed
            if display_frame is None:
                display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(display_frame, "🔧 Waiting for camera feed...", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                stats = system.get_stats()
                status_text = f"Frames: {stats['frames_captured']} | Detections: {stats['detections_completed']}"
                cv2.putText(display_frame, status_text, (50, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if stats['gpu']['available']:
                    gpu_text = f"GPU Memory: {stats['gpu']['memory_percent']:.1f}%"
                    cv2.putText(display_frame, gpu_text, (50, 350), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                cv2.putText(display_frame, "Press 'q' to quit", (50, 600), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Add status overlay to camera feed
                stats = system.get_stats()
                current_time = time.time()
                
                # Calculate display FPS
                if current_time - last_display_time > 0:
                    display_fps = frame_counter / (current_time - getattr(main, 'start_display_time', current_time))
                else:
                    display_fps = 0
                
                # Status overlay
                if last_result:
                    tracked_objects = last_result['tracked_objects']
                    worker_id = last_result['worker_id']
                    frame_age = last_result['frame_age']
                    
                    status_text = f"Objects: {len(tracked_objects)} | Worker: {worker_id} | Age: {frame_age*1000:.0f}ms"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # GPU status
                    if stats['gpu']['available']:
                        gpu_text = f"GPU: {stats['gpu']['memory_percent']:.1f}% ({stats['gpu']['memory_gb']:.1f}GB)"
                        cv2.putText(display_frame, gpu_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Performance
                    perf_text = f"Detection: {stats['detection_rate']:.1f} Hz | Display: {display_fps:.1f} FPS"
                    cv2.putText(display_frame, perf_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    # Fixed indicator
                    fix_text = f"FIXED VERSION | {stats['workers']} Workers | Frame {last_result['frame_id']}"
                    cv2.putText(display_frame, fix_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Print periodic updates
                    if last_result['frame_id'] % 30 == 0:
                        print(f"\n🔧 Frame {last_result['frame_id']} | Worker {worker_id}")
                        print(f"    GPU: {stats['gpu']['memory_percent']:.1f}% | Detection: {stats['detection_rate']:.1f} Hz")
                        print(f"    Objects: {len(tracked_objects)} | Process: {stats['avg_process_time']*1000:.0f}ms")
                        print(f"    Display: {display_fps:.1f} FPS | Age: {frame_age*1000:.0f}ms")
                        
                        # Show coordinates
                        for i, obj in enumerate(tracked_objects):
                            real_center = obj.get('real_center')
                            confidence = obj.get('confidence', 0)
                            if real_center and real_center[0] is not None:
                                x, y = real_center
                                print(f"    📍 Object {i+1}: ({x:.1f}, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display if too large
            if display_frame is not None:
                height, width = display_frame.shape[:2]
                if width > 1400 or height > 900:
                    scale = min(1400/width, 900/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Display
                cv2.imshow('Camera 8 - Fixed GPU Test', display_frame)
            
            # Store first display time
            if not hasattr(main, 'start_display_time'):
                main.start_display_time = time.time()
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested")
                break
    
    finally:
        # Cleanup
        system.stop_system()
        cv2.destroyAllWindows()
        
        # Final results
        total_time = time.time() - system.start_time
        final_stats = system.get_stats()
        
        print(f"\n🔧 Fixed GPU Test Results:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Frames Captured: {final_stats['frames_captured']}")
        print(f"   Detections: {final_stats['detections_completed']}")
        print(f"   Detection Rate: {final_stats['detection_rate']:.1f} Hz")
        print(f"   Avg Process Time: {final_stats['avg_process_time']*1000:.0f}ms")
        print(f"   Workers: {final_stats['workers']}")
        
        if final_stats['gpu']['available']:
            print(f"   Final GPU Memory: {final_stats['gpu']['memory_percent']:.1f}%")
        
        print("✅ Fixed version completed successfully!")

if __name__ == "__main__":
    main() 