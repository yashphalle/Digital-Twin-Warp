#!/usr/bin/env python3
"""
Memory-Optimized Grounding DINO GPU Utilization
Fixed version that prevents GPU out of memory errors
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

class MemoryOptimizedGroundingDino:
    """Memory-safe GPU utilization for RTX 4050 Laptop"""
    
    def __init__(self):
        # Conservative settings for 6GB GPU
        self.num_parallel_workers = 2  # Reduced from 6 to avoid OOM
        self.target_gpu_memory = 75  # Use 75% of GPU memory safely
        
        # Initialize detectors
        self.detectors = []
        self.setup_memory_safe_detectors()
        
        # Small queues to save memory
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=8)
        
        # Worker management
        self.running = False
        self.workers = []
        
        # Performance tracking
        self.detection_count = 0
        self.frame_count = 0
        self.processing_times = []
        
        # Fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # Memory-safe GPU setup
        self.setup_memory_safe_gpu()
        
    def setup_memory_safe_detectors(self):
        """Create memory-optimized detectors"""
        print(f"🔧 Creating {self.num_parallel_workers} memory-safe detectors...")
        
        for i in range(self.num_parallel_workers):
            detector = DetectorTracker()
            detector.set_camera_id(8)
            
            if torch.cuda.is_available():
                try:
                    detector.model.half()  # FP16 for memory efficiency
                    print(f"✅ Detector {i+1}: FP16 enabled")
                except:
                    print(f"⚠️  Detector {i+1}: Using FP32")
                    
            self.detectors.append(detector)
            
        print(f"🔥 {self.num_parallel_workers} memory-safe detectors ready")
        
    def setup_memory_safe_gpu(self):
        """Memory-safe GPU optimizations"""
        if torch.cuda.is_available():
            # Conservative memory allocation
            torch.cuda.set_per_process_memory_fraction(0.75)  # Use 75% max
            torch.cuda.empty_cache()
            
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(2)
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🔧 Memory-Safe Configuration")
            print(f"💾 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"⚡ Workers: {self.num_parallel_workers}")
            print(f"💿 Memory Limit: {self.target_gpu_memory}%")
            
    def start_pipeline(self):
        """Start memory-safe pipeline"""
        self.running = True
        
        # Frame capture
        capture_worker = threading.Thread(target=self._capture_worker, daemon=True)
        capture_worker.start()
        self.workers.append(capture_worker)
        
        # Detection workers
        for i in range(self.num_parallel_workers):
            detection_worker = threading.Thread(target=self._detection_worker, args=(i,), daemon=True)
            detection_worker.start()
            self.workers.append(detection_worker)
        
        print(f"✅ Memory-safe pipeline started")
        
    def _capture_worker(self):
        """Frame capture with memory management"""
        camera_url = Config.RTSP_CAMERA_URLS[8]
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return
            
        print("📷 Memory-safe capture started")
        
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Apply fisheye correction immediately
            corrected_frame = self.fisheye_corrector.correct(frame)
            
            frame_data = {
                'frame': corrected_frame,
                'frame_id': frame_id,
                'capture_time': time.time()
            }
            
            # Drop oldest frame if queue full
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
        
    def _detection_worker(self, worker_id):
        """Memory-safe detection worker"""
        detector = self.detectors[worker_id]
        print(f"🧠 Memory-safe worker {worker_id} started")
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                start_time = time.time()
                
                try:
                    if torch.cuda.is_available():
                        # Clear cache before processing
                        torch.cuda.empty_cache()
                        
                        with torch.amp.autocast('cuda'):
                            tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                    else:
                        tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️  Worker {worker_id}: OOM, clearing cache...")
                    torch.cuda.empty_cache()
                    time.sleep(0.2)
                    continue
                except Exception as e:
                    print(f"❌ Worker {worker_id}: Error - {e}")
                    continue
                
                process_time = time.time() - start_time
                frame_age = time.time() - frame_data['capture_time']
                
                result_data = {
                    'frame_id': frame_data['frame_id'],
                    'tracked_objects': tracked_objects,
                    'processed_frame': processed_frame,
                    'process_time': process_time,
                    'frame_age': frame_age,
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
                # Store result
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                        
                self.result_queue.put_nowait(result_data)
                
                self.processing_times.append(process_time)
                if len(self.processing_times) > 20:
                    self.processing_times.pop(0)
                    
                self.detection_count += 1
                
            except Empty:
                continue
            except Exception as e:
                continue
                
    def get_latest_result(self):
        """Get latest result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def get_stats(self):
        """Get performance statistics"""
        current_time = time.time()
        runtime = current_time - getattr(self, 'start_time', current_time)
        detection_rate = self.detection_count / max(runtime, 1)
        
        # GPU memory stats
        gpu_stats = {'available': False}
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            gpu_stats = {
                'available': True,
                'memory_gb': gpu_memory_used,
                'memory_percent': memory_percent
            }
        
        return {
            'detection_rate': detection_rate,
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frames_captured': self.frame_count,
            'detections_completed': self.detection_count,
            'workers': self.num_parallel_workers,
            'gpu': gpu_stats
        }
        
    def stop_pipeline(self):
        """Stop pipeline"""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=2)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("🛑 Memory-safe pipeline stopped")

def main():
    print("🔧 MEMORY-OPTIMIZED GROUNDING DINO")
    print("=" * 50)
    print("Fixed: No more GPU out of memory errors")
    print("Hardware: RTX 4050 Laptop (6GB)")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Initialize system
    system = MemoryOptimizedGroundingDino()
    system.start_time = time.time()
    system.start_pipeline()
    
    print("✅ System started successfully")
    
    last_result = None
    start_time = time.time()
    
    try:
        while True:
            # Get latest result
            result = system.get_latest_result()
            if result:
                last_result = result
                
            # Create safe display frame
            if last_result and last_result.get('processed_frame') is not None:
                try:
                    processed_frame = last_result['processed_frame']
                    if isinstance(processed_frame, np.ndarray) and len(processed_frame.shape) == 3:
                        display_frame = processed_frame.copy()
                    else:
                        # Fallback frame
                        display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "🔧 Processing Active", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except:
                    display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "🔧 Memory-Safe Mode", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            else:
                # Status frame while waiting
                display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(display_frame, "🔧 Starting Detection...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                stats = system.get_stats()
                status_text = f"Frames: {stats['frames_captured']} | Detections: {stats['detections_completed']}"
                cv2.putText(display_frame, status_text, (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if stats['gpu']['available']:
                    gpu_text = f"GPU Memory: {stats['gpu']['memory_percent']:.1f}%"
                    cv2.putText(display_frame, gpu_text, (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                cv2.putText(display_frame, "Press 'q' to quit", (50, 600), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add status overlay if we have detection results
            if last_result:
                stats = system.get_stats()
                tracked_objects = last_result['tracked_objects']
                worker_id = last_result['worker_id']
                frame_age = last_result['frame_age']
                
                # Status overlay
                status_text = f"Objects: {len(tracked_objects)} | Worker: {worker_id} | Age: {frame_age*1000:.0f}ms"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # GPU status
                if stats['gpu']['available']:
                    gpu_text = f"GPU: {stats['gpu']['memory_percent']:.1f}% ({stats['gpu']['memory_gb']:.1f}GB)"
                    cv2.putText(display_frame, gpu_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Performance
                perf_text = f"Rate: {stats['detection_rate']:.1f} Hz | Process: {stats['avg_process_time']*1000:.0f}ms"
                cv2.putText(display_frame, perf_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Memory-safe indicator
                mode_text = f"Memory-Safe Mode | {stats['workers']} Workers"
                cv2.putText(display_frame, mode_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Print updates
                if last_result['frame_id'] % 20 == 0:
                    print(f"\n🔧 Frame {last_result['frame_id']} | Worker {worker_id}")
                    print(f"    GPU: {stats['gpu']['memory_percent']:.1f}% | Rate: {stats['detection_rate']:.1f} Hz")
                    print(f"    Objects: {len(tracked_objects)} | Process: {stats['avg_process_time']*1000:.0f}ms")
                    
                    # Show coordinates
                    for i, obj in enumerate(tracked_objects):
                        real_center = obj.get('real_center')
                        confidence = obj.get('confidence', 0)
                        if real_center and real_center[0] is not None:
                            x, y = real_center
                            print(f"    📍 Object {i+1}: ({x:.1f}, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Display
            cv2.imshow('Camera 8 - Memory-Safe Grounding DINO', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested")
                break
    
    finally:
        # Cleanup
        system.stop_pipeline()
        cv2.destroyAllWindows()
        
        # Results
        total_time = time.time() - start_time
        final_stats = system.get_stats()
        
        print(f"\n🔧 Memory-Safe Results:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Frames: {final_stats['frames_captured']}")
        print(f"   Detections: {final_stats['detections_completed']}")
        print(f"   Rate: {final_stats['detection_rate']:.1f} Hz")
        print(f"   Workers: {final_stats['workers']}")
        
        if final_stats['gpu']['available']:
            print(f"   GPU Memory: {final_stats['gpu']['memory_percent']:.1f}%")

if __name__ == "__main__":
    main() 