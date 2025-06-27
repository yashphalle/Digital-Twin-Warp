#!/usr/bin/env python3
"""
Maximum Grounding DINO GPU Utilization - Single Camera
Push RTX 4050 Laptop to 80-90% GPU utilization with Grounding DINO
Using aggressive parallel processing and frame batching
"""

import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

class MaxGroundingDinoGPU:
    """Maximum GPU utilization system for Grounding DINO"""
    
    def __init__(self):
        # Aggressive parallel processing settings
        self.num_parallel_workers = 6  # Multiple parallel inference workers
        self.max_concurrent_frames = 12  # Process many frames simultaneously
        self.target_detection_rate = 15  # Aggressive 15 Hz target
        
        # Initialize multiple detector instances for parallel processing
        self.detectors = []
        self.setup_parallel_detectors()
        
        # Aggressive pipeline queues
        self.frame_queue = Queue(maxsize=30)
        self.corrected_queue = Queue(maxsize=30)
        self.detection_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        
        # Worker management
        self.running = False
        self.workers = []
        
        # Performance tracking
        self.gpu_utilization_history = []
        self.processing_times = []
        self.detection_count = 0
        self.frame_count = 0
        
        # Fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # GPU optimization
        self.setup_maximum_gpu_optimization()
        
    def setup_parallel_detectors(self):
        """Create multiple Grounding DINO detector instances for parallel processing"""
        print(f"🚀 Creating {self.num_parallel_workers} Grounding DINO detectors for parallel processing...")
        
        for i in range(self.num_parallel_workers):
            detector = DetectorTracker()
            detector.set_camera_id(8)
            
            # GPU optimization for each detector
            if torch.cuda.is_available():
                try:
                    detector.model.half()  # FP16 for speed
                    print(f"✅ Detector {i+1}: FP16 enabled")
                except:
                    print(f"⚠️  Detector {i+1}: Using FP32")
                    
            self.detectors.append(detector)
            
        print(f"🔥 {self.num_parallel_workers} Grounding DINO detectors ready for maximum GPU usage")
        
    def setup_maximum_gpu_optimization(self):
        """Extreme GPU optimizations for maximum utilization"""
        if torch.cuda.is_available():
            # Extreme optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Maximum GPU memory allocation
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Pre-warm GPU memory
            dummy_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
            del dummy_tensor
            torch.cuda.empty_cache()
            
            # Optimize threading for GPU work
            torch.set_num_threads(2)  # Reduce CPU competition
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 MAXIMUM Grounding DINO GPU optimizations")
            print(f"💾 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"🎯 Target: 80-90% GPU utilization")
            print(f"⚡ {self.num_parallel_workers} parallel workers")
            print(f"🔥 {self.target_detection_rate} Hz detection rate")
            
    def start_maximum_pipeline(self):
        """Start aggressive parallel processing pipeline"""
        self.running = True
        
        # Frame capture worker (high rate)
        capture_worker = threading.Thread(target=self._aggressive_capture_worker, daemon=True)
        capture_worker.start()
        self.workers.append(capture_worker)
        
        # Multiple frame correction workers
        for i in range(3):
            correction_worker = threading.Thread(target=self._correction_worker, args=(i,), daemon=True)
            correction_worker.start()
            self.workers.append(correction_worker)
        
        # Parallel detection workers (one per detector)
        for i in range(self.num_parallel_workers):
            detection_worker = threading.Thread(target=self._parallel_detection_worker, args=(i,), daemon=True)
            detection_worker.start()
            self.workers.append(detection_worker)
        
        # Aggressive frame dispatcher
        dispatcher_worker = threading.Thread(target=self._frame_dispatcher, daemon=True)
        dispatcher_worker.start()
        self.workers.append(dispatcher_worker)
        
        # GPU monitoring
        monitor_worker = threading.Thread(target=self._gpu_monitor, daemon=True)
        monitor_worker.start()
        self.workers.append(monitor_worker)
        
        print(f"✅ Started maximum GPU pipeline with {len(self.workers)} workers")
        
    def _aggressive_capture_worker(self):
        """Capture frames at maximum rate"""
        camera_url = Config.RTSP_CAMERA_URLS[8]
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return
            
        print("✅ Aggressive frame capture started - maximum rate")
        
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_data = {
                'frame': frame.copy(),
                'frame_id': frame_id,
                'capture_time': time.time()
            }
            
            # Aggressive queue management - always keep newest frames
            while self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Drop oldest
                except Empty:
                    break
                    
            try:
                self.frame_queue.put_nowait(frame_data)
                frame_id += 1
                self.frame_count += 1
            except:
                pass
                
        cap.release()
        
    def _correction_worker(self, worker_id):
        """Parallel fisheye correction"""
        print(f"✅ Correction worker {worker_id} started")
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                # Apply correction
                corrected_frame = self.fisheye_corrector.correct(frame_data['frame'])
                
                corrected_data = {
                    'frame': corrected_frame,
                    'frame_id': frame_data['frame_id'],
                    'capture_time': frame_data['capture_time'],
                    'correction_time': time.time()
                }
                
                # Smart queue management
                while self.corrected_queue.full():
                    try:
                        self.corrected_queue.get_nowait()
                    except Empty:
                        break
                        
                self.corrected_queue.put_nowait(corrected_data)
                
            except Empty:
                continue
            except Exception as e:
                continue
                
    def _frame_dispatcher(self):
        """Dispatch frames to detection workers aggressively"""
        print("🎯 Frame dispatcher started - aggressive scheduling")
        
        while self.running:
            try:
                # Get corrected frame
                frame_data = self.corrected_queue.get(timeout=1)
                
                # Dispatch to detection queue aggressively
                while self.detection_queue.full():
                    try:
                        self.detection_queue.get_nowait()  # Drop oldest
                    except Empty:
                        break
                        
                self.detection_queue.put_nowait(frame_data)
                
            except Empty:
                continue
            except Exception as e:
                continue
                
    def _parallel_detection_worker(self, worker_id):
        """Parallel Grounding DINO detection worker"""
        detector = self.detectors[worker_id]
        print(f"🧠 Parallel Grounding DINO worker {worker_id} started")
        
        while self.running:
            try:
                frame_data = self.detection_queue.get(timeout=1)
                
                # Process with dedicated detector
                start_time = time.time()
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                else:
                    tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                
                process_time = time.time() - start_time
                
                # Calculate frame age
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
                
                # Smart result queue management
                while self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()  # Drop oldest
                    except Empty:
                        break
                        
                self.result_queue.put_nowait(result_data)
                
                # Track performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                    
                self.detection_count += 1
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                continue
                
    def _gpu_monitor(self):
        """Monitor GPU utilization continuously"""
        print("📊 GPU monitor started - continuous monitoring")
        
        while self.running:
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                # Try to get actual GPU utilization
                gpu_percent = None
                gpu_temp = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass
                
                utilization_data = {
                    'memory_gb': gpu_memory_used,
                    'memory_percent': gpu_memory_percent,
                    'gpu_percent': gpu_percent,
                    'temperature': gpu_temp,
                    'timestamp': time.time()
                }
                
                self.gpu_utilization_history.append(utilization_data)
                if len(self.gpu_utilization_history) > 100:
                    self.gpu_utilization_history.pop(0)
                    
            time.sleep(0.1)  # Monitor every 100ms
            
    def get_latest_result(self):
        """Get latest detection result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        current_time = time.time()
        
        # Detection rate
        runtime = current_time - getattr(self, 'start_time', current_time)
        detection_rate = self.detection_count / max(runtime, 1)
        
        # GPU stats
        gpu_stats = {'available': False}
        if self.gpu_utilization_history:
            recent_gpu = self.gpu_utilization_history[-10:]
            gpu_stats = {
                'available': True,
                'memory_gb': recent_gpu[-1]['memory_gb'],
                'memory_percent': np.mean([g['memory_percent'] for g in recent_gpu]),
                'gpu_percent': np.mean([g['gpu_percent'] for g in recent_gpu if g['gpu_percent'] is not None]) if any(g['gpu_percent'] is not None for g in recent_gpu) else None,
                'temperature': recent_gpu[-1]['temperature'] if recent_gpu[-1]['temperature'] is not None else None
            }
        
        # Queue stats
        queue_stats = {
            'frame_queue': self.frame_queue.qsize(),
            'corrected_queue': self.corrected_queue.qsize(),
            'detection_queue': self.detection_queue.qsize(),
            'result_queue': self.result_queue.qsize()
        }
        
        return {
            'detection_rate': detection_rate,
            'target_rate': self.target_detection_rate,
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frames_captured': self.frame_count,
            'detections_completed': self.detection_count,
            'parallel_workers': self.num_parallel_workers,
            'gpu': gpu_stats,
            'queues': queue_stats
        }
        
    def stop_pipeline(self):
        """Stop all workers"""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=3)
            
        print("🛑 Maximum GPU pipeline stopped")

def main():
    print("🚀 MAXIMUM GROUNDING DINO GPU UTILIZATION")
    print("=" * 60)
    print("Goal: Push RTX 4050 Laptop to 80-90% GPU usage")
    print("Method: Aggressive parallel Grounding DINO processing")
    print("Target: Maximum accuracy + Maximum GPU utilization")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize maximum GPU system
    max_system = MaxGroundingDinoGPU()
    max_system.start_time = time.time()
    max_system.start_maximum_pipeline()
    
    print("✅ Maximum Grounding DINO system started")
    print("🎯 Pushing GPU to maximum utilization...")
    
    # Display state
    last_result = None
    display_frame = None
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    last_stats_time = start_time
    
    try:
        while True:
            # Get latest result
            result = max_system.get_latest_result()
            if result:
                last_result = result
                display_frame = result['processed_frame'].copy()
                
            if display_frame is None:
                time.sleep(0.01)
                continue
                
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            else:
                fps = fps_counter / max(current_time - last_fps_time, 0.001)
                
            # Get stats
            if current_time - last_stats_time >= 1.0:
                stats = max_system.get_performance_stats()
                last_stats_time = current_time
            else:
                stats = max_system.get_performance_stats()
            
            # Draw status
            if last_result:
                tracked_objects = last_result['tracked_objects']
                frame_age = last_result['frame_age']
                worker_id = last_result['worker_id']
                
                # Status display
                status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Age: {frame_age*1000:.1f}ms | Worker: {worker_id}"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # GPU utilization
                if stats['gpu']['available']:
                    gpu_text = f"GPU: {stats['gpu']['memory_percent']:.1f}% ({stats['gpu']['memory_gb']:.1f}GB)"
                    if stats['gpu']['gpu_percent'] is not None:
                        gpu_text += f" | Util: {stats['gpu']['gpu_percent']:.1f}%"
                    if stats['gpu']['temperature'] is not None:
                        gpu_text += f" | {stats['gpu']['temperature']}°C"
                    cv2.putText(display_frame, gpu_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Detection performance
                detection_text = f"Detection: {stats['detection_rate']:.1f}/{stats['target_rate']} Hz | Process: {stats['avg_process_time']*1000:.1f}ms"
                cv2.putText(display_frame, detection_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Parallel processing info
                parallel_text = f"Workers: {stats['parallel_workers']} | Detections: {stats['detections_completed']}"
                cv2.putText(display_frame, parallel_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Queue status
                queue_text = f"Q: F{stats['queues']['frame_queue']} C{stats['queues']['corrected_queue']} D{stats['queues']['detection_queue']} R{stats['queues']['result_queue']}"
                cv2.putText(display_frame, queue_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Print periodic updates
                if last_result['frame_id'] % 50 == 0:
                    print(f"\n🚀 MAX GPU - Frame {last_result['frame_id']} | Worker {worker_id}")
                    print(f"    GPU: {stats['gpu']['memory_percent']:.1f}%", end="")
                    if stats['gpu']['gpu_percent'] is not None:
                        print(f" | Util: {stats['gpu']['gpu_percent']:.1f}%", end="")
                    print(f" | Detection: {stats['detection_rate']:.1f} Hz")
                    print(f"    Process: {stats['avg_process_time']*1000:.1f}ms | Age: {frame_age*1000:.1f}ms | Objects: {len(tracked_objects)}")
                    
                    # Show coordinates
                    for i, obj in enumerate(tracked_objects):
                        real_center = obj.get('real_center')
                        confidence = obj.get('confidence', 0)
                        if real_center and real_center[0] is not None:
                            x, y = real_center
                            print(f"    Object {i+1}: ({x:.1f}, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display
            if display_frame is not None:
                height, width = display_frame.shape[:2]
                display_width = 1280
                display_height = int(height * (display_width / width))
                display_frame = cv2.resize(display_frame, (display_width, display_height))
                
                # Display
                cv2.imshow('Camera 8 - Maximum Grounding DINO GPU', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested")
                break
    
    finally:
        # Cleanup
        max_system.stop_pipeline()
        cv2.destroyAllWindows()
        
        # Final performance summary
        total_time = time.time() - start_time
        final_stats = max_system.get_performance_stats()
        
        print(f"\n🚀 Maximum Grounding DINO GPU Test Results:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Frames Captured: {final_stats['frames_captured']}")
        print(f"   Detections: {final_stats['detections_completed']}")
        print(f"   Detection Rate: {final_stats['detection_rate']:.1f} Hz")
        print(f"   Avg Process Time: {final_stats['avg_process_time']*1000:.1f}ms")
        print(f"   Parallel Workers: {final_stats['parallel_workers']}")
        
        if final_stats['gpu']['available']:
            print(f"   Final GPU Memory: {final_stats['gpu']['memory_percent']:.1f}%")
            if final_stats['gpu']['gpu_percent'] is not None:
                print(f"   Final GPU Utilization: {final_stats['gpu']['gpu_percent']:.1f}%")

if __name__ == "__main__":
    main() 