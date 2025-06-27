#!/usr/bin/env python3
"""
Maximum GPU Utilization Test - Remove ALL Bottlenecks
Push GPU to 80-95% utilization by removing software limitations
"""

import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

class MaxGPUUtilizationSystem:
    """System designed to maximize GPU utilization"""
    
    def __init__(self):
        # Multiple detector instances for parallel processing
        self.num_detector_instances = 4  # Multiple model instances
        self.detectors = []
        self.setup_multiple_detectors()
        
        # Aggressive processing settings
        self.max_concurrent_frames = 8  # Process 8 frames simultaneously
        self.target_detection_rate = 20  # 20 Hz detection rate
        
        # Pipeline queues (remove all bottlenecks)
        self.frame_queue = Queue(maxsize=20)
        self.corrected_frame_queue = Queue(maxsize=20)
        self.detection_queue = Queue(maxsize=20)
        self.result_queue = Queue(maxsize=20)
        
        # Worker threads
        self.running = False
        self.workers = []
        
        # Performance tracking
        self.gpu_utilization_history = []
        self.processing_times = []
        self.detection_rate_actual = []
        
        # GPU optimization
        self.setup_maximum_gpu_optimization()
        
    def setup_multiple_detectors(self):
        """Create multiple detector instances for parallel processing"""
        print("🚀 Creating multiple detector instances for maximum GPU usage...")
        
        for i in range(self.num_detector_instances):
            detector = DetectorTracker()
            detector.set_camera_id(8)
            
            # Configure each detector for GPU optimization
            if torch.cuda.is_available():
                try:
                    detector.model.half()  # FP16
                    print(f"✅ Detector {i+1} configured with FP16")
                except:
                    print(f"⚠️  Detector {i+1} using FP32")
                    
            self.detectors.append(detector)
            
        print(f"🔥 Created {self.num_detector_instances} detector instances")
        
    def setup_maximum_gpu_optimization(self):
        """Configure system for absolute maximum GPU utilization"""
        if torch.cuda.is_available():
            # Extreme GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Maximum memory allocation
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Pre-allocate GPU memory to avoid allocation overhead
            dummy_tensor = torch.zeros((1000, 1000), device='cuda')
            del dummy_tensor
            torch.cuda.empty_cache()
            
            # Enable all optimizations
            torch.set_num_threads(1)  # Don't compete with GPU
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 MAXIMUM GPU optimizations enabled")
            print(f"💾 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"🔥 Target: 80-95% GPU utilization")
            print(f"⚡ {self.num_detector_instances} parallel detectors")
            print(f"🎯 {self.target_detection_rate} Hz detection rate")
            
    def start_maximum_pipeline(self):
        """Start all pipeline workers for maximum throughput"""
        self.running = True
        
        # Frame capture worker (no bottleneck)
        capture_worker = threading.Thread(target=self._frame_capture_worker, daemon=True)
        capture_worker.start()
        self.workers.append(capture_worker)
        
        # Frame correction workers (parallel fisheye correction)
        for i in range(2):
            correction_worker = threading.Thread(target=self._frame_correction_worker, args=(i,), daemon=True)
            correction_worker.start()
            self.workers.append(correction_worker)
        
        # Detection workers (maximum parallel detection)
        for i in range(self.num_detector_instances):
            detection_worker = threading.Thread(target=self._detection_worker, args=(i,), daemon=True)
            detection_worker.start()
            self.workers.append(detection_worker)
        
        # GPU monitoring worker
        monitor_worker = threading.Thread(target=self._gpu_monitor_worker, daemon=True)
        monitor_worker.start()
        self.workers.append(monitor_worker)
        
        print(f"✅ Started maximum throughput pipeline with {len(self.workers)} workers")
        
    def _frame_capture_worker(self):
        """Capture frames at maximum rate"""
        camera_url = Config.RTSP_CAMERA_URLS[8]
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return
            
        print("✅ Frame capture worker started - no rate limiting")
        
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_data = {
                'frame': frame,
                'frame_id': frame_id,
                'timestamp': time.time()
            }
            
            try:
                self.frame_queue.put(frame_data, timeout=0.001)
                frame_id += 1
            except:
                pass  # Drop frame if queue full
                
        cap.release()
        
    def _frame_correction_worker(self, worker_id):
        """Apply fisheye correction in parallel"""
        fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        print(f"✅ Frame correction worker {worker_id} started")
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                # Apply correction
                corrected_frame = fisheye_corrector.correct(frame_data['frame'])
                
                corrected_data = {
                    'frame': corrected_frame,
                    'frame_id': frame_data['frame_id'],
                    'timestamp': frame_data['timestamp'],
                    'correction_time': time.time()
                }
                
                self.corrected_frame_queue.put(corrected_data, timeout=0.001)
                
            except Empty:
                continue
            except:
                pass
                
    def _detection_worker(self, worker_id):
        """Detection worker with dedicated detector instance"""
        detector = self.detectors[worker_id]
        print(f"🧠 Detection worker {worker_id} started with dedicated detector")
        
        while self.running:
            try:
                frame_data = self.corrected_frame_queue.get(timeout=1)
                
                # Process with dedicated detector
                start_time = time.time()
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                else:
                    tracked_objects, processed_frame = detector.process_frame(frame_data['frame'])
                
                process_time = time.time() - start_time
                
                result_data = {
                    'frame_id': frame_data['frame_id'],
                    'tracked_objects': tracked_objects,
                    'processed_frame': processed_frame,
                    'process_time': process_time,
                    'worker_id': worker_id,
                    'timestamp': frame_data['timestamp'],
                    'frame_age': time.time() - frame_data['timestamp']
                }
                
                self.result_queue.put(result_data, timeout=0.001)
                
                # Track performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Detection worker {worker_id} error: {e}")
                continue
                
    def _gpu_monitor_worker(self):
        """Monitor GPU utilization continuously"""
        print("📊 GPU monitoring worker started")
        
        while self.running:
            if torch.cuda.is_available():
                # Get GPU stats
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                
                # Get GPU temperature and usage if available
                try:
                    # This requires nvidia-ml-py3: pip install nvidia-ml-py3
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    utilization_data = {
                        'memory_percent': gpu_utilization,
                        'gpu_percent': gpu_util.gpu,
                        'memory_percent_nvml': gpu_util.memory,
                        'temperature': gpu_temp,
                        'timestamp': time.time()
                    }
                except:
                    # Fallback without nvidia-ml
                    utilization_data = {
                        'memory_percent': gpu_utilization,
                        'gpu_percent': None,
                        'memory_percent_nvml': None,
                        'temperature': None,
                        'timestamp': time.time()
                    }
                
                self.gpu_utilization_history.append(utilization_data)
                if len(self.gpu_utilization_history) > 100:
                    self.gpu_utilization_history.pop(0)
                    
            time.sleep(0.1)  # Monitor every 100ms
            
    def get_max_performance_stats(self):
        """Get comprehensive performance statistics"""
        current_time = time.time()
        
        # Calculate detection rate
        recent_detections = [r for r in self.processing_times if current_time - r < 5.0]
        detection_rate = len(recent_detections) / 5.0 if recent_detections else 0
        
        # GPU stats
        gpu_stats = {}
        if self.gpu_utilization_history:
            recent_gpu = self.gpu_utilization_history[-10:]  # Last 10 readings
            gpu_stats = {
                'memory_utilization': np.mean([g['memory_percent'] for g in recent_gpu]),
                'gpu_utilization': np.mean([g['gpu_percent'] for g in recent_gpu if g['gpu_percent'] is not None]),
                'temperature': recent_gpu[-1]['temperature'] if recent_gpu[-1]['temperature'] else None
            }
        
        # Queue stats
        queue_stats = {
            'frame_queue': self.frame_queue.qsize(),
            'corrected_queue': self.corrected_frame_queue.qsize(),
            'detection_queue': self.detection_queue.qsize(),
            'result_queue': self.result_queue.qsize()
        }
        
        # Processing stats
        processing_stats = {
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'detection_rate': detection_rate,
            'target_rate': self.target_detection_rate,
            'utilization_efficiency': (detection_rate / self.target_detection_rate) * 100 if self.target_detection_rate > 0 else 0
        }
        
        return {
            'gpu': gpu_stats,
            'queues': queue_stats,
            'processing': processing_stats,
            'detectors': self.num_detector_instances
        }
        
    def get_latest_result(self):
        """Get latest detection result if available"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def stop_maximum_pipeline(self):
        """Stop all workers"""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=2)
            
        print("🛑 Maximum GPU pipeline stopped")

def main():
    print("🚀 MAXIMUM GPU UTILIZATION TEST")
    print("=" * 60)
    print("Goal: Push GPU to 80-95% utilization")
    print("Method: Remove ALL software bottlenecks")
    print("Features: Multiple detectors + Parallel pipeline")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize maximum GPU system
    max_gpu_system = MaxGPUUtilizationSystem()
    max_gpu_system.start_maximum_pipeline()
    
    print("✅ Maximum GPU system started")
    print("🎯 Pushing GPU to maximum utilization...")
    
    # Display state
    frame_count = 0
    last_result = None
    last_stats_time = time.time()
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    
    try:
        while True:
            # Get latest result for display
            result = max_gpu_system.get_latest_result()
            if result:
                last_result = result
            
            if last_result is None:
                time.sleep(0.01)
                continue
                
            # Use the processed frame for display
            display_frame = last_result['processed_frame'].copy()
            tracked_objects = last_result['tracked_objects']
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            else:
                fps = fps_counter / max(current_time - last_fps_time, 0.001)
                
            # Get performance stats every second
            if current_time - last_stats_time >= 1.0:
                stats = max_gpu_system.get_max_performance_stats()
                last_stats_time = current_time
            else:
                stats = max_gpu_system.get_max_performance_stats()
            
            # Status display
            status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Age: {last_result['frame_age']*1000:.1f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # GPU utilization display
            gpu_text = f"GPU Mem: {stats['gpu']['memory_utilization']:.1f}%"
            if stats['gpu']['gpu_utilization'] is not None:
                gpu_text += f" | GPU: {stats['gpu']['gpu_utilization']:.1f}%"
            if stats['gpu']['temperature'] is not None:
                gpu_text += f" | Temp: {stats['gpu']['temperature']}°C"
            cv2.putText(display_frame, gpu_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Processing performance
            detection_rate = stats['processing']['detection_rate']
            target_rate = stats['processing']['target_rate']
            efficiency = stats['processing']['utilization_efficiency']
            process_text = f"Detection: {detection_rate:.1f}/{target_rate} Hz ({efficiency:.1f}%) | Workers: {stats['detectors']}"
            cv2.putText(display_frame, process_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Queue status
            queue_text = f"Queues: F{stats['queues']['frame_queue']} C{stats['queues']['corrected_queue']} D{stats['queues']['detection_queue']} R{stats['queues']['result_queue']}"
            cv2.putText(display_frame, queue_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Print performance updates
            if result and result['frame_id'] % 50 == 0:  # Every 50 frames
                worker_id = result['worker_id']
                process_time = result['process_time']
                frame_age = result['frame_age']
                
                print(f"\n🚀 MAX GPU - Frame {result['frame_id']} | Worker {worker_id}")
                print(f"    GPU Memory: {stats['gpu']['memory_utilization']:.1f}%", end="")
                if stats['gpu']['gpu_utilization'] is not None:
                    print(f" | GPU Util: {stats['gpu']['gpu_utilization']:.1f}%", end="")
                print(f" | Detection: {detection_rate:.1f} Hz")
                print(f"    Process: {process_time*1000:.1f}ms | Age: {frame_age*1000:.1f}ms | Objects: {len(tracked_objects)}")
                
                # Show coordinates
                for i, obj in enumerate(tracked_objects):
                    real_center = obj.get('real_center')
                    confidence = obj.get('confidence', 0)
                    if real_center and real_center[0] is not None:
                        x, y = real_center
                        print(f"    Object {i+1}: ({x:.1f}, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display
            height, width = display_frame.shape[:2]
            display_width = 1280
            display_height = int(height * (display_width / width))
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Display
            cv2.imshow('Camera 8 - Maximum GPU Utilization', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                break
            elif key == ord('+') or key == ord('='):
                # Increase target detection rate
                max_gpu_system.target_detection_rate = min(max_gpu_system.target_detection_rate + 2, 50)
                print(f"🔧 Target detection rate: {max_gpu_system.target_detection_rate} Hz")
            elif key == ord('-'):
                # Decrease target detection rate
                max_gpu_system.target_detection_rate = max(max_gpu_system.target_detection_rate - 2, 5)
                print(f"🔧 Target detection rate: {max_gpu_system.target_detection_rate} Hz")
    
    finally:
        # Cleanup
        max_gpu_system.stop_maximum_pipeline()
        cv2.destroyAllWindows()
        
        # Final performance summary
        total_time = time.time() - start_time
        final_stats = max_gpu_system.get_max_performance_stats()
        
        print(f"\n🚀 Maximum GPU test completed")
        print(f"📊 Final Performance Summary:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Display FPS: {frame_count/total_time:.1f}")
        print(f"   Detection Rate: {final_stats['processing']['detection_rate']:.1f} Hz")
        print(f"   GPU Memory: {final_stats['gpu']['memory_utilization']:.1f}%")
        if final_stats['gpu']['gpu_utilization'] is not None:
            print(f"   GPU Utilization: {final_stats['gpu']['gpu_utilization']:.1f}%")
        print(f"   Avg Process Time: {final_stats['processing']['avg_process_time']*1000:.1f}ms")
        print(f"   Detector Instances: {final_stats['detectors']}")

if __name__ == "__main__":
    main() 