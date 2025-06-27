#!/usr/bin/env python3
"""
Optimized Single GPU Test - Efficient Pipeline
Focus on pipeline efficiency and GPU utilization optimization
rather than multiple competing model instances
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

class OptimizedGPUPipeline:
    """Optimized GPU pipeline with efficient processing"""
    
    def __init__(self):
        # Optimized pipeline settings
        self.target_detection_rate = 10  # Start with 10 Hz
        self.max_queue_size = 10
        
        # Single optimized detector
        self.detector = DetectorTracker()
        self.detector.set_camera_id(8)
        self.setup_gpu_optimizations()
        
        # Pipeline queues
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        self.corrected_queue = Queue(maxsize=self.max_queue_size)
        self.detection_queue = Queue(maxsize=self.max_queue_size)
        self.result_queue = Queue(maxsize=self.max_queue_size)
        
        # Workers
        self.running = False
        self.workers = []
        
        # Performance tracking
        self.gpu_stats = []
        self.processing_times = []
        self.detection_count = 0
        self.frame_count = 0
        
        # Fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
    def setup_gpu_optimizations(self):
        """Setup GPU optimizations for single model instance"""
        if torch.cuda.is_available():
            # Optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory management
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable optimizations
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Model optimization
            try:
                self.detector.model.half()  # FP16
                print("🔥 Model converted to FP16")
            except:
                print("⚠️  Using FP32")
                
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 Optimized GPU setup: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"🎯 Target: {self.target_detection_rate} Hz detection rate")
            
    def start_optimized_pipeline(self):
        """Start optimized processing pipeline"""
        self.running = True
        
        # Frame capture worker
        capture_worker = threading.Thread(target=self._capture_worker, daemon=True)
        capture_worker.start()
        self.workers.append(capture_worker)
        
        # Frame correction worker
        correction_worker = threading.Thread(target=self._correction_worker, daemon=True)
        correction_worker.start()
        self.workers.append(correction_worker)
        
        # Detection worker (single optimized)
        detection_worker = threading.Thread(target=self._detection_worker, daemon=True)
        detection_worker.start()
        self.workers.append(detection_worker)
        
        # GPU monitoring
        monitor_worker = threading.Thread(target=self._gpu_monitor, daemon=True)
        monitor_worker.start()
        self.workers.append(monitor_worker)
        
        print(f"✅ Started optimized pipeline with {len(self.workers)} workers")
        
    def _capture_worker(self):
        """Capture frames efficiently"""
        camera_url = Config.RTSP_CAMERA_URLS[8]
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return
            
        print("✅ Frame capture started")
        
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_data = {
                'frame': frame,
                'frame_id': frame_id,
                'capture_time': time.time()
            }
            
            # Smart queue management
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Drop oldest
                except Empty:
                    pass
                    
            try:
                self.frame_queue.put_nowait(frame_data)
                frame_id += 1
                self.frame_count += 1
            except:
                pass
                
        cap.release()
        
    def _correction_worker(self):
        """Apply fisheye correction efficiently"""
        print("✅ Frame correction worker started")
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                # Apply correction
                start_time = time.time()
                corrected_frame = self.fisheye_corrector.correct(frame_data['frame'])
                correction_time = time.time() - start_time
                
                corrected_data = {
                    'frame': corrected_frame,
                    'frame_id': frame_data['frame_id'],
                    'capture_time': frame_data['capture_time'],
                    'correction_time': correction_time,
                    'timestamp': time.time()
                }
                
                # Smart queue management
                if self.corrected_queue.full():
                    try:
                        self.corrected_queue.get_nowait()  # Drop oldest
                    except Empty:
                        pass
                        
                self.corrected_queue.put_nowait(corrected_data)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Correction error: {e}")
                continue
                
    def _detection_worker(self):
        """Single optimized detection worker"""
        print("🧠 Detection worker started")
        
        last_detection_time = 0
        detection_interval = 1.0 / self.target_detection_rate
        
        while self.running:
            try:
                # Rate limiting for target detection rate
                current_time = time.time()
                if current_time - last_detection_time < detection_interval:
                    time.sleep(0.01)
                    continue
                
                frame_data = self.corrected_queue.get(timeout=1)
                
                # Process frame
                start_time = time.time()
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        tracked_objects, processed_frame = self.detector.process_frame(frame_data['frame'])
                else:
                    tracked_objects, processed_frame = self.detector.process_frame(frame_data['frame'])
                
                process_time = time.time() - start_time
                last_detection_time = current_time
                
                # Calculate frame age
                frame_age = current_time - frame_data['capture_time']
                
                result_data = {
                    'frame_id': frame_data['frame_id'],
                    'tracked_objects': tracked_objects,
                    'processed_frame': processed_frame,
                    'process_time': process_time,
                    'frame_age': frame_age,
                    'timestamp': current_time
                }
                
                # Smart queue management
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()  # Drop oldest
                    except Empty:
                        pass
                        
                self.result_queue.put_nowait(result_data)
                
                # Track performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 50:
                    self.processing_times.pop(0)
                    
                self.detection_count += 1
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Detection error: {e}")
                continue
                
    def _gpu_monitor(self):
        """Monitor GPU utilization"""
        print("📊 GPU monitor started")
        
        while self.running:
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                # Try to get actual GPU utilization
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    gpu_percent = None
                    gpu_temp = None
                
                gpu_data = {
                    'memory_mb': gpu_memory_used,
                    'memory_percent': gpu_memory_percent,
                    'gpu_percent': gpu_percent,
                    'temperature': gpu_temp,
                    'timestamp': time.time()
                }
                
                self.gpu_stats.append(gpu_data)
                if len(self.gpu_stats) > 100:
                    self.gpu_stats.pop(0)
                    
            time.sleep(0.5)  # Monitor every 500ms
            
    def get_latest_result(self):
        """Get latest result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def get_performance_stats(self):
        """Get performance statistics"""
        current_time = time.time()
        
        # Calculate detection rate
        recent_time_window = 5.0  # 5 seconds
        recent_detections = [t for t in self.processing_times if current_time - t < recent_time_window]
        actual_detection_rate = self.detection_count / max(current_time - getattr(self, 'start_time', current_time), 1)
        
        # GPU stats
        gpu_stats = {'available': False}
        if self.gpu_stats:
            recent_gpu = self.gpu_stats[-10:]
            gpu_stats = {
                'available': True,
                'memory_percent': np.mean([g['memory_percent'] for g in recent_gpu]),
                'gpu_percent': np.mean([g['gpu_percent'] for g in recent_gpu if g['gpu_percent'] is not None]) if any(g['gpu_percent'] is not None for g in recent_gpu) else None,
                'temperature': recent_gpu[-1]['temperature'] if recent_gpu[-1]['temperature'] is not None else None
            }
        
        # Queue stats
        queue_stats = {
            'frame_queue': self.frame_queue.qsize(),
            'corrected_queue': self.corrected_queue.qsize(),
            'result_queue': self.result_queue.qsize()
        }
        
        return {
            'detection_rate': actual_detection_rate,
            'target_rate': self.target_detection_rate,
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frames_captured': self.frame_count,
            'detections_completed': self.detection_count,
            'gpu': gpu_stats,
            'queues': queue_stats
        }
        
    def adjust_detection_rate(self, change):
        """Adjust target detection rate"""
        self.target_detection_rate = max(1, min(50, self.target_detection_rate + change))
        return self.target_detection_rate
        
    def stop_pipeline(self):
        """Stop all workers"""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=2)
            
        print("🛑 Optimized pipeline stopped")

def main():
    print("🚀 OPTIMIZED SINGLE GPU TEST")
    print("=" * 50)
    print("Goal: Efficient GPU utilization with single model")
    print("Method: Optimized pipeline + smart queue management")
    print("Controls: +/- adjust detection rate, q=quit")
    print("=" * 50)
    
    # Initialize optimized pipeline
    pipeline = OptimizedGPUPipeline()
    pipeline.start_time = time.time()
    pipeline.start_optimized_pipeline()
    
    print("✅ Optimized pipeline started")
    
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
            result = pipeline.get_latest_result()
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
                
            # Get stats periodically
            if current_time - last_stats_time >= 1.0:
                stats = pipeline.get_performance_stats()
                last_stats_time = current_time
            else:
                stats = pipeline.get_performance_stats()
            
            # Draw status on frame
            if last_result:
                tracked_objects = last_result['tracked_objects']
                frame_age = last_result['frame_age']
                
                # Status text
                status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Age: {frame_age*1000:.1f}ms"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # GPU stats
                if stats['gpu']['available']:
                    gpu_text = f"GPU Mem: {stats['gpu']['memory_percent']:.1f}%"
                    if stats['gpu']['gpu_percent'] is not None:
                        gpu_text += f" | GPU: {stats['gpu']['gpu_percent']:.1f}%"
                    if stats['gpu']['temperature'] is not None:
                        gpu_text += f" | {stats['gpu']['temperature']}°C"
                    cv2.putText(display_frame, gpu_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Detection performance
                detection_text = f"Detection: {stats['detection_rate']:.1f}/{stats['target_rate']} Hz | Process: {stats['avg_process_time']*1000:.1f}ms"
                cv2.putText(display_frame, detection_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Queue status
                queue_text = f"Queues: F{stats['queues']['frame_queue']} C{stats['queues']['corrected_queue']} R{stats['queues']['result_queue']}"
                cv2.putText(display_frame, queue_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Print periodic updates
                if last_result['frame_id'] % 100 == 0:
                    print(f"\n📊 Frame {last_result['frame_id']} | Detection: {stats['detection_rate']:.1f} Hz")
                    print(f"    GPU Memory: {stats['gpu']['memory_percent']:.1f}%", end="")
                    if stats['gpu']['gpu_percent'] is not None:
                        print(f" | GPU Util: {stats['gpu']['gpu_percent']:.1f}%", end="")
                    print(f" | Process: {stats['avg_process_time']*1000:.1f}ms")
                    
                    # Show object coordinates
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
            cv2.imshow('Camera 8 - Optimized GPU Pipeline', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested")
                break
            elif key == ord('+') or key == ord('='):
                new_rate = pipeline.adjust_detection_rate(2)
                print(f"🔧 Detection rate: {new_rate} Hz")
            elif key == ord('-'):
                new_rate = pipeline.adjust_detection_rate(-2)
                print(f"🔧 Detection rate: {new_rate} Hz")
            elif key == ord('r'):
                pipeline.target_detection_rate = 10
                print(f"🔧 Detection rate reset: 10 Hz")
    
    finally:
        # Cleanup
        pipeline.stop_pipeline()
        cv2.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        final_stats = pipeline.get_performance_stats()
        
        print(f"\n📊 Optimized GPU Test Results:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Frames Captured: {final_stats['frames_captured']}")
        print(f"   Detections: {final_stats['detections_completed']}")
        print(f"   Final Detection Rate: {final_stats['detection_rate']:.1f} Hz")
        print(f"   Avg Process Time: {final_stats['avg_process_time']*1000:.1f}ms")
        
        if final_stats['gpu']['available']:
            print(f"   GPU Memory: {final_stats['gpu']['memory_percent']:.1f}%")
            if final_stats['gpu']['gpu_percent'] is not None:
                print(f"   GPU Utilization: {final_stats['gpu']['gpu_percent']:.1f}%")

if __name__ == "__main__":
    main() 