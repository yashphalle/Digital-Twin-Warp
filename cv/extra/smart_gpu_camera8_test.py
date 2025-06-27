#!/usr/bin/env python3
"""
Smart GPU Camera 8 Test - Ultimate Performance
Combines smart queue system with GPU optimization for maximum GPU utilization
and real-time frame processing without latency buildup
"""

import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

class SmartFrameQueue:
    """Smart frame queue that always provides the newest frame"""
    
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.frame_buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frames_dropped = 0
        self.frames_added = 0
        
    def put_frame(self, frame):
        """Add frame to queue, dropping old frames if necessary"""
        with self.lock:
            if len(self.frame_buffer) >= self.max_size:
                # Queue is full, drop oldest frame
                dropped_frame = self.frame_buffer.popleft()
                self.frames_dropped += 1
            
            self.frame_buffer.append({
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self.frames_added
            })
            self.frames_added += 1
    
    def get_newest_frames(self, count=1):
        """Get the newest N frames, return empty list if not enough frames"""
        with self.lock:
            if len(self.frame_buffer) < count:
                return []
            
            # Get the newest N frames
            newest_frames = list(self.frame_buffer)[-count:]
            
            # Clear all frames to prevent processing backlog
            self.frame_buffer.clear()
            
            return newest_frames
    
    def get_stats(self):
        """Get queue statistics"""
        with self.lock:
            return {
                'queue_size': len(self.frame_buffer),
                'frames_added': self.frames_added,
                'frames_dropped': self.frames_dropped,
                'drop_rate': self.frames_dropped / max(self.frames_added, 1)
            }

class SmartGPUDetector:
    """GPU-optimized detector with smart queue integration"""
    
    def __init__(self):
        self.tracker = DetectorTracker()
        self.tracker.set_camera_id(8)
        
        # GPU Optimization Settings
        self.setup_gpu_optimizations()
        
        # Smart batch processing
        self.batch_size = 2  # Process multiple newest frames simultaneously
        self.adaptive_batch_size = True  # Automatically adjust batch size
        
        # Parallel processing with smart queuing
        self.num_workers = 2
        self.detection_queue = Queue(maxsize=5)  # Smaller queue to prevent buildup
        self.result_queue = Queue(maxsize=5)
        self.workers = []
        self.running = False
        
        # Performance tracking
        self.gpu_utilization = []
        self.processing_times = []
        self.frame_age_stats = []
        
        # Adaptive processing
        self.target_processing_time = 0.3  # Target 300ms processing time
        self.last_batch_adjustment = time.time()
        
    def setup_gpu_optimizations(self):
        """Configure GPU for maximum utilization"""
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory management
            torch.cuda.empty_cache()
            
            # Enable tensor core usage if available
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Increase GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 Smart GPU optimizations enabled for: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print(f"🧠 Smart Queue + GPU Parallel Processing")
            
            # Configure model for optimal GPU usage
            try:
                self.tracker.model.half()  # Use FP16 for 2x speed
                print("🔥 Model converted to FP16 for 2x speed boost")
            except:
                print("⚠️  FP16 conversion failed, using FP32")
                
        else:
            print("⚠️  CUDA not available - falling back to CPU")
    
    def start_smart_workers(self):
        """Start smart parallel detection workers"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._smart_detection_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
            
        print(f"✅ Started {self.num_workers} smart GPU workers")
    
    def stop_smart_workers(self):
        """Stop smart detection workers"""
        self.running = False
        
        # Clear queues
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except Empty:
                break
                
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2)
            
        print("🛑 Smart GPU workers stopped")
    
    def _smart_detection_worker(self, worker_id):
        """Smart worker that processes newest frames with GPU optimization"""
        print(f"🧠 Smart GPU worker {worker_id} started")
        
        while self.running:
            try:
                # Get newest frame data to process
                frame_data = self.detection_queue.get(timeout=1)
                if frame_data is None:  # Shutdown signal
                    break
                
                frames_info, batch_timestamp = frame_data
                
                # Calculate frame ages
                current_time = time.time()
                frame_ages = [current_time - info['timestamp'] for info in frames_info]
                avg_frame_age = np.mean(frame_ages)
                
                # Process newest frames with GPU optimization
                start_time = time.time()
                frames = [info['frame'] for info in frames_info]
                batch_results = self._process_smart_batch(frames, worker_id, avg_frame_age)
                process_time = time.time() - start_time
                
                # Track processing performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                self.frame_age_stats.append(avg_frame_age)
                if len(self.frame_age_stats) > 50:
                    self.frame_age_stats.pop(0)
                
                # Send results back
                result_data = {
                    'frame_ids': [info['frame_id'] for info in frames_info],
                    'results': batch_results,
                    'process_time': process_time,
                    'frame_age': avg_frame_age,
                    'worker_id': worker_id,
                    'batch_size': len(frames)
                }
                
                self.result_queue.put(result_data)
                
                # Adaptive batch size adjustment
                self._adjust_batch_size(process_time)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Smart worker {worker_id} error: {e}")
                continue
    
    def _process_smart_batch(self, frames, worker_id, avg_frame_age):
        """Process batch of newest frames with GPU optimization"""
        if not frames:
            return []
        
        try:
            # Monitor GPU utilization
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated()
                torch.cuda.synchronize()
            
            batch_results = []
            
            if len(frames) == 1:
                # Single frame processing with GPU optimization
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        tracked_objects, _ = self.tracker.process_frame(frames[0])
                else:
                    tracked_objects, _ = self.tracker.process_frame(frames[0])
                batch_results.append(tracked_objects)
            else:
                # Parallel processing for multiple newest frames
                with ThreadPoolExecutor(max_workers=min(len(frames), 3)) as executor:
                    futures = []
                    for frame in frames:
                        future = executor.submit(self._process_optimized_frame, frame)
                        futures.append(future)
                    
                    for future in futures:
                        result = future.result()
                        batch_results.append(result)
            
            # Monitor GPU utilization
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated()
                gpu_memory_used = (gpu_memory_after - gpu_memory_before) / 1024**2
                
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                utilization = (gpu_memory_used / total_memory) * 100
                
                self.gpu_utilization.append(utilization)
                if len(self.gpu_utilization) > 50:
                    self.gpu_utilization.pop(0)
            
            return batch_results
            
        except Exception as e:
            print(f"❌ Smart batch processing error in worker {worker_id}: {e}")
            return [[] for _ in frames]
    
    def _process_optimized_frame(self, frame):
        """Process single frame with GPU optimizations"""
        try:
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    tracked_objects, _ = self.tracker.process_frame(frame)
            else:
                tracked_objects, _ = self.tracker.process_frame(frame)
            return tracked_objects
        except Exception as e:
            print(f"❌ Optimized frame processing error: {e}")
            return []
    
    def _adjust_batch_size(self, process_time):
        """Dynamically adjust batch size based on performance"""
        if not self.adaptive_batch_size:
            return
            
        current_time = time.time()
        if current_time - self.last_batch_adjustment < 5.0:  # Adjust every 5 seconds
            return
            
        if process_time > self.target_processing_time and self.batch_size > 1:
            # Processing too slow, reduce batch size
            self.batch_size = max(1, self.batch_size - 1)
            print(f"🔧 Reduced batch size to {self.batch_size} (processing time: {process_time:.2f}s)")
        elif process_time < self.target_processing_time * 0.7 and self.batch_size < 4:
            # Processing fast enough, increase batch size
            self.batch_size = min(4, self.batch_size + 1)
            print(f"🔧 Increased batch size to {self.batch_size} (processing time: {process_time:.2f}s)")
            
        self.last_batch_adjustment = current_time
    
    def submit_newest_frames(self, frame_queue):
        """Submit newest frames from smart queue for processing"""
        try:
            # Get newest frames based on current batch size
            newest_frames = frame_queue.get_newest_frames(self.batch_size)
            if not newest_frames:
                return False
                
            batch_data = (newest_frames, time.time())
            self.detection_queue.put(batch_data, timeout=0.05)  # Very short timeout
            return True
        except:
            return False  # Queue full or timeout
    
    def get_results(self):
        """Get processing results if available"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_frame_age': np.mean(self.frame_age_stats) if self.frame_age_stats else 0,
            'detection_queue_size': self.detection_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            stats.update({
                'gpu_memory_mb': current_memory,
                'gpu_memory_percent': (current_memory / total_memory) * 100,
                'avg_gpu_utilization': np.mean(self.gpu_utilization) if self.gpu_utilization else 0
            })
        
        return stats

class SmartFrameCapture:
    """Smart threaded frame capture for newest frame processing"""
    
    def __init__(self, camera_url, smart_queue):
        self.camera_url = camera_url
        self.smart_queue = smart_queue
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.frames_captured = 0
        self.capture_errors = 0
        
    def start(self):
        """Start smart capture thread"""
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal camera buffer
        
        if not self.cap.isOpened():
            print("❌ Failed to connect to camera")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._smart_capture_loop, daemon=True)
        self.capture_thread.start()
        print("✅ Smart frame capture thread started")
        return True
    
    def _smart_capture_loop(self):
        """Smart capture loop that feeds newest frames to queue"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                self.capture_errors += 1
                
                if consecutive_failures >= max_failures:
                    print("🔄 Smart capture attempting to reconnect...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.camera_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                    
                    if self.cap.isOpened():
                        print("✅ Smart capture reconnected!")
                    else:
                        print("❌ Smart capture reconnection failed...")
                        time.sleep(3)
                continue
            
            consecutive_failures = 0
            self.frames_captured += 1
            
            # Add frame to smart queue (will auto-drop old frames)
            self.smart_queue.put_frame(frame)
    
    def stop(self):
        """Stop smart capture thread"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("🛑 Smart frame capture stopped")
    
    def get_stats(self):
        """Get capture statistics"""
        return {
            'frames_captured': self.frames_captured,
            'capture_errors': self.capture_errors,
            'error_rate': self.capture_errors / max(self.frames_captured, 1)
        }

def main():
    print("🧠🚀 SMART GPU CAMERA 8 TEST - ULTIMATE PERFORMANCE")
    print("=" * 70)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Features: Smart Queue + GPU Optimization + Parallel Processing")
    print("Always processes NEWEST frames with MAXIMUM GPU utilization")
    print("Press 'q' to quit")
    print("=" * 70)
    
    # Initialize smart components
    smart_queue = SmartFrameQueue(max_size=5)
    smart_detector = SmartGPUDetector()
    smart_detector.start_smart_workers()
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Smart frame capture
    camera_url = Config.RTSP_CAMERA_URLS[8]
    smart_capture = SmartFrameCapture(camera_url, smart_queue)
    
    if not smart_capture.start():
        print("❌ Failed to start smart camera capture")
        smart_detector.stop_smart_workers()
        return
    
    print("✅ Smart GPU camera system started")
    print("🎯 Processing newest frames with maximum GPU utilization...")
    
    # Processing state
    frame_count = 0
    detection_count = 0
    last_results = []
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    
    # Smart processing timing
    last_detection_time = 0
    detection_interval = 1.0 / 4  # Target 4 detections per second
    
    try:
        while True:
            # Get newest frames from smart queue for display
            newest_frames = smart_queue.get_newest_frames(1)
            
            if not newest_frames:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            frame_data = newest_frames[0]
            frame = frame_data['frame']
            frame_timestamp = frame_data['timestamp']
            frame_age = time.time() - frame_timestamp
            
            frame_count += 1
            fps_counter += 1
            
            # Apply fisheye correction
            corrected_frame = fisheye_corrector.correct(frame)
            
            # Smart detection timing - based on time, not frame count
            current_time = time.time()
            should_detect = (current_time - last_detection_time) >= detection_interval
            
            if should_detect:
                # Submit newest frames for GPU processing
                submitted = smart_detector.submit_newest_frames(smart_queue)
                if submitted:
                    detection_count += 1
                    last_detection_time = current_time
            
            # Check for results from GPU workers
            result_data = smart_detector.get_results()
            if result_data:
                last_results = result_data['results']
                if last_results and len(last_results) > 0:
                    # Use the latest result from the batch
                    tracked_objects = last_results[-1]
                else:
                    tracked_objects = []
            else:
                # Use previous results if no new results available
                tracked_objects = last_results[-1] if last_results else []
            
            # Draw detections on corrected frame
            display_frame = corrected_frame.copy()
            
            # Draw objects manually with smart GPU results
            for obj in tracked_objects:
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
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            else:
                fps = fps_counter / max(current_time - last_fps_time, 0.001)
            
            # Get comprehensive statistics
            queue_stats = smart_queue.get_stats()
            capture_stats = smart_capture.get_stats()
            perf_stats = smart_detector.get_performance_stats()
            
            # Status display with smart GPU stats
            status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Age: {frame_age*1000:.1f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Smart queue stats
            queue_text = f"Queue: {queue_stats['queue_size']}/5 | Dropped: {queue_stats['frames_dropped']} ({queue_stats['drop_rate']:.1%})"
            cv2.putText(display_frame, queue_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # GPU performance stats
            if 'gpu_memory_percent' in perf_stats:
                gpu_text = f"GPU: {perf_stats['gpu_memory_percent']:.1f}% | Util: {perf_stats['avg_gpu_utilization']:.1f}% | Batch: {perf_stats['batch_size']}"
                cv2.putText(display_frame, gpu_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Processing performance
            process_text = f"Process: {perf_stats['avg_processing_time']*1000:.1f}ms | Workers: {perf_stats['num_workers']} | Frame Age: {perf_stats['avg_frame_age']*1000:.1f}ms"
            cv2.putText(display_frame, process_text, (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Print coordinates when we have new results
            if result_data and tracked_objects:
                worker_id = result_data.get('worker_id', 0)
                batch_size = result_data.get('batch_size', 1)
                process_time = result_data.get('process_time', 0)
                frame_age_result = result_data.get('frame_age', 0)
                
                print(f"\n🧠🚀 Worker {worker_id} processed {batch_size} newest frames: {len(tracked_objects)} objects")
                print(f"    Process: {process_time*1000:.1f}ms | Frame Age: {frame_age_result*1000:.1f}ms | GPU Batch: {perf_stats['batch_size']}")
                for i, obj in enumerate(tracked_objects):
                    real_center = obj.get('real_center')
                    confidence = obj.get('confidence', 0)
                    if real_center and real_center[0] is not None:
                        x, y = real_center
                        print(f"    Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display
            height, width = display_frame.shape[:2]
            display_width = 1280
            display_height = int(height * (display_width / width))
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Display
            cv2.imshow('Camera 8 - Smart GPU (Ultimate Performance)', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                break
            elif key == ord('+') or key == ord('='):
                # Increase detection rate
                detection_interval = max(detection_interval / 1.2, 0.1)
                print(f"🔧 Detection rate: {1/detection_interval:.1f} Hz")
            elif key == ord('-'):
                # Decrease detection rate
                detection_interval = min(detection_interval * 1.2, 2.0)
                print(f"🔧 Detection rate: {1/detection_interval:.1f} Hz")
            elif key == ord('r'):
                # Reset detection rate
                detection_interval = 1.0 / 4
                print(f"🔧 Detection rate reset: {1/detection_interval:.1f} Hz")
            elif key == ord('a'):
                # Toggle adaptive batch sizing
                smart_detector.adaptive_batch_size = not smart_detector.adaptive_batch_size
                print(f"🔧 Adaptive batch sizing: {'ON' if smart_detector.adaptive_batch_size else 'OFF'}")
    
    finally:
        # Cleanup
        smart_detector.stop_smart_workers()
        smart_capture.stop()
        cv2.destroyAllWindows()
        
        # Performance summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        final_perf_stats = smart_detector.get_performance_stats()
        final_queue_stats = smart_queue.get_stats()
        final_capture_stats = smart_capture.get_stats()
        
        print(f"\n🧠🚀 Smart GPU test completed")
        print(f"📊 Performance Summary:")
        print(f"   Display FPS: {avg_fps:.1f}")
        print(f"   Detection Rate: {detection_count/max(total_time, 1):.1f} Hz")
        print(f"   Avg Processing Time: {final_perf_stats['avg_processing_time']*1000:.1f}ms")
        print(f"   Avg Frame Age: {final_perf_stats['avg_frame_age']*1000:.1f}ms")
        print(f"   Final Batch Size: {final_perf_stats['batch_size']}")
        
        if 'gpu_memory_percent' in final_perf_stats:
            print(f"   GPU Memory Usage: {final_perf_stats['gpu_memory_percent']:.1f}%")
            print(f"   Avg GPU Utilization: {final_perf_stats['avg_gpu_utilization']:.1f}%")
        
        print(f"   Frames Captured: {final_capture_stats['frames_captured']}")
        print(f"   Queue Drops: {final_queue_stats['frames_dropped']} ({final_queue_stats['drop_rate']:.1%})")
        print(f"   Total Runtime: {total_time:.1f}s")

if __name__ == "__main__":
    main() 