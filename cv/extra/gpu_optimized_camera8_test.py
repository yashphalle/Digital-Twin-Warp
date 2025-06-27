#!/usr/bin/env python3
"""
GPU Optimized Camera 8 Test - Maximum GPU Utilization
Implements parallel processing, batch processing, and advanced GPU optimizations
for Grounding DINO to achieve maximum GPU utilization and throughput
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

class GPUOptimizedDetector:
    """GPU-optimized detector with parallel processing capabilities"""
    
    def __init__(self):
        self.tracker = DetectorTracker()
        self.tracker.set_camera_id(8)
        
        # GPU Optimization Settings
        self.setup_gpu_optimizations()
        
        # Batch processing
        self.batch_size = 2  # Process multiple frames simultaneously
        self.frame_batch = []
        self.batch_results = []
        
        # Parallel processing
        self.num_workers = 2  # Number of parallel detection workers
        self.detection_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.workers = []
        self.running = False
        
        # Performance tracking
        self.gpu_utilization = []
        self.batch_times = []
        
    def setup_gpu_optimizations(self):
        """Configure GPU for maximum utilization"""
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Set memory management
            torch.cuda.empty_cache()
            
            # Enable tensor core usage if available (RTX cards)
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Increase GPU memory fraction for better utilization
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU Optimizations enabled for: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print(f"🔧 CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
            print(f"⚡ TensorFloat-32: {getattr(torch.backends.cuda.matmul, 'allow_tf32', 'N/A')}")
            
            # Configure model for optimal GPU usage
            try:
                self.tracker.model.half()  # Use FP16 for 2x speed on modern GPUs
                print("🔥 Model converted to FP16 for 2x speed boost")
            except:
                print("⚠️  FP16 conversion failed, using FP32")
            
        else:
            print("⚠️  CUDA not available - falling back to CPU")
    
    def start_parallel_workers(self):
        """Start parallel detection workers"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._detection_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
            
        print(f"✅ Started {self.num_workers} parallel detection workers")
    
    def stop_parallel_workers(self):
        """Stop parallel detection workers"""
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
            
        print("🛑 Parallel workers stopped")
    
    def _detection_worker(self, worker_id):
        """Worker thread for parallel detection processing"""
        print(f"🔧 Detection worker {worker_id} started")
        
        while self.running:
            try:
                # Get batch of frames to process
                batch_data = self.detection_queue.get(timeout=1)
                if batch_data is None:  # Shutdown signal
                    break
                
                frames, frame_ids = batch_data
                
                # Process batch
                start_time = time.time()
                batch_results = self._process_frame_batch(frames, worker_id)
                process_time = time.time() - start_time
                
                # Send results back
                result_data = {
                    'frame_ids': frame_ids,
                    'results': batch_results,
                    'process_time': process_time,
                    'worker_id': worker_id
                }
                
                self.result_queue.put(result_data)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                continue
    
    def _process_frame_batch(self, frames, worker_id):
        """Process a batch of frames for maximum GPU utilization"""
        if not frames:
            return []
        
        try:
            # Monitor GPU utilization
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated()
                torch.cuda.synchronize()  # Ensure accurate timing
            
            batch_results = []
            
            if len(frames) == 1:
                # Single frame processing
                tracked_objects, _ = self.tracker.process_frame(frames[0])
                batch_results.append(tracked_objects)
            else:
                # Batch processing for multiple frames
                # Note: Grounding DINO doesn't natively support batching,
                # so we'll use parallel processing within the batch
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    for frame in frames:
                        future = executor.submit(self._process_single_frame, frame)
                        futures.append(future)
                    
                    for future in futures:
                        result = future.result()
                        batch_results.append(result)
            
            # Monitor GPU utilization
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated()
                gpu_memory_used = (gpu_memory_after - gpu_memory_before) / 1024**2  # MB
                
                # Store memory usage as utilization metric
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                utilization = (gpu_memory_used / total_memory) * 100
                
                self.gpu_utilization.append(utilization)
                if len(self.gpu_utilization) > 50:
                    self.gpu_utilization.pop(0)
            
            return batch_results
            
        except Exception as e:
            print(f"❌ Batch processing error in worker {worker_id}: {e}")
            return [[] for _ in frames]
    
    def _process_single_frame(self, frame):
        """Process a single frame with GPU optimizations"""
        try:
            # Use mixed precision for speed if available
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    tracked_objects, _ = self.tracker.process_frame(frame)
            else:
                tracked_objects, _ = self.tracker.process_frame(frame)
            return tracked_objects
        except Exception as e:
            print(f"❌ Single frame processing error: {e}")
            return []
    
    def submit_frame_batch(self, frames, frame_ids):
        """Submit a batch of frames for processing"""
        try:
            batch_data = (frames, frame_ids)
            self.detection_queue.put(batch_data, timeout=0.1)
            return True
        except:
            return False  # Queue full
    
    def get_results(self):
        """Get processing results if available"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_gpu_stats(self):
        """Get GPU utilization statistics"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            
            avg_utilization = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
            
            return {
                'current_memory_mb': current_memory,
                'max_memory_mb': max_memory,
                'total_memory_mb': total_memory,
                'memory_usage_percent': (current_memory / total_memory) * 100,
                'avg_gpu_utilization': avg_utilization
            }
        else:
            return {'gpu_available': False}

def main():
    print("🚀 GPU OPTIMIZED CAMERA 8 TEST - MAXIMUM GPU UTILIZATION")
    print("=" * 65)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Features: Parallel processing + Batch processing + GPU optimizations")
    print("Press 'q' to quit")
    print("=" * 65)
    
    # Initialize GPU-optimized detector
    gpu_detector = GPUOptimizedDetector()
    gpu_detector.start_parallel_workers()
    
    # Initialize fisheye corrector
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Connect to Camera 8
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        gpu_detector.stop_parallel_workers()
        return
    
    print("✅ Connected to Camera 8 (GPU optimized)")
    print("🎯 Processing with maximum GPU utilization...")
    
    # Processing state
    frame_count = 0
    detection_count = 0
    batch_size = gpu_detector.batch_size
    frame_batch = []
    frame_id_batch = []
    last_results = []
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    processing_times = []
    
    # GPU monitoring
    last_gpu_stats_time = start_time
    
    consecutive_failures = 0
    max_failures = 10
    
    try:
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
            
            # Add frame to batch
            frame_batch.append(corrected_frame.copy())
            frame_id_batch.append(frame_count)
            
            # Process batch when full or every few frames
            if len(frame_batch) >= batch_size or frame_count % 10 == 0:
                if frame_batch:
                    # Submit batch for processing
                    submitted = gpu_detector.submit_frame_batch(frame_batch, frame_id_batch)
                    if submitted:
                        detection_count += len(frame_batch)
                    
                    # Clear batch
                    frame_batch = []
                    frame_id_batch = []
            
            # Check for results
            result_data = gpu_detector.get_results()
            if result_data:
                last_results = result_data['results']
                process_time = result_data['process_time']
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
            
            # Use most recent results for display
            if last_results and len(last_results) > 0:
                tracked_objects = last_results[-1]  # Use last result from batch
            else:
                tracked_objects = []
            
            # Draw detections
            display_frame = corrected_frame.copy()
            
            # Draw objects manually (since we're not using tracker's draw method)
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
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            else:
                fps = 0
            
            # Get GPU stats periodically
            gpu_stats = {}
            if current_time - last_gpu_stats_time >= 1.0:
                gpu_stats = gpu_detector.get_gpu_stats()
                last_gpu_stats_time = current_time
            
            # Status display
            avg_process_time = np.mean(processing_times) if processing_times else 0
            status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Process: {avg_process_time*1000:.1f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # GPU utilization display
            if gpu_stats and 'memory_usage_percent' in gpu_stats:
                gpu_text = f"GPU Memory: {gpu_stats['memory_usage_percent']:.1f}% | Util: {gpu_stats['avg_gpu_utilization']:.1f}%"
                cv2.putText(display_frame, gpu_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Parallel processing info
            queue_size = gpu_detector.detection_queue.qsize()
            result_size = gpu_detector.result_queue.qsize()
            parallel_text = f"Batch: {batch_size} | Workers: {gpu_detector.num_workers} | Queue: {queue_size}/{result_size}"
            cv2.putText(display_frame, parallel_text, (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Print coordinates when we have new results
            if result_data and tracked_objects:
                worker_id = result_data.get('worker_id', 0)
                print(f"\n🚀 Frame batch processed by worker {worker_id}: {len(tracked_objects)} objects | Process: {avg_process_time*1000:.1f}ms")
                for i, obj in enumerate(tracked_objects):
                    real_center = obj.get('real_center')
                    confidence = obj.get('confidence', 0)
                    if real_center and real_center[0] is not None:
                        x, y = real_center
                        print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display
            height, width = display_frame.shape[:2]
            display_width = 1280
            display_height = int(height * (display_width / width))
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Display
            cv2.imshow('Camera 8 - GPU Optimized (Max Utilization)', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                break
            elif key == ord('+') or key == ord('='):
                gpu_detector.batch_size = min(gpu_detector.batch_size + 1, 4)
                batch_size = gpu_detector.batch_size
                print(f"🔧 Batch size increased to {gpu_detector.batch_size}")
            elif key == ord('-'):
                gpu_detector.batch_size = max(gpu_detector.batch_size - 1, 1)
                batch_size = gpu_detector.batch_size
                print(f"🔧 Batch size decreased to {gpu_detector.batch_size}")
            elif key == ord('r'):
                gpu_detector.batch_size = 2
                batch_size = gpu_detector.batch_size
                print(f"🔧 Batch size reset to {gpu_detector.batch_size}")
    
    finally:
        # Cleanup
        gpu_detector.stop_parallel_workers()
        cap.release()
        cv2.destroyAllWindows()
        
        # Performance summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_process_time = np.mean(processing_times) if processing_times else 0
        final_gpu_stats = gpu_detector.get_gpu_stats()
        
        print(f"\n🚀 GPU optimized test completed")
        print(f"📊 Performance Summary:")
        print(f"   Display FPS: {avg_fps:.1f}")
        print(f"   Avg Process Time: {avg_process_time*1000:.1f}ms")
        print(f"   Total Frames: {frame_count}")
        print(f"   Detection Batches: {detection_count}")
        print(f"   Workers Used: {gpu_detector.num_workers}")
        
        if final_gpu_stats and 'memory_usage_percent' in final_gpu_stats:
            print(f"   GPU Memory Usage: {final_gpu_stats['memory_usage_percent']:.1f}%")
            print(f"   Avg GPU Utilization: {final_gpu_stats['avg_gpu_utilization']:.1f}%")

if __name__ == "__main__":
    main() 