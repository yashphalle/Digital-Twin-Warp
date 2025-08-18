#!/usr/bin/env python3
"""
Batch Detection Thread Pool
Enhanced version with YOLOv8 batch processing support
"""

import threading
import time
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List
import sys
import os
from collections import deque

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.yolo_detector import YOLOv8PalletDetector
from .queue_manager import QueueManager, FrameData
from config.yolo_config import get_detection_config

logger = logging.getLogger(__name__)

class BatchDetectionThreadPool:
    """Enhanced detection pool with batch processing support"""
    
    def __init__(self, num_workers: int = 3, batch_size: int = 30, queue_manager: QueueManager = None):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.queue_manager = queue_manager
        self.running = False
        
        # Thread pool for detection workers
        self.detection_pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="BatchDetection")
        
        # Batch buffers for each worker
        self.batch_buffers = [deque() for _ in range(num_workers)]
        self.buffer_locks = [threading.Lock() for _ in range(num_workers)]
        
        # GPU context management
        self.gpu_contexts = []
        self._initialize_gpu_contexts()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'batch_detections': 0,
            'detection_times': [],
            'batch_times': [],
            'worker_loads': {i: 0 for i in range(num_workers)}
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"✅ Batch Detection Thread Pool initialized")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Batch Size: {batch_size}")

    def _initialize_gpu_contexts(self):
        """Initialize GPU contexts for each worker"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU detection")
            
        for worker_id in range(self.num_workers):
            try:
                # Create YOLOv8 detector instance for this worker using config
                config = get_detection_config()
                device = f"cuda:{worker_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
                detector = YOLOv8PalletDetector(
                    model_path=config['model_path'],
                    device=device,
                    conf_threshold=config['confidence_threshold']
                )
                
                context = {
                    'worker_id': worker_id,
                    'detector': detector,
                    'device': device,
                    'batch_buffer': self.batch_buffers[worker_id],
                    'buffer_lock': self.buffer_locks[worker_id]
                }
                
                self.gpu_contexts.append(context)
                logger.info(f"🎯 Detection worker {worker_id} initialized on {device}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize detection worker {worker_id}: {e}")
                raise

    def start_detection_workers(self):
        """Start all detection workers"""
        if self.running:
            logger.warning("Detection workers already running")
            return
            
        self.running = True
        
        # Start batch detection workers
        for worker_id in range(self.num_workers):
            future = self.detection_pool.submit(self._batch_detection_worker, worker_id)
            logger.info(f"🚀 Started batch detection worker {worker_id}")

    def _batch_detection_worker(self, worker_id: int):
        """Enhanced worker with batch processing"""
        context = self.gpu_contexts[worker_id]
        detector = context['detector']
        batch_buffer = context['batch_buffer']
        buffer_lock = context['buffer_lock']
        
        logger.info(f"✅ Batch detection worker {worker_id} started on {context['device']}")
        
        while self.running:
            try:
                # Try to get frame from queue
                frame_data = self.queue_manager.get_frame_for_detection(timeout=0.1)
                
                if frame_data is None:
                    # No frame available, process any buffered frames
                    self._process_batch_if_ready(worker_id, force=True)
                    continue
                
                # Add frame to batch buffer
                with buffer_lock:
                    batch_buffer.append(frame_data)
                
                # Process batch if ready
                if len(batch_buffer) >= self.batch_size:
                    self._process_batch_if_ready(worker_id, force=True)
                else:
                    # Check if we should process partial batch (timeout-based)
                    self._process_batch_if_ready(worker_id, force=False)
                    
            except Exception as e:
                logger.error(f"❌ Batch detection worker {worker_id} error: {e}")
                time.sleep(0.1)
        
        logger.info(f"🛑 Batch detection worker {worker_id} stopped")

    def _process_batch_if_ready(self, worker_id: int, force: bool = False):
        """Process batch if ready or forced"""
        context = self.gpu_contexts[worker_id]
        detector = context['detector']
        batch_buffer = context['batch_buffer']
        buffer_lock = context['buffer_lock']
        
        with buffer_lock:
            if len(batch_buffer) == 0:
                return
                
            # Decide whether to process batch
            should_process = force or len(batch_buffer) >= self.batch_size
            
            if not should_process:
                return
            
            # Extract frames and frame_data objects
            batch_frame_data = list(batch_buffer)
            batch_frames = [fd.frame for fd in batch_frame_data]
            batch_buffer.clear()
        
        if len(batch_frames) == 0:
            return
        
        try:
            # Record start time
            start_time = time.time()
            
            # Perform batch detection
            if len(batch_frames) == 1:
                # Single frame detection
                detections_list = [detector.detect_pallets(batch_frames[0])]
            else:
                # Batch detection
                detections_list = detector.detect_pallets_batch(batch_frames)
            
            # Record detection time
            detection_time = time.time() - start_time
            
            # Update frame data with detection results
            for frame_data, detections in zip(batch_frame_data, detections_list):
                frame_data.detections = detections
                frame_data.detection_time = detection_time / len(batch_frames)  # Average time per frame
                
                # Send to processing queue
                self.queue_manager.put_frame_for_processing(frame_data)
            
            # Update statistics
            with self._stats_lock:
                self.detection_stats['total_detections'] += len(batch_frames)
                if len(batch_frames) > 1:
                    self.detection_stats['batch_detections'] += 1
                    self.detection_stats['batch_times'].append(detection_time)
                self.detection_stats['detection_times'].append(detection_time)
                self.detection_stats['worker_loads'][worker_id] += len(batch_frames)
            
            # Log performance
            fps = len(batch_frames) / detection_time if detection_time > 0 else 0
            if len(batch_frames) > 1:
                logger.info(f"🎯 Worker {worker_id}: Batch processed {len(batch_frames)} frames in {detection_time:.3f}s ({fps:.2f} FPS)")
            else:
                logger.info(f"🔍 Worker {worker_id}: Single frame processed in {detection_time:.3f}s ({fps:.2f} FPS)")
                
        except Exception as e:
            logger.error(f"❌ Batch processing error in worker {worker_id}: {e}")
            # Return frames to processing queue without detections
            for frame_data in batch_frame_data:
                frame_data.detections = []
                self.queue_manager.put_frame_for_processing(frame_data)

    def stop(self):
        """Stop all detection workers"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping batch detection workers...")
        self.running = False
        
        # Process any remaining frames in buffers
        for worker_id in range(self.num_workers):
            self._process_batch_if_ready(worker_id, force=True)
        
        # Shutdown thread pool
        self.detection_pool.shutdown(wait=True)
        logger.info("✅ Batch detection workers stopped")

    def get_statistics(self):
        """Get detection statistics"""
        with self._stats_lock:
            stats = self.detection_stats.copy()
            
        # Calculate averages
        if stats['detection_times']:
            stats['avg_detection_time'] = sum(stats['detection_times']) / len(stats['detection_times'])
            stats['avg_fps'] = 1.0 / stats['avg_detection_time']
        else:
            stats['avg_detection_time'] = 0
            stats['avg_fps'] = 0
            
        if stats['batch_times']:
            stats['avg_batch_time'] = sum(stats['batch_times']) / len(stats['batch_times'])
            stats['avg_batch_fps'] = self.batch_size / stats['avg_batch_time']
        else:
            stats['avg_batch_time'] = 0
            stats['avg_batch_fps'] = 0
            
        return stats
