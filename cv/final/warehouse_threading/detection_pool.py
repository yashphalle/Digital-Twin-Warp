#!/usr/bin/env python3
"""
Detection Thread Pool
Manages GPU detection threads with load balancing
"""

import threading
import time
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.yolo_detector import YOLOv8PalletDetector
from config.yolo_config import get_detection_config
from .queue_manager import QueueManager, FrameData

logger = logging.getLogger(__name__)

class DetectionThreadPool:
    """Manages pool of GPU detection threads"""
    
    def __init__(self, num_workers: int = 1, queue_manager: QueueManager = None):
        self.num_workers = num_workers
        self.queue_manager = queue_manager
        self.running = False
        
        # Thread pool for detection workers
        self.detection_pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="Detection")
        
        # GPU context management
        self.gpu_contexts = []
        self._initialize_gpu_contexts()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'detection_times': [],
            'worker_loads': {i: 0 for i in range(num_workers)}
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"✅ Detection Thread Pool initialized with {num_workers} workers")

    def _initialize_gpu_contexts(self):
        """Initialize separate GPU contexts for each worker"""
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
                    'device': detector.device,
                    'total_processed': 0,
                    'last_activity': time.time()
                }
                
                self.gpu_contexts.append(context)
                logger.info(f"🔍 Detection worker {worker_id} initialized on {detector.device}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize detection worker {worker_id}: {e}")

    def start_detection_workers(self):
        """Start all detection worker threads"""
        self.running = True
        
        # Submit worker tasks to thread pool
        for worker_id in range(self.num_workers):
            future = self.detection_pool.submit(self._detection_worker, worker_id)
            logger.info(f"🚀 Started detection worker {worker_id}")

    def _detection_worker(self, worker_id: int):
        """Worker function for detection threads"""
        context = self.gpu_contexts[worker_id]
        detector = context['detector']
        
        logger.info(f"✅ Detection worker {worker_id} started on {context['device']}")
        
        while self.running:
            try:
                # Get frame from camera queue
                frame_data = self.queue_manager.get_frame('camera_to_detection', timeout=1.0)
                if frame_data is None:
                    continue

                # DEBUG: Log which camera is being processed
                #logger.info(f"🔍 Worker {worker_id}: Processing Camera {frame_data.camera_id} frame {frame_data.frame_number}")

                # Record start time
                start_time = time.time()
                
                # Perform GPU detection with tracking (NEW: YOLOv8 tracking enabled)
                detections = detector.detect_pallets_with_tracking(frame_data.frame)
                
                # Record detection time
                detection_time = time.time() - start_time
                
                # Update frame data with detection results (ONLY NEW PART)
                frame_data.stage = "detected"
                frame_data.metadata.update({
                    'raw_detections': detections,
                    'detection_worker': worker_id,
                    'detection_time': detection_time,
                    'detection_device': str(context['device'])
                })
                
                # Queue for processing threads (ONLY NEW PART)
                success = self.queue_manager.put_frame('detection_to_processing', frame_data, timeout=0.5)
                if not success:
                    logger.debug(f"🔍 Worker {worker_id}: Frame dropped (processing queue full)")
                
                # Update statistics
                with self._stats_lock:
                    self.detection_stats['total_detections'] += 1
                    self.detection_stats['detection_times'].append(detection_time)
                    self.detection_stats['worker_loads'][worker_id] += 1
                    
                    # Keep only recent detection times (last 100)
                    if len(self.detection_stats['detection_times']) > 100:
                        self.detection_stats['detection_times'] = self.detection_stats['detection_times'][-100:]
                
                # Update context
                context['total_processed'] += 1
                context['last_activity'] = time.time()
                
                #logger.debug(f"🔍 Worker {worker_id}: Processed frame from camera {frame_data.camera_id} in {detection_time:.3f}s")
                
            except Exception as e:
                logger.error(f"❌ Detection worker {worker_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error

    def stop_detection_workers(self):
        """Stop all detection workers"""
        self.running = False
        
        # Shutdown thread pool
        self.detection_pool.shutdown(wait=True, timeout=5.0)
        
        logger.info("🛑 All detection workers stopped")

    def get_detection_stats(self) -> dict:
        """Get detection performance statistics"""
        with self._stats_lock:
            stats = self.detection_stats.copy()
            
        # Calculate average detection time
        if stats['detection_times']:
            stats['avg_detection_time'] = sum(stats['detection_times']) / len(stats['detection_times'])
            stats['max_detection_time'] = max(stats['detection_times'])
            stats['min_detection_time'] = min(stats['detection_times'])
        else:
            stats['avg_detection_time'] = 0
            stats['max_detection_time'] = 0
            stats['min_detection_time'] = 0
            
        # Add worker context info
        stats['worker_contexts'] = []
        for context in self.gpu_contexts:
            stats['worker_contexts'].append({
                'worker_id': context['worker_id'],
                'device': str(context['device']),
                'total_processed': context['total_processed'],
                'last_activity': context['last_activity']
            })
            
        return stats
