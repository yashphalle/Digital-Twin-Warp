#!/usr/bin/env python3
"""
Per-Camera Detection Thread Pool
Uses per-camera queues to ensure fair frame processing across all cameras
"""

import logging
import threading
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from .per_camera_queue_manager import PerCameraQueueManager

logger = logging.getLogger(__name__)

class PerCameraDetectionThreadPool:
    """
    Manages pool of GPU detection threads with per-camera queue support
    Ensures fair processing across all cameras using round-robin selection
    """
    
    def __init__(self, num_workers: int = 3, queue_manager: PerCameraQueueManager = None):
        self.num_workers = num_workers
        self.queue_manager = queue_manager
        self.running = False
        
        # Thread pool for detection workers
        self.detection_pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="PerCameraDetection")
        
        # GPU context management
        self.gpu_contexts = []
        self._initialize_gpu_contexts()
        
        # Performance tracking with per-camera stats
        self.detection_stats = {
            'total_detections': 0,
            'detection_times': [],
            'worker_loads': {i: 0 for i in range(num_workers)},
            'per_camera_detections': {
                camera_id: 0 for camera_id in queue_manager.active_cameras
            } if queue_manager else {}
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"✅ Per-Camera Detection Thread Pool initialized with {num_workers} workers")
        logger.info(f"📊 Tracking detections for cameras: {list(self.detection_stats['per_camera_detections'].keys())}")

    def _initialize_gpu_contexts(self):
        """Initialize GPU contexts for each worker"""
        try:
            # Import detection modules (same as working system)
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from modules.detector import CPUSimplePalletDetector
            
            # Check available GPUs
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info(f"🔥 Found {num_gpus} GPU(s) available")
                
                for worker_id in range(self.num_workers):
                    # Distribute workers across available GPUs
                    gpu_id = worker_id % num_gpus
                    device = torch.device(f'cuda:{gpu_id}')
                    
                    # Initialize detector for this worker (same as working system)
                    detector = CPUSimplePalletDetector()
                    
                    context = {
                        'worker_id': worker_id,
                        'device': device,
                        'detector': detector,
                        'gpu_id': gpu_id
                    }
                    
                    self.gpu_contexts.append(context)
                    logger.info(f"🚀 Worker {worker_id} initialized on GPU {gpu_id} ({device})")
            else:
                logger.warning("⚠️ No CUDA GPUs available, using CPU")
                # Fallback to CPU (same as working system)
                for worker_id in range(self.num_workers):
                    detector = CPUSimplePalletDetector()
                    
                    context = {
                        'worker_id': worker_id,
                        'device': torch.device('cpu'),
                        'detector': detector,
                        'gpu_id': -1
                    }
                    
                    self.gpu_contexts.append(context)
                    logger.info(f"🚀 Worker {worker_id} initialized on CPU")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize GPU contexts: {e}")
            raise

    def start_detection_workers(self):
        """Start all detection worker threads"""
        self.running = True
        
        # Submit worker tasks to thread pool
        for worker_id in range(self.num_workers):
            future = self.detection_pool.submit(self._per_camera_detection_worker, worker_id)
            logger.info(f"🚀 Started per-camera detection worker {worker_id}")

    def _per_camera_detection_worker(self, worker_id: int):
        """
        Worker function for detection threads with per-camera queue support
        Uses round-robin frame selection to ensure fair camera processing
        """
        context = self.gpu_contexts[worker_id]
        detector = context['detector']
        
        logger.info(f"✅ Per-camera detection worker {worker_id} started on {context['device']}")
        
        processed_cameras = set()  # Track which cameras this worker has processed
        
        while self.running:
            try:
                # Get frame using round-robin selection across cameras
                frame_data = self.queue_manager.get_camera_frame_round_robin(timeout=1.0)
                if frame_data is None:
                    logger.debug(f"🔍 Worker {worker_id}: No frames available from any camera")
                    continue
                
                camera_id = frame_data.camera_id
                processed_cameras.add(camera_id)
                
                # Record start time
                start_time = time.time()
                
                # Perform GPU detection (SAME method as main.py)
                detections = detector.detect_pallets(frame_data.frame)
                
                # Record detection time
                detection_time = time.time() - start_time
                
                # Update frame data with detection results
                frame_data.stage = "detected"
                frame_data.metadata.update({
                    'raw_detections': detections,
                    'detection_worker': worker_id,
                    'detection_time': detection_time,
                    'detection_device': str(context['device']),
                    'gpu_id': context['gpu_id'],
                    'per_camera_processing': True
                })
                
                # Queue for processing threads
                success = self.queue_manager.put_frame('detection_to_processing', frame_data, timeout=0.5)
                if not success:
                    logger.debug(f"🔍 Worker {worker_id}: Frame from Camera {camera_id} dropped (processing queue full)")
                
                # Update statistics
                with self._stats_lock:
                    self.detection_stats['total_detections'] += 1
                    self.detection_stats['detection_times'].append(detection_time)
                    self.detection_stats['worker_loads'][worker_id] += 1
                    self.detection_stats['per_camera_detections'][camera_id] += 1
                    
                    # Keep only recent detection times (last 100)
                    if len(self.detection_stats['detection_times']) > 100:
                        self.detection_stats['detection_times'] = self.detection_stats['detection_times'][-100:]
                
                logger.debug(f"🔍 Worker {worker_id}: Processed frame from Camera {camera_id} "
                           f"({len(detections)} detections, {detection_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"❌ Per-camera detection worker {worker_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        logger.info(f"🛑 Per-camera detection worker {worker_id} stopped. "
                   f"Processed cameras: {sorted(processed_cameras)}")

    def stop(self):
        """Stop all detection workers"""
        logger.info("🛑 Stopping per-camera detection thread pool...")
        self.running = False
        
        # Shutdown thread pool
        self.detection_pool.shutdown(wait=True)
        logger.info("✅ Per-camera detection thread pool stopped")

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        with self._stats_lock:
            stats = self.detection_stats.copy()
            
            # Calculate additional metrics
            if stats['detection_times']:
                stats['avg_detection_time'] = sum(stats['detection_times']) / len(stats['detection_times'])
                stats['max_detection_time'] = max(stats['detection_times'])
                stats['min_detection_time'] = min(stats['detection_times'])
            else:
                stats['avg_detection_time'] = 0
                stats['max_detection_time'] = 0
                stats['min_detection_time'] = 0
            
            # Calculate per-camera processing balance
            camera_detections = list(stats['per_camera_detections'].values())
            if camera_detections:
                stats['camera_balance'] = {
                    'max_detections': max(camera_detections),
                    'min_detections': min(camera_detections),
                    'avg_detections': sum(camera_detections) / len(camera_detections),
                    'balance_ratio': min(camera_detections) / max(camera_detections) if max(camera_detections) > 0 else 1.0
                }
            
        return stats

    def log_detection_stats(self):
        """Log detailed detection statistics"""
        stats = self.get_detection_stats()
        
        logger.info("=" * 70)
        logger.info("PER-CAMERA DETECTION STATISTICS")
        logger.info("=" * 70)
        
        # Overall stats
        logger.info(f"Total Detections: {stats['total_detections']}")
        logger.info(f"Avg Detection Time: {stats['avg_detection_time']:.3f}s")
        logger.info(f"Detection FPS: {1/stats['avg_detection_time']:.2f}" if stats['avg_detection_time'] > 0 else "N/A")
        
        # Worker load distribution
        logger.info("\nWorker Load Distribution:")
        for worker_id, load in stats['worker_loads'].items():
            percentage = (load / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
            logger.info(f"  Worker {worker_id}: {load:4d} detections ({percentage:.1f}%)")
        
        # Per-camera detection balance
        logger.info("\nPer-Camera Detection Balance:")
        for camera_id, detections in stats['per_camera_detections'].items():
            percentage = (detections / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
            logger.info(f"  Camera {camera_id:2d}: {detections:4d} detections ({percentage:.1f}%)")
        
        # Balance metrics
        if 'camera_balance' in stats:
            balance = stats['camera_balance']
            logger.info(f"\nCamera Balance Metrics:")
            logger.info(f"  Balance Ratio: {balance['balance_ratio']:.3f} (1.0 = perfect balance)")
            logger.info(f"  Max/Min Detections: {balance['max_detections']}/{balance['min_detections']}")
        
        logger.info("=" * 70)
