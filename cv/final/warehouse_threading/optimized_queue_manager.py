#!/usr/bin/env python3
"""
Optimized Queue Manager
Enhanced queue management with better sizing for GPU utilization
Uses SAME FrameData structure - only optimizes queue flow
"""

import queue
import time
import threading
import logging
from typing import Dict, Any, Optional

# Import existing tested structure (SAME as original)
from .queue_manager import QueueManager, FrameData

logger = logging.getLogger(__name__)

class OptimizedQueueManager(QueueManager):
    """
    Optimized version of QueueManager
    Uses SAME FrameData structure and methods
    ONLY CHANGE: Optimized queue sizes for better GPU feeding
    """
    
    def __init__(self, max_cameras: int = 11):
        self.max_cameras = max_cameras
        
        # OPTIMIZED queue sizes for better GPU utilization
        self.queues = {
            # Camera → Detection (INCREASED for better GPU feeding)
            'camera_to_detection': queue.Queue(maxsize=max_cameras * 3),  # Was 2, now 3
            
            # Detection → Processing (INCREASED for smoother flow)
            'detection_to_processing': queue.Queue(maxsize=max_cameras * 3),  # Was 2, now 3
            
            # Processing → Database (SAME as original)
            'processing_to_database': queue.Queue(maxsize=max_cameras * 1),
            
            # Processing → GUI (SAME as original)
            'processing_to_gui': queue.Queue(maxsize=6),
        }
        
        # Enhanced statistics tracking
        self.stats = {
            'frames_queued': 0,
            'frames_processed': 0,
            'queue_overflows': 0,
            'processing_times': {},
            'optimization_metrics': {
                'gpu_feed_efficiency': 0.0,
                'queue_utilization_avg': 0.0,
                'bottleneck_detection': 'none'
            }
        }
        
        self._stats_lock = threading.Lock()
        
        logger.info("✅ Optimized Queue Manager initialized")
        logger.info(f"🎯 OPTIMIZED Queue sizes: {[(name, q.maxsize) for name, q in self.queues.items()]}")
        logger.info("📈 Enhanced monitoring for GPU utilization tracking")
    
    def put_frame(self, queue_name: str, frame_data: FrameData, timeout: float = 1.0) -> bool:
        """
        Enhanced frame queuing with optimization metrics
        SAME functionality as original, ENHANCED monitoring
        """
        try:
            start_time = time.time()
            self.queues[queue_name].put(frame_data, timeout=timeout)
            put_time = time.time() - start_time
            
            with self._stats_lock:
                self.stats['frames_queued'] += 1
                
                # Track queue performance for optimization
                if queue_name not in self.stats['processing_times']:
                    self.stats['processing_times'][queue_name] = []
                self.stats['processing_times'][queue_name].append(put_time)
                
                # Keep only recent times (last 100)
                if len(self.stats['processing_times'][queue_name]) > 100:
                    self.stats['processing_times'][queue_name] = self.stats['processing_times'][queue_name][-100:]
            
            return True
            
        except queue.Full:
            with self._stats_lock:
                self.stats['queue_overflows'] += 1
                
                # Detect bottlenecks for optimization
                if queue_name == 'camera_to_detection':
                    self.stats['optimization_metrics']['bottleneck_detection'] = 'gpu_detection'
                elif queue_name == 'detection_to_processing':
                    self.stats['optimization_metrics']['bottleneck_detection'] = 'processing'
                
            logger.warning(f"Queue {queue_name} full, dropping frame from camera {frame_data.camera_id}")
            return False
        except KeyError:
            logger.error(f"Unknown queue: {queue_name}")
            return False

    def get_frame(self, queue_name: str, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Enhanced frame retrieval with optimization metrics
        SAME functionality as original, ENHANCED monitoring
        """
        try:
            start_time = time.time()
            frame_data = self.queues[queue_name].get(timeout=timeout)
            get_time = time.time() - start_time
            
            with self._stats_lock:
                self.stats['frames_processed'] += 1
                
                # Track retrieval performance
                retrieval_key = f"{queue_name}_retrieval"
                if retrieval_key not in self.stats['processing_times']:
                    self.stats['processing_times'][retrieval_key] = []
                self.stats['processing_times'][retrieval_key].append(get_time)
                
                # Keep only recent times
                if len(self.stats['processing_times'][retrieval_key]) > 100:
                    self.stats['processing_times'][retrieval_key] = self.stats['processing_times'][retrieval_key][-100:]
            
            return frame_data
            
        except queue.Empty:
            return None
        except KeyError:
            logger.error(f"Unknown queue: {queue_name}")
            return None

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get enhanced optimization statistics"""
        stats = self.get_queue_stats()  # Get base stats from parent
        
        with self._stats_lock:
            # Calculate GPU feeding efficiency
            camera_queue_util = stats.get('camera_to_detection', {}).get('utilization', 0)
            detection_queue_util = stats.get('detection_to_processing', {}).get('utilization', 0)
            
            # GPU feed efficiency: how well we're keeping GPU busy
            gpu_feed_efficiency = (camera_queue_util + detection_queue_util) / 2
            
            # Average queue utilization
            utilizations = [q.get('utilization', 0) for q in stats.values() if isinstance(q, dict) and 'utilization' in q]
            avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
            
            # Update optimization metrics
            self.stats['optimization_metrics']['gpu_feed_efficiency'] = gpu_feed_efficiency
            self.stats['optimization_metrics']['queue_utilization_avg'] = avg_utilization
            
            # Enhanced stats
            stats['optimization'] = {
                'gpu_feed_efficiency': f"{gpu_feed_efficiency:.2f}",
                'avg_queue_utilization': f"{avg_utilization:.2f}",
                'bottleneck_detected': self.stats['optimization_metrics']['bottleneck_detection'],
                'total_overflows': self.stats['queue_overflows'],
                'optimization_type': 'enhanced_gpu_feeding'
            }
            
            # Processing time averages
            stats['performance'] = {}
            for queue_name, times in self.stats['processing_times'].items():
                if times:
                    avg_time = sum(times) / len(times)
                    stats['performance'][queue_name] = f"{avg_time:.4f}s"
        
        return stats

    def log_optimization_status(self):
        """Log current optimization status"""
        stats = self.get_optimization_stats()
        opt_stats = stats.get('optimization', {})
        
        logger.info(f"🎯 QUEUE OPTIMIZATION STATUS:")
        logger.info(f"   GPU Feed Efficiency: {opt_stats.get('gpu_feed_efficiency', 'N/A')}")
        logger.info(f"   Avg Queue Utilization: {opt_stats.get('avg_queue_utilization', 'N/A')}")
        logger.info(f"   Bottleneck Detected: {opt_stats.get('bottleneck_detected', 'none')}")
        logger.info(f"   Total Overflows: {opt_stats.get('total_overflows', 0)}")
