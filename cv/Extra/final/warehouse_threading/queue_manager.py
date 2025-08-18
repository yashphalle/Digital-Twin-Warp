#!/usr/bin/env python3
"""
Queue Management System
Handles all inter-thread communication safely
"""

import queue
import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Standardized frame data structure for pipeline"""
    camera_id: int
    frame: Any  # numpy array
    timestamp: float
    frame_number: int
    stage: str = "raw"  # raw, preprocessed, detected, processed, final
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class QueueManager:
    """Manages all queues in the threading pipeline"""
    
    def __init__(self, max_cameras: int = 11):
        self.max_cameras = max_cameras
        
        # Pipeline queues OPTIMIZED FOR GPU UTILIZATION
        self.queues = {
            # Camera → Detection (REDUCED to force GPU processing)
            'camera_to_detection': queue.Queue(maxsize=max_cameras * 2),  # Was 5, now 2

            # Detection → Processing (REDUCED for faster flow)
            'detection_to_processing': queue.Queue(maxsize=max_cameras * 2),  # Was 3, now 2

            # Processing → Database (REDUCED for faster flow)
            'processing_to_database': queue.Queue(maxsize=max_cameras * 1),  # Was 2, now 1

            # Processing → GUI (minimal buffer, real-time display)
            'processing_to_gui': queue.Queue(maxsize=6),  # Was 10, now 6
        }
        
        # Statistics tracking
        self.stats = {
            'frames_queued': 0,
            'frames_processed': 0,
            'queue_overflows': 0,
            'processing_times': {}
        }
        
        self._stats_lock = threading.Lock()
        
        logger.info("✅ Queue Manager initialized")
        logger.info(f"📊 Queue sizes: {[(name, q.maxsize) for name, q in self.queues.items()]}")

    def put_frame(self, queue_name: str, frame_data: FrameData, timeout: float = 1.0) -> bool:
        """Safely put frame data into specified queue"""
        try:
            self.queues[queue_name].put(frame_data, timeout=timeout)
            
            with self._stats_lock:
                self.stats['frames_queued'] += 1
                
            return True
            
        except queue.Full:
            with self._stats_lock:
                self.stats['queue_overflows'] += 1
                
            logger.warning(f"Queue {queue_name} full, dropping frame from camera {frame_data.camera_id}")
            return False
        except KeyError:
            logger.error(f"Unknown queue: {queue_name}")
            return False

    def get_frame(self, queue_name: str, timeout: float = 1.0) -> Optional[FrameData]:
        """Safely get frame data from specified queue"""
        try:
            frame_data = self.queues[queue_name].get(timeout=timeout)
            
            with self._stats_lock:
                self.stats['frames_processed'] += 1
                
            return frame_data
            
        except queue.Empty:
            return None
        except KeyError:
            logger.error(f"Unknown queue: {queue_name}")
            return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        stats = {}
        
        for name, q in self.queues.items():
            stats[name] = {
                'size': q.qsize(),
                'maxsize': q.maxsize,
                'utilization': q.qsize() / q.maxsize if q.maxsize > 0 else 0
            }
        
        with self._stats_lock:
            stats['global'] = self.stats.copy()
            
        return stats

    def clear_all_queues(self):
        """Clear all queues (for shutdown)"""
        for name, q in self.queues.items():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        logger.info("🧹 All queues cleared")
