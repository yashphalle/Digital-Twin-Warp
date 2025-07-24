#!/usr/bin/env python3
"""
Per-Camera Queue Management System
Solves frame ordering issues by using separate queues for each camera
"""

import queue
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from queue_manager import FrameData, QueueManager

logger = logging.getLogger(__name__)

class PerCameraQueueManager(QueueManager):
    """
    Per-Camera Queue Manager to solve frame ordering issues
    KEY OPTIMIZATION: Separate detection queues per camera to prevent
    one camera's frames from dominating the processing pipeline
    """
    
    def __init__(self, max_cameras: int = 11, active_cameras: List[int] = None):
        self.max_cameras = max_cameras
        self.active_cameras = active_cameras or list(range(1, max_cameras + 1))
        
        # PER-CAMERA detection queues to prevent frame ordering issues
        self.camera_detection_queues = {
            camera_id: queue.Queue(maxsize=5)  # Small queue per camera for real-time processing
            for camera_id in self.active_cameras
        }
        
        # PER-CAMERA processing queues (NEW - Phase 1)
        self.camera_processing_queues = {
            f'camera_{camera_id}_detection_to_processing': queue.Queue(maxsize=20)
            for camera_id in self.active_cameras
        }

        # SHARED queues for other pipeline stages (KEEP existing for compatibility)
        self.queues = {
            # Detection → Processing (KEEP for backward compatibility)
            'detection_to_processing': queue.Queue(maxsize=max_cameras * 20),

            # Processing → Database (SAME as original)
            'processing_to_database': queue.Queue(maxsize=max_cameras * 20),

            # Processing → GUI (SMALLER for real-time display)
            'processing_to_gui': queue.Queue(maxsize=5),
        }
        
        # Enhanced statistics tracking
        self.stats = {
            'frames_queued': 0,
            'frames_processed': 0,
            'queue_overflows': 0,
            'processing_times': {},
            'frames_replaced': 0,
            'per_camera_stats': {
                camera_id: {
                    'frames_queued': 0,
                    'frames_processed': 0,
                    'queue_overflows': 0,
                    'frames_replaced': 0
                } for camera_id in self.active_cameras
            }
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"✅ Per-Camera Queue Manager initialized for cameras: {self.active_cameras}")
        logger.info(f"📊 Created {len(self.camera_detection_queues)} separate detection queues")
    
    def put_camera_frame(self, camera_id: int, frame_data: FrameData, timeout: float = 0.1) -> bool:
        """
        Put frame into camera-specific detection queue with frame replacement
        """
        if camera_id not in self.camera_detection_queues:
            logger.error(f"No detection queue for Camera {camera_id}")
            return False
            
        try:
            start_time = time.time()
            camera_queue = self.camera_detection_queues[camera_id]

            # NEW FEATURE: Replace oldest frame with newest when queue is full
            if camera_queue.full():
                try:
                    # Remove oldest frame (FIFO - get removes from front)
                    old_frame = camera_queue.get_nowait()
                    logger.debug(f"[QUEUE] Camera {camera_id}: Replaced old frame with newer frame")

                    # Track frame replacements for monitoring
                    with self._stats_lock:
                        self.stats['frames_replaced'] += 1
                        self.stats['per_camera_stats'][camera_id]['frames_replaced'] += 1

                except queue.Empty:
                    pass  # Queue became empty between full() check and get_nowait()

            # Add new frame
            camera_queue.put(frame_data, timeout=timeout)
            put_time = time.time() - start_time

            # Update statistics
            with self._stats_lock:
                self.stats['frames_queued'] += 1
                self.stats['per_camera_stats'][camera_id]['frames_queued'] += 1
                
                # Track put times for optimization
                if 'put_times' not in self.stats:
                    self.stats['put_times'] = []
                self.stats['put_times'].append(put_time)
                
                # Keep only recent put times (last 100)
                if len(self.stats['put_times']) > 100:
                    self.stats['put_times'] = self.stats['put_times'][-100:]

            return True
            
        except queue.Full:
            with self._stats_lock:
                self.stats['queue_overflows'] += 1
                self.stats['per_camera_stats'][camera_id]['queue_overflows'] += 1
                
            logger.debug(f"Camera {camera_id} detection queue full, dropping frame")
            return False
        except Exception as e:
            logger.error(f"Error putting frame for Camera {camera_id}: {e}")
            return False
    
    def get_camera_frame_round_robin(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Get frame using round-robin selection across cameras
        Ensures fair processing of all cameras
        """
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
            
        start_camera_index = self._round_robin_index
        
        # Try each camera in round-robin order
        for _ in range(len(self.active_cameras)):
            camera_id = self.active_cameras[self._round_robin_index]
            self._round_robin_index = (self._round_robin_index + 1) % len(self.active_cameras)

            try:
                camera_queue = self.camera_detection_queues[camera_id]
                queue_size = camera_queue.qsize()
                logger.debug(f"[ROUND-ROBIN] Trying Camera {camera_id}, queue size: {queue_size}")

                # Use longer timeout per camera
                frame_data = camera_queue.get(timeout=0.5)  # Fixed timeout instead of divided

                # Update statistics
                with self._stats_lock:
                    self.stats['frames_processed'] += 1
                    self.stats['per_camera_stats'][camera_id]['frames_processed'] += 1

                logger.info(f"[ROUND-ROBIN] ✅ Got frame from Camera {camera_id}")
                return frame_data

            except queue.Empty:
                logger.debug(f"[ROUND-ROBIN] Camera {camera_id} queue empty")
                continue  # Try next camera
                
        # No frames available from any camera
        return None
    
    def get_camera_frame_any_available(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Get frame from any camera that has frames available
        Faster than round-robin but may favor faster cameras
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            for camera_id in self.active_cameras:
                try:
                    camera_queue = self.camera_detection_queues[camera_id]
                    frame_data = camera_queue.get_nowait()
                    
                    # Update statistics
                    with self._stats_lock:
                        self.stats['frames_processed'] += 1
                        self.stats['per_camera_stats'][camera_id]['frames_processed'] += 1
                    
                    logger.debug(f"[ANY-AVAILABLE] Got frame from Camera {camera_id}")
                    return frame_data
                    
                except queue.Empty:
                    continue  # Try next camera
                    
            # Small delay before trying again
            time.sleep(0.001)
            
        return None
    
    def put_frame(self, queue_name: str, frame_data: FrameData, timeout: float = 1.0) -> bool:
        """
        Enhanced frame queuing - routes camera frames to per-camera queues
        """
        # Route camera frames to per-camera queues
        if queue_name == 'camera_to_detection':
            logger.debug(f"[PUT] Routing Camera {frame_data.camera_id} frame to per-camera queue")
            return self.put_camera_frame(frame_data.camera_id, frame_data, timeout)

        # Route per-camera processing frames (NEW - Phase 1)
        if queue_name in self.camera_processing_queues:
            try:
                self.camera_processing_queues[queue_name].put(frame_data, timeout=timeout)
                return True
            except queue.Full:
                logger.warning(f"[QUEUE] {queue_name} full, dropping frame")
                return False

        # Use parent implementation for other queues
        return super().put_frame(queue_name, frame_data, timeout)
    
    def get_frame(self, queue_name: str, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Enhanced frame retrieval - uses round-robin for detection frames
        """
        # Use round-robin for detection frames
        if queue_name == 'camera_to_detection':
            return self.get_camera_frame_round_robin(timeout)

        # Get from per-camera processing queues (NEW - Phase 1)
        if queue_name in self.camera_processing_queues:
            try:
                return self.camera_processing_queues[queue_name].get(timeout=timeout)
            except queue.Empty:
                return None

        # Use parent implementation for other queues
        return super().get_frame(queue_name, timeout)
    
    def get_per_camera_stats(self) -> Dict[str, Any]:
        """Get detailed per-camera statistics"""
        with self._stats_lock:
            stats = {
                'global': self.stats.copy(),
                'per_camera': self.stats['per_camera_stats'].copy(),
                'queue_sizes': {
                    camera_id: self.camera_detection_queues[camera_id].qsize()
                    for camera_id in self.active_cameras
                },
                'queue_utilization': {
                    camera_id: self.camera_detection_queues[camera_id].qsize() / 5.0  # maxsize=5
                    for camera_id in self.active_cameras
                }
            }
            
        return stats
    
    def log_camera_stats(self):
        """Log detailed per-camera statistics"""
        stats = self.get_per_camera_stats()
        
        logger.info("=" * 60)
        logger.info("PER-CAMERA QUEUE STATISTICS")
        logger.info("=" * 60)
        
        for camera_id in self.active_cameras:
            camera_stats = stats['per_camera'][camera_id]
            queue_size = stats['queue_sizes'][camera_id]
            utilization = stats['queue_utilization'][camera_id]
            
            logger.info(f"Camera {camera_id:2d}: "
                       f"Queued={camera_stats['frames_queued']:4d}, "
                       f"Processed={camera_stats['frames_processed']:4d}, "
                       f"Replaced={camera_stats['frames_replaced']:3d}, "
                       f"QueueSize={queue_size}/5 ({utilization:.1%})")
        
        logger.info("=" * 60)
