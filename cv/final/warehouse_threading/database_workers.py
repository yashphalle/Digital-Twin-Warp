#!/usr/bin/env python3
"""
Database Workers Module
Async database operations for parallel pipeline system
"""

import logging
import time
import threading
import queue
from typing import Dict, List

logger = logging.getLogger(__name__)

class DatabaseWorkerPool:
    """
    Pool of database workers for async database operations
    Each worker handles database saves for specific cameras
    """
    
    def __init__(self, active_cameras: List[int]):
        self.active_cameras = active_cameras
        self.running = False
        
        # Per-camera database queues and workers
        self.database_queues = {}
        self.database_workers = {}
        self.database_handlers = {}
        
        # Performance statistics
        self.stats = {
            'saves_per_camera': {camera_id: 0 for camera_id in active_cameras},
            'errors_per_camera': {camera_id: 0 for camera_id in active_cameras},
            'avg_save_time_per_camera': {camera_id: [] for camera_id in active_cameras}
        }
        
        # Initialize per-camera database queues
        for camera_id in active_cameras:
            self.database_queues[camera_id] = queue.Queue(maxsize=20)
        
        logger.info(f"✅ Database Worker Pool initialized for {len(active_cameras)} cameras")

    def start_workers(self):
        """Start all database workers"""
        if self.running:
            logger.warning("Database workers already running")
            return
            
        self.running = True
        
        for camera_id in self.active_cameras:
            worker = threading.Thread(
                target=self._database_worker,
                args=(camera_id,),
                name=f"DatabaseWorker-Camera-{camera_id}",
                daemon=True
            )
            self.database_workers[camera_id] = worker
            worker.start()
            logger.info(f"🗄️ Started database worker for Camera {camera_id}")

    def stop_workers(self):
        """Stop all database workers"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping database workers...")
        self.running = False
        
        # Workers will stop automatically (daemon threads)
        # Clean up database handlers
        for camera_id, handler in self.database_handlers.items():
            if handler:
                try:
                    handler.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up database handler for Camera {camera_id}: {e}")
        
        logger.info("✅ Database workers stopped")

    def queue_database_save(self, camera_id: int, detection: Dict, frame_number: int = None) -> bool:
        """
        Queue a database save task for specific camera
        Returns True if queued successfully, False if queue full
        """
        if camera_id not in self.database_queues:
            logger.error(f"No database queue for Camera {camera_id}")
            return False
            
        database_task = {
            'camera_id': camera_id,
            'detection': detection,
            'timestamp': time.time(),
            'frame_number': frame_number
        }
        
        try:
            self.database_queues[camera_id].put_nowait(database_task)
            return True
        except queue.Full:
            logger.warning(f"Database queue full for Camera {camera_id}, dropping detection")
            return False

    def _database_worker(self, camera_id: int):
        """Database worker for specific camera"""
        logger.info(f"[DATABASE] Camera {camera_id} database worker started")
        
        # Create database handler
        db_handler = self._create_database_handler(camera_id)
        self.database_handlers[camera_id] = db_handler
        
        if not db_handler or not db_handler.is_connected():
            logger.warning(f"[DATABASE] Camera {camera_id}: No database connection, worker will skip saves")
        
        while self.running:
            try:
                # Get database task from camera-specific queue
                task = self.database_queues[camera_id].get(timeout=1.0)
                
                if db_handler and db_handler.is_connected():
                    # Measure save time
                    save_start = time.time()
                    
                    # Save to database
                    db_handler.save_detection_to_db(task['camera_id'], task['detection'])
                    
                    save_time = time.time() - save_start
                    
                    # Update statistics
                    self.stats['saves_per_camera'][camera_id] += 1
                    self.stats['avg_save_time_per_camera'][camera_id].append(save_time)
                    
                    # Keep only recent save times (last 100)
                    if len(self.stats['avg_save_time_per_camera'][camera_id]) > 100:
                        self.stats['avg_save_time_per_camera'][camera_id] = \
                            self.stats['avg_save_time_per_camera'][camera_id][-100:]
                    
                    logger.debug(f"[DATABASE] Camera {camera_id}: Saved detection in {save_time*1000:.1f}ms")
                else:
                    logger.debug(f"[DATABASE] Camera {camera_id}: Skipped save (no database connection)")
                    
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"[DATABASE] Camera {camera_id} database error: {e}")
                self.stats['errors_per_camera'][camera_id] += 1
                time.sleep(0.1)  # Brief pause on error

    def _create_database_handler(self, camera_id: int):
        """Create database handler for specific camera"""
        try:
            from ..warehouse_database_handler import WarehouseDatabaseHandler
            return WarehouseDatabaseHandler()
        except Exception as e:
            logger.error(f"Failed to create database handler for Camera {camera_id}: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Get database worker statistics"""
        stats = {
            'total_saves': sum(self.stats['saves_per_camera'].values()),
            'total_errors': sum(self.stats['errors_per_camera'].values()),
            'per_camera': {}
        }
        
        for camera_id in self.active_cameras:
            saves = self.stats['saves_per_camera'][camera_id]
            errors = self.stats['errors_per_camera'][camera_id]
            save_times = self.stats['avg_save_time_per_camera'][camera_id]
            avg_save_time = sum(save_times) / len(save_times) if save_times else 0
            
            stats['per_camera'][camera_id] = {
                'saves': saves,
                'errors': errors,
                'avg_save_time_ms': avg_save_time * 1000,
                'success_rate': saves / (saves + errors) if (saves + errors) > 0 else 0
            }
        
        return stats

    def log_statistics(self):
        """Log database worker statistics"""
        stats = self.get_statistics()
        
        logger.info("[DATABASE] DATABASE WORKER STATISTICS:")
        logger.info(f"   Total Saves: {stats['total_saves']}")
        logger.info(f"   Total Errors: {stats['total_errors']}")
        
        for camera_id in self.active_cameras:
            camera_stats = stats['per_camera'][camera_id]
            logger.info(f"   Camera {camera_id:2d}: {camera_stats['saves']:4d} saves | "
                       f"{camera_stats['errors']:2d} errors | "
                       f"{camera_stats['avg_save_time_ms']:5.1f}ms avg | "
                       f"{camera_stats['success_rate']*100:5.1f}% success")
