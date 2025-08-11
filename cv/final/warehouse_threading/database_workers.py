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
        
        # Simple statistics
        self.stats = {
            'total_queued': 0,
            'total_saved': 0,
            'total_dropped': 0
        }
        
        # Initialize per-camera database queues (300 items each for high detection count)
        for camera_id in active_cameras:
            self.database_queues[camera_id] = queue.Queue(maxsize=300)
        
        logger.info(f"✅ Async Database: {len(active_cameras)} cameras, 300 queue size each")

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
        """Queue detection for database save with oldest replacement when full"""

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
            # Try normal put first
            self.database_queues[camera_id].put_nowait(database_task)
            self.stats['total_queued'] += 1
            return True

        except queue.Full:
            # Queue is full (300 items) - REPLACE OLDEST
            try:
                # Remove oldest detection (FIFO)
                old_task = self.database_queues[camera_id].get_nowait()
                old_detection_id = old_task['detection'].get('global_id', 'unknown')

                # Add new detection
                self.database_queues[camera_id].put_nowait(database_task)

                # Update statistics
                self.stats['total_dropped'] += 1
                self.stats['total_queued'] += 1

                # Log replacement
                new_detection_id = detection.get('global_id', 'unknown')
                logger.warning(f"[DATABASE] Camera {camera_id}: Queue full, replaced old detection {old_detection_id} with new {new_detection_id}")

                return True

            except queue.Empty:
                # Race condition: try again
                try:
                    self.database_queues[camera_id].put_nowait(database_task)
                    self.stats['total_queued'] += 1
                    return True
                except queue.Full:
                    logger.error(f"[DATABASE] Camera {camera_id}: Failed to replace, dropping detection {detection.get('global_id')}")
                    return False

    def _database_worker(self, camera_id: int):
        """Database worker for specific camera"""
        logger.info(f"[DATABASE] Camera {camera_id} database worker started")
        
        # Create database handler
        db_handler = self._create_database_handler(camera_id)
        self.database_handlers[camera_id] = db_handler
        
        if not db_handler or not db_handler.is_connected():
            logger.error(f"[DATABASE] Camera {camera_id}: No database connection, worker will skip saves")
        else:
            logger.info(f"[DATABASE] Camera {camera_id}: Database connection confirmed, ready to save")
        
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
                    self.stats['total_saved'] += 1
                    
                    logger.info(f"[DATABASE] Camera {camera_id}: Saved detection {task['detection'].get('global_id', 'unknown')} in {save_time*1000:.1f}ms")
                else:
                    logger.warning(f"[DATABASE] Camera {camera_id}: Skipped save (no database connection)")
                    
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"[DATABASE] Camera {camera_id} database error: {e}")
                import traceback
                logger.error(f"[DATABASE] Camera {camera_id} traceback: {traceback.format_exc()}")
                time.sleep(0.1)  # Brief pause on error

    def _create_database_handler(self, camera_id: int):
        """Create database handler for specific camera"""
        try:
            logger.info(f"[DATABASE] Camera {camera_id}: Importing WarehouseDatabaseHandler...")

            # Fix import path for worker thread
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

            from warehouse_database_handler import WarehouseDatabaseHandler
            logger.info(f"[DATABASE] Camera {camera_id}: Creating database handler...")
            handler = WarehouseDatabaseHandler()
            logger.info(f"[DATABASE] Camera {camera_id}: Database handler created successfully")
            return handler
        except Exception as e:
            logger.error(f"[DATABASE] Camera {camera_id}: Failed to create database handler: {e}")
            import traceback
            logger.error(f"[DATABASE] Camera {camera_id}: Traceback: {traceback.format_exc()}")
            return None

    def get_queue_status(self) -> Dict:
        """Get simple queue status for monitoring"""
        status = {}
        for camera_id in self.active_cameras:
            queue_size = self.database_queues[camera_id].qsize()
            status[camera_id] = {
                'queue_size': queue_size,
                'queue_utilization_percent': (queue_size / 300) * 100
            }
        return status
