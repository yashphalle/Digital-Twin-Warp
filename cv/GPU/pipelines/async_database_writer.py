# cv/GPU/pipelines/async_database_writer.py

import threading
import time
import logging
from queue import Queue, Empty, Full
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from pymongo import MongoClient, UpdateOne, InsertOne
from pymongo.errors import BulkWriteError
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.modules.coordinate_mapper import CoordinateMapper
from GPU.configs.config import Config

logger = logging.getLogger(__name__)

class AsyncDatabaseWriter:
    """
    Asynchronous database writer that runs in a separate thread
    Never blocks the main detection pipeline
    """
    
    def __init__(self, 
                 queue_size: int = 5000,
                 batch_write_interval: float = 2.5,
                 max_batch_size: int = 1000,
                 coordinate_mapper: Optional[CoordinateMapper] = None):
        """
        Initialize async database writer
        
        Args:
            queue_size: Maximum detections to queue
            batch_write_interval: Seconds between bulk writes
            max_batch_size: Maximum detections per bulk write
            coordinate_mapper: Optional coordinate mapper instance
        """
        # Queue for thread-safe detection passing
        self.detection_queue = Queue(maxsize=queue_size)
        self.queue_size = queue_size
        
        # Timing parameters
        self.batch_write_interval = batch_write_interval
        self.max_batch_size = max_batch_size
        
        # Coordinate mapper
        self.coordinate_mapper = coordinate_mapper
        if self.coordinate_mapper:
            logger.info("✅ Coordinate mapper provided for physical coordinate conversion")
        else:
            logger.info("⚠️ No coordinate mapper - physical coordinates will be NULL")
        
        # MongoDB connection
        self.mongo_client = None
        self.db = None
        self.collection = None
        self._connect_mongodb()
        
        # Writer thread
        self.writer_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_dropped': 0,
            'total_written': 0,
            'total_new': 0,
            'total_updated': 0,
            'total_errors': 0,
            'last_write_time': 0,
            'last_write_count': 0
        }
        
        logger.info(f"✅ AsyncDatabaseWriter initialized (queue_size={queue_size})")
    
    def _connect_mongodb(self):
        """Connect to MongoDB"""
        try:
            # Use config settings
            self.mongo_client = MongoClient(Config.MONGO_URI)
            self.db = self.mongo_client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
            
            # Create indexes for performance
            self.collection.create_index([("global_id", 1)])
            self.collection.create_index([("camera_id", 1), ("global_id", 1)])
            self.collection.create_index([("last_seen", -1)])
            logger.info("✅ MongoDB indexes created")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
    
    def start(self):
        """Start the writer thread"""
        if self.running:
            logger.warning("Writer thread already running")
            return
        
        self.running = True
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="AsyncDatabaseWriter",
            daemon=True
        )
        self.writer_thread.start()
        logger.info("✅ Database writer thread started")
    
    def stop(self, timeout: float = 5.0):
        """Stop the writer thread gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping database writer thread...")
        self.running = False
        
        # Wait for thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=timeout)
            
        # Final flush
        self._flush_remaining()
        
        # Close MongoDB connection
        if self.mongo_client:
            self.mongo_client.close()
            
        logger.info(f"✅ Database writer stopped. Stats: {self.get_stats()}")
    
    def enqueue_detection(self, detection: Dict, camera_id: int, frame_shape: Tuple[int, int] = None):
        """
        Add detection to queue for database writing
        
        Args:
            detection: Detection dict from GPU processor
            camera_id: Camera ID
            frame_shape: Frame dimensions for coordinate mapping
        """
        try:
            # Enrich detection for database
            enriched = self._enrich_detection(detection, camera_id, frame_shape)
            
            # Non-blocking put
            self.detection_queue.put_nowait(enriched)
            self.stats['total_queued'] += 1
            
        except Full:
            # Queue full - drop detection
            self.stats['total_dropped'] += 1
            if self.stats['total_dropped'] % 100 == 0:
                logger.warning(f"⚠️ Dropped {self.stats['total_dropped']} detections (queue full)")
        except Exception as e:
            logger.error(f"Error enqueueing detection: {e}")
    
    def _enrich_detection(self, detection: Dict, camera_id: int, frame_shape: Tuple[int, int] = None) -> Dict:
        """
        Enrich detection with all required fields for database
        
        Args:
            detection: Raw detection from GPU processor
            camera_id: Camera ID
            frame_shape: Frame dimensions (height, width)
        """
        # Extract bbox
        bbox = detection.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        # Calculate corners from bbox
        corners = [
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ]
        
        # Calculate center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center = [center_x, center_y]
        
        # Calculate area
        area = float((x2 - x1) * (y2 - y1))
        
        # Physical coordinates (if mapper available)
        physical_x_ft = None
        physical_y_ft = None
        physical_corners = None
        real_center = None
        coordinate_status = 'NOT_MAPPED'
        
        if self.coordinate_mapper and self.coordinate_mapper.is_calibrated:
            try:
                # Map center to physical coordinates
                phys_x, phys_y = self.coordinate_mapper.pixel_to_real(center_x, center_y)
                if phys_x is not None and phys_y is not None:
                    physical_x_ft = round(phys_x, 2)
                    physical_y_ft = round(phys_y, 2)
                    real_center = [physical_x_ft, physical_y_ft]
                    coordinate_status = 'SUCCESS'
                    
                    # Map all corners to physical
                    physical_corners = []
                    for corner in corners:
                        px, py = self.coordinate_mapper.pixel_to_real(corner[0], corner[1])
                        if px is not None and py is not None:
                            physical_corners.append([round(px, 2), round(py, 2)])
                        else:
                            physical_corners.append([None, None])
                            
            except Exception as e:
                logger.debug(f"Coordinate mapping failed: {e}")
                coordinate_status = 'FAILED'
        
        # Determine tracking status
        track_age = detection.get('track_age', 0)
        tracking_status = 'new' if track_age <= 1 else 'existing'
        
        # Build enriched document
        enriched = {
            # Core fields
            'camera_id': camera_id,
            'global_id': detection.get('track_id'),  # Direct from YOLOv8 tracking
            'warp_id': None,  # Placeholder
            
            # Timestamps (will be set during write)
            'timestamp': datetime.utcnow(),
            
            # Bounding box and geometry
            'bbox': bbox,
            'corners': corners,
            'physical_corners': physical_corners,
            'shape_type': 'quadrangle',
            'real_center': real_center,
            'confidence': float(detection.get('confidence', 0.0)),
            'area': area,
            'center': center,
            
            # Physical coordinates
            'physical_x_ft': physical_x_ft,
            'physical_y_ft': physical_y_ft,
            'coordinate_status': coordinate_status,
            
            # Tracking info
            'tracking_status': tracking_status,
            'track_age': track_age,
            'similarity_score': 1.0,  # Placeholder
            
            # Color placeholders (brown)
            'color_rgb': [139, 69, 19],
            'color_hsv': [25, 86, 55],
            'color_hex': '#8B4513',
            'color_name': 'brown',
            'color_confidence': 1.0,
            'extraction_method': 'placeholder'
        }
        
        return enriched
    
    def _writer_loop(self):
        """Main writer thread loop"""
        buffer = []
        last_write_time = time.time()
        
        logger.info("📝 Database writer thread started")
        
        while self.running or not self.detection_queue.empty():
            try:
                # Collect detections from queue
                deadline = time.time() + 0.1  # Check every 100ms
                
                while time.time() < deadline and len(buffer) < self.max_batch_size:
                    try:
                        detection = self.detection_queue.get_nowait()
                        buffer.append(detection)
                    except Empty:
                        break
                
                # Write if interval elapsed or buffer full
                elapsed = time.time() - last_write_time
                should_write = (
                    elapsed >= self.batch_write_interval or 
                    len(buffer) >= self.max_batch_size or
                    (not self.running and buffer)  # Final flush
                )
                
                if should_write and buffer:
                    self._bulk_write(buffer)
                    buffer.clear()
                    last_write_time = time.time()
                
                # Log queue status periodically
                if int(time.time()) % 10 == 0 and self.stats['last_write_time'] != int(time.time()):
                    queue_depth = self.detection_queue.qsize()
                    if queue_depth > 0:
                        logger.info(f"📊 Queue depth: {queue_depth}/{self.queue_size} "
                                  f"({queue_depth/self.queue_size*100:.1f}% full)")
                
            except Exception as e:
                logger.error(f"Error in writer loop: {e}")
                time.sleep(0.1)
        
        logger.info("📝 Database writer thread finished")
    
    def _bulk_write(self, detections: List[Dict]):
        """
        Perform bulk write to MongoDB
        
        Args:
            detections: List of enriched detection documents
        """
        if not self.mongo_client or not detections:
            return
        
        start_time = time.time()
        
        try:
            # Separate new and existing detections
            new_detections = []
            update_operations = []
            current_time = datetime.utcnow()
            
            for det in detections:
                if det['tracking_status'] == 'new':
                    # Prepare new document
                    det['first_seen'] = current_time
                    det['last_seen'] = current_time
                    det['times_seen'] = 1
                    new_detections.append(det)
                else:
                    # Prepare update operation
                    update_op = UpdateOne(
                        {
                            'global_id': det['global_id'],
                            'camera_id': det['camera_id']
                        },
                        {
                            '$set': {
                                'last_seen': current_time,
                                'bbox': det['bbox'],
                                'corners': det['corners'],
                                'physical_corners': det['physical_corners'],
                                'real_center': det['real_center'],
                                'confidence': det['confidence'],
                                'area': det['area'],
                                'center': det['center'],
                                'physical_x_ft': det['physical_x_ft'],
                                'physical_y_ft': det['physical_y_ft'],
                                'coordinate_status': det['coordinate_status'],
                                'similarity_score': det['similarity_score']
                            },
                            '$inc': {
                                'times_seen': 1
                            }
                        },
                        upsert=False  # Don't create if doesn't exist
                    )
                    update_operations.append(update_op)
            
            # Perform bulk operations
            total_written = 0
            
            # Insert new detections
            if new_detections:
                result = self.collection.insert_many(new_detections, ordered=False)
                self.stats['total_new'] += len(result.inserted_ids)
                total_written += len(result.inserted_ids)
            
            # Update existing detections
            if update_operations:
                result = self.collection.bulk_write(update_operations, ordered=False)
                self.stats['total_updated'] += result.modified_count
                total_written += result.modified_count
            
            # Update statistics
            self.stats['total_written'] += total_written
            self.stats['last_write_time'] = time.time()
            self.stats['last_write_count'] = len(detections)
            
            # Log performance
            write_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Bulk write complete: {len(new_detections)} new, "
                       f"{len(update_operations)} updates in {write_time:.1f}ms")
            
        except BulkWriteError as bwe:
            # Some operations succeeded, some failed
            write_errors = bwe.details.get('writeErrors', [])
            self.stats['total_errors'] += len(write_errors)
            logger.warning(f"⚠️ Partial bulk write: {len(write_errors)} errors")
            
        except Exception as e:
            self.stats['total_errors'] += len(detections)
            logger.error(f"❌ Bulk write failed: {e}")
    
    def _flush_remaining(self):
        """Flush any remaining detections in queue"""
        remaining = []
        
        try:
            while True:
                detection = self.detection_queue.get_nowait()
                remaining.append(detection)
                if len(remaining) >= self.max_batch_size:
                    self._bulk_write(remaining)
                    remaining.clear()
        except Empty:
            pass
        
        # Write final batch
        if remaining:
            self._bulk_write(remaining)
            logger.info(f"✅ Flushed {len(remaining)} remaining detections")
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'queue_depth': self.detection_queue.qsize(),
            'queue_utilization': f"{self.detection_queue.qsize()/self.queue_size*100:.1f}%",
            **self.stats
        }
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self.mongo_client is not None