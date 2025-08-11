#!/usr/bin/env python3
"""
Processing Thread Pool
11 dedicated processing workers - one per camera
Replaces single processing consumer thread for maximum parallelization
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class ProcessingThreadPool:
    """
    Pool of 11 processing workers - one dedicated worker per camera
    Each worker handles filtering, coordinates, tracking for its assigned camera
    """
    
    def __init__(self, active_cameras: List[int], queue_manager, database_workers):
        self.active_cameras = active_cameras
        self.queue_manager = queue_manager
        self.database_workers = database_workers
        self.running = False
        
        # One processing worker per camera
        self.num_workers = len(active_cameras)
        
        # Thread pool for processing workers
        self.processing_pool = ThreadPoolExecutor(
            max_workers=self.num_workers, 
            thread_name_prefix="Processing"
        )
        
        # Per-camera frame processors (maintains camera-specific state)
        self.frame_processors = {}
        self.processor_locks = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'processing_times': [],
            'worker_loads': {camera_id: 0 for camera_id in active_cameras}
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"✅ Processing Thread Pool initialized with {self.num_workers} workers (1 per camera)")
        logger.info(f"📹 Camera assignments: {active_cameras}")
    
    def start_processing_workers(self):
        """Start all processing worker threads - one per camera"""
        if self.running:
            logger.warning("Processing workers already running")
            return
            
        self.running = True
        
        # Start one dedicated worker per camera
        for camera_id in self.active_cameras:
            future = self.processing_pool.submit(self._processing_worker, camera_id)
            logger.info(f"🔧 Started processing worker for Camera {camera_id}")
    
    def stop_processing_workers(self):
        """Stop all processing workers"""
        self.running = False
        
        # Shutdown thread pool
        self.processing_pool.shutdown(wait=True, timeout=10.0)
        
        logger.info("🛑 All processing workers stopped")
    
    def _processing_worker(self, camera_id: int):
        """
        Dedicated processing worker for specific camera
        Handles filtering, coordinates, tracking for assigned camera only
        """
        logger.info(f"✅ Processing worker started for Camera {camera_id}")
        
        # Get or create processor for this camera
        processor = self._get_processor_for_camera(camera_id)
        
        processed_count = 0
        
        while self.running:
            try:
                # Get frame from detection_to_processing queue
                frame_data = self.queue_manager.get_frame('detection_to_processing', timeout=1.0)
                
                if frame_data is None:
                    continue
                
                # Only process frames from assigned camera
                if frame_data.camera_id != camera_id:
                    # Put frame back for correct worker
                    self.queue_manager.put_frame('detection_to_processing', frame_data, timeout=0.1)
                    continue
                
                # Record start time
                start_time = time.time()
                
                # Set raw detections from detection worker
                processor.raw_detections = frame_data.metadata.get('raw_detections', [])
                
                # Process frame (filtering, coordinates, tracking)
                success = processor.process_frame_yolo_tracking(frame_data.frame)
                
                if success:
                    # Add all final detections to database queue
                    for detection in processor.final_tracked_detections:
                        self.database_workers.queue_database_save(
                            camera_id, detection, frame_data.frame_number
                        )
                    
                    # Update performance stats
                    processing_time = time.time() - start_time
                    with self._stats_lock:
                        self.processing_stats['total_processed'] += 1
                        self.processing_stats['processing_times'].append(processing_time)
                        self.processing_stats['worker_loads'][camera_id] += 1
                        
                        # Keep only recent times
                        if len(self.processing_stats['processing_times']) > 100:
                            self.processing_stats['processing_times'] = self.processing_stats['processing_times'][-100:]
                    
                    processed_count += 1
                    
                    # Log progress every 50 frames
                    if processed_count % 50 == 0:
                        avg_time = sum(self.processing_stats['processing_times'][-10:]) / min(10, len(self.processing_stats['processing_times']))
                        logger.info(f"🔧 Processing Worker Camera {camera_id}: {processed_count} frames, avg {avg_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"❌ Processing worker Camera {camera_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        logger.info(f"🛑 Processing worker stopped for Camera {camera_id} (processed {processed_count} frames)")
    
    def _get_processor_for_camera(self, camera_id: int):
        """Get or create frame processor for specific camera"""
        if camera_id not in self.frame_processors:
            # Import here to avoid circular imports
            from modules.yolo_frame_processor import YOLOv8FrameProcessor
            from modules.yolo_tracking_database import YOLOv8TrackingDatabase
            from modules.filtering import DetectionFiltering
            from modules.coordinate_mapper import CoordinateMapper
            from modules.color_extractor import ColorExtractor
            from warehouse_database_handler import WarehouseDatabaseHandler
            
            # Create frame processor for this camera
            frame_processor = YOLOv8FrameProcessor(camera_id)
            
            # Create YOLOv8 tracking database
            yolo_tracking_db = YOLOv8TrackingDatabase(f"yolo_camera_{camera_id}_tracking.pkl", camera_id)
            
            # Create supporting components
            filtering = DetectionFiltering()
            coord_converter = CoordinateMapper(camera_id)
            color_extractor = ColorExtractor()
            db_handler = WarehouseDatabaseHandler()
            
            # Inject components into frame processor
            frame_processor.inject_components(
                fisheye_corrector=None,  # Already done in camera thread
                pallet_detector=None,    # Already done in detection thread
                filtering=filtering,
                coordinate_mapper=coord_converter,
                coordinate_mapper_initialized=True,
                global_db=yolo_tracking_db,
                yolo_tracking_db=yolo_tracking_db,
                color_extractor=color_extractor,
                db_handler=db_handler,
                display_manager=None
            )
            
            self.frame_processors[camera_id] = frame_processor
            self.processor_locks[camera_id] = threading.Lock()
            
            logger.info(f"🔧 Created dedicated processor for Camera {camera_id}")
        
        return self.frame_processors[camera_id]
    
    def get_performance_stats(self) -> Dict:
        """Get processing performance statistics"""
        with self._stats_lock:
            stats = {
                'total_processed': self.processing_stats['total_processed'],
                'worker_loads': self.processing_stats['worker_loads'].copy(),
                'avg_processing_time': 0.0,
                'processing_fps': 0.0
            }
            
            if self.processing_stats['processing_times']:
                avg_time = sum(self.processing_stats['processing_times']) / len(self.processing_stats['processing_times'])
                stats['avg_processing_time'] = avg_time
                stats['processing_fps'] = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return stats
