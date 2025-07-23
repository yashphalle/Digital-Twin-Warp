#!/usr/bin/env python3
"""
Per-Camera Optimized Pipeline Threading System
DUPLICATE of optimized_pipeline_system.py with per-camera queue support
Uses SAME detection and processing modules with per-camera queues for fair processing
"""

import logging
import time
import signal
import sys
import threading
import cv2
from typing import List

# Import per-camera components
from .optimized_camera_threads import OptimizedCameraThreadManager
from .per_camera_queue_manager import PerCameraQueueManager
from .per_camera_detection_pool import PerCameraDetectionThreadPool

logger = logging.getLogger(__name__)

class PerCameraOptimizedPipelineSystem:
    """
    Per-Camera Optimized Pipeline Threading System with GUI Support
    Uses SAME modules and functionality as original optimized system
    ONLY CHANGE: Per-camera queues instead of shared queue for fair processing
    """

    def __init__(self, active_cameras: List[int] = [1, 2], enable_gui: bool = False, gui_cameras: List[int] = None):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.gui_cameras = gui_cameras or []
        self.running = False

        # Initialize PER-CAMERA threading components
        self.queue_manager = PerCameraQueueManager(max_cameras=len(active_cameras), active_cameras=active_cameras)
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = PerCameraDetectionThreadPool(num_workers=11, queue_manager=self.queue_manager)  # TESTING: 1 worker

        # Initialize GUI display managers if enabled (SAME as optimized system)
        self.display_managers = {}
        if self.enable_gui:
            logger.info(f"[GUI] Initializing display managers for cameras: {self.gui_cameras}")
            try:
                from modules.gui_display import CPUDisplayManager
                for cam_id in self.gui_cameras:
                    camera_name = f"Camera {cam_id}"
                    self.display_managers[cam_id] = CPUDisplayManager(cam_id, camera_name)
                    logger.info(f"[GUI] Display manager initialized for {camera_name}")
            except Exception as e:
                logger.error(f"[GUI] Failed to initialize display managers: {e}")
                self.enable_gui = False

        # Initialize frame processors for each camera (SAME as optimized system)
        self.frame_processors = {}

        # Performance monitoring
        self.stats_thread = None
        self.last_stats_time = time.time()

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"[OK] PER-CAMERA Optimized Pipeline Threading System initialized for cameras: {active_cameras}")
        logger.info("[PER-CAMERA] KEY FEATURES:")
        logger.info("   - Per-camera queues for fair processing (solves frame ordering issue)")
        logger.info("   - Round-robin detection workers")
        logger.info("   - Smart frame skipping BEFORE processing (95% CPU savings)")
        logger.info("   - Enhanced queue management for GPU feeding")
        logger.info("   - SAME tested modules: detection, filtering, coordinates, database")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down per-camera optimized system...")
        self.stop()

    def start(self):
        """Start the complete per-camera optimized pipeline system"""
        try:
            logger.info("[START] Starting PER-CAMERA Optimized Pipeline Threading System...")
            self.running = True

            # Start camera threads (SAME as optimized system)
            logger.info("[CAMERAS] Starting optimized camera threads...")
            self.camera_manager.start_camera_threads()

            # Start per-camera detection pool
            logger.info("[DETECTION] Starting per-camera detection thread pool...")
            self.detection_pool.start_detection_workers()

            # Start processing consumer (SAME as optimized system)
            logger.info("[PROCESSING] Starting processing consumer...")
            self._start_processing_consumer()

            # Start performance monitoring
            logger.info("[MONITOR] Starting performance monitoring...")
            self._start_performance_monitoring()

            logger.info("[SUCCESS] PER-CAMERA Optimized Pipeline Threading System started successfully!")
            logger.info("[MONITOR] Monitor CPU usage - should drop from 100% to ~20-30%")
            logger.info("[MONITOR] Monitor GPU usage - should increase from 20% to 80-90%")
            logger.info("[BALANCE] Watch for camera balance ratio in stats (1.0 = perfect balance)")

            # Keep main thread alive and monitor system
            self._monitor_system()

        except Exception as e:
            logger.error(f"[ERROR] Failed to start per-camera optimized pipeline system: {e}")
            self.stop()
            raise

    def _start_processing_consumer(self):
        """Start processing consumer thread (SAME as optimized system)"""
        self.processing_thread = threading.Thread(
            target=self._processing_consumer_worker,
            name="PerCameraProcessingConsumer",
            daemon=True
        )
        self.processing_thread.start()
        logger.info("[START] Per-camera optimized processing consumer started (using existing modules)")

    def _processing_consumer_worker(self):
        """Processing consumer worker (SAME logic as optimized system)"""
        logger.info("[PROCESSING] Per-camera processing consumer worker started")
        
        while self.running:
            try:
                # Get processed frame from detection workers
                frame_data = self.queue_manager.get_frame('detection_to_processing', timeout=1.0)
                if frame_data is None:
                    continue

                camera_id = frame_data.camera_id
                
                # Initialize frame processor for this camera if not exists (SAME as optimized)
                if camera_id not in self.frame_processors:
                    self._initialize_frame_processor(camera_id)

                # Process frame using SAME modules as optimized system
                frame_processor = self.frame_processors[camera_id]
                
                # Process using individual stages (SAME as optimized system)
                raw_detections = frame_data.metadata.get('raw_detections', [])
                frame = frame_data.frame
                frame_height, frame_width = frame.shape[:2]

                # Set raw detections (SAME as optimized system)
                frame_processor.raw_detections = raw_detections

                # Stage 1: Apply filtering (SAME as optimized system)
                frame_processor.area_filtered_detections = frame_processor.filtering.apply_area_filter(raw_detections)
                frame_processor.grid_filtered_detections = frame_processor.filtering.apply_grid_cell_filter(frame_processor.area_filtered_detections)

                # Stage 2: Physical coordinate translation (SAME as optimized system)
                frame_processor.grid_filtered_detections = frame_processor.translate_to_physical_coordinates(
                    frame_processor.grid_filtered_detections, frame_width, frame_height
                )

                # Stage 3: Physical size filtering (SAME as optimized system)
                frame_processor.size_filtered_detections = frame_processor.filtering.apply_physical_size_filter(frame_processor.grid_filtered_detections)

                # Stage 4: SIFT feature matching and global ID assignment (SAME as optimized system)
                frame_processor.final_tracked_detections = frame_processor.assign_global_ids(frame_processor.size_filtered_detections, frame)

                # Stage 5: Save to database (SAME as optimized system)
                if frame_processor.db_handler and frame_processor.db_handler.is_connected():
                    for detection in frame_processor.final_tracked_detections:
                        frame_processor.db_handler.save_detection_to_db(camera_id, detection)

                # Update frame data with processed results
                frame_data.stage = "processed"
                frame_data.metadata.update({
                    'final_tracked_detections': frame_processor.final_tracked_detections,
                    'detection_counts': frame_processor.get_detection_counts(),
                    'per_camera_processing': True
                })

                # Send to GUI if enabled (SAME as optimized system)
                if self.enable_gui and camera_id in self.display_managers:
                    self.queue_manager.put_frame('processing_to_gui', frame_data, timeout=0.1)

                # Database operations handled by frame processor (SAME as optimized system)
                
            except Exception as e:
                logger.error(f"[ERROR] Per-camera processing consumer error: {e}")
                time.sleep(0.1)

        logger.info("[STOP] Per-camera processing consumer worker stopped")

    def _initialize_frame_processor(self, camera_id: int):
        """Initialize frame processor for camera (SAME as optimized system)"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

            from modules.frame_processor import CPUFrameProcessor
            from modules.filtering import DetectionFiltering
            from modules.coordinate_mapper import CoordinateMapper
            from modules.feature_database import CPUGlobalFeatureDatabase
            from modules.color_extractor import ObjectColorExtractor
            from warehouse_database_handler import create_database_handler

            # Initialize frame processor
            frame_processor = CPUFrameProcessor(camera_id)

            # Initialize components (SAME as optimized system)
            filtering = DetectionFiltering(camera_id)
            coord_converter = CoordinateMapper(camera_id=camera_id)
            coord_converter.load_calibration()  # ← MISSING! This is why coordinates fail
            global_db = CPUGlobalFeatureDatabase(f"cpu_camera_{camera_id}_global_features.pkl", camera_id)
            color_extractor = ObjectColorExtractor()

            # Use factory function to create database handler with Config values (from .env)
            db_handler = create_database_handler()

            # Inject components into frame processor (SAME as original + GUI support)
            display_manager = self.display_managers.get(camera_id) if self.enable_gui else None
            frame_processor.inject_components(
                fisheye_corrector=None,  # Already done in camera thread
                pallet_detector=None,    # Already done in detection thread
                filtering=filtering,
                coordinate_mapper=coord_converter,
                coordinate_mapper_initialized=True,
                global_db=global_db,
                color_extractor=color_extractor,
                db_handler=db_handler,
                display_manager=display_manager  # GUI display manager if enabled
            )

            self.frame_processors[camera_id] = frame_processor
            logger.info(f"[OK] Frame processor initialized for Camera {camera_id}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize frame processor for Camera {camera_id}: {e}")
            raise

    def _start_performance_monitoring(self):
        """Start performance monitoring thread (SAME as optimized system)"""
        self.stats_thread = threading.Thread(
            target=self._performance_monitor_worker,
            name="PerCameraPerformanceMonitor",
            daemon=True
        )
        self.stats_thread.start()

    def _performance_monitor_worker(self):
        """Performance monitoring worker with per-camera stats"""
        logger.info("[MONITOR] Per-camera performance monitoring started")
        
        while self.running:
            try:
                time.sleep(30)  # Log stats every 30 seconds
                
                if not self.running:
                    break
                
                logger.info("\n" + "=" * 80)
                logger.info("PER-CAMERA OPTIMIZED SYSTEM PERFORMANCE REPORT")
                logger.info("=" * 80)
                
                # Per-camera queue statistics
                self.queue_manager.log_camera_stats()
                
                # Per-camera detection statistics
                self.detection_pool.log_detection_stats()
                
                # System uptime
                uptime = time.time() - self.last_stats_time
                logger.info(f"System Uptime: {uptime:.1f} seconds")
                
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"[ERROR] Performance monitoring error: {e}")

    def _monitor_system(self):
        """Monitor system health and handle GUI events (SAME as optimized system)"""
        try:
            while self.running:
                # Handle GUI events if enabled
                if self.enable_gui:
                    for cam_id, display_manager in self.display_managers.items():
                        try:
                            # Get frame for display
                            frame_data = self.queue_manager.get_frame('processing_to_gui', timeout=0.1)
                            if frame_data and frame_data.camera_id == cam_id:
                                # Display frame (SAME as optimized system)
                                key = display_manager.display_frame(
                                    frame_data.frame,
                                    frame_data.metadata.get('processed_detections', []),
                                    frame_data.metadata.get('detection_counts', {})
                                )
                                
                                if key == ord('q'):
                                    logger.info("[GUI] 'q' key pressed, shutting down...")
                                    self.running = False
                                    break
                        except Exception as e:
                            logger.debug(f"[GUI] Display error for Camera {cam_id}: {e}")
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("[INTERRUPT] Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self):
        """Stop the complete per-camera optimized pipeline system"""
        if not self.running:
            return
            
        logger.info("[STOP] Stopping Per-Camera Optimized Pipeline System...")
        self.running = False

        try:
            # Stop components in reverse order (SAME as optimized system)
            logger.info("[STOP] Stopping per-camera detection pool...")
            self.detection_pool.stop()

            logger.info("[STOP] Stopping optimized camera threads...")
            self.camera_manager.stop()

            # Close GUI displays
            if self.enable_gui:
                logger.info("[STOP] Closing GUI displays...")
                for display_manager in self.display_managers.values():
                    display_manager.cleanup()

            # Wait for stats thread to finish
            if self.stats_thread and self.stats_thread.is_alive():
                logger.info("[STOP] Stopping performance monitoring...")
                self.stats_thread.join(timeout=2)

            logger.info("[SUCCESS] Per-Camera Optimized Pipeline System stopped successfully")

        except Exception as e:
            logger.error(f"[ERROR] Error during shutdown: {e}")

    def get_system_stats(self):
        """Get comprehensive system statistics"""
        stats = {
            'system': {
                'running': self.running,
                'active_cameras': self.active_cameras,
                'gui_enabled': self.enable_gui,
                'uptime': time.time() - self.last_stats_time
            },
            'queues': self.queue_manager.get_per_camera_stats(),
            'detection': self.detection_pool.get_detection_stats(),
        }
        
        return stats
