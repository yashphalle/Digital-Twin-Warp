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
        self.detection_pool = PerCameraDetectionThreadPool(num_workers=2, queue_manager=self.queue_manager)  # OPTIMIZED: 2 workers for RTX 4050 6GB GPU

        # NEW: Per-camera processing threads (copied from parallel_pipeline_system.py)
        self.processing_threads = {}

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

        # Frame processors now handled by individual processing threads

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

            # Start processing threads (11 parallel processing threads)
            logger.info("[PROCESSING] Starting parallel processing threads...")
            self._start_processing_threads()

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

    def _start_processing_threads(self):
        """Start 11 parallel processing threads (copied from parallel_pipeline_system.py)"""
        for camera_id in self.active_cameras:
            thread = threading.Thread(
                target=self._camera_processing_worker,
                args=(camera_id,),
                name=f"Processing-Camera-{camera_id}",
                daemon=True
            )
            self.processing_threads[camera_id] = thread
            thread.start()
            logger.info(f"🚀 Started processing thread for Camera {camera_id}")

    def _camera_processing_worker(self, camera_id: int):
        """Process frames for specific camera only (copied from parallel_pipeline_system.py)"""
        logger.info(f"[PROCESSING] Camera {camera_id} processing worker started")

        # Import processors (SAME as optimized system) - Fixed to use correct processor
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from modules.frame_processor import CPUFrameProcessor
        from modules.filtering import DetectionFiltering
        from modules.coordinate_mapper import CoordinateMapper
        from modules.feature_database import CPUGlobalFeatureDatabase
        from modules.color_extractor import ObjectColorExtractor
        from warehouse_database_handler import create_database_handler

        # Create camera-specific processor (SAME as optimized system)
        try:
            # Create frame processor (SAME as original)
            frame_processor = CPUFrameProcessor(camera_id)

            # Initialize all components (SAME as original)
            filtering = DetectionFiltering(camera_id)
            coord_converter = CoordinateMapper(camera_id=camera_id)
            coord_converter.load_calibration()

            sift_tracker = CPUGlobalFeatureDatabase(f"cpu_camera_{camera_id}_global_features.pkl", camera_id)
            color_extractor = ObjectColorExtractor()
            db_handler = create_database_handler()

            # Inject components into frame processor (SAME as original)
            frame_processor.inject_components(
                fisheye_corrector=None,  # Already done in camera thread
                pallet_detector=None,    # Already done in detection thread
                filtering=filtering,
                coordinate_mapper=coord_converter,
                coordinate_mapper_initialized=True,
                global_db=sift_tracker,
                color_extractor=color_extractor,
                db_handler=db_handler,
                display_manager=None  # No GUI in processing thread
            )

            processor = frame_processor
            logger.info(f"[PROCESSING] Camera {camera_id} processor initialized")
        except Exception as e:
            logger.error(f"[PROCESSING] Failed to initialize processor for Camera {camera_id}: {e}")
            return

        processed_count = 0

        while self.running:
            try:
                # Get frame from per-camera processing queue (NEW - Phase 1)
                per_camera_queue = f'camera_{camera_id}_detection_to_processing'
                frame_data = self.queue_manager.get_frame(per_camera_queue, timeout=1.0)
                if frame_data is None:
                    continue

                # No camera filtering needed - queue is camera-specific!

                processing_start = time.time()

                # Extract detection results (SAME as optimized system)
                raw_detections = frame_data.metadata.get('raw_detections', [])
                frame = frame_data.frame

                # REUSE EXISTING PROCESSING PIPELINE (SAME as optimized system)
                try:
                    # Set raw detections
                    processor.raw_detections = raw_detections

                    # Apply existing processing pipeline
                    frame_height, frame_width = frame.shape[:2]

                    # Stage 1: Apply filtering
                    processor.area_filtered_detections = processor.filtering.apply_area_filter(raw_detections)
                    processor.grid_filtered_detections = processor.filtering.apply_grid_cell_filter(processor.area_filtered_detections)

                    # Stage 2: Physical coordinate translation
                    processor.grid_filtered_detections = processor.translate_to_physical_coordinates(
                        processor.grid_filtered_detections, frame_width, frame_height
                    )

                    # Stage 3: Physical size filtering
                    processor.size_filtered_detections = processor.filtering.apply_physical_size_filter(processor.grid_filtered_detections)

                    # Stage 4: SIFT feature matching and global ID assignment
                    processor.final_tracked_detections = processor.assign_global_ids(processor.size_filtered_detections, frame)

                    # Stage 5: Save to database (SAME as optimized system)
                    if processor.db_handler and processor.db_handler.is_connected():
                        for detection in processor.final_tracked_detections:
                            processor.db_handler.save_detection_to_db(camera_id, detection)

                    processed_count += 1
                    processing_time = time.time() - processing_start

                    logger.debug(f"[PROCESSING] Camera {camera_id}: Processed frame {frame_data.frame_number} in {processing_time:.3f}s")

                except Exception as e:
                    logger.error(f"[PROCESSING] Camera {camera_id} processing error: {e}")
                    continue

            except Exception as e:
                logger.error(f"[PROCESSING] Camera {camera_id} worker error: {e}")
                time.sleep(0.1)  # Brief pause on error

        logger.info(f"[PROCESSING] Camera {camera_id} processing worker stopped")

    # Frame processor initialization now handled by individual processing threads

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
