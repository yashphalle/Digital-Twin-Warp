#!/usr/bin/env python3
"""
Optimized Pipeline Threading System
Uses SAME detection and processing modules with optimized camera threading
Zero changes to tested functionality - only threading optimization
"""

import logging
import time
import signal
import sys
import threading
import cv2
from typing import List

# Import optimized components
from .optimized_camera_threads import OptimizedCameraThreadManager
from .optimized_queue_manager import OptimizedQueueManager
from .detection_pool import DetectionThreadPool  # SAME as original (already working)
from .batch_detection_pool import BatchDetectionThreadPool  # Enhanced batch processing
from .database_workers import DatabaseWorkerPool
from .processing_thread_pool import ProcessingThreadPool  # NEW: 11 processing workers

logger = logging.getLogger(__name__)

# ===== PERFORMANCE TESTING FLAGS =====
# Set to True to bypass SIFT and filtering for performance testing
# Set to False for full processing (production mode)
BYPASS_PROCESSING_FOR_TESTING = False  # Change this to False for full processing

# Set to True to bypass only SIFT (keep filtering enabled)
BYPASS_SIFT_ONLY = True  # Change this to False to enable SIFT

# ===== YOLOV8 TRACKING FLAGS =====
# Set to True to use YOLOv8 built-in tracking instead of SIFT
USE_YOLOV8_TRACKING = True  # NEW: Enable YOLOv8 tracking

class OptimizedPipelineSystem:
    """
    Optimized Pipeline Threading System with GUI Support
    Uses SAME modules and functionality as original
    ONLY CHANGE: Optimized camera threading and queue management + GUI display
    """

    def __init__(self, active_cameras: List[int] = [1, 2], enable_gui: bool = False, gui_cameras: List[int] = None):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.gui_cameras = gui_cameras or []
        self.running = False

        # Initialize OPTIMIZED threading components
        self.queue_manager = OptimizedQueueManager(max_cameras=len(active_cameras))
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = DetectionThreadPool(num_workers=2, queue_manager=self.queue_manager)  # OPTIMIZED: 2 workers for RTX 4050 6GB GPU

        # Initialize ASYNC database workers
        self.database_workers = DatabaseWorkerPool(active_cameras)

        # Initialize PROCESSING thread pool - 11 workers (1 per camera)
        self.processing_pool = ProcessingThreadPool(active_cameras, self.queue_manager, self.database_workers)

        # Initialize GUI display managers if enabled
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

        # Performance monitoring
        self.performance_stats = {
            'start_time': None,
            'frames_processed': 0,
            'optimization_active': True
        }
        
        # Setup signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"[OK] OPTIMIZED Pipeline Threading System initialized for cameras: {active_cameras}")

        # Show processing mode
        if BYPASS_PROCESSING_FOR_TESTING:
            logger.info("🚀 [TESTING MODE] BYPASS PROCESSING ENABLED - Raw detections only (SIFT & filtering disabled)")
            logger.info("   - Expected: 2-3x FPS improvement")
            logger.info("   - Database: Raw detections with placeholder data")
            logger.info("   - To enable full processing: Set BYPASS_PROCESSING_FOR_TESTING = False")
        elif USE_YOLOV8_TRACKING:
            logger.info("🎯 [YOLOV8 TRACKING MODE] YOLOv8 built-in tracking enabled")
            logger.info("   - YOLOv8 ByteTrack: Enabled")
            logger.info("   - SIFT feature matching: Disabled")
            logger.info("   - Filtering pipeline: Enabled")
            logger.info("   - Physical coordinates: Enabled")
            logger.info("   - Expected: 10-20x faster than SIFT")
            logger.info("   - To disable: Set USE_YOLOV8_TRACKING = False")
        elif BYPASS_SIFT_ONLY:
            logger.info("🔧 [SIFT-BYPASS MODE] FILTERING ENABLED - SIFT tracking disabled")
            logger.info("   - SIFT feature matching: Disabled")
            logger.info("   - Filtering pipeline: Enabled")
            logger.info("   - Physical coordinates: Enabled")
            logger.info("   - Expected: 1.5-2x FPS improvement")
            logger.info("   - To enable SIFT: Set BYPASS_SIFT_ONLY = False")
        else:
            logger.info("🔧 [PRODUCTION MODE] FULL PROCESSING ENABLED - Complete pipeline")
            logger.info("   - SIFT feature matching: Enabled")
            logger.info("   - Filtering pipeline: Enabled")
            logger.info("   - Physical coordinates: Enabled")

        logger.info("[OPTIMIZATIONS] KEY FEATURES:")
        logger.info("   - Smart frame skipping BEFORE processing (95% CPU savings)")
        logger.info("   - Enhanced queue management for GPU feeding")
        logger.info("   - Async database operations (non-blocking)")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down optimized system...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the optimized threading system"""
        if self.running:
            logger.warning("[WARNING] Optimized system already running")
            return

        self.running = True
        self.performance_stats['start_time'] = time.time()

        logger.info("[START] Starting OPTIMIZED Pipeline Threading System...")

        try:
            # Start camera threads (OPTIMIZED)
            logger.info("[CAMERAS] Starting optimized camera threads...")
            self.camera_manager.start_camera_threads()

            # Start detection pool (SAME as original)
            logger.info("[DETECTION] Starting detection thread pool...")
            self.detection_pool.start_detection_workers()

            # Start ASYNC database workers
            logger.info("[DATABASE] Starting async database workers...")
            self.database_workers.start_workers()

            # Start processing thread pool (NEW: 11 workers, 1 per camera)
            logger.info("[PROCESSING] Starting processing thread pool (11 workers)...")
            self.processing_pool.start_processing_workers()

            # Start performance monitoring
            logger.info("[MONITOR] Starting performance monitoring...")
            self.monitor_thread = threading.Thread(
                target=self._performance_monitor,
                name="OptimizedMonitor",
                daemon=True
            )
            self.monitor_thread.start()

            logger.info("[SUCCESS] OPTIMIZED Pipeline Threading System started successfully!")
            logger.info("[MONITOR] Monitor CPU usage - should drop from 100% to ~20-30%")
            logger.info("[MONITOR] Monitor GPU usage - should increase from 20% to 80-90%")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("[INTERRUPT] Keyboard interrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"[ERROR] Error in optimized pipeline system: {e}")
            self.stop()
    
    def stop(self):
        """Stop the optimized threading system"""
        if not self.running:
            return

        logger.info("[STOP] Stopping OPTIMIZED Pipeline Threading System...")
        self.running = False

        # Close GUI windows if enabled
        if self.enable_gui:
            try:
                cv2.destroyAllWindows()
                logger.info("[GUI] All GUI windows closed")
            except Exception as e:
                logger.error(f"[GUI] Error closing windows: {e}")

        # Stop components in reverse order
        if hasattr(self, 'processing_pool'):
            logger.info("[PROCESSING] Stopping processing thread pool...")
            self.processing_pool.stop_processing_workers()

        if hasattr(self, 'detection_pool'):
            self.detection_pool.stop()

        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop_camera_threads()

        # Stop ASYNC database workers
        if hasattr(self, 'database_workers'):
            logger.info("[DATABASE] Stopping async database workers...")
            self.database_workers.stop_workers()

        # Log final performance stats
        self._log_final_stats()

        logger.info("[COMPLETE] OPTIMIZED Pipeline Threading System stopped")
    
    # OLD METHOD - COMMENTED OUT - REPLACED BY ProcessingThreadPool
    def _basic_processing_consumer_OLD_DISABLED(self):
        """
        OLD METHOD - COMMENTED OUT - REPLACED BY ProcessingThreadPool
        Processing consumer - SAME as original system
        Uses SAME modules: CPUFrameProcessor, filtering, coordinates, database
        """
        pass  # This method is disabled - using ProcessingThreadPool instead
    #     # Create frame processors for each camera based on tracking mode
    #     processors = {}
    #     for camera_id in self.active_cameras:
    #         try:
    #             # Create frame processor based on tracking mode
    #             if USE_YOLOV8_TRACKING:
    #                 frame_processor = YOLOv8FrameProcessor(camera_id)
    #                 # Create YOLOv8 tracking database
    #                 yolo_tracking_db = YOLOv8TrackingDatabase(f"yolo_camera_{camera_id}_tracking.pkl", camera_id)
    #             else:
    #                 frame_processor = CPUFrameProcessor(camera_id)
    #                 # Create SIFT feature database
    #                 global_db = CPUGlobalFeatureDatabase(f"cpu_camera_{camera_id}_global_features.pkl", camera_id)
    #
    #             # Initialize common components
    #             filtering = DetectionFiltering(camera_id)
    #             coord_converter = CoordinateMapper(camera_id=camera_id)
    #             coord_converter.load_calibration()
    #             color_extractor = ObjectColorExtractor()
    #             # Use factory function to create database handler with Config values (from .env)
    #             db_handler = create_database_handler()
    #
    #             # Inject components into frame processor based on tracking mode
    #             display_manager = self.display_managers.get(camera_id) if self.enable_gui else None
    #
    #             if USE_YOLOV8_TRACKING:
    #                 # YOLOv8 tracking mode - inject yolo_tracking_db as global_db for compatibility
    #                 frame_processor.inject_components(
    #                     fisheye_corrector=None,  # Already done in camera thread
    #                     pallet_detector=None,    # Already done in detection thread
    #                     filtering=filtering,
    #                     coordinate_mapper=coord_converter,
    #                     coordinate_mapper_initialized=True,
    #                     global_db=yolo_tracking_db,  # YOLOv8 tracking database (mapped to global_db)
    #                     yolo_tracking_db=yolo_tracking_db,  # Also set the specific attribute
    #                     color_extractor=color_extractor,
    #                     db_handler=db_handler,
    #                     display_manager=display_manager
    #                 )
    #             else:
    #                 # SIFT mode - inject SIFT feature database
    #                 frame_processor.inject_components(
    #                     fisheye_corrector=None,  # Already done in camera thread
    #                     pallet_detector=None,    # Already done in detection thread
    #                     filtering=filtering,
    #                     coordinate_mapper=coord_converter,
    #                     coordinate_mapper_initialized=True,
    #                     global_db=global_db,  # SIFT feature database
    #                     color_extractor=color_extractor,
    #                     db_handler=db_handler,
    #                     display_manager=display_manager
    #                 )
    #
    #             processors[camera_id] = frame_processor
    #             logger.info(f"[OK] Frame processor initialized for Camera {camera_id}")
    #
    #         except Exception as e:
    #             logger.error(f"[ERROR] Failed to initialize processor for Camera {camera_id}: {e}")
    #
    #     processed_count = 0
    #     saved_count = 0
    #
    #     while self.running:
    #         try:
    #             # Get frame with detection results (SAME as original)
    #             frame_data = self.queue_manager.get_frame('detection_to_processing', timeout=1.0)
    #             if frame_data is None:
    #                 continue
    #
    #             # Extract detection results (SAME as original)
    #             raw_detections = frame_data.metadata.get('raw_detections', [])
    #             camera_id = frame_data.camera_id
    #             frame = frame_data.frame
    #
    #             # Always show GUI even if no detections (FIXED: GUI was not showing when no objects detected)
    #             if not raw_detections:
    #                 logger.debug(f"[DETECTION] Camera {camera_id}: No detections found")
    #                 # Still show GUI with empty frame
    #                 if self.enable_gui and camera_id in self.gui_cameras:
    #                     try:
    #                         display_manager = self.display_managers[camera_id]
    #
    #                         # Create mock tracker with empty detections
    #                         mock_tracker = type('MockTracker', (), {
    #                             'frame_processor': type('MockFrameProcessor', (), {
    #                                 'final_tracked_detections': [],
    #                                 'get_detection_counts': lambda self: {'total': 0, 'new': 0, 'tracked': 0},
    #                                 'raw_detections': [],
    #                                 'filtered_detections': [],
    #                                 'grid_filtered_detections': []
    #                             })(),
    #                             'camera_id': camera_id,
    #                             'camera_name': f"Camera {camera_id}"
    #                         })()
    #
    #                         # Show GUI window with empty detections
    #                         display_frame = display_manager.render_frame(frame, mock_tracker)
    #                         window_name = f"CPU Tracking - Camera {camera_id}"
    #                         cv2.imshow(window_name, display_frame)
    #
    #                         # Handle keyboard input
    #                         key = cv2.waitKey(1) & 0xFF
    #                         if key == ord('q'):
    #                             logger.info("[GUI] 'q' key pressed - shutting down system")
    #                             self.stop()
    #                             break
    #
    #                     except Exception as e:
    #                         logger.error(f"[GUI] Display error for Camera {camera_id}: {e}")
    #                 continue

    #             # Get processor for this camera (SAME as original)
    #             if camera_id not in processors:
    #                 logger.warning(f"[WARNING] No processor for Camera {camera_id}")
    #                 continue
    #
    #             processor = processors[camera_id]
    #
    #             # REUSE EXISTING PROCESSING PIPELINE (SAME as original)
    #             try:
    #                 # Set raw detections (SAME as original)
    #                 processor.raw_detections = raw_detections
    #
    #                 # ===== PROCESSING MODE SELECTION (FIXED PRIORITY ORDER) =====
    #                 if BYPASS_PROCESSING_FOR_TESTING:
    #                     # ===== BYPASS MODE: Raw detections only (for performance testing) =====
    #                     logger.info(f"🚀 [BYPASS] Camera {camera_id}: Using raw detections (SIFT & filtering disabled)")
    #                     processor.final_tracked_detections = processor.raw_detections.copy()
    #
    #                     # Add minimal required fields for database compatibility
    #                     for i, detection in enumerate(processor.final_tracked_detections):
    #                         detection['global_id'] = f"{camera_id}{1000 + i}"  # Simple ID: 7001, 7002, etc.
    #                         detection['tracking_status'] = 'raw_test'
    #                         detection['physical_x_ft'] = 0.0  # Placeholder
    #                         detection['physical_y_ft'] = 0.0  # Placeholder
    #                         detection['color_rgb'] = None  # Placeholder
    #                         detection['color_hsv'] = None  # Placeholder
    #                         detection['color_hex'] = None  # Placeholder

    #                 elif USE_YOLOV8_TRACKING:
    #                     # ===== YOLOV8 TRACKING MODE: Complete pipeline with YOLOv8 tracking =====
    #                     logger.info(f"🎯 [YOLOV8-TRACKING] Camera {camera_id}: Using YOLOv8 tracking pipeline")
    #                     frame_height, frame_width = frame.shape[:2]
    #
    #                     # Stage 1: Apply filtering (SAME as original)
    #                     processor.area_filtered_detections = processor.filtering.apply_area_filter(raw_detections)
    #                     processor.grid_filtered_detections = processor.filtering.apply_grid_cell_filter(processor.area_filtered_detections)
    #
    #                     # Stage 2: Physical coordinate translation (SAME as original)
    #                     processor.grid_filtered_detections = processor.translate_to_physical_coordinates(
    #                         processor.grid_filtered_detections, frame_width, frame_height
    #                     )
    #
    #                     # Stage 3: Physical size filtering (SAME as original)
    #                     processor.size_filtered_detections = processor.filtering.apply_physical_size_filter(processor.grid_filtered_detections)
    #
    #                     # Stage 4: YOLOv8 tracking-based global ID assignment (NEW)
    #                     processor.final_tracked_detections = processor.assign_global_ids_yolo_tracking(processor.size_filtered_detections, frame)
    #
    #                 elif BYPASS_SIFT_ONLY:
    #                     # ===== SIFT-BYPASS MODE: Filtering enabled, SIFT disabled =====
    #                     logger.info(f"🚀 [SIFT-BYPASS] Camera {camera_id}: Using filtered detections (SIFT disabled)")
    #                     frame_height, frame_width = frame.shape[:2]
    #
    #                     # Stage 1: Apply filtering (SAME as original)
    #                     processor.area_filtered_detections = processor.filtering.apply_area_filter(raw_detections)
    #                     processor.grid_filtered_detections = processor.filtering.apply_grid_cell_filter(processor.area_filtered_detections)
    #
    #                     # Stage 2: Physical coordinate translation (SAME as original)
    #                     processor.grid_filtered_detections = processor.translate_to_physical_coordinates(
    #                         processor.grid_filtered_detections, frame_width, frame_height
    #                     )
    #
    #                     # Stage 3: Physical size filtering (SAME as original)
    #                     processor.size_filtered_detections = processor.filtering.apply_physical_size_filter(processor.grid_filtered_detections)
    #
    #                     # Stage 4: Simple ID assignment without SIFT
    #                     processor.final_tracked_detections = processor.size_filtered_detections.copy()
    #
    #                     # Assign real global IDs without SIFT matching
    #                     for detection in processor.final_tracked_detections:
    #                         # Generate new global ID without SIFT feature matching
    #                         new_global_id = processor.global_db.next_global_id
    #                         processor.global_db.next_global_id += 1
    #
    #                         detection.update({
    #                             'global_id': new_global_id,  # Real ID instead of -1
    #                             'tracking_status': 'new',  # Use properly handled status
    #                             'similarity_score': 0.0
    #                         })
    #
    #                 else:
    #                     # ===== SIFT MODE: Full SIFT feature matching =====
    #                     logger.info(f"🔧 [SIFT] Camera {camera_id}: Using SIFT feature matching")
    #                     frame_height, frame_width = frame.shape[:2]
    #
    #                     # Stage 1: Apply filtering (SAME as original)
    #                     processor.area_filtered_detections = processor.filtering.apply_area_filter(raw_detections)
    #                     processor.grid_filtered_detections = processor.filtering.apply_grid_cell_filter(processor.area_filtered_detections)
    #
    #                     # Stage 2: Physical coordinate translation (SAME as original)
    #                     processor.grid_filtered_detections = processor.translate_to_physical_coordinates(
    #                         processor.grid_filtered_detections, frame_width, frame_height
    #                     )
    #
    #                     # Stage 3: Physical size filtering (SAME as original)
    #                     processor.size_filtered_detections = processor.filtering.apply_physical_size_filter(processor.grid_filtered_detections)
    #
    #                     # Stage 4: SIFT feature matching and global ID assignment
    #                     processor.final_tracked_detections = processor.assign_global_ids(processor.size_filtered_detections, frame)
    #
    #                 # Stage 5: Save to database (ASYNC - NON-BLOCKING)
    #                 db_start_time = time.time()
    #                 if processor.db_handler and processor.db_handler.is_connected():
    #                     for detection in processor.final_tracked_detections:
    #                         success = self.database_workers.queue_database_save(camera_id, detection, frame_data.frame_number)
    #                         if success:
    #                             saved_count += 1
    #                 db_time = time.time() - db_start_time
    #
    #                 # Simple logging every 10 frames for performance comparison
    #                 if processed_count % 10 == 0:
    #                     queue_size = self.database_workers.database_queues[camera_id].qsize()
    #                     logger.info(f"[ASYNC] Camera {camera_id}: DB time {db_time*1000:.1f}ms, Queue: {queue_size}/300, Detections: {len(processor.final_tracked_detections)}")
    #
    #                 processed_count += 1
    #                 self.performance_stats['frames_processed'] = processed_count
    #
    #                 # GUI Display (SAME as main.py)
    #                 if self.enable_gui and camera_id in self.gui_cameras:
    #                     try:
    #                         display_manager = self.display_managers[camera_id]
    #
    #                         # Create mock tracker object for display (same structure as main.py)
    #                         mock_tracker = type('MockTracker', (), {
    #                             'frame_processor': processor,
    #                             'camera_id': camera_id,
    #                             'camera_name': f"Camera {camera_id}"
    #                         })()
    #
    #                         # Render frame with detections and overlay
    #                         display_frame = display_manager.render_frame(frame, mock_tracker)
    #
    #                         # Show GUI window
    #                         window_name = f"CPU Tracking - Camera {camera_id}"
    #                         cv2.imshow(window_name, display_frame)
    #
    #                         # Handle keyboard input (same as main.py)
    #                         key = cv2.waitKey(1) & 0xFF
    #                         if key == ord('q'):
    #                             logger.info("[GUI] 'q' key pressed - shutting down system")
    #                             self.stop()
    #                             break
    #                         elif key == ord('n'):
    #                             logger.info("[GUI] 'n' key pressed - next detection prompt")
    #                         elif key == ord('p'):
    #                             logger.info("[GUI] 'p' key pressed - previous detection prompt")
    #
    #                     except Exception as e:
    #                         logger.error(f"[GUI] Display error for Camera {camera_id}: {e}")
    #
    #                 # Log results using existing method (SAME as original)
    #                 counts = processor.get_detection_counts()
    #                 logger.info(f"[RESULTS] Camera {camera_id}: {counts['raw_detections']} raw → {counts['area_filtered_detections']} area → {counts['grid_filtered_detections']} grid → {counts['size_filtered_detections']} size → {counts['final_tracked_detections']} final")
    #                 logger.info(f"[STATS] OPTIMIZED SYSTEM: Processed {processed_count}, Saved {saved_count}, New: {counts['new_objects']}, Existing: {counts['existing_objects']}")
    #
    #             except Exception as e:
    #                 logger.error(f"[ERROR] Processing pipeline error for Camera {camera_id}: {e}")
    #
    #         except Exception as e:
    #             logger.error(f"[ERROR] Processing consumer error: {e}")
    #             time.sleep(0.1)
    #
    #     logger.info("[STOP] Optimized processing consumer stopped")
    
    def _performance_monitor(self):
        """Monitor system performance and optimization effectiveness"""
        logger.info("[MONITOR] Performance monitoring started")

        while self.running:
            try:
                time.sleep(30)  # Monitor every 30 seconds

                # Get optimization stats
                camera_stats = self.camera_manager.get_optimization_stats()
                queue_stats = self.queue_manager.get_optimization_stats()

                # Log performance summary
                logger.info("[PERFORMANCE] OPTIMIZATION PERFORMANCE SUMMARY:")
                logger.info(f"   Expected CPU Savings: {camera_stats.get('expected_cpu_savings', 'N/A')}")
                logger.info(f"   GPU Feed Efficiency: {queue_stats.get('optimization', {}).get('gpu_feed_efficiency', 'N/A')}")
                logger.info(f"   Frames Processed: {self.performance_stats['frames_processed']}")

                # Log queue status
                self.queue_manager.log_optimization_status()

                # Log simple database stats
                stats = self.database_workers.stats
                logger.info(f"[STATS] Database: Queued: {stats['total_queued']}, Saved: {stats['total_saved']}, Dropped: {stats['total_dropped']}")

                # Log queue status for each camera
                for camera_id in self.active_cameras:
                    queue_size = self.database_workers.database_queues[camera_id].qsize()
                    logger.info(f"[QUEUE] Camera {camera_id}: {queue_size}/300 items")

            except Exception as e:
                logger.error(f"[ERROR] Performance monitoring error: {e}")
                time.sleep(5)
    
    def _log_final_stats(self):
        """Log final performance statistics"""
        if self.performance_stats['start_time']:
            runtime = time.time() - self.performance_stats['start_time']
            fps = self.performance_stats['frames_processed'] / runtime if runtime > 0 else 0

            logger.info("[FINAL] FINAL OPTIMIZATION RESULTS:")
            logger.info(f"   Runtime: {runtime:.1f} seconds")
            logger.info(f"   Frames Processed: {self.performance_stats['frames_processed']}")
            logger.info(f"   Average FPS: {fps:.2f}")
            logger.info("[MONITOR] Check system monitor for CPU/GPU utilization improvements!")
