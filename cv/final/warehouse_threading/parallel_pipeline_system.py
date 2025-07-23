#!/usr/bin/env python3
"""
Parallel Pipeline Threading System
11 parallel processing threads + async database operations
Based on OptimizedPipelineSystem but with parallel processing architecture
"""

import logging
import time
import signal
import sys
import threading
import cv2
import queue
from typing import List, Dict

# Import optimized components (reuse existing)
from .optimized_camera_threads import OptimizedCameraThreadManager
from .optimized_queue_manager import OptimizedQueueManager
from .detection_pool import DetectionThreadPool

logger = logging.getLogger(__name__)

class ParallelPipelineSystem:
    """
    Parallel processing system with:
    - 11 camera threads (preprocessing)
    - 11 detection workers (GPU)
    - 11 processing threads (post-detection)
    - 11 database workers (async saves)
    """
    
    def __init__(self, active_cameras: List[int] = [1, 2], enable_gui: bool = False, gui_cameras: List[int] = None):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.gui_cameras = gui_cameras or []
        self.running = False

        # Initialize SAME threading components as optimized system
        self.queue_manager = OptimizedQueueManager(max_cameras=len(active_cameras))
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = DetectionThreadPool(num_workers=11, queue_manager=self.queue_manager)

        # NEW: Per-camera database queues and workers
        self.database_queues = {}
        self.database_workers = {}
        self.processing_threads = {}
        
        # Initialize per-camera database queues
        for camera_id in active_cameras:
            self.database_queues[camera_id] = queue.Queue(maxsize=20)  # 20 database tasks per camera
        
        # Performance monitoring
        self.performance_stats = {
            'start_time': None,
            'frames_processed_per_camera': {camera_id: 0 for camera_id in active_cameras},
            'database_saves_per_camera': {camera_id: 0 for camera_id in active_cameras},
            'processing_times_per_camera': {camera_id: [] for camera_id in active_cameras}
        }
        
        # Setup signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"✅ Parallel Pipeline System initialized for cameras: {active_cameras}")
        logger.info(f"🔧 Architecture: {len(active_cameras)} cameras → {len(active_cameras)} processing threads → {len(active_cameras)} database workers")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the parallel pipeline system"""
        if self.running:
            logger.warning("System is already running")
            return

        self.running = True
        self.performance_stats['start_time'] = time.time()
        
        logger.info("🚀 STARTING PARALLEL PIPELINE SYSTEM")
        logger.info("=" * 80)

        try:
            # Start camera threads (SAME as optimized system)
            logger.info("[CAMERAS] Starting optimized camera threads...")
            self.camera_manager.start_camera_threads()

            # Start detection pool (SAME as optimized system)
            logger.info("[DETECTION] Starting detection thread pool...")
            self.detection_pool.start_detection_workers()

            # NEW: Start per-camera processing threads
            logger.info("[PROCESSING] Starting parallel processing threads...")
            self._start_processing_threads()

            # NEW: Start per-camera database workers
            logger.info("[DATABASE] Starting async database workers...")
            self._start_database_workers()

            # Start performance monitoring
            logger.info("[MONITOR] Starting performance monitoring...")
            self.monitor_thread = threading.Thread(
                target=self._performance_monitor,
                name="ParallelMonitor",
                daemon=True
            )
            self.monitor_thread.start()

            logger.info("=" * 80)
            logger.info("✅ PARALLEL PIPELINE SYSTEM STARTED SUCCESSFULLY")
            logger.info(f"📊 Monitoring {len(self.active_cameras)} cameras with parallel processing")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Failed to start parallel pipeline system: {e}")
            self.stop()
            raise

    def _start_processing_threads(self):
        """Start 11 parallel processing threads"""
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

    def _start_database_workers(self):
        """Start 11 async database workers"""
        for camera_id in self.active_cameras:
            worker = threading.Thread(
                target=self._database_worker,
                args=(camera_id,),
                name=f"Database-Camera-{camera_id}",
                daemon=True
            )
            self.database_workers[camera_id] = worker
            worker.start()
            logger.info(f"🗄️ Started database worker for Camera {camera_id}")

    def _camera_processing_worker(self, camera_id: int):
        """Process frames for specific camera only"""
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

            sift_tracker = CPUGlobalFeatureDatabase(f"camera_{camera_id}_features.pkl", camera_id)
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
                # Get frame from shared detection queue
                frame_data = self.queue_manager.get_frame('detection_to_processing', timeout=1.0)
                if frame_data is None:
                    continue

                # ONLY process frames from assigned camera
                if frame_data.camera_id != camera_id:
                    continue

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

                    # Stage 5: ASYNC database save (NEW!)
                    for detection in processor.final_tracked_detections:
                        database_task = {
                            'camera_id': camera_id,
                            'detection': detection,
                            'timestamp': time.time(),
                            'frame_number': frame_data.frame_number
                        }

                        # Non-blocking database queue put
                        try:
                            self.database_queues[camera_id].put_nowait(database_task)
                        except queue.Full:
                            logger.warning(f"[DATABASE] Camera {camera_id} database queue full, dropping detection")

                    processed_count += 1
                    processing_time = time.time() - processing_start

                    # Update performance stats
                    self.performance_stats['frames_processed_per_camera'][camera_id] = processed_count
                    self.performance_stats['processing_times_per_camera'][camera_id].append(processing_time)

                    # Keep only recent processing times (last 100)
                    if len(self.performance_stats['processing_times_per_camera'][camera_id]) > 100:
                        self.performance_stats['processing_times_per_camera'][camera_id] = \
                            self.performance_stats['processing_times_per_camera'][camera_id][-100:]

                    logger.debug(f"[PROCESSING] Camera {camera_id}: Processed frame {frame_data.frame_number} in {processing_time:.3f}s")

                except Exception as e:
                    logger.error(f"[PROCESSING] Camera {camera_id} processing error: {e}")
                    continue

            except Exception as e:
                logger.error(f"[PROCESSING] Camera {camera_id} worker error: {e}")
                time.sleep(0.1)  # Brief pause on error

    def _database_worker(self, camera_id: int):
        """Handle database saves for specific camera asynchronously"""
        logger.info(f"[DATABASE] Camera {camera_id} database worker started")

        # Create database handler (SAME as optimized system) - Fixed to use create_database_handler
        from warehouse_database_handler import create_database_handler

        try:
            db_handler = create_database_handler()
            if not db_handler or not db_handler.is_connected():
                logger.warning(f"[DATABASE] Camera {camera_id}: Database not connected, worker will skip saves")
                db_handler = None
        except Exception as e:
            logger.error(f"[DATABASE] Camera {camera_id}: Failed to create database handler: {e}")
            db_handler = None

        saved_count = 0

        while self.running:
            try:
                # Get database task from camera-specific queue
                task = self.database_queues[camera_id].get(timeout=1.0)

                if db_handler and db_handler.is_connected():
                    # Save to database (blocking, but in separate thread!)
                    db_handler.save_detection_to_db(task['camera_id'], task['detection'])
                    saved_count += 1

                    # Update performance stats
                    self.performance_stats['database_saves_per_camera'][camera_id] = saved_count

                    logger.debug(f"[DATABASE] Camera {camera_id}: Saved detection {saved_count}")
                else:
                    logger.debug(f"[DATABASE] Camera {camera_id}: Skipped save (no database connection)")

            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"[DATABASE] Camera {camera_id} database error: {e}")
                time.sleep(0.1)  # Brief pause on error

    def _performance_monitor(self):
        """Monitor and log performance statistics"""
        logger.info("[MONITOR] Performance monitoring started")

        while self.running:
            time.sleep(30)  # Monitor every 30 seconds

            if not self.running:
                break

            try:
                logger.info("[PERFORMANCE] PARALLEL PIPELINE PERFORMANCE SUMMARY:")
                logger.info("=" * 60)

                # Calculate runtime
                runtime = time.time() - self.performance_stats['start_time'] if self.performance_stats['start_time'] else 0

                total_frames = 0
                total_database_saves = 0

                # Per-camera statistics
                for camera_id in self.active_cameras:
                    frames_processed = self.performance_stats['frames_processed_per_camera'][camera_id]
                    database_saves = self.performance_stats['database_saves_per_camera'][camera_id]
                    processing_times = self.performance_stats['processing_times_per_camera'][camera_id]

                    # Calculate FPS for this camera
                    camera_fps = frames_processed / runtime if runtime > 0 else 0

                    # Calculate average processing time
                    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

                    logger.info(f"   Camera {camera_id:2d}: {camera_fps:5.2f} FPS | {frames_processed:4d} frames | {database_saves:4d} DB saves | {avg_processing_time*1000:5.1f}ms avg")

                    total_frames += frames_processed
                    total_database_saves += database_saves

                # Total system statistics
                total_fps = total_frames / runtime if runtime > 0 else 0
                logger.info("=" * 60)
                logger.info(f"   TOTAL SYSTEM: {total_fps:5.2f} FPS | {total_frames:4d} frames | {total_database_saves:4d} DB saves")
                logger.info(f"   Runtime: {runtime:.1f}s | Cameras: {len(self.active_cameras)}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"[MONITOR] Performance monitoring error: {e}")

    def stop(self):
        """Stop the parallel pipeline system"""
        if not self.running:
            logger.warning("System is not running")
            return

        logger.info("🛑 STOPPING PARALLEL PIPELINE SYSTEM")
        self.running = False

        try:
            # Stop camera threads
            logger.info("[SHUTDOWN] Stopping camera threads...")
            self.camera_manager.stop_camera_threads()

            # Stop detection pool
            logger.info("[SHUTDOWN] Stopping detection workers...")
            self.detection_pool.stop_detection_workers()

            # Processing threads will stop automatically (daemon threads)
            logger.info("[SHUTDOWN] Processing threads stopping...")

            # Database workers will stop automatically (daemon threads)
            logger.info("[SHUTDOWN] Database workers stopping...")

            # Final performance summary
            self._log_final_stats()

            logger.info("✅ PARALLEL PIPELINE SYSTEM STOPPED")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    def _log_final_stats(self):
        """Log final performance statistics"""
        if self.performance_stats['start_time']:
            runtime = time.time() - self.performance_stats['start_time']

            total_frames = sum(self.performance_stats['frames_processed_per_camera'].values())
            total_database_saves = sum(self.performance_stats['database_saves_per_camera'].values())
            total_fps = total_frames / runtime if runtime > 0 else 0

            logger.info("=" * 80)
            logger.info("FINAL PARALLEL PIPELINE PERFORMANCE SUMMARY:")
            logger.info(f"   Total Runtime: {runtime:.1f} seconds")
            logger.info(f"   Total Frames Processed: {total_frames}")
            logger.info(f"   Total Database Saves: {total_database_saves}")
            logger.info(f"   Average System FPS: {total_fps:.2f}")
            logger.info(f"   Cameras Processed: {len(self.active_cameras)}")

            # Per-camera final stats
            for camera_id in self.active_cameras:
                frames = self.performance_stats['frames_processed_per_camera'][camera_id]
                saves = self.performance_stats['database_saves_per_camera'][camera_id]
                fps = frames / runtime if runtime > 0 else 0
                logger.info(f"   Camera {camera_id}: {fps:.2f} FPS ({frames} frames, {saves} saves)")

            logger.info("=" * 80)

    def run(self):
        """Run the parallel pipeline system (blocking)"""
        try:
            self.start()

            logger.info("🔄 PARALLEL PIPELINE SYSTEM RUNNING")
            logger.info("Press Ctrl+C to stop...")

            # Keep main thread alive
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n[INTERRUPT] Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.stop()
