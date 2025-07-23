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

# DeepSORT import (optional) - Try multiple import paths
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ DeepSORT available for tracking (deepsort_tracker)")
except ImportError:
    try:
        from deep_sort_realtime import DeepSort
        DEEPSORT_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("✅ DeepSORT available for tracking (direct import)")
    except ImportError as e:
        DEEPSORT_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.info(f"⚠️ DeepSORT import failed: {e}")
        logger.info("Will use SIFT tracking only")
except Exception as e:
    DEEPSORT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ DeepSORT import error: {e}")
    logger.info("Will use SIFT tracking only")

logger = logging.getLogger(__name__)

class ParallelPipelineSystem:
    """
    Parallel processing system with:
    - 11 camera threads (preprocessing)
    - 11 detection workers (GPU)
    - 11 processing threads (post-detection)
    - 11 database workers (async saves)
    """
    
    def __init__(self, active_cameras: List[int] = [1, 2], enable_gui: bool = False, gui_cameras: List[int] = None, use_deepsort: bool = False):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.gui_cameras = gui_cameras or []
        self.running = False
        self.use_deepsort = use_deepsort and DEEPSORT_AVAILABLE

        # Initialize SAME threading components as optimized system
        self.queue_manager = OptimizedQueueManager(max_cameras=len(active_cameras))
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = DetectionThreadPool(num_workers=1, queue_manager=self.queue_manager)

        # NEW: Per-camera database queues and workers
        self.database_queues = {}
        self.database_workers = {}
        self.processing_threads = {}

        # Initialize per-camera database queues
        for camera_id in active_cameras:
            self.database_queues[camera_id] = queue.Queue(maxsize=50)  # 50 database tasks per camera

        # Initialize DeepSORT trackers (if enabled)
        self.deepsort_trackers = {}
        self.seen_track_ids = {}  # Track which DeepSORT track IDs we've seen before
        if self.use_deepsort:
            for camera_id in active_cameras:
                try:
                    # Try different parameter combinations for different DeepSORT versions
                    # ✅ CRITICAL FIX: Force DeepSORT to use CPU to avoid GPU competition with Grounding DINO
                    try:
                        # Version 1: Try with model_type and CPU-only
                        self.deepsort_trackers[camera_id] = DeepSort(
                            model_type="osnet_x1_0",
                            max_age=30,
                            n_init=3,
                            max_iou_distance=0.7,
                            embedder_gpu=False,  # Force CPU for embedder
                            half_precision=False  # Disable FP16 for CPU
                        )
                    except TypeError:
                        # Version 2: Try without model_type but still CPU-only
                        self.deepsort_trackers[camera_id] = DeepSort(
                            max_age=30,
                            n_init=3,
                            max_iou_distance=0.7,
                            embedder_gpu=False,  # Force CPU for embedder
                            half_precision=False  # Disable FP16 for CPU
                        )
                    except Exception:
                        # Version 3: Minimal parameters with CPU-only
                        self.deepsort_trackers[camera_id] = DeepSort(
                            embedder_gpu=False,  # Force CPU for embedder
                            half_precision=False  # Disable FP16 for CPU
                        )
                    self.seen_track_ids[camera_id] = set()  # Track seen track IDs per camera
                    logger.info(f"✅ DeepSORT tracker initialized for Camera {camera_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize DeepSORT for Camera {camera_id}: {e}")
                    self.use_deepsort = False
                    break

            if self.use_deepsort:
                logger.info(f"🎯 DeepSORT tracking enabled for {len(active_cameras)} cameras")
            else:
                logger.info("⚠️ DeepSORT initialization failed, falling back to SIFT tracking")
        else:
            logger.info("📊 Using SIFT tracking (original system)")
        
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

    def _convert_to_deepsort_format(self, detections):
        """Convert detections to DeepSORT format"""
        deepsort_dets = []
        for det in detections:
            # DeepSORT expects: ([x1, y1, x2, y2], confidence)
            # Extract bbox from the 'bbox' key which contains [x1, y1, x2, y2]
            bbox = det.get('bbox', [0, 0, 0, 0])
            confidence = det.get('confidence', 0.8)
            deepsort_dets.append((bbox, confidence))
        return deepsort_dets

    def _convert_from_deepsort_format(self, tracks, camera_id: int, original_detections=None):
        """Convert DeepSORT tracks back to detection format, preserving physical coordinates"""
        final_detections = []
        for track in tracks:
            if not (hasattr(track, 'is_deleted') and track.is_deleted()):  # Include ALL active tracks (tentative + confirmed)
                bbox = track.to_ltwh()  # [x, y, width, height]

                # DEBUG: Log bbox values
                logger.debug(f"[DEEPSORT_BBOX] Camera {camera_id}: Track {track.track_id} bbox: {bbox}")

                # Generate proper global_id as integer (not string)
                global_id = int(track.track_id) + (camera_id * 1000)

                # Determine if this is a new or existing track
                track_id = int(track.track_id)
                if track_id in self.seen_track_ids[camera_id]:
                    tracking_status = 'existing'
                else:
                    tracking_status = 'new'
                    self.seen_track_ids[camera_id].add(track_id)

                # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                area = (x2 - x1) * (y2 - y1)

                # Create detection in the EXPECTED format for frontend/backend compatibility
                detection = {
                    'global_id': global_id,
                    'tracking_status': tracking_status,
                    'bbox': [x1, y1, x2, y2],                    # ✅ REQUIRED for frontend
                    'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], # ✅ REQUIRED for frontend
                    'confidence': track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.8,
                    'area': area,                                # ✅ REQUIRED for frontend
                    'shape_type': 'quadrangle',                  # ✅ REQUIRED for frontend
                    'track_id': track.track_id,
                    'tracking_method': 'deepsort'
                }

                # ✅ CRITICAL FIX: Copy physical coordinates from original detections
                # Since DeepSORT maintains detection order, use track index to match
                track_index = len(final_detections)  # Current index in the list

                if original_detections and track_index < len(original_detections):
                    # Use the detection at the same index
                    orig_det = original_detections[track_index]
                    detection.update({
                        'physical_x_ft': orig_det.get('physical_x_ft'),
                        'physical_y_ft': orig_det.get('physical_y_ft'),
                        'coordinate_status': orig_det.get('coordinate_status', 'SUCCESS'),
                        'physical_corners': orig_det.get('physical_corners'),
                        'real_center': orig_det.get('real_center')
                    })
                    logger.debug(f"[DEEPSORT] Camera {camera_id}: Preserved coordinates for track {track.track_id}: ({detection.get('physical_x_ft')}, {detection.get('physical_y_ft')})")
                else:
                    # No original detection available
                    detection.update({
                        'physical_x_ft': None,
                        'physical_y_ft': None,
                        'coordinate_status': 'NO_ORIGINAL',
                        'physical_corners': None,
                        'real_center': None
                    })
                    logger.warning(f"[DEEPSORT] Camera {camera_id}: No original detection for track {track.track_id} (index {track_index})")

                # DEBUG: Log final detection format
                logger.debug(f"[DEEPSORT_DETECTION] Camera {camera_id}: {detection['global_id']} -> bbox={detection['bbox']}, coords=({detection.get('physical_x_ft')}, {detection.get('physical_y_ft')})")

                final_detections.append(detection)
        return final_detections

    def _deepsort_tracking(self, camera_id: int, detections, frame):
        """Perform DeepSORT tracking"""
        try:
            # CRITICAL FIX: Handle empty detections
            if len(detections) == 0:
                logger.debug(f"[DEEPSORT] Camera {camera_id}: No detections to track, updating with empty list")
                # Still update tracker to age existing tracks
                tracks = self.deepsort_trackers[camera_id].update_tracks([], frame=frame)
                final_detections = self._convert_from_deepsort_format(tracks, camera_id, detections)
                return final_detections

            # Convert to DeepSORT format
            deepsort_dets = self._convert_to_deepsort_format(detections)

            # Update tracks
            tracks = self.deepsort_trackers[camera_id].update_tracks(deepsort_dets, frame=frame)

            # Convert back to our format, preserving original detection data
            final_detections = self._convert_from_deepsort_format(tracks, camera_id, detections)

            logger.info(f"[DEEPSORT_DEBUG] Camera {camera_id}: Final pipeline result: {len(final_detections)} detections going to database")
            logger.debug(f"[DEEPSORT] Camera {camera_id}: Tracked {len(final_detections)} objects from {len(detections)} detections")
            return final_detections

        except Exception as e:
            logger.error(f"[DEEPSORT] Camera {camera_id}: DeepSORT tracking failed: {e}")
            raise

    def _safe_tracking(self, camera_id: int, detections, frame, processor):
        """Safe tracking with fallback to SIFT"""
        if self.use_deepsort:
            try:
                # ✅ CRITICAL FIX: Frame is already fisheye-corrected AND resized by camera thread!
                # No additional processing needed - use frame as-is
                processed_frame = frame

                # Use DeepSORT tracking (now with empty detection handling)
                tracked_detections = self._deepsort_tracking(camera_id, detections, processed_frame)

                # ✅ CRITICAL FIX: Physical coordinates are ALREADY translated in main pipeline!
                # Do NOT translate again - this was causing double transformation!

                # Step 2: Color extraction using PROCESSED frame (coordinates match processed frame)
                for detection in tracked_detections:
                    try:
                        # Extract image region for color analysis using PROCESSED frame
                        bbox = detection['bbox']
                        x1, y1, x2, y2 = bbox
                        image_region = processed_frame[y1:y2, x1:x2]  # ✅ FIXED: Use processed frame with matching coordinates

                        # Extract dominant color from detected object (same as SIFT)
                        color_info = processor.color_extractor.extract_dominant_color(image_region)
                        detection.update(color_info)  # Add color data to detection

                        # Add similarity_score (DeepSORT doesn't have this, so set to 1.0)
                        if 'similarity_score' not in detection:
                            detection['similarity_score'] = 1.0

                    except Exception as e:
                        logger.error(f"[DEEPSORT] Camera {camera_id}: Color extraction failed for detection: {e}")
                        # Add default color info if extraction fails
                        detection.update({
                            'color_rgb': None,
                            'color_hsv': None,
                            'color_hex': None,
                            'color_confidence': 0.0,
                            'color_name': 'unknown',
                            'extraction_method': 'failed',
                            'similarity_score': 1.0
                        })

                return tracked_detections

            except Exception as e:
                logger.error(f"[FALLBACK] Camera {camera_id}: DeepSORT failed, using SIFT: {e}")
                # Fallback to SIFT
                return processor.assign_global_ids(detections, frame)
        else:
            # Use original SIFT tracking
            return processor.assign_global_ids(detections, frame)

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

                    # Stage 4: Object tracking (DeepSORT or SIFT)
                    logger.debug(f"[PROCESSING] Camera {camera_id}: Starting tracking with {len(processor.size_filtered_detections)} size-filtered detections")

                    processor.final_tracked_detections = self._safe_tracking(
                        camera_id, processor.size_filtered_detections, frame, processor
                    )

                    logger.debug(f"[PROCESSING] Camera {camera_id}: Tracking completed, got {len(processor.final_tracked_detections)} final detections")

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
