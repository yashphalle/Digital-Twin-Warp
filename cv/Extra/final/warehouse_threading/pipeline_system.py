#!/usr/bin/env python3
"""
Basic Pipeline Threading System
Orchestrates camera preprocessing and detection threading
"""

import logging
import time
import signal
import sys
from typing import List

from .queue_manager import QueueManager
from .camera_threads import CameraThreadManager  
from .detection_pool import DetectionThreadPool

logger = logging.getLogger(__name__)

class PipelineThreadingSystem:
    """Basic threaded system for Phase 1 testing"""
    
    def __init__(self, active_cameras: List[int] = [1, 2]):
        self.active_cameras = active_cameras
        self.running = False
        
        # Initialize threading components
        self.queue_manager = QueueManager(max_cameras=len(active_cameras))
        self.camera_manager = CameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = DetectionThreadPool(num_workers=3, queue_manager=self.queue_manager)
        
        # Setup signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"✅ Pipeline Threading System initialized for cameras: {active_cameras}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("🛑 Shutdown signal received")
        self.stop()

    def start(self):
        """Start the basic threaded system"""
        logger.info("🚀 Starting Pipeline Threading System...")
        logger.info("=" * 60)
        logger.info("PHASE 1: Camera Preprocessing + Detection Threading")
        logger.info(f"📹 Cameras: {self.active_cameras}")
        logger.info("🔍 Detection: 3 GPU threads")
        logger.info("🔄 Processing: Basic detection consumer")
        logger.info("⚡ Frame Skipping: Every 20th frame (20x speed boost)")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        self.running = True

        try:
            # Start detection workers first
            self.detection_pool.start_detection_workers()

            # Start camera threads
            self.camera_manager.start_camera_threads()

            # Start basic processing consumer
            import threading
            processing_thread = threading.Thread(
                target=self._basic_processing_consumer,
                name="BasicProcessing",
                daemon=True
            )
            processing_thread.start()
            logger.info("🔄 Started basic processing consumer")

            # Monitor system performance
            self._monitor_system()

        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            self.stop()

    def _basic_processing_consumer(self):
        """Complete processing consumer using existing CPUFrameProcessor"""
        logger.info("✅ Complete processing consumer started (using existing modules)")

        # Import existing frame processor
        from modules.frame_processor import CPUFrameProcessor
        from modules.filtering import DetectionFiltering
        from modules.coordinate_mapper import CoordinateMapper
        from modules.color_extractor import ObjectColorExtractor
        from modules.feature_database import CPUGlobalFeatureDatabase
        from warehouse_database_handler import WarehouseDatabaseHandler

        # Create frame processors for each camera (REUSE EXISTING LOGIC)
        processors = {}
        for camera_id in self.active_cameras:
            try:
                # Create frame processor
                frame_processor = CPUFrameProcessor(camera_id)

                # Initialize all components
                filtering = DetectionFiltering(camera_id)
                coord_converter = CoordinateMapper(camera_id=camera_id)
                coord_converter.load_calibration()
                color_extractor = ObjectColorExtractor()
                global_db = CPUGlobalFeatureDatabase(f"camera_{camera_id}_features.pkl")
                db_handler = WarehouseDatabaseHandler()

                # Inject components into frame processor
                frame_processor.inject_components(
                    fisheye_corrector=None,  # Already done in camera thread
                    pallet_detector=None,    # Already done in detection thread
                    filtering=filtering,
                    coordinate_mapper=coord_converter,
                    coordinate_mapper_initialized=True,
                    global_db=global_db,
                    color_extractor=color_extractor,
                    db_handler=db_handler,
                    display_manager=None     # Using web GUI instead
                )

                processors[camera_id] = frame_processor
                logger.info(f"✅ Frame processor initialized for Camera {camera_id}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize processor for Camera {camera_id}: {e}")

        processed_count = 0
        saved_count = 0

        while self.running:
            try:
                # Get frame with detection results
                frame_data = self.queue_manager.get_frame('detection_to_processing', timeout=1.0)
                if frame_data is None:
                    continue

                # Extract detection results
                raw_detections = frame_data.metadata.get('raw_detections', [])
                camera_id = frame_data.camera_id
                frame = frame_data.frame

                if not raw_detections:
                    logger.debug(f"🔍 Camera {camera_id}: No detections found")
                    continue

                # Get processor for this camera
                if camera_id not in processors:
                    logger.warning(f"❌ No processor for Camera {camera_id}")
                    continue

                processor = processors[camera_id]

                # REUSE EXISTING PROCESSING PIPELINE
                try:
                    # Set raw detections (skip detection step since already done)
                    processor.raw_detections = raw_detections

                    # Apply existing processing pipeline (filtering + coordinates + features + database)
                    frame_height, frame_width = frame.shape[:2]

                    # Stage 1: Apply filtering
                    processor.area_filtered_detections = processor.filtering.apply_area_filter(raw_detections)
                    processor.grid_filtered_detections = processor.filtering.apply_grid_cell_filter(processor.area_filtered_detections)

                    # Stage 2: Physical coordinate translation (REUSE EXISTING METHOD)
                    processor.grid_filtered_detections = processor.translate_to_physical_coordinates(
                        processor.grid_filtered_detections, frame_width, frame_height
                    )

                    # Stage 3: Physical size filtering
                    processor.size_filtered_detections = processor.filtering.apply_physical_size_filter(processor.grid_filtered_detections)

                    # Stage 4: SIFT feature matching and global ID assignment (REUSE EXISTING METHOD)
                    processor.final_tracked_detections = processor.assign_global_ids(processor.size_filtered_detections, frame)

                    # Stage 5: Save to database (REUSE EXISTING METHOD)
                    if processor.db_handler and processor.db_handler.is_connected():
                        for detection in processor.final_tracked_detections:
                            processor.db_handler.save_detection_to_db(camera_id, detection)
                            saved_count += 1

                    processed_count += 1

                    # Log results using existing method
                    counts = processor.get_detection_counts()
                    logger.info(f"🔍 Camera {camera_id}: {counts['raw_detections']} raw → {counts['area_filtered_detections']} area → {counts['grid_filtered_detections']} grid → {counts['size_filtered_detections']} size → {counts['final_tracked_detections']} final")
                    logger.info(f"📊 Total processed: {processed_count}, Total saved: {saved_count}, New: {counts['new_objects']}, Existing: {counts['existing_objects']}")

                except Exception as e:
                    logger.error(f"❌ Processing pipeline error for Camera {camera_id}: {e}")

            except Exception as e:
                logger.error(f"❌ Processing consumer error: {e}")
                time.sleep(0.1)

    def _monitor_system(self):
        """Monitor system performance and log statistics"""
        last_stats_time = time.time()
        
        while self.running:
            try:
                time.sleep(1)  # Monitor every second
                
                # Log statistics every 10 seconds
                if time.time() - last_stats_time >= 10:
                    self._log_performance_stats()
                    self._log_gpu_utilization()  # Add GPU monitoring
                    last_stats_time = time.time()
                    
            except KeyboardInterrupt:
                break

    def _log_performance_stats(self):
        """Log current performance statistics"""
        # Queue statistics
        queue_stats = self.queue_manager.get_queue_stats()
        
        # Camera statistics  
        camera_stats = self.camera_manager.get_camera_stats()
        
        # Detection statistics
        detection_stats = self.detection_pool.get_detection_stats()
        
        logger.info("📊 PERFORMANCE STATS:")
        logger.info(f"   Frames Queued: {queue_stats['global']['frames_queued']}")
        logger.info(f"   Frames Processed: {queue_stats['global']['frames_processed']}")
        logger.info(f"   Queue Overflows: {queue_stats['global']['queue_overflows']}")
        
        # Queue utilization
        for queue_name, stats in queue_stats.items():
            if queue_name != 'global':
                utilization = stats['utilization'] * 100
                logger.info(f"   {queue_name}: {stats['size']}/{stats['maxsize']} ({utilization:.1f}%)")
        
        # Detection performance
        if detection_stats['avg_detection_time'] > 0:
            logger.info(f"   Avg Detection Time: {detection_stats['avg_detection_time']:.3f}s")
            logger.info(f"   Total Detections: {detection_stats['total_detections']}")
            
            # Worker load distribution
            total_processed = sum(detection_stats['worker_loads'].values())
            if total_processed > 0:
                load_distribution = []
                for worker_id, load in detection_stats['worker_loads'].items():
                    percentage = (load / total_processed) * 100
                    load_distribution.append(f"W{worker_id}: {percentage:.1f}%")
                logger.info(f"   Worker Loads: {', '.join(load_distribution)}")
        
        # Camera status
        connected_cameras = sum(1 for stats in camera_stats.values() if stats['connected'])
        logger.info(f"   Connected Cameras: {connected_cameras}/{len(self.active_cameras)}")
        
        # Calculate approximate FPS
        if queue_stats['global']['frames_processed'] > 0:
            runtime = time.time() - getattr(self, 'start_time', time.time())
            if runtime > 0:
                fps = queue_stats['global']['frames_processed'] / runtime
                logger.info(f"   Approximate FPS: {fps:.2f}")

        # Store start time if not set
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()

    def stop(self):
        """Stop the system gracefully"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping Pipeline Threading System...")
        self.running = False
        
        # Stop components in reverse order
        self.camera_manager.stop_camera_threads()
        self.detection_pool.stop_detection_workers()
        
        # Clear queues
        self.queue_manager.clear_all_queues()
        
        logger.info("✅ Pipeline Threading System stopped successfully")

    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'running': self.running,
            'active_cameras': self.active_cameras,
            'queue_stats': self.queue_manager.get_queue_stats(),
            'camera_stats': self.camera_manager.get_camera_stats(),
            'detection_stats': self.detection_pool.get_detection_stats()
        }

    def _log_gpu_utilization(self):
        """Log GPU utilization statistics"""
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory usage
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100

                logger.info("🚀 GPU UTILIZATION:")
                logger.info(f"   Memory Used: {gpu_memory_used:.2f}GB / {gpu_memory_total:.1f}GB")
                logger.info(f"   Memory Utilization: {gpu_utilization:.1f}%")

                # Try to get GPU compute utilization via nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        compute_util = result.stdout.strip()
                        logger.info(f"   Compute Utilization: {compute_util}%")
                except:
                    pass  # nvidia-smi not available

            else:
                logger.warning("❌ CUDA not available for GPU monitoring")

        except Exception as e:
            logger.error(f"❌ GPU monitoring error: {e}")
