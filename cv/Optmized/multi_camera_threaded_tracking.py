#!/usr/bin/env python3
"""
Multi-Camera Multi-Threaded Warehouse Tracking System
Handles 3 cameras (8, 9, 10) simultaneously with optimized threading
Architecture:
- Per Camera: Capture → Detection → Features → Database
- Shared: Global Database, Display Grid, Performance Monitor
- Total: ~21 threads (7 per camera) + shared threads
"""

import cv2
import numpy as np
import logging
import sys
import os
import pickle
import time
import json
import torch
import threading
import queue
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraPerformanceMonitor:
    """Performance monitor for multiple cameras"""
    
    def __init__(self, camera_ids: List[int]):
        self.camera_ids = camera_ids
        self.camera_stats = {}
        
        # Initialize stats for each camera
        for camera_id in camera_ids:
            self.camera_stats[camera_id] = {
                'frame_count': 0,
                'detection_count': 0,
                'feature_count': 0,
                'fps': 0.0,
                'detection_fps': 0.0,
                'feature_fps': 0.0,
                'thread_status': {
                    'capture': 'idle',
                    'detection': 'idle',
                    'features': 'idle',
                    'filtering': 'idle'
                }
            }
        
        # Global stats
        self.total_fps = 0.0
        self.total_detections = 0
        self.total_objects = 0
        self.gpu_memory_usage = 0.0
        self.cpu_usage = 0.0
        
        self.start_time = time.time()
        self.last_report_time = time.time()
        self._lock = threading.Lock()
    
    def update_camera_stats(self, camera_id: int, stat_type: str, value: int = 1):
        """Update camera-specific statistics"""
        with self._lock:
            if camera_id in self.camera_stats:
                if stat_type == 'frame':
                    self.camera_stats[camera_id]['frame_count'] += value
                elif stat_type == 'detection':
                    self.camera_stats[camera_id]['detection_count'] += value
                elif stat_type == 'feature':
                    self.camera_stats[camera_id]['feature_count'] += value
    
    def update_thread_status(self, camera_id: int, thread_name: str, status: str):
        """Update thread status for specific camera"""
        with self._lock:
            if camera_id in self.camera_stats:
                self.camera_stats[camera_id]['thread_status'][thread_name] = status
    
    def calculate_performance(self):
        """Calculate and log performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_report_time
        
        if elapsed >= 5.0:  # Report every 5 seconds
            with self._lock:
                total_fps = 0.0
                total_detection_fps = 0.0
                total_feature_fps = 0.0
                
                logger.info("🎥 MULTI-CAMERA PERFORMANCE REPORT:")
                logger.info("=" * 60)
                
                for camera_id in self.camera_ids:
                    stats = self.camera_stats[camera_id]
                    
                    # Calculate FPS for this camera
                    camera_fps = stats['frame_count'] / elapsed
                    detection_fps = stats['detection_count'] / elapsed
                    feature_fps = stats['feature_count'] / elapsed
                    
                    # Update camera stats
                    stats['fps'] = camera_fps
                    stats['detection_fps'] = detection_fps
                    stats['feature_fps'] = feature_fps
                    
                    # Add to totals
                    total_fps += camera_fps
                    total_detection_fps += detection_fps
                    total_feature_fps += feature_fps
                    
                    # Log camera performance
                    logger.info(f"📹 Camera {camera_id}:")
                    logger.info(f"   FPS: {camera_fps:.1f} | Detection: {detection_fps:.1f} | Features: {feature_fps:.1f}")
                    
                    # Thread status
                    thread_status = " | ".join([f"{k}:{v}" for k, v in stats['thread_status'].items()])
                    logger.info(f"   Threads: {thread_status}")
                    
                    # Reset counters
                    stats['frame_count'] = 0
                    stats['detection_count'] = 0
                    stats['feature_count'] = 0
                
                # Global performance
                self.total_fps = total_fps
                
                # GPU memory
                if torch.cuda.is_available():
                    self.gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
                
                logger.info("=" * 60)
                logger.info(f"🚀 TOTAL PERFORMANCE:")
                logger.info(f"   Combined FPS: {total_fps:.1f}")
                logger.info(f"   Combined Detection FPS: {total_detection_fps:.1f}")
                logger.info(f"   Combined Feature FPS: {total_feature_fps:.1f}")
                logger.info(f"   GPU Memory: {self.gpu_memory_usage:.2f}GB")
                logger.info("=" * 60)
                
                self.last_report_time = current_time

class SharedGlobalDatabase:
    """Shared global database for all cameras"""
    
    def __init__(self, database_file: str = "multi_camera_global_features.pkl"):
        self.database_file = database_file
        self.features = {}
        self.next_global_id = 2000  # Start from 2000 for multi-camera
        self._lock = threading.Lock()
        
        # Load existing database
        self.load_database()
        
        # SIFT detector (shared across cameras)
        self.sift = cv2.SIFT_create(
            nfeatures=250,  # Reduced for multi-camera performance
            contrastThreshold=0.05,
            edgeThreshold=12
        )
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=25)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Optimized parameters for multi-camera
        self.similarity_threshold = 0.2
        self.min_matches = 6
        self.max_disappeared_frames = 45  # Longer for multi-camera
        
        # Thread pool for database operations
        self.db_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="DatabaseWorker")
        
        logger.info(f"🗄️ Shared global database initialized with {len(self.features)} objects")
    
    def load_database(self):
        """Load shared database"""
        try:
            if os.path.exists(self.database_file):
                with self._lock:
                    with open(self.database_file, 'rb') as f:
                        data = pickle.load(f)
                        self.features = data.get('features', {})
                        self.next_global_id = data.get('next_id', 2000)
                logger.info(f"Loaded {len(self.features)} objects from shared database")
            else:
                self.features = {}
                self.next_global_id = 2000
        except Exception as e:
            logger.error(f"Error loading shared database: {e}")
            self.features = {}
            self.next_global_id = 2000
    
    def save_database_async(self):
        """Asynchronous database saving"""
        def save_worker():
            try:
                with self._lock:
                    data = {
                        'features': self.features.copy(),
                        'next_id': self.next_global_id,
                        'last_updated': datetime.now().isoformat(),
                        'multi_camera': True,
                        'cameras': [8, 9, 10]
                    }
                
                with open(self.database_file, 'wb') as f:
                    pickle.dump(data, f)
                    
            except Exception as e:
                logger.error(f"Error saving shared database: {e}")
        
        self.db_executor.submit(save_worker)
    
    def extract_features(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT features (thread-safe)"""
        if image_region is None or image_region.size == 0:
            return None
        
        try:
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region
            
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) >= self.min_matches:
                return descriptors
            return None
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate feature similarity"""
        if features1 is None or features2 is None:
            return 0.0
        
        if len(features1) < 2 or len(features2) < 2:
            return 0.0
        
        try:
            # Limit features for performance
            max_features = 80
            if len(features1) > max_features:
                features1 = features1[:max_features]
            if len(features2) > max_features:
                features2 = features2[:max_features]
            
            matches = self.flann.knnMatch(features1, features2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= self.min_matches:
                similarity = len(good_matches) / min(len(features1), len(features2))
                return min(similarity, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def find_matching_object(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Find matching object in shared database"""
        best_match_id = None
        best_similarity = 0.0
        
        with self._lock:
            feature_items = list(self.features.items())
        
        for global_id, feature_data in feature_items:
            stored_features = feature_data['features']
            similarity = self.calculate_similarity(query_features, stored_features)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match_id = global_id
                best_similarity = similarity
        
        return best_match_id, best_similarity
    
    def add_new_object(self, features: np.ndarray, detection_info: Dict, camera_id: int) -> int:
        """Add new object to shared database"""
        with self._lock:
            global_id = self.next_global_id
            self.next_global_id += 1
            
            feature_data = {
                'features': features,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'times_seen': 1,
                'disappeared_frames': 0,
                'detection_info': detection_info,
                'physical_locations': [],
                'camera_detections': {camera_id: 1},  # Track which cameras see this object
                'multi_camera': True
            }
            
            # Add physical coordinates if available
            if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                    feature_data['physical_locations'].append({
                        'timestamp': datetime.now().isoformat(),
                        'x_ft': detection_info['physical_x_ft'],
                        'y_ft': detection_info['physical_y_ft'],
                        'camera_id': camera_id
                    })
            
            self.features[global_id] = feature_data
        
        self.save_database_async()
        logger.info(f"🆕 NEW MULTI-CAM ID: {global_id} (Camera {camera_id})")
        return global_id
    
    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict, camera_id: int):
        """Update existing object in shared database"""
        with self._lock:
            if global_id in self.features:
                feature_data = self.features[global_id]
                
                feature_data['features'] = features
                feature_data['last_seen'] = datetime.now().isoformat()
                feature_data['times_seen'] += 1
                feature_data['disappeared_frames'] = 0
                
                # Update camera detection count
                if camera_id in feature_data['camera_detections']:
                    feature_data['camera_detections'][camera_id] += 1
                else:
                    feature_data['camera_detections'][camera_id] = 1
                
                # Add physical coordinates
                if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                    if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                        feature_data['physical_locations'].append({
                            'timestamp': datetime.now().isoformat(),
                            'x_ft': detection_info['physical_x_ft'],
                            'y_ft': detection_info['physical_y_ft'],
                            'camera_id': camera_id
                        })
                        
                        # Keep only recent locations
                        if len(feature_data['physical_locations']) > 30:
                            feature_data['physical_locations'] = feature_data['physical_locations'][-30:]
        
        self.save_database_async()
        logger.info(f"🔄 UPDATED MULTI-CAM ID: {global_id} (Camera {camera_id})")
    
    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict, camera_id: int) -> Tuple[int, str, float]:
        """Assign global ID for multi-camera tracking"""
        features = self.extract_features(image_region)
        if features is None:
            return -1, 'failed', 0.0
        
        match_id, similarity = self.find_matching_object(features)
        
        if match_id is not None:
            self.update_object(match_id, features, detection_info, camera_id)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info, camera_id)
            return new_id, 'new', 1.0
    
    def cleanup(self):
        """Cleanup database thread pool"""
        self.db_executor.shutdown(wait=True)

class CameraCoordinateMapper:
    """Thread-safe coordinate mapping for individual cameras"""

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        self._lock = threading.Lock()

        self.load_calibration()

    def load_calibration(self):
        """Load camera-specific calibration"""
        filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"

        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)

            with self._lock:
                image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
                real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)

                if len(image_corners) != 4 or len(real_world_corners) != 4:
                    raise ValueError("Calibration must contain exactly 4 corner points")

                self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
                self.is_calibrated = True

                logger.info(f"✅ Camera {self.camera_id} coordinate mapper loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load calibration for camera {self.camera_id}: {e}")
            self.is_calibrated = False

    def pixel_to_real_batch(self, pixel_points: np.ndarray) -> np.ndarray:
        """Batch coordinate transformation"""
        if not self.is_calibrated:
            return np.full((len(pixel_points), 2), np.nan)

        try:
            with self._lock:
                points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
                transformed_points = cv2.perspectiveTransform(points_reshaped, self.homography_matrix)
                return transformed_points.reshape(-1, 2)
        except Exception as e:
            logger.error(f"Camera {self.camera_id} coordinate transformation failed: {e}")
            return np.full((len(pixel_points), 2), np.nan)

class MultiThreadedDetectionPool:
    """Multi-threaded detection pool with multiple detector instances"""

    def __init__(self, num_detectors=2, camera_ids=None):
        self.num_detectors = num_detectors
        self.detectors = []
        self.detector_locks = []
        self.detector_available = []
        self.initialized = False
        self.camera_ids = camera_ids or []  # Store camera IDs for queue initialization

        # Performance optimization - reduce detection frequency
        self.detection_interval = 1  # Process every frame for maximum detection
        self.frame_counters = {}  # Track frame counts per camera

        # Detection queue and worker threads
        self.detection_queue = queue.Queue(maxsize=50)
        self.result_queues = {}  # Per-camera result queues
        self.detection_workers = []
        self.running = False

        self._initialize_detector_pool()
        self._initialize_camera_queues()  # Initialize queues for all cameras

    def _initialize_camera_queues(self):
        """Initialize result queues for all cameras to prevent race conditions"""
        for camera_id in self.camera_ids:
            self.frame_counters[camera_id] = 0
            self.result_queues[camera_id] = queue.Queue(maxsize=10)
            logger.info(f"🔧 Initialized result queue for Camera {camera_id}")

    def _initialize_detector_pool(self):
        """Initialize SHARED detector with multiple worker threads for GPU optimization"""
        try:
            # Import here to avoid issues
            import sys
            sys.path.append('..')
            from pallet_detector_simple import SimplePalletDetector

            logger.info(f"🔧 Initializing SHARED detector with {self.num_detectors} worker threads...")
            logger.info("💡 GPU OPTIMIZATION: Using 1 shared model instead of multiple models")

            # Create ONE shared detector instance to save GPU memory
            self.shared_detector = SimplePalletDetector()
            self.shared_detector.confidence_threshold = 0.1  # Lower threshold for better detection
            self.shared_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet", "pallet with cargo", "loaded pallet"]
            self.shared_detector.current_prompt = "pallet wrapped in plastic"

            # Create locks for thread-safe access to the shared detector
            self.detector_lock = threading.Lock()

            # Clear the old detector arrays (not needed with shared model)
            self.detectors = [self.shared_detector] * self.num_detectors  # Reference to same detector
            self.detector_locks = [self.detector_lock] * self.num_detectors  # Same lock
            self.detector_available = [True] * self.num_detectors

            logger.info(f"✅ Shared detector initialized for {self.num_detectors} worker threads")

            self.initialized = True
            logger.info(f"🚀 GPU-optimized detection pool initialized with 1 shared model + {self.num_detectors} workers")

        except Exception as e:
            logger.error(f"❌ Failed to initialize detection pool: {e}")
            self.initialized = False

    def start_detection_workers(self):
        """Start detection worker threads"""
        self.running = True

        # Create detection worker threads
        for i in range(self.num_detectors):
            worker = threading.Thread(target=self._detection_worker, args=(i,), daemon=True)
            worker.start()
            self.detection_workers.append(worker)

        logger.info(f"🧵 Started {self.num_detectors} detection worker threads")

    def stop_detection_workers(self):
        """Stop detection worker threads"""
        self.running = False

        # Add poison pills to stop workers
        for _ in range(self.num_detectors):
            try:
                self.detection_queue.put(None, timeout=1.0)
            except:
                pass

    def _detection_worker(self, detector_id: int):
        """GPU-optimized detection worker thread using shared model"""
        logger.info(f"🔍 Detection worker {detector_id} started (GPU-OPTIMIZED)")

        while self.running:
            try:
                # Get detection task
                task = self.detection_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break

                frame, camera_id, task_id = task

                # Use shared detector with thread-safe locking
                with self.detector_lock:
                    # Perform detection using shared model
                    detections = self.shared_detector.detect_pallets(frame)

                # Debug logging for detection
                if len(detections) > 0:
                    logger.info(f"🔍 Worker {detector_id} - Camera {camera_id}: Detected {len(detections)} objects")

                # Add camera ID to each detection
                for detection in detections:
                    detection['camera_id'] = camera_id
                    # Calculate center if not present
                    if 'center' not in detection:
                        bbox = detection['bbox']
                        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        detection['center'] = center

                # Send result back
                if camera_id in self.result_queues:
                    try:
                        self.result_queues[camera_id].put((task_id, detections), timeout=0.1)
                        logger.info(f"✅ Worker {detector_id}: Sent {len(detections)} detections to Camera {camera_id}")
                    except queue.Full:
                        logger.warning(f"⚠️ Worker {detector_id}: Result queue full for Camera {camera_id}")
                    except Exception as e:
                        logger.error(f"❌ Worker {detector_id}: Failed to send results to Camera {camera_id}: {e}")
                else:
                    logger.error(f"❌ Worker {detector_id}: No result queue for Camera {camera_id}!")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection worker {detector_id} error: {e}")

        logger.info(f"🔍 Detection worker {detector_id} stopped")

    def detect_pallets_async(self, frame: np.ndarray, camera_id: int) -> Optional[List[Dict]]:
        """Asynchronous detection using worker pool"""
        if not self.initialized or not self.running:
            return []

        # Frame counter should already be initialized
        if camera_id not in self.frame_counters:
            logger.warning(f"⚠️ Camera {camera_id} not properly initialized in detection pool!")
            return []

        self.frame_counters[camera_id] += 1

        # Process every frame for maximum detection rate
        if self.frame_counters[camera_id] % self.detection_interval != 0:
            return []  # Return empty list for skipped frames

        try:
            # Submit detection task
            task_id = self.frame_counters[camera_id]
            task = (frame.copy(), camera_id, task_id)
            self.detection_queue.put(task, timeout=0.01)  # Non-blocking

            # Try to get ANY available result (async - may be from previous frame)
            try:
                result_task_id, detections = self.result_queues[camera_id].get(timeout=0.01)
                logger.info(f"📥 Camera {camera_id}: Received {len(detections)} detections from worker (task {result_task_id})")
                return detections
            except queue.Empty:
                # No result ready yet - this is normal for async processing
                return []

        except queue.Full:
            return []  # Detection queue full, skip frame
        except Exception as e:
            logger.error(f"Async detection failed for camera {camera_id}: {e}")
            return []

class CameraWorker:
    """Individual camera processing worker"""

    def __init__(self, camera_id: int, shared_db: SharedGlobalDatabase, performance_monitor: MultiCameraPerformanceMonitor, shared_detector: MultiThreadedDetectionPool, parent_tracker=None):
        self.camera_id = camera_id
        self.shared_db = shared_db
        self.performance_monitor = performance_monitor
        self.shared_detector = shared_detector
        self.parent_tracker = parent_tracker

        # Get camera configuration
        from cv.configs.config import Config
        from cv.configs.warehouse_config import get_warehouse_config

        warehouse_config = get_warehouse_config()

        if str(camera_id) in warehouse_config.camera_zones:
            camera_zone = warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = camera_zone.camera_name
            self.rtsp_url = camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

        # Camera-specific components
        self.coordinate_mapper = CameraCoordinateMapper(camera_id)

        # Detection parameters
        self.MIN_AREA = 8000   # Slightly reduced for multi-camera
        self.MAX_AREA = 120000
        self.CELL_SIZE = 45    # Slightly larger for multi-camera

        # Threading - Optimized 2-thread architecture per camera
        self.running = False
        self.threads = {}

        # Queues for optimized pipeline
        self.raw_frame_queue = queue.Queue(maxsize=5)      # Raw frames from capture
        self.detection_result_queue = queue.Queue(maxsize=5)  # Detection results
        self.result_queue = queue.Queue(maxsize=3)         # Final processed results

        # Statistics
        self.detections_processed = 0
        self.objects_tracked = 0

        # Frame skipping for performance
        self.capture_frame_skip = 2  # Skip every 2nd frame in capture
        self.capture_frame_counter = 0

        logger.info(f"🎥 Camera {camera_id} worker initialized: {self.camera_name}")

    def capture_thread(self):
        """Frame capture thread for this camera"""
        self.performance_monitor.update_thread_status(self.camera_id, 'capture', 'starting')

        # Connect to camera
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error(f"Camera {self.camera_id}: Failed to connect to {self.rtsp_url}")
            return

        from cv.configs.config import Config
        cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.RTSP_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)

        logger.info(f"🎥 Camera {self.camera_id} capture started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status(self.camera_id, 'capture', 'capturing')

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # Frame skipping for performance
                self.capture_frame_counter += 1
                if self.capture_frame_counter % self.capture_frame_skip != 0:
                    continue  # Skip this frame

                # Resize for processing - smaller for better performance
                height, width = frame.shape[:2]
                if width > 1200:  # Even smaller for multi-camera performance
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Add to queue (optimized for detection throughput)
                try:
                    self.raw_frame_queue.put(frame, timeout=0.01)  # Faster timeout for higher throughput
                    self.performance_monitor.update_camera_stats(self.camera_id, 'frame')
                except:
                    pass  # Queue full, skip frame

                # Also store frame for display continuity (every 10th frame to avoid memory issues)
                if self.capture_frame_counter % 10 == 0:
                    # Store in parent tracker for display
                    if hasattr(self, 'parent_tracker') and self.parent_tracker:
                        self.parent_tracker.last_frames[self.camera_id] = frame.copy()

                self.performance_monitor.update_thread_status(self.camera_id, 'capture', 'idle')
                time.sleep(0.02)  # Small delay

            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {e}")
                time.sleep(0.1)

        cap.release()
        logger.info(f"🎥 Camera {self.camera_id} capture stopped")

    def detection_thread(self):
        """Detection thread for this camera using shared detector"""
        self.performance_monitor.update_thread_status(self.camera_id, 'detection', 'starting')
        logger.info(f"🔍 Camera {self.camera_id} detection started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status(self.camera_id, 'detection', 'waiting')

                frame = self.raw_frame_queue.get(timeout=1.0)

                self.performance_monitor.update_thread_status(self.camera_id, 'detection', 'detecting')

                # Use multi-threaded detection pool
                detections = self.shared_detector.detect_pallets_async(frame, self.camera_id)

                # DEBUG: Log what detect_pallets_async actually returns
                logger.info(f"🔍 Camera {self.camera_id} detection_thread: detect_pallets_async returned {len(detections) if detections else 0} detections")

                # Package result
                detection_result = {
                    'frame': frame,
                    'detections': detections,
                    'camera_id': self.camera_id,
                    'timestamp': time.time()
                }

                try:
                    self.detection_result_queue.put(detection_result, timeout=0.01)  # Faster for throughput
                    self.performance_monitor.update_camera_stats(self.camera_id, 'detection')
                except:
                    pass

                self.performance_monitor.update_thread_status(self.camera_id, 'detection', 'idle')

            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Camera {self.camera_id} detection error: {e}")
                time.sleep(0.1)

        logger.info(f"🔍 Camera {self.camera_id} detection stopped")

    def processing_thread(self):
        """Processing thread for filtering, coordinates, and tracking"""
        self.performance_monitor.update_thread_status(self.camera_id, 'features', 'starting')
        logger.info(f"🔧 Camera {self.camera_id} processing started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status(self.camera_id, 'features', 'waiting')

                detection_result = self.detection_result_queue.get(timeout=1.0)
                frame = detection_result['frame']
                detections = detection_result['detections']

                self.performance_monitor.update_thread_status(self.camera_id, 'features', 'processing')

                # Apply filtering with debug logging
                logger.info(f"🔧 Camera {self.camera_id}: Processing {len(detections)} raw detections")
                area_filtered = [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]
                logger.info(f"🔧 Camera {self.camera_id}: After area filter: {len(area_filtered)} detections")
                grid_filtered = self._apply_grid_filter(area_filtered)
                logger.info(f"🔧 Camera {self.camera_id}: After grid filter: {len(grid_filtered)} detections")

                # Coordinate translation
                frame_height, frame_width = frame.shape[:2]
                coordinate_filtered = self._translate_coordinates_batch(grid_filtered, frame_width, frame_height)
                logger.info(f"🔧 Camera {self.camera_id}: After coordinate translation: {len(coordinate_filtered)} detections")

                # Feature extraction and global ID assignment
                tracked_detections = []
                for detection in coordinate_filtered:
                    # Extract region for features
                    region = self._extract_detection_region(frame, detection['bbox'])

                    # Assign global ID using shared database
                    global_id, status, similarity = self.shared_db.assign_global_id(
                        region, detection, self.camera_id
                    )

                    detection['global_id'] = global_id
                    detection['tracking_status'] = status
                    detection['similarity_score'] = similarity
                    detection['camera_id'] = self.camera_id

                    tracked_detections.append(detection)
                    self.detections_processed += 1

                # Package final result
                final_result = {
                    'camera_id': self.camera_id,
                    'camera_name': self.camera_name,
                    'frame': frame,
                    'raw_detections': detections,
                    'area_filtered': area_filtered,
                    'grid_filtered': grid_filtered,
                    'tracked_detections': tracked_detections,
                    'timestamp': detection_result['timestamp']
                }

                # Debug logging for display
                if len(tracked_detections) > 0:
                    logger.info(f"📺 Camera {self.camera_id}: Sending {len(tracked_detections)} tracked detections to display")

                try:
                    self.result_queue.put(final_result, timeout=0.05)
                    self.performance_monitor.update_camera_stats(self.camera_id, 'feature')
                except:
                    pass

                self.performance_monitor.update_thread_status(self.camera_id, 'features', 'idle')

            except queue.Empty:
                # No detection result available, continue
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Camera {self.camera_id} processing error: {e}")
                time.sleep(0.1)

        logger.info(f"🔧 Camera {self.camera_id} processing stopped")

    def _apply_grid_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply grid cell filtering"""
        if len(detections) <= 1:
            return detections

        # Calculate grid cells
        for detection in detections:
            center = detection['center']
            cell_x = int(center[0] // self.CELL_SIZE)
            cell_y = int(center[1] // self.CELL_SIZE)
            detection['grid_cell'] = (cell_x, cell_y)

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        occupied_cells = set()
        accepted = []

        for detection in sorted_detections:
            cell = detection['grid_cell']

            # Check 3x3 neighborhood
            conflict = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbor_cell in occupied_cells:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                occupied_cells.add(cell)
                accepted.append(detection)

        return accepted

    def _translate_coordinates_batch(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """Batch coordinate translation"""
        if not self.coordinate_mapper.is_calibrated or not detections:
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'MAPPER_NOT_AVAILABLE'
            return detections

        try:
            centers = [detection['center'] for detection in detections]
            centers_array = np.array(centers, dtype=np.float32)

            # Scale to calibration frame size
            scale_x = 3840 / frame_width
            scale_y = 2160 / frame_height
            scaled_centers = centers_array * np.array([scale_x, scale_y])

            # Batch transformation
            physical_coords = self.coordinate_mapper.pixel_to_real_batch(scaled_centers)

            # Assign results
            for i, detection in enumerate(detections):
                if i < len(physical_coords) and not np.isnan(physical_coords[i]).any():
                    detection['physical_x_ft'] = round(float(physical_coords[i][0]), 2)
                    detection['physical_y_ft'] = round(float(physical_coords[i][1]), 2)
                    detection['coordinate_status'] = 'SUCCESS'
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'CONVERSION_FAILED'

            return detections

        except Exception as e:
            logger.error(f"Camera {self.camera_id} coordinate translation failed: {e}")
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'ERROR'
            return detections

    def _extract_detection_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract detection region"""
        try:
            x1, y1, x2, y2 = bbox
            height, width = frame.shape[:2]

            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))

            if x2 <= x1 or y2 <= y1:
                return np.array([])

            return frame[y1:y2, x1:x2]

        except Exception as e:
            logger.error(f"Camera {self.camera_id} region extraction failed: {e}")
            return np.array([])

    def start_camera_threads(self):
        """Start all threads for this camera"""
        self.running = True

        # Keep current 3-thread architecture but optimize queues
        self.threads['capture'] = threading.Thread(target=self.capture_thread, name=f"Cam{self.camera_id}_Capture")
        self.threads['detection'] = threading.Thread(target=self.detection_thread, name=f"Cam{self.camera_id}_Detection")
        self.threads['processing'] = threading.Thread(target=self.processing_thread, name=f"Cam{self.camera_id}_Processing")

        for thread_name, thread in self.threads.items():
            thread.daemon = True
            thread.start()
            logger.info(f"✅ Camera {self.camera_id} {thread_name} thread started")
            time.sleep(0.1)

    def stop_camera_threads(self):
        """Stop all threads for this camera"""
        self.running = False

        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=1.0)

        logger.info(f"🛑 Camera {self.camera_id} threads stopped")

class MultiCameraThreadedTracker:
    """Main multi-camera tracking system"""

    def __init__(self, camera_ids: List[int] = [8, 9, 10]):
        self.camera_ids = camera_ids
        self.camera_workers = {}
        self.running = False

        # Shared components
        self.shared_db = SharedGlobalDatabase()
        self.shared_detector = MultiThreadedDetectionPool(num_detectors=2, camera_ids=camera_ids)  # Pass camera_ids to fix race condition
        self.performance_monitor = MultiCameraPerformanceMonitor(camera_ids)

        # Display queue for all cameras
        self.display_queue = queue.Queue(maxsize=10)

        # Store last frames for display continuity
        self.last_frames = {}
        self.last_detection_displays = {}  # Store last detection-only displays
        self.last_detection_counts = {}    # Track detection counts per camera
        self.last_detection_times = {}     # Track when last detection occurred

        # Initialize camera workers
        for camera_id in camera_ids:
            self.camera_workers[camera_id] = CameraWorker(
                camera_id, self.shared_db, self.performance_monitor, self.shared_detector, self
            )

        # Statistics
        self.total_detections = 0
        self.total_objects = 0

        logger.info(f"🎥 Multi-camera tracker initialized for cameras: {camera_ids}")
        logger.info(f"🧵 Total threads: {len(camera_ids) * 3} camera threads + 3 shared threads")

    def result_collection_thread(self):
        """Collect results from all cameras"""
        logger.info("📊 Result collection thread started")

        while self.running:
            try:
                camera_results = {}

                # Collect results from all cameras with shorter timeout
                for camera_id in self.camera_ids:
                    try:
                        result = self.camera_workers[camera_id].result_queue.get(timeout=0.05)
                        camera_results[camera_id] = result
                    except queue.Empty:
                        # Create a placeholder result for display continuity
                        if hasattr(self, 'last_frames') and camera_id in self.last_frames:
                            camera_results[camera_id] = {
                                'camera_id': camera_id,
                                'camera_name': f"Camera {camera_id}",
                                'frame': self.last_frames[camera_id],
                                'tracked_detections': [],
                                'timestamp': time.time()
                            }
                    except Exception as e:
                        if self.running:
                            logger.error(f"Error getting result from camera {camera_id}: {e}")

                if camera_results:
                    # Store last frames for display continuity
                    for camera_id, result in camera_results.items():
                        if 'frame' in result:
                            self.last_frames[camera_id] = result['frame']

                    # Package multi-camera result
                    multi_result = {
                        'camera_results': camera_results,
                        'timestamp': time.time(),
                        'total_cameras': len(camera_results)
                    }

                    logger.info(f"📊 Result collection: Sending {len(camera_results)} camera results to display")

                    try:
                        self.display_queue.put(multi_result, timeout=0.1)
                    except queue.Full:
                        pass  # Display queue full, skip this frame
                    except Exception as e:
                        if self.running:
                            logger.error(f"Error putting to display queue: {e}")

                time.sleep(0.05)  # Small delay

            except Exception as e:
                if self.running:
                    logger.error(f"Result collection error: {e}")
                time.sleep(0.1)

        logger.info("📊 Result collection thread stopped")

    def display_thread(self):
        """Multi-camera display thread"""
        logger.info("🖥️ Multi-camera display thread started")

        # Create display window
        window_name = "Multi-Camera Threaded Warehouse Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1800, 1200)

        # Create a default display frame
        default_frame = np.zeros((1200, 1800, 3), dtype=np.uint8)
        cv2.putText(default_frame, "Multi-Camera Warehouse Tracking", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(default_frame, "Waiting for camera feeds...", (50, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(default_frame, "Cameras: 8, 9, 10", (50, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        while self.running:
            try:
                try:
                    multi_result = self.display_queue.get(timeout=0.05)  # Shorter timeout for more responsive display
                    logger.info(f"🖥️ Display: Got multi_result with {len(multi_result.get('camera_results', {}))} cameras")
                    # Create multi-camera display
                    display_frame = self._create_multi_camera_display(multi_result)
                except queue.Empty:
                    # No new data - show last detection frames (PERSISTENT DISPLAY)
                    logger.info("🖥️ Display: No new data, showing last frames")
                    display_frame = self._create_display_from_last_frames()
                    if display_frame is None:
                        # Still no data - show waiting screen
                        logger.info("🖥️ Display: No last frames, showing waiting screen")
                        display_frame = default_frame.copy()
                        cv2.putText(display_frame, f"Cameras: {self.camera_ids}", (50, 220),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(display_frame, "Waiting for first detections...", (50, 260),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.running = False
                    break

            except Exception as e:
                if self.running:
                    logger.error(f"Display error: {e}")
                time.sleep(0.1)

        cv2.destroyAllWindows()
        logger.info("🖥️ Multi-camera display thread stopped")

    def performance_monitoring_thread(self):
        """Performance monitoring for all cameras"""
        logger.info("📊 Multi-camera performance monitoring started")

        while self.running:
            try:
                self.performance_monitor.calculate_performance()
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(1.0)

        logger.info("📊 Multi-camera performance monitoring stopped")

    def _create_display_from_last_frames(self) -> Optional[np.ndarray]:
        """Create display from stored last detection displays - PERSISTENT DETECTION VIEW"""
        if not self.last_detection_displays:
            return None

        # Create 2x2 grid
        grid_width = 1800
        grid_height = 1200
        cell_width = grid_width // 2
        cell_height = grid_height // 2

        # Create black background
        display_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Position cameras in grid
        positions = {
            8: (0, 0),      # Top-left
            9: (1, 0),      # Top-right
            10: (0, 1)      # Bottom-left
        }

        for camera_id, detection_display in self.last_detection_displays.items():
            if camera_id in positions:
                col, row = positions[camera_id]

                # Resize to fit grid cell
                frame_resized = cv2.resize(detection_display, (cell_width-10, cell_height-10))

                # Place in grid
                y_start = row * cell_height + 5
                y_end = y_start + frame_resized.shape[0]
                x_start = col * cell_width + 5
                x_end = x_start + frame_resized.shape[1]

                display_frame[y_start:y_end, x_start:x_end] = frame_resized

                # Add camera label with detection count and age
                detection_count = self.last_detection_counts.get(camera_id, 0)
                last_time = self.last_detection_times.get(camera_id, time.time())
                age_seconds = int(time.time() - last_time)

                label = f"Camera {camera_id}: {detection_count} objects"
                cv2.putText(display_frame, label, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                age_label = f"Last: {age_seconds}s ago"
                cv2.putText(display_frame, age_label, (x_start + 10, y_start + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Add status info
        cv2.putText(display_frame, "Status: Showing last detection frames", (1220, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "Waiting for new detections...", (1220, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return display_frame

    def _create_multi_camera_display(self, multi_result: Dict) -> np.ndarray:
        """Create grid display for multiple cameras"""
        camera_results = multi_result['camera_results']

        # Create 2x2 grid (with space for 4 cameras, using 3)
        grid_width = 1800
        grid_height = 1200
        cell_width = grid_width // 2
        cell_height = grid_height // 2

        # Create black background
        display_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Position cameras in grid
        positions = {
            8: (0, 0),      # Top-left
            9: (1, 0),      # Top-right
            10: (0, 1)      # Bottom-left
        }

        for camera_id, result in camera_results.items():
            if camera_id in positions:
                col, row = positions[camera_id]

                # Debug: Check what's in the result
                tracked_detections = result.get('tracked_detections', [])
                logger.info(f"🎨 Camera {camera_id}: Creating display with {len(tracked_detections)} tracked detections")

                # Create detection-only display (black background with only detections)
                detection_display = self._create_detection_only_display(result)

                # Store this detection display for persistence
                self.last_detection_displays[camera_id] = detection_display.copy()
                self.last_detection_counts[camera_id] = len(result['tracked_detections'])
                self.last_detection_times[camera_id] = time.time()

                # Resize to fit grid cell
                frame_resized = cv2.resize(detection_display, (cell_width-10, cell_height-10))

                # Place in grid
                y_start = row * cell_height + 5
                y_end = y_start + frame_resized.shape[0]
                x_start = col * cell_width + 5
                x_end = x_start + frame_resized.shape[1]

                display_frame[y_start:y_end, x_start:x_end] = frame_resized

                # Add camera label
                tracked_detections = result['tracked_detections']
                label = f"Camera {camera_id}: {len(tracked_detections)} objects"
                cv2.putText(display_frame, label, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Add global info overlay
        self._draw_global_info_overlay(display_frame, multi_result)

        return display_frame

    def _create_detection_only_display(self, result: Dict) -> np.ndarray:
        """Create detection display - show camera feed with detections overlaid"""
        tracked_detections = result['tracked_detections']
        camera_id = result['camera_id']
        original_frame = result['frame']

        # Use camera feed as background instead of black screen
        display_height, display_width = 600, 900  # Smaller size for grid
        detection_frame = cv2.resize(original_frame, (display_width, display_height))

        # Debug logging for display
        if len(tracked_detections) > 0:
            logger.info(f"🎨 Creating camera view: {len(tracked_detections)} objects for Camera {camera_id}")
        else:
            logger.info(f"🎨 Creating camera view: Showing live feed for Camera {camera_id} (no detections)")

        # Scale factor to fit detections in display area
        scale_x = display_width / original_frame.shape[1]
        scale_y = display_height / original_frame.shape[0]

        for detection in tracked_detections:
            bbox = detection['bbox']
            center = detection['center']
            global_id = detection.get('global_id', -1)
            tracking_status = detection.get('tracking_status', 'unknown')
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')

            # Scale coordinates to fit display
            x1, y1, x2, y2 = bbox
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            center_scaled = (int(center[0] * scale_x), int(center[1] * scale_y))

            # Color coding
            if tracking_status == 'new':
                color = (0, 255, 0)  # Green
            elif tracking_status == 'existing':
                color = (255, 165, 0)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Draw filled rectangle (more visible on black background)
            cv2.rectangle(detection_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 3)
            cv2.circle(detection_frame, center_scaled, 8, color, -1)

            # Labels with better visibility
            if global_id != -1:
                id_label = f"ID:{global_id}"
                cv2.putText(detection_frame, id_label, (x1_scaled, y1_scaled-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if physical_x is not None and physical_y is not None:
                coord_label = f"({physical_x:.1f},{physical_y:.1f})"
                cv2.putText(detection_frame, coord_label, (x1_scaled, y2_scaled+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Add camera info
        cv2.putText(detection_frame, f"Camera {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(detection_frame, f"Objects: {len(tracked_detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return detection_frame

    def _draw_global_info_overlay(self, frame: np.ndarray, multi_result: Dict):
        """Draw global information overlay"""
        # Info panel on the right side
        panel_x = 1200
        panel_width = 600

        # Black background for info panel
        cv2.rectangle(frame, (panel_x, 0), (panel_x + panel_width, 600), (0, 0, 0), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)

        y_offset = 30
        cv2.putText(frame, "🎥 MULTI-CAMERA TRACKING", (panel_x + 20, y_offset), font, 0.7, color, 2)

        y_offset += 40
        cv2.putText(frame, f"Cameras: {len(multi_result['camera_results'])}/3", (panel_x + 20, y_offset), font, 0.5, color, 1)

        y_offset += 30
        cv2.putText(frame, "PERFORMANCE:", (panel_x + 20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 25
        cv2.putText(frame, f"Total FPS: {self.performance_monitor.total_fps:.1f}", (panel_x + 20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"GPU Memory: {self.performance_monitor.gpu_memory_usage:.2f}GB", (panel_x + 20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 30
        cv2.putText(frame, "CAMERA STATS:", (panel_x + 20, y_offset), font, 0.5, (255, 255, 255), 1)

        # Individual camera stats
        for camera_id in self.camera_ids:
            if camera_id in self.performance_monitor.camera_stats:
                stats = self.performance_monitor.camera_stats[camera_id]
                y_offset += 25
                cv2.putText(frame, f"Cam {camera_id}: {stats['fps']:.1f} FPS", (panel_x + 20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 30
        cv2.putText(frame, f"Database Objects: {len(self.shared_db.features)}", (panel_x + 20, y_offset), font, 0.4, (0, 255, 0), 1)

        # Controls
        y_offset += 40
        cv2.putText(frame, "Controls:", (panel_x + 20, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, "Q - Quit", (panel_x + 20, y_offset), font, 0.4, (255, 255, 255), 1)

    def start_tracking(self):
        """Start multi-camera tracking system"""
        self.running = True

        logger.info("🚀 STARTING MULTI-CAMERA THREADED TRACKING")
        logger.info("=" * 80)
        logger.info(f"🎥 Cameras: {self.camera_ids}")
        logger.info(f"🧵 OPTIMIZED THREADING ARCHITECTURE:")
        logger.info(f"   📹 Camera threads: {len(self.camera_ids)} cameras × 3 threads = {len(self.camera_ids) * 3}")
        logger.info(f"   🔍 Detection workers: {self.shared_detector.num_detectors} parallel detectors (GPU-OPTIMIZED)")
        logger.info(f"   🔄 Shared threads: 3 (collection, display, monitoring)")
        logger.info(f"   📊 TOTAL THREADS: {len(self.camera_ids) * 3 + 3 + self.shared_detector.num_detectors}")
        logger.info(f"   🎯 GPU FOCUS: {self.shared_detector.num_detectors} threads dedicated to detection")
        logger.info(f"   💾 MEMORY: 32GB RAM + 6GB GPU available")
        logger.info("🗄️ Shared global database for cross-camera tracking")
        logger.info("🎨 Detection-only display (black background + detections)")
        logger.info("=" * 80)

        # Start detection workers
        self.shared_detector.start_detection_workers()

        # Start camera workers
        for camera_id in self.camera_ids:
            self.camera_workers[camera_id].start_camera_threads()
            time.sleep(0.2)  # Stagger camera starts

        # Start shared threads
        shared_threads = {
            'result_collection': threading.Thread(target=self.result_collection_thread, name="ResultCollection"),
            'display': threading.Thread(target=self.display_thread, name="MultiDisplay"),
            'performance': threading.Thread(target=self.performance_monitoring_thread, name="MultiPerformance")
        }

        for thread_name, thread in shared_threads.items():
            thread.daemon = True
            thread.start()
            logger.info(f"✅ Started {thread_name} thread")
            time.sleep(0.1)

        logger.info("🚀 All multi-camera threads started!")
        logger.info("Press 'q' in the display window to quit")

        # Wait for display thread
        try:
            shared_threads['display'].join()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        self.stop_tracking()

    def stop_tracking(self):
        """Stop all tracking"""
        logger.info("🛑 Stopping multi-camera tracking...")

        self.running = False

        # Stop detection workers
        self.shared_detector.stop_detection_workers()

        # Stop camera workers
        for camera_id in self.camera_ids:
            self.camera_workers[camera_id].stop_camera_threads()

        # Cleanup
        self.shared_db.cleanup()

        logger.info("🏁 Multi-camera tracking stopped")


def main():
    """Main function for multi-camera tracking"""
    print("🎥 MULTI-CAMERA THREADED WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("SIMULTANEOUS 3-CAMERA PROCESSING:")
    print("🎥 Camera 8 (Column 3 - Bottom)")
    print("🎥 Camera 9 (Column 3 - Middle)")
    print("🎥 Camera 10 (Column 3 - Top)")
    print("=" * 80)
    print("ARCHITECTURE:")
    print("🧵 Per Camera: Capture → Detection → Processing (3 threads each)")
    print("🧵 Shared: Result Collection → Display → Performance (3 threads)")
    print("🧵 Total: ~12 threads + shared database")
    print("=" * 80)
    print("FEATURES:")
    print("✅ Shared Global Database - Cross-camera object tracking")
    print("✅ Real-time Grid Display - All cameras simultaneously")
    print("✅ Performance Monitoring - Per-camera and total FPS")
    print("✅ Physical Coordinates - Warehouse coordinate system")
    print("✅ Multi-threaded Pipeline - Maximum CPU/GPU utilization")
    print("=" * 80)

    # System info
    import multiprocessing as mp
    cpu_count = mp.cpu_count()

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 System: {cpu_count} CPU cores, {gpu_count} GPU - {gpu_name}")
    else:
        print(f"⚠️ System: {cpu_count} CPU cores, No GPU available")

    print("🎯 Target: 8-12 FPS per camera, 24-36 total FPS")
    print("=" * 80)

    tracker = MultiCameraThreadedTracker(camera_ids=[8, 9, 10])

    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("\nShutting down multi-camera tracker...")
    except Exception as e:
        logger.error(f"Error running multi-camera tracker: {e}")
    finally:
        tracker.stop_tracking()


if __name__ == "__main__":
    main()
