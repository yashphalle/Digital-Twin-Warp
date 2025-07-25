#!/usr/bin/env python3
"""
Multi-Threaded Complete Warehouse Tracking System
Maximizes CPU and GPU utilization through parallel processing
Architecture:
- Thread 1: Frame Capture (Camera I/O)
- Thread 2: Detection (GPU - Grounding DINO) 
- Thread 3: Feature Extraction (CPU - SIFT)
- Thread 4: Filtering & Coordinates (CPU)
- Thread 5: Database Operations (I/O)
- Thread 6: Display Rendering (CPU)
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

class ThreadSafeCounter:
    """Thread-safe counter for performance monitoring"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self):
        with self._lock:
            return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0

class PerformanceMonitor:
    """Monitor system performance across threads"""
    def __init__(self):
        self.frame_counter = ThreadSafeCounter()
        self.detection_counter = ThreadSafeCounter()
        self.feature_counter = ThreadSafeCounter()
        self.database_counter = ThreadSafeCounter()
        
        self.start_time = time.time()
        self.last_report_time = time.time()
        
        # Performance metrics
        self.fps = 0.0
        self.detection_fps = 0.0
        self.feature_fps = 0.0
        self.database_fps = 0.0
        
        # System utilization
        self.gpu_memory_usage = 0.0
        self.cpu_usage = 0.0
        
        # Thread status
        self.thread_status = {
            'capture': 'idle',
            'detection': 'idle', 
            'features': 'idle',
            'filtering': 'idle',
            'database': 'idle',
            'display': 'idle'
        }
        
        self._lock = threading.Lock()
    
    def update_thread_status(self, thread_name: str, status: str):
        with self._lock:
            self.thread_status[thread_name] = status
    
    def update_performance(self):
        """Update performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_report_time
        
        if elapsed >= 5.0:  # Report every 5 seconds
            with self._lock:
                # Calculate FPS
                frames = self.frame_counter.get()
                detections = self.detection_counter.get()
                features = self.feature_counter.get()
                database_ops = self.database_counter.get()
                
                self.fps = frames / elapsed
                self.detection_fps = detections / elapsed
                self.feature_fps = features / elapsed
                self.database_fps = database_ops / elapsed
                
                # Reset counters
                self.frame_counter.reset()
                self.detection_counter.reset()
                self.feature_counter.reset()
                self.database_counter.reset()
                
                self.last_report_time = current_time
                
                # Monitor GPU memory
                if torch.cuda.is_available():
                    self.gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
                
                # Log performance
                logger.info(f"🚀 MULTI-THREAD PERFORMANCE:")
                logger.info(f"   📹 Capture FPS: {self.fps:.1f}")
                logger.info(f"   🔍 Detection FPS: {self.detection_fps:.1f}")
                logger.info(f"   🎯 Feature FPS: {self.feature_fps:.1f}")
                logger.info(f"   💾 Database FPS: {self.database_fps:.1f}")
                logger.info(f"   🖥️  GPU Memory: {self.gpu_memory_usage:.2f}GB")
                logger.info(f"   🧵 Threads: {', '.join([f'{k}:{v}' for k,v in self.thread_status.items()])}")

class OptimizedFisheyeCorrector:
    """CPU-optimized fisheye correction with caching"""
    
    def __init__(self, lens_mm: float = 2.8):
        self.lens_mm = lens_mm
        self.correction_maps = {}  # Cache maps for different frame sizes
        self._lock = threading.Lock()
    
    def get_correction_maps(self, frame_shape):
        """Get or create correction maps for frame size"""
        key = f"{frame_shape[0]}x{frame_shape[1]}"
        
        with self._lock:
            if key not in self.correction_maps:
                height, width = frame_shape[:2]
                
                # Camera matrix and distortion coefficients
                camera_matrix = np.array([
                    [width * 0.7, 0, width / 2],
                    [0, height * 0.7, height / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                dist_coeffs = np.array([-0.3, 0.1, 0, 0, -0.02], dtype=np.float32)
                
                # Generate correction maps
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    camera_matrix, dist_coeffs, np.eye(3), camera_matrix,
                    (width, height), cv2.CV_16SC2
                )
                
                self.correction_maps[key] = (map1, map2)
                logger.info(f"Fisheye correction maps cached for {key}")
            
            return self.correction_maps[key]
    
    def correct(self, frame: np.ndarray) -> np.ndarray:
        """Apply fisheye correction with cached maps"""
        try:
            map1, map2 = self.get_correction_maps(frame.shape)
            corrected = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            return corrected
        except Exception as e:
            logger.error(f"Fisheye correction failed: {e}")
            return frame

class ThreadedPalletDetector:
    """Thread-safe pallet detector with GPU optimization"""
    
    def __init__(self):
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt_index = 0
        self.current_prompt = self.sample_prompts[0]
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔍 Threaded detector initialized on {self.device}")
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize detection model with thread safety"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            with self._lock:
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Mixed precision for better performance
                if self.device.type == 'cuda':
                    self.model = self.model.half()
                    logger.info("✅ Mixed precision enabled for threaded detection")
                
        except Exception as e:
            logger.error(f"Failed to initialize threaded detector: {e}")
            self.processor = None
            self.model = None
    
    def detect_pallets_threaded(self, frame: np.ndarray) -> List[Dict]:
        """Thread-safe pallet detection"""
        if self.model is None or self.processor is None:
            return []
        
        try:
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            with self._lock:
                # Process inputs
                inputs = self.processor(
                    images=pil_image,
                    text=self.current_prompt,
                    return_tensors="pt"
                )
                
                # Move to GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference with mixed precision
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                
                # Process results
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.confidence_threshold,
                    text_threshold=self.confidence_threshold,
                    target_sizes=[pil_image.size[::-1]]
                )
            
            # Convert to detection format
            detections = []
            if results and len(results) > 0:
                boxes = results[0]["boxes"].cpu().numpy()
                scores = results[0]["scores"].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'area': area,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Threaded detection failed: {e}")
            return []

class CoordinateMapper:
    """Thread-safe coordinate mapping"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        self._lock = threading.Lock()
        
        logger.info(f"Threaded coordinate mapper initialized - Camera: {camera_id}")

    def load_calibration(self, filename=None):
        """Thread-safe calibration loading"""
        if filename is None:
            if self.camera_id:
                filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            else:
                filename = "../configs/warehouse_calibration.json"
        
        if not os.path.isabs(filename) and not filename.startswith('../'):
            filename = f"../{filename}"
            
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            with self._lock:
                warehouse_dims = calibration_data.get('warehouse_dimensions', {})
                self.floor_width_ft = warehouse_dims.get('width_feet', self.floor_width_ft)
                self.floor_length_ft = warehouse_dims.get('length_feet', self.floor_length_ft)
                
                image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
                real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
                
                if len(image_corners) != 4 or len(real_world_corners) != 4:
                    raise ValueError("Calibration must contain exactly 4 corner points")
                
                self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
                self.is_calibrated = True
                
                logger.info(f"Threaded coordinate calibration loaded: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load threaded calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real_batch(self, pixel_points: np.ndarray) -> np.ndarray:
        """Thread-safe batch coordinate transformation"""
        if not self.is_calibrated:
            return np.full((len(pixel_points), 2), np.nan)
        
        try:
            with self._lock:
                points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
                transformed_points = cv2.perspectiveTransform(points_reshaped, self.homography_matrix)
                return transformed_points.reshape(-1, 2)
        except Exception as e:
            logger.error(f"Threaded coordinate transformation failed: {e}")
            return np.full((len(pixel_points), 2), np.nan)

    def pixel_to_real(self, pixel_x, pixel_y):
        """Single point coordinate transformation"""
        points = np.array([[pixel_x, pixel_y]])
        result = self.pixel_to_real_batch(points)
        if len(result) > 0 and not np.isnan(result[0]).any():
            return float(result[0][0]), float(result[0][1])
        return None, None

class ThreadedGlobalFeatureDatabase:
    """Thread-safe global feature database with optimized SIFT"""

    def __init__(self, database_file: str = "threaded_warehouse_global_features.pkl"):
        self.database_file = database_file
        self.features = {}
        self.next_global_id = 1000
        self._lock = threading.Lock()

        # Load existing database
        self.load_database()

        # Initialize SIFT with optimized parameters for threading
        self.sift = cv2.SIFT_create(
            nfeatures=300,  # Reduced for speed
            contrastThreshold=0.04,
            edgeThreshold=10
        )

        # FLANN matcher for feature comparison
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=30)  # Reduced for speed
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Matching parameters optimized for threading
        self.similarity_threshold = 0.25  # Slightly lower for speed
        self.min_matches = 8  # Reduced for speed
        self.max_disappeared_frames = 30

        # Thread pool for parallel feature processing
        self.feature_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FeatureWorker")

        logger.info(f"Threaded feature database initialized with {len(self.features)} objects")

    def load_database(self):
        """Thread-safe database loading"""
        try:
            if os.path.exists(self.database_file):
                with self._lock:
                    with open(self.database_file, 'rb') as f:
                        data = pickle.load(f)
                        self.features = data.get('features', {})
                        self.next_global_id = data.get('next_id', 1000)
                logger.info(f"Loaded {len(self.features)} objects from threaded database")
            else:
                self.features = {}
                self.next_global_id = 1000
        except Exception as e:
            logger.error(f"Error loading threaded database: {e}")
            self.features = {}
            self.next_global_id = 1000

    def save_database_async(self):
        """Asynchronous database saving"""
        def save_worker():
            try:
                with self._lock:
                    data = {
                        'features': self.features.copy(),
                        'next_id': self.next_global_id,
                        'last_updated': datetime.now().isoformat(),
                        'multithreaded': True
                    }

                with open(self.database_file, 'wb') as f:
                    pickle.dump(data, f)

            except Exception as e:
                logger.error(f"Error saving threaded database: {e}")

        # Submit to thread pool for async execution
        self.feature_executor.submit(save_worker)

    def extract_features_parallel(self, image_regions: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Parallel feature extraction for multiple regions"""
        def extract_single_feature(image_region):
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

        # Process features in parallel
        futures = []
        for region in image_regions:
            future = self.feature_executor.submit(extract_single_feature, region)
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures, timeout=5.0):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel feature extraction failed: {e}")
                results.append(None)

        return results

    def calculate_similarity_fast(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Fast similarity calculation optimized for threading"""
        if features1 is None or features2 is None:
            return 0.0

        if len(features1) < 2 or len(features2) < 2:
            return 0.0

        try:
            # Use reduced feature sets for speed
            max_features = 100
            if len(features1) > max_features:
                features1 = features1[:max_features]
            if len(features2) > max_features:
                features2 = features2[:max_features]

            matches = self.flann.knnMatch(features1, features2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Slightly relaxed for speed
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                similarity = len(good_matches) / min(len(features1), len(features2))
                return min(similarity, 1.0)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Fast similarity calculation failed: {e}")
            return 0.0

    def find_matching_object_parallel(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Parallel object matching"""
        if query_features is None:
            return None, 0.0

        best_match_id = None
        best_similarity = 0.0

        # Create similarity calculation tasks
        similarity_tasks = []

        with self._lock:
            feature_items = list(self.features.items())

        def calculate_similarity_task(global_id, feature_data):
            stored_features = feature_data['features']
            similarity = self.calculate_similarity_fast(query_features, stored_features)
            return global_id, similarity

        # Submit similarity calculations to thread pool
        futures = []
        for global_id, feature_data in feature_items:
            future = self.feature_executor.submit(calculate_similarity_task, global_id, feature_data)
            futures.append(future)

        # Collect results
        for future in as_completed(futures, timeout=3.0):
            try:
                global_id, similarity = future.result()
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_match_id = global_id
                    best_similarity = similarity
            except Exception as e:
                logger.error(f"Parallel matching failed: {e}")

        return best_match_id, best_similarity

    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Thread-safe new object addition"""
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
                'multithreaded': True
            }

            # Add physical coordinates if available
            if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                    feature_data['physical_locations'].append({
                        'timestamp': datetime.now().isoformat(),
                        'x_ft': detection_info['physical_x_ft'],
                        'y_ft': detection_info['physical_y_ft']
                    })

            self.features[global_id] = feature_data

        # Async database save
        self.save_database_async()

        logger.info(f"🆕 NEW THREADED ID: {global_id}")
        return global_id

    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Thread-safe object update"""
        with self._lock:
            if global_id in self.features:
                feature_data = self.features[global_id]

                feature_data['features'] = features
                feature_data['last_seen'] = datetime.now().isoformat()
                feature_data['times_seen'] += 1
                feature_data['disappeared_frames'] = 0

                # Add physical coordinates if available
                if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                    if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                        feature_data['physical_locations'].append({
                            'timestamp': datetime.now().isoformat(),
                            'x_ft': detection_info['physical_x_ft'],
                            'y_ft': detection_info['physical_y_ft']
                        })

                        # Keep only recent locations
                        if len(feature_data['physical_locations']) > 50:
                            feature_data['physical_locations'] = feature_data['physical_locations'][-50:]

        # Async database save
        self.save_database_async()

        logger.info(f"🔄 UPDATED THREADED ID: {global_id}")

    def mark_disappeared_objects(self, seen_ids: Set[int]):
        """Thread-safe disappeared object management"""
        to_remove = []

        with self._lock:
            for global_id in self.features:
                if global_id not in seen_ids:
                    self.features[global_id]['disappeared_frames'] += 1

                    if self.features[global_id]['disappeared_frames'] >= self.max_disappeared_frames:
                        to_remove.append(global_id)

            # Remove old objects
            for global_id in to_remove:
                logger.info(f"🗑️ REMOVED THREADED ID: {global_id}")
                del self.features[global_id]

        if to_remove:
            self.save_database_async()

    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """Thread-safe global ID assignment"""
        if image_region is None or image_region.size == 0:
            return -1, 'failed', 0.0

        # Extract features
        features = self.extract_features_parallel([image_region])[0]
        if features is None:
            return -1, 'failed', 0.0

        # Find matching object
        match_id, similarity = self.find_matching_object_parallel(features)

        if match_id is not None:
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

    def cleanup(self):
        """Cleanup thread pool"""
        self.feature_executor.shutdown(wait=True)

class MultiThreadedWarehouseTracker:
    """Multi-threaded complete warehouse tracking system"""

    def __init__(self, camera_id: int = 8):
        self.camera_id = camera_id

        # Import configurations here to avoid circular imports
        from cv.configs.config import Config
        from cv.configs.warehouse_config import get_warehouse_config

        self.warehouse_config = get_warehouse_config()

        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

        # Threading components
        self.running = False
        self.threads = {}

        # Thread-safe queues for pipeline
        import queue
        self.frame_queue = queue.Queue(maxsize=5)  # Capture -> Detection
        self.detection_queue = queue.Queue(maxsize=5)  # Detection -> Features
        self.feature_queue = queue.Queue(maxsize=5)  # Features -> Filtering
        self.coordinate_queue = queue.Queue(maxsize=5)  # Filtering -> Coordinates
        self.database_queue = queue.Queue(maxsize=20)  # Database operations
        self.display_queue = queue.Queue(maxsize=3)  # Display frames

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Detection components (thread-safe)
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = ThreadedPalletDetector()
        self.coordinate_mapper = CoordinateMapper(camera_id=camera_id)
        self.global_db = ThreadedGlobalFeatureDatabase(f"threaded_camera_{camera_id}_features.pkl")

        # Initialize coordinate mapper
        self._initialize_coordinate_mapper()

        # Detection parameters
        self.MIN_AREA = 10000
        self.MAX_AREA = 100000
        self.CELL_SIZE = 40

        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.new_objects = 0
        self.existing_objects = 0

        logger.info(f"🧵 Multi-threaded tracker initialized for {self.camera_name}")
        logger.info(f"🚀 Thread pool ready with {mp.cpu_count()} CPU cores available")

    def _initialize_coordinate_mapper(self):
        """Initialize coordinate mapper"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                logger.info(f"✅ Threaded coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ Threaded coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize threaded coordinate mapper: {e}")

    def frame_capture_thread(self):
        """Thread 1: Frame capture from camera"""
        self.performance_monitor.update_thread_status('capture', 'starting')

        # Connect to camera
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to connect to camera: {self.rtsp_url}")
            return

        from cv.configs.config import Config
        cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.RTSP_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)

        logger.info(f"🎥 Frame capture thread started for {self.camera_name}")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('capture', 'capturing')

                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Apply fisheye correction
                from cv.configs.config import Config
                if Config.FISHEYE_CORRECTION_ENABLED:
                    frame = self.fisheye_corrector.correct(frame)

                # Resize for processing
                height, width = frame.shape[:2]
                if width > 1600:
                    scale = 1600 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                    self.performance_monitor.frame_counter.increment()
                except:
                    # Queue full, skip frame
                    pass

                self.performance_monitor.update_thread_status('capture', 'idle')
                time.sleep(0.01)  # Small delay to prevent CPU overload

            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(0.1)

        cap.release()
        logger.info("🎥 Frame capture thread stopped")

    def detection_thread(self):
        """Thread 2: GPU detection processing"""
        self.performance_monitor.update_thread_status('detection', 'starting')
        logger.info("🔍 Detection thread started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('detection', 'waiting')

                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)

                self.performance_monitor.update_thread_status('detection', 'detecting')

                # Run detection
                detections = self.pallet_detector.detect_pallets_threaded(frame)

                # Package result
                detection_result = {
                    'frame': frame,
                    'detections': detections,
                    'timestamp': time.time()
                }

                # Add to next queue
                try:
                    self.detection_queue.put(detection_result, timeout=0.1)
                    self.performance_monitor.detection_counter.increment()
                except:
                    # Queue full, skip
                    pass

                self.performance_monitor.update_thread_status('detection', 'idle')

            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logger.error(f"Detection thread error: {e}")
                time.sleep(0.1)

        logger.info("🔍 Detection thread stopped")

    def filtering_thread(self):
        """Thread 3: Filtering and coordinate processing"""
        self.performance_monitor.update_thread_status('filtering', 'starting')
        logger.info("🔧 Filtering thread started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('filtering', 'waiting')

                # Get detection result
                detection_result = self.detection_queue.get(timeout=1.0)
                frame = detection_result['frame']
                detections = detection_result['detections']

                self.performance_monitor.update_thread_status('filtering', 'filtering')

                # Apply area filtering
                area_filtered = [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]

                # Apply grid cell filtering
                grid_filtered = self._apply_grid_filter(area_filtered)

                # Batch coordinate translation
                frame_height, frame_width = frame.shape[:2]
                coordinate_filtered = self._translate_coordinates_batch(grid_filtered, frame_width, frame_height)

                # Package result
                filtered_result = {
                    'frame': frame,
                    'raw_detections': detections,
                    'area_filtered': area_filtered,
                    'grid_filtered': grid_filtered,
                    'final_detections': coordinate_filtered,
                    'timestamp': detection_result['timestamp']
                }

                # Add to feature queue
                try:
                    self.feature_queue.put(filtered_result, timeout=0.1)
                except:
                    pass

                self.performance_monitor.update_thread_status('filtering', 'idle')

            except Exception as e:
                if self.running:
                    logger.error(f"Filtering thread error: {e}")
                time.sleep(0.1)

        logger.info("🔧 Filtering thread stopped")

    def feature_tracking_thread(self):
        """Thread 4: Feature extraction and global ID assignment"""
        self.performance_monitor.update_thread_status('features', 'starting')
        logger.info("🎯 Feature tracking thread started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('features', 'waiting')

                # Get filtered result
                filtered_result = self.feature_queue.get(timeout=1.0)
                frame = filtered_result['frame']
                final_detections = filtered_result['final_detections']

                self.performance_monitor.update_thread_status('features', 'extracting')

                # Extract regions for feature analysis
                image_regions = []
                for detection in final_detections:
                    region = self._extract_detection_region(frame, detection['bbox'])
                    image_regions.append(region)

                # Parallel feature extraction and ID assignment
                tracked_detections = []
                seen_ids = set()

                for i, detection in enumerate(final_detections):
                    self.total_detections += 1

                    # Assign global ID
                    global_id, status, similarity = self.global_db.assign_global_id(
                        image_regions[i], detection
                    )

                    # Update statistics
                    if status == 'new':
                        self.new_objects += 1
                    elif status == 'existing':
                        self.existing_objects += 1

                    # Add tracking info
                    detection['global_id'] = global_id
                    detection['tracking_status'] = status
                    detection['similarity_score'] = similarity

                    if global_id != -1:
                        seen_ids.add(global_id)

                    tracked_detections.append(detection)

                # Mark disappeared objects
                self.global_db.mark_disappeared_objects(seen_ids)

                # Package final result
                final_result = {
                    'frame': frame,
                    'raw_detections': filtered_result['raw_detections'],
                    'area_filtered': filtered_result['area_filtered'],
                    'grid_filtered': filtered_result['grid_filtered'],
                    'tracked_detections': tracked_detections,
                    'timestamp': filtered_result['timestamp']
                }

                # Add to display queue
                try:
                    self.display_queue.put(final_result, timeout=0.1)
                    self.performance_monitor.feature_counter.increment()
                except:
                    pass

                self.performance_monitor.update_thread_status('features', 'idle')

            except Exception as e:
                if self.running:
                    logger.error(f"Feature tracking thread error: {e}")
                time.sleep(0.1)

        logger.info("🎯 Feature tracking thread stopped")

    def database_thread(self):
        """Thread 5: Database operations"""
        self.performance_monitor.update_thread_status('database', 'starting')
        logger.info("💾 Database thread started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('database', 'waiting')

                # Process database operations from queue
                try:
                    operation = self.database_queue.get(timeout=1.0)

                    self.performance_monitor.update_thread_status('database', 'processing')

                    # Process the database operation
                    if operation['type'] == 'save':
                        # Database save operations are handled by the feature database
                        pass
                    elif operation['type'] == 'cleanup':
                        # Periodic cleanup operations
                        pass

                    self.performance_monitor.database_counter.increment()
                    self.performance_monitor.update_thread_status('database', 'idle')

                except:
                    # No operations in queue
                    time.sleep(0.5)

            except Exception as e:
                if self.running:
                    logger.error(f"Database thread error: {e}")
                time.sleep(0.1)

        logger.info("💾 Database thread stopped")

    def display_thread(self):
        """Thread 6: Display rendering"""
        self.performance_monitor.update_thread_status('display', 'starting')

        # Create display window
        window_name = f"Multi-Threaded Warehouse Tracking - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        logger.info("🖥️ Display thread started")

        while self.running:
            try:
                self.performance_monitor.update_thread_status('display', 'waiting')

                # Get result for display
                final_result = self.display_queue.get(timeout=1.0)

                self.performance_monitor.update_thread_status('display', 'rendering')

                # Draw results
                display_frame = self._draw_results(final_result)

                # Show frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.running = False
                    break

                self.performance_monitor.update_thread_status('display', 'idle')

            except Exception as e:
                if self.running:
                    logger.error(f"Display thread error: {e}")
                time.sleep(0.1)

        cv2.destroyAllWindows()
        logger.info("🖥️ Display thread stopped")

    def performance_monitoring_thread(self):
        """Performance monitoring thread"""
        logger.info("📊 Performance monitoring thread started")

        while self.running:
            try:
                self.performance_monitor.update_performance()
                time.sleep(5.0)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(1.0)

        logger.info("📊 Performance monitoring thread stopped")

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
            # Extract centers
            centers = [detection['center'] for detection in detections]
            centers_array = np.array(centers, dtype=np.float32)

            # Scale to calibration frame size
            scale_x = 3840 / frame_width
            scale_y = 2160 / frame_height
            scaled_centers = centers_array * np.array([scale_x, scale_y])

            # Batch coordinate transformation
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
            logger.error(f"Batch coordinate translation failed: {e}")
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'ERROR'
            return detections

    def _extract_detection_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract detection region for feature analysis"""
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
            logger.error(f"Region extraction failed: {e}")
            return np.array([])

    def _draw_results(self, final_result: Dict) -> np.ndarray:
        """Draw tracking results on frame"""
        frame = final_result['frame'].copy()
        tracked_detections = final_result['tracked_detections']

        # Draw detections
        for detection in tracked_detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            area = detection.get('area', 0)

            # Global tracking info
            global_id = detection.get('global_id', -1)
            tracking_status = detection.get('tracking_status', 'unknown')

            # Physical coordinates
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')

            x1, y1, x2, y2 = bbox

            # Color coding
            if tracking_status == 'new':
                color = (0, 255, 0)  # Green
                status_text = "NEW"
            elif tracking_status == 'existing':
                color = (255, 165, 0)  # Orange
                status_text = "THREADED"
            else:
                color = (0, 0, 255)  # Red
                status_text = "FAILED"

            # Draw bounding box and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, center, 8, color, -1)
            cv2.circle(frame, center, 8, (255, 255, 255), 2)

            # Labels
            y_offset = y1 - 5
            line_height = 20

            # Global ID
            if global_id != -1:
                id_label = f"🧵ID:{global_id} ({status_text})"
                cv2.putText(frame, id_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= line_height

            # Confidence and area
            conf_label = f"Conf:{confidence:.3f} Area:{area:.0f}"
            cv2.putText(frame, conf_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset -= line_height

            # Pixel coordinates
            pixel_label = f"Pixel:({center[0]},{center[1]})"
            cv2.putText(frame, pixel_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset -= line_height

            # Physical coordinates
            if physical_x is not None and physical_y is not None:
                coord_label = f"Physical:({physical_x:.1f}ft,{physical_y:.1f}ft)"
                coord_color = (0, 255, 255)  # Cyan
            else:
                coord_label = f"Physical:N/A"
                coord_color = (0, 0, 255)  # Red

            cv2.putText(frame, coord_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_color, 1)

        # Draw info overlay
        frame = self._draw_info_overlay(frame, final_result)

        return frame

    def _draw_info_overlay(self, frame: np.ndarray, final_result: Dict) -> np.ndarray:
        """Draw information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 400), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)

        y_offset = 30
        cv2.putText(frame, f"🧵 MULTI-THREADED WAREHOUSE TRACKING", (20, y_offset), font, 0.7, color, 2)

        y_offset += 30
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"PIPELINE PERFORMANCE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"📹 Capture FPS: {self.performance_monitor.fps:.1f}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"🔍 Detection FPS: {self.performance_monitor.detection_fps:.1f}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"🎯 Feature FPS: {self.performance_monitor.feature_fps:.1f}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"💾 Database FPS: {self.performance_monitor.database_fps:.1f}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 25
        cv2.putText(frame, f"DETECTION PIPELINE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"Raw: {len(final_result['raw_detections'])}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"Area Filtered: {len(final_result['area_filtered'])}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"Grid Filtered: {len(final_result['grid_filtered'])}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"🧵 Final Tracked: {len(final_result['tracked_detections'])}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 25
        cv2.putText(frame, f"TRACKING STATS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"New Objects: {self.new_objects}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"🧵 Tracked Objects: {self.existing_objects}", (20, y_offset), font, 0.4, (255, 165, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Database Objects: {len(self.global_db.features)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        # Thread status
        y_offset += 25
        cv2.putText(frame, f"THREAD STATUS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        thread_status_text = " | ".join([f"{k}:{v}" for k, v in self.performance_monitor.thread_status.items()])
        cv2.putText(frame, thread_status_text[:80], (20, y_offset), font, 0.3, (255, 255, 255), 1)

        # GPU memory
        y_offset += 20
        cv2.putText(frame, f"🖥️ GPU Memory: {self.performance_monitor.gpu_memory_usage:.2f}GB", (20, y_offset), font, 0.4, (0, 255, 255), 1)

        return frame

    def start_tracking(self):
        """Start multi-threaded tracking system"""
        self.running = True

        logger.info("🚀 STARTING MULTI-THREADED WAREHOUSE TRACKING")
        logger.info("=" * 70)
        logger.info("🧵 Thread Architecture:")
        logger.info("  Thread 1: 📹 Frame Capture (Camera I/O)")
        logger.info("  Thread 2: 🔍 Detection (GPU - Grounding DINO)")
        logger.info("  Thread 3: 🔧 Filtering & Coordinates (CPU)")
        logger.info("  Thread 4: 🎯 Feature Extraction & Tracking (CPU)")
        logger.info("  Thread 5: 💾 Database Operations (I/O)")
        logger.info("  Thread 6: 🖥️ Display Rendering (CPU)")
        logger.info("  Thread 7: 📊 Performance Monitoring")
        logger.info("=" * 70)

        # Start all threads
        self.threads['capture'] = threading.Thread(target=self.frame_capture_thread, name="FrameCapture")
        self.threads['detection'] = threading.Thread(target=self.detection_thread, name="Detection")
        self.threads['filtering'] = threading.Thread(target=self.filtering_thread, name="Filtering")
        self.threads['features'] = threading.Thread(target=self.feature_tracking_thread, name="Features")
        self.threads['database'] = threading.Thread(target=self.database_thread, name="Database")
        self.threads['display'] = threading.Thread(target=self.display_thread, name="Display")
        self.threads['performance'] = threading.Thread(target=self.performance_monitoring_thread, name="Performance")

        # Start threads in order
        for thread_name, thread in self.threads.items():
            thread.daemon = True
            thread.start()
            logger.info(f"✅ Started {thread_name} thread")
            time.sleep(0.1)  # Small delay between thread starts

        logger.info("🚀 All threads started successfully!")
        logger.info("Press 'q' in the display window to quit")

        # Wait for display thread to finish (user quit)
        try:
            self.threads['display'].join()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        self.stop_tracking()

    def stop_tracking(self):
        """Stop all threads and cleanup"""
        logger.info("🛑 Stopping multi-threaded tracking...")

        self.running = False

        # Wait for all threads to finish
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread...")
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"{thread_name} thread did not stop gracefully")

        # Cleanup
        self.global_db.cleanup()

        logger.info(f"🏁 Multi-threaded tracking stopped")
        logger.info(f"📊 Final stats - New: {self.new_objects}, Existing: {self.existing_objects}, Total: {self.total_detections}")


def main():
    """Main function for multi-threaded warehouse tracking"""
    print("🧵 MULTI-THREADED COMPLETE WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("MAXIMUM CPU & GPU UTILIZATION through parallel processing:")
    print("🧵 Thread 1: 📹 Frame Capture (Camera I/O)")
    print("🧵 Thread 2: 🔍 Detection (GPU - Grounding DINO)")
    print("🧵 Thread 3: 🔧 Filtering & Coordinates (CPU)")
    print("🧵 Thread 4: 🎯 Feature Extraction & Tracking (CPU)")
    print("🧵 Thread 5: 💾 Database Operations (I/O)")
    print("🧵 Thread 6: 🖥️ Display Rendering (CPU)")
    print("🧵 Thread 7: 📊 Performance Monitoring")
    print("=" * 80)
    print("Camera: 8 (Column 3 - Bottom)")
    print("Performance Target: 8-15 FPS with 80%+ CPU/GPU utilization")
    print("Architecture: Producer-Consumer pipeline with thread pools")
    print("=" * 80)
    print("\n🚀 Multi-Threaded Pipeline:")
    print("Capture → Detection → Filtering → Features → Database → Display")
    print("\nColor Coding:")
    print("- Green: New objects")
    print("- Orange: Multi-threaded tracked objects")
    print("- Red: Failed tracking")
    print("- Cyan: Physical coordinate labels")
    print("=" * 80)

    # System info
    import multiprocessing as mp
    cpu_count = mp.cpu_count()

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 System Ready: {cpu_count} CPU cores, {gpu_count} GPU - {gpu_name}")
    else:
        print(f"⚠️ System: {cpu_count} CPU cores, No GPU available")

    print("=" * 80)

    tracker = MultiThreadedWarehouseTracker(camera_id=8)

    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("\nShutting down multi-threaded tracker...")
    except Exception as e:
        logger.error(f"Error running multi-threaded tracker: {e}")
    finally:
        tracker.stop_tracking()


if __name__ == "__main__":
    main()
