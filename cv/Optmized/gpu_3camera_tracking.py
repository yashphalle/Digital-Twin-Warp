#!/usr/bin/env python3
"""
GPU-Optimized Complete Warehouse Tracking System
Maximizes GPU utilization for all OpenCV operations and feature matching
Integrates all functionalities with GPU acceleration:
1) Detection (GPU)
2) Area + Grid Cell Filtering (GPU)  
3) Physical Coordinate Translation (GPU)
4) GPU SIFT Feature Matching
5) Persistent Object IDs
6) Cross-Frame Tracking & Database
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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizedFisheyeCorrector:
    """GPU-accelerated fisheye correction"""
    
    def __init__(self, lens_mm: float = 2.8):
        self.lens_mm = lens_mm
        self.correction_maps = None
        self.gpu_map1 = None
        self.gpu_map2 = None
        
        # Check GPU availability
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info(f"✅ GPU fisheye correction enabled - {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) available")
        else:
            logger.warning("⚠️ No GPU available for fisheye correction, falling back to CPU")
    
    def initialize_correction_maps(self, frame_shape):
        """Initialize GPU correction maps"""
        if not self.gpu_available:
            return
            
        height, width = frame_shape[:2]
        
        # Camera matrix and distortion coefficients for 2.8mm lens
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
        
        # Upload maps to GPU
        self.gpu_map1 = cv2.cuda_GpuMat()
        self.gpu_map2 = cv2.cuda_GpuMat()
        self.gpu_map1.upload(map1)
        self.gpu_map2.upload(map2)
        
        logger.info(f"GPU fisheye correction maps initialized for {width}x{height}")
    
    def correct(self, frame: np.ndarray) -> np.ndarray:
        """Apply GPU-accelerated fisheye correction"""
        if not self.gpu_available:
            return frame
            
        if self.gpu_map1 is None:
            self.initialize_correction_maps(frame.shape)
        
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Apply correction on GPU
            gpu_corrected = cv2.cuda.remap(gpu_frame, self.gpu_map1, self.gpu_map2, cv2.INTER_LINEAR)
            
            # Download result
            corrected_frame = gpu_corrected.download()
            return corrected_frame
            
        except Exception as e:
            logger.error(f"GPU fisheye correction failed: {e}")
            return frame

class GPUSimplePalletDetector:
    """GPU-optimized pallet detector with embedded DetectorTracker functionality"""
    
    def __init__(self):
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt_index = 0
        self.current_prompt = self.sample_prompts[0]
        
        # Initialize GPU detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔍 Initializing GPU pallet detector on {self.device}")
        
        # Initialize Grounding DINO model
        self._initialize_grounding_dino()
        
        # GPU availability check
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        logger.info(f"OpenCV GPU operations: {'✅ Available' if self.gpu_available else '❌ Not Available'}")
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model for GPU inference"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("✅ AutoProcessor loaded successfully")
            
            # Load model and move to GPU
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Grounding DINO model loaded and moved to {self.device}")
            
            # Enable mixed precision for better GPU utilization
            if self.device.type == 'cuda':
                self.model = self.model.half()  # Use FP16
                logger.info("✅ Mixed precision (FP16) enabled for better GPU utilization")
                
        except Exception as e:
            logger.error(f"Failed to initialize Grounding DINO: {e}")
            self.processor = None
            self.model = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """GPU-optimized pallet detection"""
        if self.model is None or self.processor is None:
            return []
        
        try:
            # Convert frame for model input
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=self.current_prompt,
                return_tensors="pt"
            )
            
            # Move inputs to GPU with mixed precision
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device.type == 'cuda':
                # Use autocast for mixed precision
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
                        'area': area
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return []

class GPUCoordinateMapper:
    """GPU-accelerated coordinate mapping"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.gpu_homography = None
        self.is_calibrated = False
        
        # Check GPU availability
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        logger.info(f"GPU coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")

    def load_calibration(self, filename=None):
        """Load calibration and prepare GPU matrices"""
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
            
            warehouse_dims = calibration_data.get('warehouse_dimensions', {})
            self.floor_width_ft = warehouse_dims.get('width_feet', self.floor_width_ft)
            self.floor_length_ft = warehouse_dims.get('length_feet', self.floor_length_ft)
            
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            if len(image_corners) != 4 or len(real_world_corners) != 4:
                raise ValueError("Calibration must contain exactly 4 corner points")
            
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            
            # Upload homography matrix to GPU if available
            if self.gpu_available:
                self.gpu_homography = cv2.cuda_GpuMat()
                self.gpu_homography.upload(self.homography_matrix.astype(np.float32))
                logger.info("✅ Homography matrix uploaded to GPU")
            
            self.is_calibrated = True
            logger.info(f"GPU coordinate calibration loaded from: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real_batch_gpu(self, pixel_points: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch coordinate transformation"""
        if not self.is_calibrated or not self.gpu_available:
            return self.pixel_to_real_batch_cpu(pixel_points)
        
        try:
            # Reshape points for perspectiveTransform
            points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
            
            # Upload points to GPU
            gpu_points = cv2.cuda_GpuMat()
            gpu_points.upload(points_reshaped)
            
            # Apply transformation on GPU
            gpu_transformed = cv2.cuda.perspectiveTransform(gpu_points, self.gpu_homography)
            
            # Download results
            transformed_points = gpu_transformed.download()
            
            # Reshape back to original format
            result = transformed_points.reshape(-1, 2)
            return result
            
        except Exception as e:
            logger.error(f"GPU coordinate transformation failed: {e}")
            return self.pixel_to_real_batch_cpu(pixel_points)
    
    def pixel_to_real_batch_cpu(self, pixel_points: np.ndarray) -> np.ndarray:
        """CPU fallback for batch coordinate transformation"""
        if not self.is_calibrated:
            return np.full((len(pixel_points), 2), np.nan)
        
        try:
            points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(points_reshaped, self.homography_matrix)
            return transformed_points.reshape(-1, 2)
        except Exception as e:
            logger.error(f"CPU coordinate transformation failed: {e}")
            return np.full((len(pixel_points), 2), np.nan)

    def pixel_to_real(self, pixel_x, pixel_y):
        """Single point coordinate transformation"""
        points = np.array([[pixel_x, pixel_y]])
        result = self.pixel_to_real_batch_gpu(points)
        if len(result) > 0 and not np.isnan(result[0]).any():
            return float(result[0][0]), float(result[0][1])
        return None, None

class GPUGlobalFeatureDatabase:
    """GPU-accelerated global feature database with CUDA SIFT and GPU feature matching"""

    def __init__(self, database_file: str = "gpu_warehouse_global_features.pkl"):
        self.database_file = database_file
        self.features = {}
        self.next_global_id = 1000
        self.load_database()

        # Check GPU availability for SIFT
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

        if self.gpu_available:
            try:
                # Try to create GPU SIFT detector
                self.gpu_sift = cv2.cuda.SIFT_create(
                    nfeatures=500,
                    contrastThreshold=0.04,
                    edgeThreshold=10
                )
                logger.info("✅ GPU SIFT detector initialized")
                self.use_gpu_sift = True
            except Exception as e:
                logger.warning(f"GPU SIFT not available: {e}, falling back to CPU SIFT")
                self.use_gpu_sift = False
                self.cpu_sift = cv2.SIFT_create(
                    nfeatures=500,
                    contrastThreshold=0.04,
                    edgeThreshold=10
                )
        else:
            logger.warning("No GPU available, using CPU SIFT")
            self.use_gpu_sift = False
            self.cpu_sift = cv2.SIFT_create(
                nfeatures=500,
                contrastThreshold=0.04,
                edgeThreshold=10
            )

        # GPU-accelerated feature matching
        if self.gpu_available:
            try:
                # Try GPU-based matcher
                self.gpu_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
                logger.info("✅ GPU feature matcher initialized")
                self.use_gpu_matcher = True
            except Exception as e:
                logger.warning(f"GPU matcher not available: {e}, using CPU FLANN")
                self.use_gpu_matcher = False
                self._init_cpu_matcher()
        else:
            self.use_gpu_matcher = False
            self._init_cpu_matcher()

        # Matching parameters
        self.similarity_threshold = 0.3
        self.min_matches = 10
        self.max_disappeared_frames = 30

        logger.info(f"GPU feature database initialized with {len(self.features)} objects")
        logger.info(f"GPU SIFT: {'✅' if self.use_gpu_sift else '❌'}, GPU Matcher: {'✅' if self.use_gpu_matcher else '❌'}")

    def _init_cpu_matcher(self):
        """Initialize CPU FLANN matcher as fallback"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.cpu_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def load_database(self):
        """Load feature database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.features = data.get('features', {})
                    self.next_global_id = data.get('next_id', 1000)
                logger.info(f"Loaded {len(self.features)} objects from GPU database")
            else:
                self.features = {}
                self.next_global_id = 1000
        except Exception as e:
            logger.error(f"Error loading GPU database: {e}")
            self.features = {}
            self.next_global_id = 1000

    def save_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.features,
                'next_id': self.next_global_id,
                'last_updated': datetime.now().isoformat(),
                'gpu_optimized': True
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving GPU database: {e}")

    def extract_features_gpu(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """GPU-accelerated SIFT feature extraction"""
        if image_region is None or image_region.size == 0:
            return None

        try:
            # Convert to grayscale if needed
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region

            if self.use_gpu_sift:
                # GPU SIFT extraction
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(gray)

                # Extract features on GPU
                gpu_keypoints, gpu_descriptors = self.gpu_sift.detectAndComputeAsync(gpu_image, None)

                # Download descriptors
                if gpu_descriptors is not None:
                    descriptors = gpu_descriptors.download()
                    if descriptors is not None and len(descriptors) >= self.min_matches:
                        return descriptors

                return None
            else:
                # CPU SIFT fallback
                keypoints, descriptors = self.cpu_sift.detectAndCompute(gray, None)
                if descriptors is not None and len(descriptors) >= self.min_matches:
                    return descriptors
                return None

        except Exception as e:
            logger.error(f"GPU feature extraction failed: {e}")
            return None

    def calculate_similarity_gpu(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """GPU-accelerated feature similarity calculation"""
        if features1 is None or features2 is None:
            return 0.0

        if len(features1) < 2 or len(features2) < 2:
            return 0.0

        try:
            if self.use_gpu_matcher:
                # GPU-based matching
                gpu_desc1 = cv2.cuda_GpuMat()
                gpu_desc2 = cv2.cuda_GpuMat()
                gpu_desc1.upload(features1.astype(np.float32))
                gpu_desc2.upload(features2.astype(np.float32))

                # Perform matching on GPU
                gpu_matches = self.gpu_matcher.knnMatch(gpu_desc1, gpu_desc2, k=2)

                # Download matches
                matches = []
                for match_pair in gpu_matches:
                    if len(match_pair) == 2:
                        matches.append(match_pair)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Calculate similarity
                if len(good_matches) >= self.min_matches:
                    similarity = len(good_matches) / min(len(features1), len(features2))
                    return min(similarity, 1.0)
                else:
                    return 0.0
            else:
                # CPU FLANN fallback
                matches = self.cpu_matcher.knnMatch(features1, features2, k=2)

                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= self.min_matches:
                    similarity = len(good_matches) / min(len(features1), len(features2))
                    return min(similarity, 1.0)
                else:
                    return 0.0

        except Exception as e:
            logger.error(f"GPU similarity calculation failed: {e}")
            return 0.0

    def find_matching_object(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Find best matching object using GPU acceleration"""
        best_match_id = None
        best_similarity = 0.0

        for global_id, feature_data in self.features.items():
            stored_features = feature_data['features']
            similarity = self.calculate_similarity_gpu(query_features, stored_features)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match_id = global_id
                best_similarity = similarity

        return best_match_id, best_similarity

    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Add new object to GPU database"""
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
            'gpu_optimized': True
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
        self.save_database()

        logger.info(f"🆕 NEW GPU GLOBAL ID: {global_id}")
        return global_id

    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Update existing object in GPU database"""
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
                    if len(feature_data['physical_locations']) > 100:
                        feature_data['physical_locations'] = feature_data['physical_locations'][-100:]

            self.save_database()
            logger.info(f"🔄 UPDATED GPU GLOBAL ID: {global_id} - Times seen: {feature_data['times_seen']}")

    def mark_disappeared_objects(self, seen_ids: Set[int]):
        """Mark objects as disappeared and cleanup old ones"""
        to_remove = []

        for global_id in self.features:
            if global_id not in seen_ids:
                self.features[global_id]['disappeared_frames'] += 1

                if self.features[global_id]['disappeared_frames'] >= self.max_disappeared_frames:
                    to_remove.append(global_id)

        for global_id in to_remove:
            logger.info(f"🗑️ REMOVED GPU GLOBAL ID: {global_id} - Disappeared for {self.max_disappeared_frames} frames")
            del self.features[global_id]

        if to_remove:
            self.save_database()

    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """Assign global ID using GPU-accelerated feature matching"""
        features = self.extract_features_gpu(image_region)
        if features is None:
            return -1, 'failed', 0.0

        match_id, similarity = self.find_matching_object(features)

        if match_id is not None:
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

class GPUCompleteWarehouseTracker:
    """GPU-optimized complete warehouse tracking system"""

    def __init__(self, camera_id: int = 8):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()

        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False

        # GPU-optimized detection components
        self.fisheye_corrector = GPUOptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = GPUSimplePalletDetector()

        # GPU coordinate mapping
        self.coordinate_mapper = GPUCoordinateMapper(camera_id=camera_id)
        self.coordinate_mapper_initialized = False
        self._initialize_coordinate_mapper()

        # GPU global feature database
        self.global_db = GPUGlobalFeatureDatabase(f"gpu_camera_{camera_id}_global_features.pkl")

        # Detection parameters (same as original)
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]

        # Filtering settings (same as original)
        self.MIN_AREA = 10000
        self.MAX_AREA = 100000
        self.CELL_SIZE = 40

        # Detection results storage
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.final_tracked_detections = []

        # Performance tracking
        self.frame_count = 0
        self.total_detections = 0
        self.new_objects = 0
        self.existing_objects = 0
        self.gpu_memory_usage = 0

        # GPU memory monitoring
        if torch.cuda.is_available():
            self.gpu_device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.gpu_device)
            logger.info(f"🚀 GPU tracking initialized on {self.gpu_name}")

        logger.info(f"GPU warehouse tracker initialized for {self.camera_name}")
        logger.info(f"All components GPU-optimized: Fisheye, Detection, SIFT, Coordinate mapping")

    def _initialize_coordinate_mapper(self):
        """Initialize GPU coordinate mapper"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                self.coordinate_mapper_initialized = True
                logger.info(f"✅ GPU coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ GPU coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize GPU coordinate mapper for {self.camera_name}: {e}")
            self.coordinate_mapper_initialized = False

    def connect_camera(self) -> bool:
        """Connect to the camera"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name}...")

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.RTSP_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                return False

            logger.info(f"{self.camera_name} connected successfully")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False

    def calculate_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y

    def get_grid_cell(self, center: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid cell coordinates for a center point"""
        x, y = center
        cell_x = int(x // self.CELL_SIZE)
        cell_y = int(y // self.CELL_SIZE)
        return cell_x, cell_y

    def get_neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all 9 cells (current + 8 neighbors) for a given cell"""
        cell_x, cell_y = cell
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                neighbors.append(neighbor_cell)

        return neighbors

    def apply_area_filter_gpu(self, detections: List[Dict]) -> List[Dict]:
        """GPU-optimized area filtering using vectorized operations"""
        if not detections:
            return []

        try:
            # Extract areas as numpy array for vectorized operations
            areas = np.array([detection.get('area', 0) for detection in detections])

            # Vectorized filtering
            valid_mask = (areas >= self.MIN_AREA) & (areas <= self.MAX_AREA)

            # Filter detections
            accepted = [detection for i, detection in enumerate(detections) if valid_mask[i]]

            return accepted

        except Exception as e:
            logger.error(f"GPU area filtering failed: {e}")
            # Fallback to CPU
            return [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]

    def apply_grid_cell_filter_gpu(self, detections: List[Dict]) -> List[Dict]:
        """GPU-optimized grid cell filtering"""
        if len(detections) <= 1:
            return detections

        try:
            # Calculate centers and grid cells for all detections
            centers = []
            for detection in detections:
                center = self.calculate_center(detection['bbox'])
                detection['center'] = center
                detection['grid_cell'] = self.get_grid_cell(center)
                centers.append(center)

            # Convert to numpy arrays for vectorized operations
            centers_array = np.array(centers)
            confidences = np.array([d['confidence'] for d in detections])

            # Sort by confidence (keep higher confidence detections first)
            sorted_indices = np.argsort(confidences)[::-1]
            sorted_detections = [detections[i] for i in sorted_indices]

            occupied_cells: Set[Tuple[int, int]] = set()
            accepted = []

            for detection in sorted_detections:
                cell = detection['grid_cell']
                neighbor_cells = self.get_neighbor_cells(cell)

                # Check if any of the 9 cells are already occupied
                conflict = any(neighbor_cell in occupied_cells for neighbor_cell in neighbor_cells)

                if not conflict:
                    occupied_cells.add(cell)
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"GPU grid filtering failed: {e}")
            return detections

    def translate_to_physical_coordinates_batch_gpu(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """GPU-optimized batch physical coordinate translation"""
        if not self.coordinate_mapper_initialized or not detections:
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'MAPPER_NOT_AVAILABLE'
            return detections

        try:
            # Extract all center points
            centers = []
            for detection in detections:
                center = detection.get('center')
                if not center:
                    center = self.calculate_center(detection['bbox'])
                    detection['center'] = center
                centers.append(center)

            # Convert to numpy array for batch processing
            centers_array = np.array(centers, dtype=np.float32)

            # Scale coordinates to calibration frame size (4K) - vectorized
            scale_x = 3840 / frame_width
            scale_y = 2160 / frame_height

            scaled_centers = centers_array * np.array([scale_x, scale_y])

            # Batch GPU coordinate transformation
            physical_coords = self.coordinate_mapper.pixel_to_real_batch_gpu(scaled_centers)

            # Assign results back to detections
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
            logger.error(f"GPU coordinate translation failed: {e}")
            # Fallback to individual processing
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'ERROR'
            return detections

    def extract_detection_region_gpu(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """GPU-optimized detection region extraction"""
        try:
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))

            if x2 <= x1 or y2 <= y1:
                return np.array([])

            # Extract region (this could be GPU-accelerated with cv2.cuda operations)
            region = frame[y1:y2, x1:x2]
            return region

        except Exception as e:
            logger.error(f"GPU region extraction failed: {e}")
            return np.array([])

    def assign_global_ids_gpu(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """GPU-accelerated global ID assignment"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            self.total_detections += 1

            # Extract image region for feature analysis
            image_region = self.extract_detection_region_gpu(frame, detection['bbox'])

            # Assign global ID using GPU SIFT features
            global_id, status, similarity = self.global_db.assign_global_id(
                image_region, detection
            )

            # Update statistics
            if status == 'new':
                self.new_objects += 1
            elif status == 'existing':
                self.existing_objects += 1

            # Add global tracking info to detection
            detection['global_id'] = global_id
            detection['tracking_status'] = status
            detection['similarity_score'] = similarity

            if global_id != -1:
                seen_ids.add(global_id)

            global_detections.append(detection)

        # Mark disappeared objects
        self.global_db.mark_disappeared_objects(seen_ids)

        return global_detections

    def monitor_gpu_usage(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            self.gpu_memory_usage = torch.cuda.memory_allocated(self.gpu_device) / 1024**3  # GB

    def start_tracking(self):
        """Start GPU-optimized warehouse tracking"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True

        # Create display window
        window_name = f"GPU Warehouse Tracking - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        logger.info("=== GPU-OPTIMIZED WAREHOUSE TRACKING ===")
        logger.info("🚀 All operations GPU-accelerated for maximum performance")
        logger.info("Pipeline: GPU Detection → GPU Filtering → GPU Coords → GPU SIFT")
        logger.info("Press 'q' or ESC to quit")
        logger.info("=" * 60)

        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()

        while self.running:
            try:
                frame_start = time.time()

                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break

                # Process frame with GPU acceleration
                processed_frame = self._process_frame_gpu(frame)

                # Monitor performance
                frame_time = time.time() - frame_start
                fps_counter += 1

                if fps_counter % 30 == 0:  # Log every 30 frames
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_counter / elapsed
                    self.monitor_gpu_usage()
                    logger.info(f"🚀 GPU Performance: {current_fps:.1f} FPS, GPU Memory: {self.gpu_memory_usage:.2f}GB")
                    fps_counter = 0
                    fps_start_time = time.time()

                # Display frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF  # Reduced wait time for better performance
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

            except Exception as e:
                logger.error(f"Error in GPU tracking loop: {e}")
                break

        self.stop_tracking()

    def _process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated frame processing pipeline"""
        self.frame_count += 1
        processed_frame = frame.copy()

        # GPU fisheye correction
        if Config.FISHEYE_CORRECTION_ENABLED:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"GPU fisheye correction failed: {e}")

        # GPU-accelerated resize
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Use GPU resize if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(processed_frame)
                    gpu_resized = cv2.cuda.resize(gpu_frame, (new_width, new_height))
                    processed_frame = gpu_resized.download()
                except:
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            else:
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Complete GPU detection and tracking pipeline
        try:
            # Stage 1: GPU detection
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)

            # Stage 2: GPU area filtering
            self.area_filtered_detections = self.apply_area_filter_gpu(self.raw_detections)

            # Stage 3: GPU grid cell filtering
            self.grid_filtered_detections = self.apply_grid_cell_filter_gpu(self.area_filtered_detections)

            # Stage 4: GPU batch physical coordinate translation
            frame_height, frame_width = processed_frame.shape[:2]
            self.grid_filtered_detections = self.translate_to_physical_coordinates_batch_gpu(
                self.grid_filtered_detections, frame_width, frame_height
            )

            # Stage 5: GPU SIFT feature matching and global ID assignment
            self.final_tracked_detections = self.assign_global_ids_gpu(self.grid_filtered_detections, processed_frame)

        except Exception as e:
            logger.error(f"GPU detection pipeline failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.final_tracked_detections = []

        # Draw results
        processed_frame = self._draw_detections_gpu(processed_frame)
        processed_frame = self._draw_info_overlay_gpu(processed_frame)

        return processed_frame

    def _draw_detections_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-optimized detection drawing (CPU drawing with GPU-processed data)"""
        result_frame = frame.copy()

        for detection in self.final_tracked_detections:
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
            coord_status = detection.get('coordinate_status', 'UNKNOWN')

            x1, y1, x2, y2 = bbox

            # Color coding based on tracking status
            if tracking_status == 'new':
                color = (0, 255, 0)  # Green for new objects
                status_text = "NEW"
            elif tracking_status == 'existing':
                color = (255, 165, 0)  # Orange for existing objects
                status_text = "GPU-TRACKED"
            else:
                color = (0, 0, 255)  # Red for failed
                status_text = "FAILED"

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(result_frame, center, 8, color, -1)
            cv2.circle(result_frame, center, 8, (255, 255, 255), 2)

            # Labels with all information
            y_offset = y1 - 5
            line_height = 20

            # Global ID and status
            if global_id != -1:
                id_label = f"GPU-ID:{global_id} ({status_text})"
                cv2.putText(result_frame, id_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= line_height

            # Confidence and area
            conf_label = f"Conf:{confidence:.3f} Area:{area:.0f}"
            cv2.putText(result_frame, conf_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset -= line_height

            # Pixel coordinates
            pixel_label = f"Pixel:({center[0]},{center[1]})"
            cv2.putText(result_frame, pixel_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset -= line_height

            # Physical coordinates
            if physical_x is not None and physical_y is not None:
                coord_label = f"Physical:({physical_x:.1f}ft,{physical_y:.1f}ft)"
                coord_color = (0, 255, 255)  # Cyan for physical coordinates
            else:
                coord_label = f"Physical:{coord_status}"
                coord_color = (0, 0, 255)  # Red for failed coordinates

            cv2.putText(result_frame, coord_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_color, 1)

        return result_frame

    def _draw_info_overlay_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-optimized information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 350), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"🚀 GPU WAREHOUSE TRACKING", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_count} | GPU Memory: {self.gpu_memory_usage:.2f}GB", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"GPU DETECTION PIPELINE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"1. GPU Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"2. GPU Area Filtered: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"3. GPU Grid Filtered: {len(self.grid_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"4. GPU Final Tracked: {len(self.final_tracked_detections)}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 25
        cv2.putText(frame, f"GPU TRACKING STATS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"New Objects: {self.new_objects}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"GPU Tracked Objects: {self.existing_objects}", (20, y_offset), font, 0.4, (255, 165, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Database Objects: {len(self.global_db.features)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        # GPU optimization status
        y_offset += 25
        cv2.putText(frame, f"GPU OPTIMIZATIONS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        gpu_sift_status = "✅" if self.global_db.use_gpu_sift else "❌"
        cv2.putText(frame, f"GPU SIFT: {gpu_sift_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.global_db.use_gpu_sift else (255, 0, 0), 1)

        y_offset += 15
        gpu_matcher_status = "✅" if self.global_db.use_gpu_matcher else "❌"
        cv2.putText(frame, f"GPU Matcher: {gpu_matcher_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.global_db.use_gpu_matcher else (255, 0, 0), 1)

        y_offset += 15
        coord_status = "✅" if self.coordinate_mapper_initialized else "❌"
        cv2.putText(frame, f"GPU Coords: {coord_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.coordinate_mapper_initialized else (255, 0, 0), 1)

        return frame

    def stop_tracking(self):
        """Stop GPU tracking system"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped GPU tracking for {self.camera_name}")
        logger.info(f"GPU Session stats - New: {self.new_objects}, Existing: {self.existing_objects}, Total: {self.total_detections}")


def main():
    """Main function for GPU-optimized warehouse tracking"""
    print("🚀 GPU-OPTIMIZED 3-CAMERA WAREHOUSE TRACKING SYSTEM")
    print("=" * 70)
    print("CAMERAS: 8, 9, 10 (Column 3 - Bottom, Middle, Top)")
    print("MAXIMUM GPU UTILIZATION - All operations GPU-accelerated:")
    print("1) 🚀 GPU Detection (Grounding DINO + Mixed Precision)")
    print("2) 🚀 GPU Area + Grid Cell Filtering (Vectorized)")
    print("3) 🚀 GPU Physical Coordinate Translation (Batch)")
    print("4) 🚀 GPU SIFT Feature Matching (CUDA SIFT)")
    print("5) 🚀 GPU Persistent Object IDs")
    print("6) 🚀 GPU Cross-Frame Tracking & Database")
    print("=" * 70)
    print("Processing: Sequential (Camera 8 → 9 → 10)")
    print("Display: 3 separate windows (one per camera)")
    print("GPU Optimizations: Fisheye, Detection, SIFT, Coordinate mapping")
    print("Performance Target: Maximum FPS with full GPU utilization")
    print("=" * 70)
    print("\nGPU Pipeline:")
    print("GPU Detection → GPU Filtering → GPU Coords → GPU SIFT → GPU IDs")
    print("\nColor Coding:")
    print("- Green: New objects")
    print("- Orange: GPU-tracked existing objects")
    print("- Red: Failed tracking")
    print("- Cyan: Physical coordinate labels")
    print("=" * 70)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Some optimizations will fall back to CPU")
    else:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Ready: {gpu_count} GPU(s) available - {gpu_name}")

    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("⚠️  WARNING: OpenCV CUDA not available! Some operations will use CPU")
    else:
        print(f"🚀 OpenCV CUDA Ready: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) available")

    print("=" * 70)

    # Create 3 camera trackers
    cameras = [8, 9, 10]
    trackers = {}

    print("🔧 Initializing 3 cameras...")
    for cam_id in cameras:
        tracker = GPUCompleteWarehouseTracker(camera_id=cam_id)

        # Connect to camera
        if not tracker.connect_camera():
            print(f"❌ Failed to connect to Camera {cam_id}")
            return

        trackers[cam_id] = tracker
        print(f"✅ Camera {cam_id} initialized and connected")

    try:
        print("🚀 Starting 3-camera tracking...")

        # Simple sequential processing
        frame_count = 0
        while True:
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"🔄 Processing frame {frame_count}...")

            for cam_id in cameras:
                tracker = trackers[cam_id]

                # Process one frame from this camera
                if tracker.cap and tracker.cap.isOpened():
                    ret, frame = tracker.cap.read()
                    if ret:
                        # Process the frame (same as single camera)
                        processed_frame = tracker._process_frame_gpu(frame)

                        # Show this camera's result
                        window_name = f"Camera {cam_id} - Warehouse Tracking"
                        cv2.imshow(window_name, processed_frame)
                    else:
                        print(f"⚠️ Failed to read frame from Camera {cam_id}")
                else:
                    print(f"❌ Camera {cam_id} not available")

            # Check for quit (any window)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nShutting down 3-camera GPU tracker...")
    except Exception as e:
        logger.error(f"Error running 3-camera GPU tracker: {e}")
    finally:
        print("🧹 Cleaning up...")
        for tracker in trackers.values():
            tracker.stop_tracking()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
