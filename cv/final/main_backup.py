#!/usr/bin/env python3
"""
CPU-Based Complete Warehouse Tracking System (Copy of GPU script)
Modified to use CPU-based detection like combined filtering script
Uses same detection method as combined_filtering_detection.py
1) Detection (CPU - post_process_grounded_object_detection)
2) Area + Grid Cell Filtering (CPU)  
3) Physical Coordinate Translation (CPU)
4) CPU SIFT Feature Matching
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
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
# sklearn import - will fallback to simple mean if not available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available, using simple color extraction")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from warehouse_database_handler import WarehouseDatabaseHandler

# Import utility modules
from utils.detection_utils import process_detection_results
from utils.coordinate_converter import create_coordinate_converter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUSimplePalletDetector:
    """Hybrid pallet detector: GPU for Grounding DINO inference, CPU for post-processing"""
    
    def __init__(self):
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt_index = 0
        self.current_prompt = self.sample_prompts[0]
        
        # Initialize detection with GPU for Grounding DINO if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"🚀 Using GPU for Grounding DINO: {torch.cuda.get_device_name()}")
            logger.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device("cpu")
            logger.info("⚠️ GPU not available, using CPU for Grounding DINO")

        logger.info(f"🔍 Initializing pallet detector on {self.device}")
        
        # Initialize Grounding DINO model
        self._initialize_grounding_dino()
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model for GPU inference (CPU fallback if GPU unavailable)"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("✅ AutoProcessor loaded successfully")
            
            # Load model and move to selected device (GPU preferred)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Grounding DINO model loaded on {self.device}")

            # Log GPU memory usage if using GPU
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"📊 GPU Memory allocated: {memory_allocated:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to initialize Grounding DINO: {e}")
            self.processor = None
            self.model = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """CPU-based pallet detection using same method as combined filtering"""
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
            
            # Move inputs to selected device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # GPU/CPU inference with automatic mixed precision if GPU available
            if self.device.type == 'cuda':
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            # Process results using SAME METHOD as combined filtering
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )

            # Clear GPU cache if using GPU to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Convert to detection format (SAME as combined filtering)
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
            logger.error(f"CPU detection failed: {e}")
            return []

    def next_prompt(self):
        """Switch to next prompt"""
        self.current_prompt_index = (self.current_prompt_index + 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
    
    def previous_prompt(self):
        """Switch to previous prompt"""
        self.current_prompt_index = (self.current_prompt_index - 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]

# Removed custom fisheye corrector - using imported OptimizedFisheyeCorrector

class ObjectColorExtractor:
    """Extract dominant colors from detected object regions"""

    def __init__(self):
        self.min_pixels = 100  # Minimum pixels needed for reliable color extraction
        logger.info("✅ Object color extractor initialized")

    def extract_dominant_color(self, image_region: np.ndarray) -> Dict:
        """Extract dominant color from object region and return HSV + RGB values"""
        if image_region is None or image_region.size == 0:
            return self._get_default_color()

        try:
            # Ensure we have enough pixels for reliable color extraction
            if image_region.size < self.min_pixels * 3:  # 3 channels
                return self._get_default_color()

            # Convert BGR to RGB for processing
            if len(image_region.shape) == 3:
                rgb_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            else:
                return self._get_default_color()

            # Reshape for color analysis
            pixels = rgb_region.reshape(-1, 3)

            # Remove very dark pixels (shadows) and very bright pixels (highlights)
            brightness = np.mean(pixels, axis=1)
            valid_pixels = pixels[(brightness > 30) & (brightness < 225)]

            if len(valid_pixels) < 50:  # Not enough valid pixels
                valid_pixels = pixels  # Use all pixels as fallback

            # Use K-means clustering if available, otherwise simple mean
            if SKLEARN_AVAILABLE:
                try:
                    n_clusters = min(3, len(valid_pixels) // 10)  # Adaptive cluster count
                    if n_clusters < 1:
                        n_clusters = 1

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(valid_pixels)

                    # Get the most frequent color cluster
                    colors = kmeans.cluster_centers_
                    labels = kmeans.labels_

                    unique, counts = np.unique(labels, return_counts=True)
                    dominant_idx = unique[np.argmax(counts)]
                    dominant_rgb = colors[dominant_idx]

                    # Calculate color confidence based on cluster dominance
                    color_confidence = np.max(counts) / len(labels)

                except Exception as e:
                    logger.warning(f"K-means clustering failed, using simple mean: {e}")
                    # Fallback: simple mean color
                    dominant_rgb = np.mean(valid_pixels, axis=0)
                    color_confidence = 0.5  # Medium confidence for fallback method
            else:
                # Simple mean color when sklearn not available
                dominant_rgb = np.mean(valid_pixels, axis=0)
                color_confidence = 0.6  # Good confidence for mean method

            # Ensure RGB values are in valid range
            dominant_rgb = np.clip(dominant_rgb, 0, 255).astype(int)

            # Convert RGB to HSV for better color representation
            rgb_normalized = dominant_rgb.reshape(1, 1, 3).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0, 0]

            return {
                'color_rgb': [int(dominant_rgb[0]), int(dominant_rgb[1]), int(dominant_rgb[2])],
                'color_hsv': [int(hsv[0]), int(hsv[1]), int(hsv[2])],
                'color_hex': f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}",
                'color_confidence': float(color_confidence),
                'color_name': self._get_color_name(hsv),
                'extraction_method': 'kmeans_clustering'
            }

        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return self._get_default_color()

    def _get_default_color(self) -> Dict:
        """Return default gray color when extraction fails"""
        return {
            'color_rgb': [128, 128, 128],
            'color_hsv': [0, 0, 128],
            'color_hex': "#808080",
            'color_confidence': 0.0,
            'color_name': 'gray',
            'extraction_method': 'default_fallback'
        }

    def _get_color_name(self, hsv: np.ndarray) -> str:
        """Get human-readable color name from HSV values"""
        h, s, v = hsv

        # Low saturation = grayscale
        if s < 30:
            if v < 50:
                return 'black'
            elif v > 200:
                return 'white'
            else:
                return 'gray'

        # Low value = dark colors
        if v < 50:
            return 'dark'

        # Categorize by hue
        if h < 10 or h > 170:
            return 'red'
        elif h < 25:
            return 'orange'
        elif h < 35:
            return 'yellow'
        elif h < 85:
            return 'green'
        elif h < 125:
            return 'blue'
        elif h < 150:
            return 'purple'
        else:
            return 'pink'

class CoordinateMapper:
    """CPU-based coordinate mapping (same as combined filtering)"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        
        logger.info(f"CPU coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")

    def load_calibration(self, filename=None):
        """Load calibration from JSON file (same as combined filtering)"""
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
            self.is_calibrated = True
            logger.info(f"CPU coordinate calibration loaded from: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real(self, pixel_x, pixel_y):
        """Single point coordinate transformation"""
        if not self.is_calibrated:
            return None, None
        
        try:
            points = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
            result = transformed_points.reshape(-1, 2)
            if len(result) > 0:
                return float(result[0][0]), float(result[0][1])
            return None, None
        except Exception as e:
            logger.error(f"CPU coordinate transformation failed: {e}")
            return None, None

class CPUGlobalFeatureDatabase:
    """CPU-based global feature database with CPU SIFT (same as combined filtering)"""

    def __init__(self, database_file: str = "cpu_warehouse_global_features.pkl", camera_id: int = 1):
        self.database_file = database_file
        self.camera_id = camera_id
        self.features = {}
        # Camera-prefixed Global IDs: Camera 8 → 8001, 8002, 8003...
        self.next_global_id = camera_id * 1000 + 1
        self.load_database()

        logger.info(f"🎯 Camera {camera_id} Global ID range: {camera_id}001 - {camera_id}999")

        # Use CPU SIFT (same as combined filtering)
        try:
            self.cpu_sift = cv2.SIFT_create(
                nfeatures=500,
                contrastThreshold=0.04,
                edgeThreshold=10
            )
            logger.info("✅ CPU SIFT detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CPU SIFT: {e}")
            self.cpu_sift = None

        # CPU-based feature matching
        self._init_cpu_matcher()

        # Matching parameters (same as combined filtering)
        self.similarity_threshold = 0.3
        self.min_matches = 10
        self.max_disappeared_frames = 30

        logger.info(f"CPU feature database initialized with {len(self.features)} objects")

    def _init_cpu_matcher(self):
        """Initialize CPU FLANN matcher"""
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
                    self.next_global_id = data.get('next_id', self.camera_id * 1000 + 1)
                logger.info(f"Loaded {len(self.features)} objects from CPU database")
            else:
                self.features = {}
                self.next_global_id = self.camera_id * 1000 + 1
        except Exception as e:
            logger.error(f"Error loading CPU database: {e}")
            self.features = {}
            self.next_global_id = self.camera_id * 1000 + 1

    def save_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.features,
                'next_id': self.next_global_id,
                'last_updated': datetime.now().isoformat(),
                'cpu_optimized': True
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving CPU database: {e}")

    def extract_features_cpu(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """CPU SIFT feature extraction (same as combined filtering)"""
        if image_region is None or image_region.size == 0 or self.cpu_sift is None:
            return None

        try:
            # Convert to grayscale if needed
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region

            # CPU SIFT extraction
            _, descriptors = self.cpu_sift.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) >= self.min_matches:
                return descriptors
            return None

        except Exception as e:
            logger.error(f"CPU feature extraction failed: {e}")
            return None

    def calculate_similarity_cpu(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """CPU feature similarity calculation (same as combined filtering)"""
        if features1 is None or features2 is None:
            return 0.0

        if len(features1) < 2 or len(features2) < 2:
            return 0.0

        try:
            # CPU FLANN matching
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
            logger.error(f"CPU similarity calculation failed: {e}")
            return 0.0

    def find_matching_object(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Find best matching object using CPU"""
        best_match_id = None
        best_similarity = 0.0

        for global_id, feature_data in self.features.items():
            stored_features = feature_data['features']
            similarity = self.calculate_similarity_cpu(query_features, stored_features)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match_id = global_id
                best_similarity = similarity

        return best_match_id, best_similarity

    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Add new object to CPU database"""
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
            'cpu_optimized': True
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

        logger.info(f"🆕 NEW CAMERA {self.camera_id} GLOBAL ID: {global_id}")
        return global_id

    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Update existing object in CPU database"""
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
            logger.info(f"🔄 UPDATED CPU GLOBAL ID: {global_id} - Times seen: {feature_data['times_seen']}")

    def mark_disappeared_objects(self, seen_ids: Set[int]):
        """Mark objects as disappeared and cleanup old ones"""
        to_remove = []

        for global_id in self.features:
            if global_id not in seen_ids:
                self.features[global_id]['disappeared_frames'] += 1

                if self.features[global_id]['disappeared_frames'] >= self.max_disappeared_frames:
                    to_remove.append(global_id)

        for global_id in to_remove:
            camera_id = global_id // 1000  # Extract camera ID from global ID
            object_num = global_id % 1000   # Extract object number
            logger.info(f"🗑️ REMOVED CAMERA {camera_id} OBJECT #{object_num} (ID: {global_id}) - Disappeared for {self.max_disappeared_frames} frames")
            del self.features[global_id]

        if to_remove:
            self.save_database()

    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """Assign global ID using CPU feature matching"""
        features = self.extract_features_cpu(image_region)
        if features is None:
            return -1, 'failed', 0.0

        match_id, similarity = self.find_matching_object(features)

        if match_id is not None:
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

class CPUCompleteWarehouseTracker:
    """CPU-based complete warehouse tracking system (same logic as combined filtering)"""

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

        # CPU-based detection components (same as combined filtering)
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = CPUSimplePalletDetector()

        # CPU coordinate mapping
        self.coordinate_mapper = CoordinateMapper(camera_id=camera_id)
        self.coordinate_mapper_initialized = False
        self._initialize_coordinate_mapper()

        # CPU global feature database with camera-specific ID ranges
        self.global_db = CPUGlobalFeatureDatabase(f"cpu_camera_{camera_id}_global_features.pkl", camera_id)

        # Color extraction for real object colors
        self.color_extractor = ObjectColorExtractor()

        # Test color extraction with a simple red image
        test_image = np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8)  # Red image
        test_color = self.color_extractor.extract_dominant_color(test_image)
        logger.info(f"🧪 Color extractor test: {test_color}")

        # Database handler for MongoDB integration (same as GPU script)
        self.db_handler = WarehouseDatabaseHandler(
            mongodb_url="mongodb://localhost:27017/",
            database_name="warehouse_tracking",
            collection_name="detections",
            batch_save_size=10,
            enable_mongodb=True
        )

        # Detection parameters (same as combined filtering)
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]

        # Filtering settings (same as combined filtering)
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

        logger.info(f"Hybrid warehouse tracker initialized for {self.camera_name}")
        logger.info(f"🚀 Detection: GPU-accelerated Grounding DINO")
        logger.info(f"🔧 Processing: CPU-based SIFT, Coordinates, Database")

    def _initialize_coordinate_mapper(self):
        """Initialize CPU coordinate mapper"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                self.coordinate_mapper_initialized = True
                logger.info(f"✅ CPU coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ CPU coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize CPU coordinate mapper for {self.camera_name}: {e}")
            self.coordinate_mapper_initialized = False

    def connect_camera(self) -> bool:
        """Connect to the camera"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name}...")

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Set aggressive timeout settings to prevent hanging
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout

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

    def apply_area_filter_cpu(self, detections: List[Dict]) -> List[Dict]:
        """CPU area filtering (same as combined filtering)"""
        if not detections:
            return []

        try:
            accepted = []
            for detection in detections:
                area = detection.get('area', 0)
                if self.MIN_AREA <= area <= self.MAX_AREA:
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"CPU area filtering failed: {e}")
            return [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]

    def apply_grid_cell_filter_cpu(self, detections: List[Dict]) -> List[Dict]:
        """CPU grid cell filtering (same as combined filtering)"""
        if len(detections) <= 1:
            return detections

        try:
            # Calculate centers and grid cells for all detections
            for detection in detections:
                center = self.calculate_center(detection['bbox'])
                detection['center'] = center
                detection['grid_cell'] = self.get_grid_cell(center)

            # Sort by confidence (keep higher confidence detections first)
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            occupied_cells: Set[Tuple[int, int]] = set()
            accepted = []

            for detection in sorted_detections:
                cell = detection['grid_cell']
                neighbor_cells = self.get_neighbor_cells(cell)

                # Check if any of the 9 cells are already occupied
                conflict = False
                for neighbor_cell in neighbor_cells:
                    if neighbor_cell in occupied_cells:
                        conflict = True
                        break

                if not conflict:
                    # Accept this detection
                    occupied_cells.add(cell)
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"CPU grid filtering failed: {e}")
            return detections

    def translate_to_physical_coordinates_cpu(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """CPU physical coordinate translation (same as combined filtering)"""
        if not self.coordinate_mapper_initialized:
            return detections

        try:
            for detection in detections:
                center = detection.get('center')
                if center is None:
                    center = self.calculate_center(detection['bbox'])
                    detection['center'] = center

                center_x, center_y = center

                # Scale coordinates to calibration frame size (4K) for accurate coordinate mapping
                # Calibration files are based on 3840x2160 resolution (same as combined filtering)
                scale_x = 3840 / frame_width
                scale_y = 2160 / frame_height

                scaled_center_x = center_x * scale_x
                scaled_center_y = center_y * scale_y

                # Transform to physical coordinates using scaled coordinates
                physical_x, physical_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

                if physical_x is not None and physical_y is not None:
                    detection['physical_x_ft'] = round(physical_x, 2)
                    detection['physical_y_ft'] = round(physical_y, 2)
                    detection['coordinate_status'] = 'SUCCESS'
                    logger.debug(f"Camera {self.camera_id}: Pixel ({center_x}, {center_y}) → Scaled ({scaled_center_x:.1f}, {scaled_center_y:.1f}) → Physical ({physical_x:.2f}ft, {physical_y:.2f}ft)")
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'CONVERSION_FAILED'
                    logger.debug(f"Camera {self.camera_id}: Coordinate conversion failed for pixel ({center_x}, {center_y})")

            return detections

        except Exception as e:
            logger.error(f"CPU coordinate translation failed: {e}")
            return detections

    def assign_global_ids_cpu(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """CPU global ID assignment (same as combined filtering)"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            # Initialize global_id to prevent scope errors
            global_id = None

            try:
                # Extract image region for SIFT features and color analysis
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                image_region = frame[y1:y2, x1:x2]

                # Extract dominant color from detected object
                color_info = self.color_extractor.extract_dominant_color(image_region)
                detection.update(color_info)  # Add color data to detection

                # Assign global ID using CPU SIFT features
                global_id, status, similarity = self.global_db.assign_global_id(
                    image_region, detection
                )

                # Debug: Log color extraction after successful ID assignment
                logger.info(f"🎨 Color extracted for object {global_id}: {color_info}")
                logger.info(f"🔍 Detection data after color update: {list(detection.keys())}")

                # Update counters
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

            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                # Set default values for failed detection
                detection['global_id'] = -1
                detection['tracking_status'] = 'failed'
                detection['similarity_score'] = 0.0

            global_detections.append(detection)

        # Mark disappeared objects
        self.global_db.mark_disappeared_objects(seen_ids)

        return global_detections

    def _process_frame_cpu(self, frame: np.ndarray) -> np.ndarray:
        """CPU-based frame processing (same pipeline as combined filtering)"""
        processed_frame = frame.copy()

        # Apply fisheye correction
        processed_frame = self.fisheye_corrector.correct(processed_frame)

        # Resize if too large (for performance)
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Complete CPU detection and tracking pipeline
        try:
            # Stage 1: CPU detection (same as combined filtering)
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)

            # Stage 2: CPU area filtering
            self.area_filtered_detections = self.apply_area_filter_cpu(self.raw_detections)

            # Stage 3: CPU grid cell filtering
            self.grid_filtered_detections = self.apply_grid_cell_filter_cpu(self.area_filtered_detections)

            # Stage 4: CPU physical coordinate translation
            frame_height, frame_width = processed_frame.shape[:2]
            self.grid_filtered_detections = self.translate_to_physical_coordinates_cpu(
                self.grid_filtered_detections, frame_width, frame_height
            )

            # Stage 5: CPU SIFT feature matching and global ID assignment
            self.final_tracked_detections = self.assign_global_ids_cpu(self.grid_filtered_detections, processed_frame)

            # Stage 6: Save detections to database (same as GPU script)
            if self.db_handler.is_connected():
                for detection in self.final_tracked_detections:
                    self.db_handler.save_detection_to_db(self.camera_id, detection)

        except Exception as e:
            logger.error(f"CPU detection pipeline failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.final_tracked_detections = []

        # Draw results
        processed_frame = self._draw_detections_cpu(processed_frame)
        processed_frame = self._draw_info_overlay_cpu(processed_frame)

        return processed_frame

    def _draw_detections_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results (same as combined filtering)"""
        result_frame = frame.copy()

        for detection in self.final_tracked_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection.get('confidence', 0.0)
            area = detection.get('area', 0)
            center = detection.get('center', self.calculate_center(bbox))

            # Color based on tracking status
            tracking_status = detection.get('tracking_status', 'unknown')
            if tracking_status == 'new':
                color = (0, 255, 0)  # Green for new objects
                status_text = "NEW"
            elif tracking_status == 'existing':
                color = (0, 165, 255)  # Orange for existing objects
                status_text = "TRACKED"
            else:
                color = (0, 0, 255)  # Red for failed tracking
                status_text = "FAILED"

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(result_frame, center, 8, color, -1)

            # Text positioning
            line_height = 20
            y_offset = y1 - 10

            # Global ID and status
            global_id = detection.get('global_id', -1)
            if global_id != -1:
                id_label = f"CPU-ID:{global_id} ({status_text})"
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
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')
            if physical_x is not None and physical_y is not None:
                coord_label = f"Physical:({physical_x:.1f},{physical_y:.1f})ft"
                coord_color = (0, 255, 255)  # Cyan for successful coordinates
            else:
                coord_label = "Physical: FAILED"
                coord_color = (0, 0, 255)  # Red for failed coordinates

            cv2.putText(result_frame, coord_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_color, 1)

        return result_frame

    def _draw_info_overlay_cpu(self, frame: np.ndarray) -> np.ndarray:
        """CPU information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 350), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"🚀 CPU WAREHOUSE TRACKING", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"CPU DETECTION PIPELINE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"1. CPU Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"2. CPU Area Filtered: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"3. CPU Grid Filtered: {len(self.grid_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"4. CPU Final Tracked: {len(self.final_tracked_detections)}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 25
        cv2.putText(frame, f"CPU TRACKING STATS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"New Objects: {self.new_objects}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"CPU Tracked Objects: {self.existing_objects}", (20, y_offset), font, 0.4, (255, 165, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Database Objects: {len(self.global_db.features)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        # CPU optimization status
        y_offset += 25
        cv2.putText(frame, f"CPU OPTIMIZATIONS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"CPU SIFT: ✅", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"CPU Matcher: ✅", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        coord_status = "✅" if self.coordinate_mapper_initialized else "❌"
        cv2.putText(frame, f"CPU Coords: {coord_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.coordinate_mapper_initialized else (255, 0, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Color Extract: ✅", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        return frame

    def start_detection(self):
        """Start CPU-based detection"""
        logger.info("=== CPU-BASED WAREHOUSE TRACKING ===")
        logger.info("🚀 All operations CPU-based for compatibility")
        logger.info("Pipeline: CPU Detection → CPU Filtering → CPU Coords → CPU SIFT")
        logger.info("Press 'q' or ESC to quit")
        logger.info("=" * 60)

        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True
        frame_count = 0

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue

                frame_count += 1
                self.frame_count = frame_count

                # Process frame
                processed_frame = self._process_frame_cpu(frame)

                # Display result
                cv2.imshow(f"CPU Tracking - {self.camera_name}", processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")

                # Log progress every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count}: {len(self.final_tracked_detections)} objects tracked")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()

        # Cleanup database handler (same as GPU script)
        if hasattr(self, 'db_handler'):
            self.db_handler.cleanup()

        cv2.destroyAllWindows()
        logger.info("CPU tracking system shutdown complete")

class MultiCameraCPUSystem:
    """Multi-camera CPU-based tracking system"""

    def __init__(self, active_cameras, gui_cameras, enable_gui=True):
        self.active_cameras = active_cameras
        self.gui_cameras = gui_cameras if enable_gui else []
        self.enable_gui = enable_gui
        self.trackers = {}
        self.running = False

        logger.info("🎛️ Multi-Camera CPU System Configuration:")
        logger.info(f"📹 Active Cameras: {self.active_cameras}")
        logger.info(f"🖥️ GUI Cameras: {self.gui_cameras}")
        logger.info(f"🎛️ GUI Mode: {'ENABLED' if self.enable_gui else 'HEADLESS'}")

    def initialize_cameras(self) -> bool:
        """Initialize all active cameras"""
        logger.info(f"🔧 Initializing {len(self.active_cameras)} cameras...")

        for cam_id in self.active_cameras:
            try:
                logger.info(f"🔧 Initializing Camera {cam_id}...")
                tracker = CPUCompleteWarehouseTracker(camera_id=cam_id)

                if tracker.connect_camera():
                    self.trackers[cam_id] = tracker
                    logger.info(f"✅ Camera {cam_id} initialized successfully")
                else:
                    logger.error(f"❌ Failed to connect Camera {cam_id}")

            except Exception as e:
                logger.error(f"❌ Error initializing Camera {cam_id}: {e}")

        connected_cameras = len(self.trackers)
        if connected_cameras == 0:
            logger.error("❌ No cameras connected successfully!")
            return False

        logger.info(f"🚀 {connected_cameras} out of {len(self.active_cameras)} cameras initialized successfully!")
        return True

    def run(self):
        """Run multi-camera CPU tracking"""
        if not self.initialize_cameras():
            return

        logger.info("🚀 Starting multi-camera CPU tracking...")
        self.running = True
        frame_count = 0

        try:
            while self.running:
                frame_count += 1

                # Process each active camera
                for cam_id in self.active_cameras:
                    if cam_id not in self.trackers:
                        continue

                    tracker = self.trackers[cam_id]
                    if tracker.cap and tracker.cap.isOpened():
                        ret, frame = tracker.cap.read()
                        if ret:
                            # Process frame
                            processed_frame = tracker._process_frame_cpu(frame)

                            # Show GUI if enabled for this camera
                            if self.enable_gui and cam_id in self.gui_cameras:
                                cv2.imshow(f"CPU Camera {cam_id}", processed_frame)

                # Handle key presses
                if self.enable_gui:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break

                # Log progress every 30 frames
                if frame_count % 30 == 0:
                    total_detections = sum(len(tracker.final_tracked_detections) for tracker in self.trackers.values())
                    total_db_saved = sum(tracker.db_handler.get_detection_count() for tracker in self.trackers.values() if hasattr(tracker, 'db_handler'))
                    logger.info(f"Frame {frame_count}: {total_detections} total objects tracked, {total_db_saved} saved to DB across {len(self.trackers)} cameras")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during multi-camera tracking: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources"""
        self.running = False

        # Get final database statistics before cleanup
        total_db_saved = sum(tracker.db_handler.get_detection_count() for tracker in self.trackers.values() if hasattr(tracker, 'db_handler'))

        for tracker in self.trackers.values():
            tracker.cleanup()

        cv2.destroyAllWindows()
        logger.info(f"💾 Total detections saved to database: {total_db_saved}")
        logger.info("✅ Multi-camera CPU system shutdown complete")

# ============================================================================
# CONFIGURATION
# ============================================================================

# 📹 SIMPLE CAMERA CONFIGURATION - EDIT THESE LISTS:
# =======================================================

# 🎯 DETECTION CAMERAS: Add camera numbers you want to run detection on
ACTIVE_CAMERAS = [1]  # Cameras that will detect objects

# 🖥️ GUI CAMERAS: Add camera numbers you want to see windows for
GUI_CAMERAS = [1]  # Cameras that will show GUI windows (subset of ACTIVE_CAMERAS)

# 🎛️ GUI CONFIGURATION
ENABLE_GUI = True  # Set to False for headless mode
ENABLE_CONSOLE_LOGGING = True  # Print logs to console

print(f"🔥 CPU RUNNING CAMERAS: {ACTIVE_CAMERAS}")
print(f"🖥️ GUI WINDOWS FOR: {GUI_CAMERAS if ENABLE_GUI else 'NONE (HEADLESS)'}")

def main():
    """Main function for CPU-based 11-camera warehouse tracking"""
    print("🚀 CPU-BASED 11-CAMERA WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("CONFIGURATION:")
    print(f"📹 Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    print(f"🖥️ GUI Cameras: {GUI_CAMERAS if ENABLE_GUI else 'DISABLED'} ({len(GUI_CAMERAS) if ENABLE_GUI else 0} windows)")
    print(f"🎛️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    print("=" * 80)
    print("CPU-BASED PROCESSING - All operations CPU-based:")
    print("1) 🚀 CPU Detection (Grounding DINO + post_process_grounded_object_detection)")
    print("2) 🚀 CPU Area + Grid Cell Filtering")
    print("3) 🚀 CPU Physical Coordinate Translation")
    print("4) 🚀 CPU SIFT Feature Matching")
    print("5) 🚀 CPU Persistent Object IDs")
    print("6) 🚀 CPU Cross-Frame Tracking & Database")
    print("=" * 80)
    if ENABLE_GUI:
        print("\nGUI Mode:")
        print("- Green: New objects")
        print("- Orange: CPU-tracked existing objects")
        print("- Red: Failed tracking")
        print("- Cyan: Physical coordinate labels")
        print("- Press 'q' to quit")
        print("- Press 'n'/'p' to change detection prompts")
    else:
        print("\nHEADLESS Mode: No GUI windows, console logging only")
    print("=" * 80)

    # Initialize multi-camera system
    multi_camera_system = MultiCameraCPUSystem(
        active_cameras=ACTIVE_CAMERAS,
        gui_cameras=GUI_CAMERAS,
        enable_gui=ENABLE_GUI
    )

    # Run the system
    multi_camera_system.run()

if __name__ == "__main__":
    main()
