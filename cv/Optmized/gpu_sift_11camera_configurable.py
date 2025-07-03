#!/usr/bin/env python3
"""
GPU SIFT-Based Complete Warehouse Tracking System
Enhanced version with GPU-accelerated SIFT for faster feature extraction
1) Detection (GPU - Grounding DINO)
2) Area + Grid Cell Filtering (CPU)  
3) Physical Coordinate Translation (CPU)
4) GPU SIFT Feature Matching (CUDA accelerated)
5) Persistent Object IDs with camera-prefixed ranges
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUSIFTPalletDetector:
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
            self.model = None
            self.processor = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """Detect pallets using Grounding DINO (same as combined filtering)"""
        if self.model is None or self.processor is None:
            return []
        
        try:
            # Convert BGR to RGB and create PIL image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Prepare inputs
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
                result = results[0]
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels = result.get("labels", [])
                
                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.tolist()
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(score),
                            'label': label,
                            'area': int((x2 - x1) * (y2 - y1)),
                            'prompt_used': self.current_prompt
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def switch_prompt(self, direction='next'):
        """Switch between prompts"""
        if direction == 'next':
            self.current_prompt_index = (self.current_prompt_index + 1) % len(self.sample_prompts)
        else:  # previous
            self.current_prompt_index = (self.current_prompt_index - 1) % len(self.sample_prompts)
        
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
        logger.info(f"🔄 Switched to prompt: '{self.current_prompt}'")

class GPUSIFTGlobalFeatureDatabase:
    """GPU-accelerated SIFT feature database with camera-specific ID ranges"""
    
    def __init__(self, database_file: str = "gpu_sift_warehouse_global_features.pkl", camera_id: int = 1):
        self.database_file = database_file
        self.camera_id = camera_id
        self.features = {}
        # Camera-prefixed Global IDs: Camera 8 → 8001, 8002, 8003...
        self.next_global_id = camera_id * 1000 + 1
        self.load_database()
        
        logger.info(f"🎯 Camera {camera_id} Global ID range: {camera_id}001 - {camera_id}999")
        
        # Initialize GPU SIFT if available
        self._initialize_gpu_sift()
        
        # Matching parameters
        self.min_matches = 10
        self.max_disappeared_frames = 30
        
    def _initialize_gpu_sift(self):
        """Initialize GPU SIFT detector if CUDA is available"""
        try:
            # Check if OpenCV was compiled with CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info(f"🚀 CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
                
                # Try to create GPU SIFT detector
                try:
                    self.gpu_sift = cv2.cuda.SIFT_create(
                        nfeatures=500,
                        nOctaveLayers=3,
                        contrastThreshold=0.04,
                        edgeThreshold=10,
                        sigma=1.6
                    )
                    self.use_gpu_sift = True
                    logger.info("✅ GPU SIFT detector initialized successfully")
                    
                    # Initialize GPU memory objects for reuse
                    self.gpu_frame = cv2.cuda_GpuMat()
                    self.gpu_gray = cv2.cuda_GpuMat()
                    
                except Exception as e:
                    logger.warning(f"⚠️ GPU SIFT creation failed: {e}")
                    self._fallback_to_cpu_sift()
            else:
                logger.info("⚠️ No CUDA devices available")
                self._fallback_to_cpu_sift()
                
        except Exception as e:
            logger.warning(f"⚠️ CUDA check failed: {e}")
            self._fallback_to_cpu_sift()
    
    def _fallback_to_cpu_sift(self):
        """Fallback to CPU SIFT if GPU SIFT is not available"""
        try:
            self.cpu_sift = cv2.SIFT_create(
                nfeatures=500,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
            self.use_gpu_sift = False
            logger.info("✅ CPU SIFT detector initialized (GPU fallback)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CPU SIFT: {e}")
            self.cpu_sift = None
            self.use_gpu_sift = False

    def load_database(self):
        """Load feature database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.features = data.get('features', {})
                    self.next_global_id = data.get('next_id', self.camera_id * 1000 + 1)
                logger.info(f"Loaded {len(self.features)} objects from GPU SIFT database")
            else:
                self.features = {}
                self.next_global_id = self.camera_id * 1000 + 1
        except Exception as e:
            logger.error(f"Error loading GPU SIFT database: {e}")
            self.features = {}
            self.next_global_id = self.camera_id * 1000 + 1

    def save_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.features,
                'next_id': self.next_global_id,
                'last_updated': datetime.now().isoformat(),
                'gpu_sift_optimized': True,
                'camera_id': self.camera_id
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving GPU SIFT database: {e}")

    def extract_features_gpu(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT features using GPU acceleration"""
        if not self.use_gpu_sift or self.gpu_sift is None:
            return self.extract_features_cpu(image_region)

        try:
            # Convert to grayscale if needed
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region

            # Upload to GPU
            self.gpu_frame.upload(gray)

            # GPU SIFT extraction
            _, descriptors = self.gpu_sift.detectAndCompute(self.gpu_frame, None)

            # Download descriptors from GPU
            if descriptors is not None:
                descriptors_cpu = descriptors.download()
                if descriptors_cpu is not None and len(descriptors_cpu) >= self.min_matches:
                    return descriptors_cpu

            return None

        except Exception as e:
            logger.warning(f"GPU SIFT extraction failed: {e}, falling back to CPU")
            return self.extract_features_cpu(image_region)

    def extract_features_cpu(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT features using CPU (fallback)"""
        if self.cpu_sift is None:
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
            logger.error(f"CPU SIFT extraction failed: {e}")
            return None

    def match_features(self, features1: np.ndarray, features2: np.ndarray) -> Tuple[int, float]:
        """Match SIFT features using FLANN matcher"""
        try:
            # FLANN parameters for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(features1, features2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            num_matches = len(good_matches)
            if num_matches > 0:
                avg_distance = sum(m.distance for m in good_matches) / num_matches
                similarity = max(0, 1 - (avg_distance / 256))  # Normalize to 0-1
                return num_matches, similarity

            return 0, 0.0

        except Exception as e:
            logger.error(f"Feature matching error: {e}")
            return 0, 0.0

    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Add new object to GPU SIFT database"""
        global_id = self.next_global_id
        self.next_global_id += 1

        feature_data = {
            'features': features,
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'disappeared_frames': 0,
            'detection_count': 1,
            'camera_id': self.camera_id,
            'extraction_method': 'gpu_sift' if self.use_gpu_sift else 'cpu_sift',
            'locations': []
        }

        # Add location data if available
        if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
            feature_data['locations'].append({
                'timestamp': datetime.now().isoformat(),
                'x_ft': detection_info['physical_x_ft'],
                'y_ft': detection_info['physical_y_ft']
            })

        self.features[global_id] = feature_data
        self.save_database()

        logger.info(f"🆕 NEW CAMERA {self.camera_id} GLOBAL ID: {global_id} ({'GPU' if self.use_gpu_sift else 'CPU'} SIFT)")
        return global_id

    def update_object(self, global_id: int, detection_info: Dict):
        """Update existing object in database"""
        if global_id in self.features:
            self.features[global_id]['last_seen'] = datetime.now().isoformat()
            self.features[global_id]['disappeared_frames'] = 0
            self.features[global_id]['detection_count'] += 1

            # Add new location
            if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                self.features[global_id]['locations'].append({
                    'timestamp': datetime.now().isoformat(),
                    'x_ft': detection_info['physical_x_ft'],
                    'y_ft': detection_info['physical_y_ft']
                })

    def cleanup_disappeared_objects(self, seen_ids: Set[int]):
        """Remove objects that haven't been seen for too long"""
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
        """Assign global ID using GPU/CPU SIFT feature matching"""
        # Extract features using GPU or CPU
        if self.use_gpu_sift:
            features = self.extract_features_gpu(image_region)
        else:
            features = self.extract_features_cpu(image_region)

        if features is None:
            return -1, 'failed', 0.0

        best_match_id = -1
        best_similarity = 0.0

        # Compare with existing objects
        for global_id, stored_data in self.features.items():
            stored_features = stored_data['features']

            try:
                match_count, similarity = self.match_features(features, stored_features)

                if match_count >= self.min_matches and similarity > best_similarity:
                    best_match_id = global_id
                    best_similarity = similarity

            except Exception as e:
                logger.warning(f"Error matching with object {global_id}: {e}")
                continue

        # Decide whether to match or create new object
        if best_match_id != -1 and best_similarity > 0.6:
            self.update_object(best_match_id, detection_info)
            return best_match_id, 'matched', best_similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

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
            if image_region.size < self.min_pixels:
                return self._get_default_color()

            # Reshape image to list of pixels
            pixels = image_region.reshape(-1, 3)

            # Remove very dark pixels (likely shadows/noise)
            brightness = np.mean(pixels, axis=1)
            bright_pixels = pixels[brightness > 30]

            if len(bright_pixels) < 10:
                bright_pixels = pixels  # Fallback to all pixels

            if SKLEARN_AVAILABLE:
                # Use K-means clustering to find dominant color
                kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
                kmeans.fit(bright_pixels)
                dominant_bgr = kmeans.cluster_centers_[0].astype(int)
            else:
                # Simple mean color if sklearn not available
                dominant_bgr = np.mean(bright_pixels, axis=0).astype(int)

            # Convert BGR to RGB and HSV
            dominant_rgb = [int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])]

            # Convert to HSV for better color analysis
            bgr_pixel = np.uint8([[dominant_bgr]])
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
            dominant_hsv = [int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])]

            # Generate hex color
            hex_color = f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}"

            # Determine color name based on HSV values
            color_name = self._get_color_name(dominant_hsv)

            return {
                'color_rgb': dominant_rgb,
                'color_hsv': dominant_hsv,
                'color_hex': hex_color,
                'color_name': color_name,
                'color_confidence': 0.8,  # Confidence score
                'extraction_method': 'kmeans' if SKLEARN_AVAILABLE else 'mean'
            }

        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
            return self._get_default_color()

    def _get_color_name(self, hsv: List[int]) -> str:
        """Determine color name from HSV values"""
        h, s, v = hsv

        # Low saturation = grayscale
        if s < 30:
            if v < 50:
                return "black"
            elif v < 130:
                return "gray"
            else:
                return "white"

        # High saturation = colored
        if h < 10 or h > 170:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 125:
            return "blue"
        elif h < 155:
            return "purple"
        else:
            return "pink"

    def _get_default_color(self) -> Dict:
        """Return default color when extraction fails"""
        return {
            'color_rgb': [128, 128, 128],
            'color_hsv': [0, 0, 128],
            'color_hex': "#808080",
            'color_name': "gray",
            'color_confidence': 0.1,
            'extraction_method': 'default'
        }

class GPUSIFTWarehouseTracker:
    """GPU SIFT-accelerated warehouse tracking system"""

    def __init__(self, camera_id: int, camera_name: str = None):
        self.camera_id = camera_id
        self.camera_name = camera_name or f"Camera {camera_id}"

        logger.info(f"🚀 Initializing GPU SIFT warehouse tracker for {self.camera_name}")

        # Detection system with GPU acceleration
        self.detector = GPUSIFTPalletDetector()

        # Fisheye correction
        self.fisheye_corrector = OptimizedFisheyeCorrector()

        # GPU SIFT global feature database with camera-specific ID ranges
        self.global_db = GPUSIFTGlobalFeatureDatabase(f"gpu_sift_camera_{camera_id}_global_features.pkl", camera_id)

        # Color extraction for real object colors
        self.color_extractor = ObjectColorExtractor()

        # Database handler for MongoDB integration
        self.db_handler = WarehouseDatabaseHandler(
            mongodb_url="mongodb://localhost:27017/",
            database_name="warehouse_tracking",
            collection_name="detections"
        )

        # Warehouse configuration
        self.warehouse_config = get_warehouse_config()

        # Filtering parameters
        self.min_area = 15000
        self.max_area = 1000000
        self.grid_size = 3

        # Tracking variables
        self.frame_count = 0
        self.detection_count = 0
        self.new_objects = 0
        self.existing_objects = 0

        # Processing stages storage
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.coordinate_mapped_detections = []
        self.final_tracked_detections = []

        logger.info(f"GPU SIFT warehouse tracker initialized for {self.camera_name}")
        logger.info(f"🚀 Detection: GPU-accelerated Grounding DINO + GPU SIFT")
        logger.info(f"🔧 Processing: CPU-based Coordinates, Database")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the complete GPU SIFT pipeline"""
        self.frame_count += 1

        # Apply fisheye correction
        processed_frame = self.fisheye_corrector.correct(frame)

        # Stage 1: GPU-accelerated pallet detection
        self.raw_detections = self.detector.detect_pallets(processed_frame)

        # Stage 2: Area filtering
        self.area_filtered_detections = self.apply_area_filter(self.raw_detections)

        # Stage 3: Grid cell filtering (remove duplicates in same cell)
        self.grid_filtered_detections = self.apply_grid_filter(self.area_filtered_detections, processed_frame)

        # Stage 4: Physical coordinate translation
        self.coordinate_mapped_detections = self.translate_to_physical_coordinates(
            self.grid_filtered_detections, processed_frame.shape
        )

        # Stage 5: GPU SIFT feature matching and global ID assignment
        self.final_tracked_detections = self.assign_global_ids_gpu_sift(self.coordinate_mapped_detections, processed_frame)

        # Stage 6: Save detections to database
        if self.db_handler.is_connected():
            for detection in self.final_tracked_detections:
                self.db_handler.save_detection_to_db(self.camera_id, detection)

        # Update statistics
        self.detection_count = len(self.final_tracked_detections)

        # Cleanup disappeared objects
        seen_ids = {det['global_id'] for det in self.final_tracked_detections if det['global_id'] != -1}
        self.global_db.cleanup_disappeared_objects(seen_ids)

        # Draw results on frame
        result_frame = self.draw_detections(processed_frame.copy())

        return result_frame

    def apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections by area"""
        if not detections:
            return []

        try:
            accepted = []
            for detection in detections:
                area = detection.get('area', 0)
                if self.min_area <= area <= self.max_area:
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"Area filtering failed: {e}")
            return detections

    def apply_grid_filter(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply grid-based filtering to remove duplicates"""
        if not detections:
            return []

        try:
            height, width = frame.shape[:2]
            cell_width = width // self.grid_size
            cell_height = height // self.grid_size

            # Group detections by grid cell
            grid_cells = {}

            for detection in detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2

                cell_x = min(center_x // cell_width, self.grid_size - 1)
                cell_y = min(center_y // cell_height, self.grid_size - 1)
                cell_key = (cell_x, cell_y)

                if cell_key not in grid_cells:
                    grid_cells[cell_key] = []
                grid_cells[cell_key].append(detection)

            # Keep only the highest confidence detection per cell
            filtered_detections = []
            for cell_detections in grid_cells.values():
                if cell_detections:
                    best_detection = max(cell_detections, key=lambda d: d.get('confidence', 0))
                    filtered_detections.append(best_detection)

            return filtered_detections

        except Exception as e:
            logger.error(f"Grid filtering failed: {e}")
            return detections

    def translate_to_physical_coordinates(self, detections: List[Dict], frame_shape) -> List[Dict]:
        """Translate pixel coordinates to physical warehouse coordinates"""
        try:
            for detection in detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Get physical coordinates using warehouse config
                physical_coords = self.warehouse_config.pixel_to_physical(
                    self.camera_id, center_x, center_y
                )

                if physical_coords:
                    detection['physical_x_ft'] = physical_coords[0]
                    detection['physical_y_ft'] = physical_coords[1]
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None

            return detections

        except Exception as e:
            logger.error(f"Coordinate translation failed: {e}")
            return detections

    def assign_global_ids_gpu_sift(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """GPU SIFT global ID assignment"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            global_id = None

            try:
                # Extract image region for GPU SIFT features and color analysis
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                image_region = frame[y1:y2, x1:x2]

                # Extract dominant color from detected object
                color_info = self.color_extractor.extract_dominant_color(image_region)
                detection.update(color_info)  # Add color data to detection

                # Assign global ID using GPU SIFT features
                global_id, status, similarity = self.global_db.assign_global_id(
                    image_region, detection
                )

                # Update statistics
                if status == 'new':
                    self.new_objects += 1
                elif status == 'matched':
                    self.existing_objects += 1

                # Add global tracking info to detection
                detection['global_id'] = global_id
                detection['tracking_status'] = status
                detection['similarity_score'] = similarity
                detection['sift_method'] = 'gpu' if self.global_db.use_gpu_sift else 'cpu'

                if global_id != -1:
                    seen_ids.add(global_id)

            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                detection['global_id'] = -1
                detection['tracking_status'] = 'failed'
                detection['similarity_score'] = 0.0
                detection['sift_method'] = 'failed'

            global_detections.append(detection)

        return global_detections

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()

        # Draw final tracked detections
        for detection in self.final_tracked_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Color based on tracking status
            status = detection.get('tracking_status', 'unknown')
            if status == 'new':
                color = (0, 255, 0)  # Green for new objects
            elif status == 'matched':
                color = (0, 255, 255)  # Yellow for matched objects
            else:
                color = (0, 0, 255)  # Red for failed

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare text information
            confidence = detection.get('confidence', 0)
            area = detection.get('area', 0)
            global_id = detection.get('global_id', -1)
            sift_method = detection.get('sift_method', 'unknown')

            # Draw text labels
            line_height = 20
            y_offset = y1 - 10

            # Global ID and SIFT method
            if global_id != -1:
                id_label = f"GPU-SIFT-ID:{global_id} ({sift_method.upper()})"
                cv2.putText(result_frame, id_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= line_height

            # Confidence and area
            info_label = f"Conf:{confidence:.2f} Area:{area}"
            cv2.putText(result_frame, info_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= line_height

            # Physical coordinates
            phys_x = detection.get('physical_x_ft')
            phys_y = detection.get('physical_y_ft')
            if phys_x is not None and phys_y is not None:
                coord_label = f"({phys_x:.1f}, {phys_y:.1f})ft"
                cv2.putText(result_frame, coord_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw statistics
        stats_text = [
            f"GPU SIFT Tracker - {self.camera_name}",
            f"Frame: {self.frame_count}",
            f"Detections: {len(self.final_tracked_detections)}",
            f"New: {self.new_objects} | Existing: {self.existing_objects}",
            f"SIFT: {'GPU' if self.global_db.use_gpu_sift else 'CPU'} accelerated"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_frame

    def handle_key_input(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('n'):  # Next prompt
            self.detector.switch_prompt('next')
        elif key == ord('p'):  # Previous prompt
            self.detector.switch_prompt('previous')
        elif key == ord('q'):  # Quit
            return False
        return True

# Main execution function
def main():
    """Main function to run GPU SIFT warehouse tracking"""
    import argparse

    parser = argparse.ArgumentParser(description='GPU SIFT Warehouse Tracking System')
    parser.add_argument('--camera', type=int, default=8, help='Camera ID to use (1-11)')
    parser.add_argument('--cameras', nargs='+', type=int, help='Multiple camera IDs (e.g., --cameras 1 2 3)')
    args = parser.parse_args()

    if args.cameras:
        camera_ids = args.cameras
    else:
        camera_ids = [args.camera]

    logger.info(f"🚀 Starting GPU SIFT warehouse tracking for cameras: {camera_ids}")

    # Initialize trackers for each camera
    trackers = {}
    for camera_id in camera_ids:
        try:
            tracker = GPUSIFTWarehouseTracker(camera_id)
            trackers[camera_id] = tracker
            logger.info(f"✅ Camera {camera_id} tracker initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera {camera_id}: {e}")

    if not trackers:
        logger.error("❌ No trackers initialized successfully")
        return

    # For demo purposes, create a simple test loop
    logger.info("🎯 GPU SIFT tracking system ready!")
    logger.info("📋 Controls:")
    logger.info("   'n' - Next prompt")
    logger.info("   'p' - Previous prompt")
    logger.info("   'q' - Quit")

    # Start camera processing
    try:
        from cv.configs.config import Config

        # Process each camera
        for camera_id, tracker in trackers.items():
            logger.info(f"🎥 Starting camera {camera_id} processing...")

            # Get camera configuration
            camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
            rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

            if not rtsp_url:
                logger.warning(f"⚠️ No RTSP URL configured for camera {camera_id}")
                continue

            # Start camera stream processing
            process_camera_stream(tracker, camera_id, camera_name, rtsp_url)

    except ImportError:
        logger.error("❌ Config not available - cannot start camera streams")
    except Exception as e:
        logger.error(f"❌ Error starting camera streams: {e}")

    logger.info("✅ GPU SIFT warehouse tracking system completed!")

def process_camera_stream(tracker, camera_id: int, camera_name: str, rtsp_url: str):
    """Process video stream from a single camera"""
    logger.info(f"🎥 Connecting to {camera_name} (ID: {camera_id}): {rtsp_url}")

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error(f"❌ Failed to connect to {camera_name}")
        return

    logger.info(f"✅ Connected to {camera_name}")

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
    cap.set(cv2.CAP_PROP_FPS, 10)  # Limit FPS for processing

    # Set resolution to 1080p for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Verify resolution settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"📺 Camera resolution set to: {actual_width}x{actual_height}")

    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"⚠️ Failed to read frame from {camera_name}")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_counter += 1

            # Resize frame to 1080p for optimal performance
            original_height, original_width = frame.shape[:2]
            if original_height > 1080 or original_width > 1920:
                # Calculate scaling to fit within 1920x1080
                scale_w = 1920 / original_width
                scale_h = 1080 / original_height
                scale = min(scale_w, scale_h)

                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Process frame through GPU SIFT pipeline
            start_time = time.time()
            result_frame = tracker.process_frame(frame)
            process_time = time.time() - start_time

            # Calculate FPS every 30 frames
            if fps_counter >= 30:
                elapsed_time = time.time() - fps_start_time
                current_fps = fps_counter / elapsed_time

                # Get current frame dimensions for performance analysis
                frame_height, frame_width = frame.shape[:2]
                total_pixels = frame_width * frame_height

                logger.info(f"📊 {camera_name} - FPS: {current_fps:.1f}, Process: {process_time:.3f}s")
                logger.info(f"📺 Original: {original_width}x{original_height} → Processed: {frame_width}x{frame_height} ({total_pixels/1000000:.1f}MP)")
                logger.info(f"🔍 Detections: {len(tracker.final_tracked_detections)}")

                # Calculate performance improvement from resolution scaling
                original_pixels = original_width * original_height
                pixel_reduction = (1 - total_pixels / original_pixels) * 100
                logger.info(f"⚡ Pixel reduction: {pixel_reduction:.1f}% (performance boost)")

                fps_counter = 0
                fps_start_time = time.time()

            # Display result (optional - comment out for headless operation)
            cv2.imshow(f"GPU SIFT - {camera_name}", result_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not tracker.handle_key_input(key):
                break

    except KeyboardInterrupt:
        logger.info(f"🛑 Stopping {camera_name} processing...")
    except Exception as e:
        logger.error(f"❌ Error processing {camera_name}: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"✅ {camera_name} processing stopped")

if __name__ == "__main__":
    main()
