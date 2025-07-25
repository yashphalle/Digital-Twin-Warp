#!/usr/bin/env python3
"""
Complete Warehouse Tracking System
Integrates all functionalities:
1) Detection
2) Area + Grid Cell Filtering  
3) Physical Coordinate Translation
4) SIFT Feature Matching
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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinateMapper:
    """Maps pixel coordinates to real-world coordinates using homography"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        """Initialize coordinate mapper with floor dimensions in FEET"""
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.is_calibrated = False
        
        logger.info(f"Coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")

    def load_calibration(self, filename=None):
        """Load calibration from JSON file"""
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
            
            logger.info(f"Coordinate calibration loaded from: {filename}")
            logger.info(f"Camera local area: {self.floor_width_ft:.1f}ft x {self.floor_length_ft:.1f}ft")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates in FEET"""
        if not self.is_calibrated or self.homography_matrix is None:
            return None, None
        
        try:
            pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
            
            global_x = float(real_point[0][0][0])
            global_y = float(real_point[0][0][1])

            return global_x, global_y
        
        except Exception as e:
            logger.error(f"Error in pixel_to_real conversion: {e}")
            return None, None

class GlobalFeatureDatabase:
    """Global feature database for warehouse tracking"""
    
    def __init__(self, database_file: str = "warehouse_global_features.pkl"):
        self.database_file = database_file
        self.features = {}  # Global ID -> Feature data
        self.next_global_id = 1000
        self.load_database()
        
        # SIFT for feature extraction
        self.sift = cv2.SIFT_create(
            nfeatures=500,
            contrastThreshold=0.04,
            edgeThreshold=10
        )
        
        # FLANN matcher for feature comparison
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Matching parameters
        self.similarity_threshold = 0.3
        self.min_matches = 10
        self.max_disappeared_frames = 30  # Remove after 30 frames
        
        logger.info(f"Global feature database initialized with {len(self.features)} objects")
    
    def load_database(self):
        """Load feature database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.features = data.get('features', {})
                    self.next_global_id = data.get('next_id', 1000)
                logger.info(f"Loaded {len(self.features)} objects from database")
            else:
                self.features = {}
                self.next_global_id = 1000
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.features = {}
            self.next_global_id = 1000
    
    def save_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.features,
                'next_id': self.next_global_id,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def extract_features(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT features from image region"""
        if image_region is None or image_region.size == 0:
            return None
        
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) < self.min_matches:
            return None
        
        return descriptors
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature sets"""
        if features1 is None or features2 is None:
            return 0.0
        
        if len(features1) < 2 or len(features2) < 2:
            return 0.0
        
        try:
            matches = self.flann.knnMatch(features1, features2, k=2)
            
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
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_matching_object(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Find best matching object in database"""
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, feature_data in self.features.items():
            stored_features = feature_data['features']
            similarity = self.calculate_similarity(query_features, stored_features)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match_id = global_id
                best_similarity = similarity
        
        return best_match_id, best_similarity
    
    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Add new object to database and return global ID"""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        feature_data = {
            'features': features,
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'times_seen': 1,
            'disappeared_frames': 0,
            'detection_info': detection_info,
            'physical_locations': []  # Store physical coordinate history
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
        
        logger.info(f"🆕 NEW GLOBAL ID: {global_id}")
        return global_id
    
    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Update existing object in database"""
        if global_id in self.features:
            feature_data = self.features[global_id]
            
            feature_data['features'] = features
            feature_data['last_seen'] = datetime.now().isoformat()
            feature_data['times_seen'] += 1
            feature_data['disappeared_frames'] = 0  # Reset disappeared counter
            
            # Add physical coordinates if available
            if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                    feature_data['physical_locations'].append({
                        'timestamp': datetime.now().isoformat(),
                        'x_ft': detection_info['physical_x_ft'],
                        'y_ft': detection_info['physical_y_ft']
                    })
                    
                    # Keep only recent locations (last 100)
                    if len(feature_data['physical_locations']) > 100:
                        feature_data['physical_locations'] = feature_data['physical_locations'][-100:]
            
            self.save_database()
            logger.info(f"🔄 UPDATED GLOBAL ID: {global_id} - Times seen: {feature_data['times_seen']}")
    
    def mark_disappeared_objects(self, seen_ids: Set[int]):
        """Mark objects as disappeared and cleanup old ones"""
        to_remove = []
        
        for global_id in self.features:
            if global_id not in seen_ids:
                self.features[global_id]['disappeared_frames'] += 1
                
                if self.features[global_id]['disappeared_frames'] >= self.max_disappeared_frames:
                    to_remove.append(global_id)
        
        # Remove old objects
        for global_id in to_remove:
            logger.info(f"🗑️ REMOVED GLOBAL ID: {global_id} - Disappeared for {self.max_disappeared_frames} frames")
            del self.features[global_id]
        
        if to_remove:
            self.save_database()
    
    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """
        Assign global ID to detection
        Returns: (global_id, status, similarity_score)
        """
        features = self.extract_features(image_region)
        if features is None:
            return -1, 'failed', 0.0
        
        match_id, similarity = self.find_matching_object(features)
        
        if match_id is not None:
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

class CompleteWarehouseTracker:
    """Complete warehouse tracking system with all functionalities"""

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

        # Detection components
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = SimplePalletDetector()

        # Coordinate mapping for physical coordinates
        self.coordinate_mapper = CoordinateMapper(camera_id=camera_id)
        self.coordinate_mapper_initialized = False
        self._initialize_coordinate_mapper()

        # Global feature database for SIFT tracking
        self.global_db = GlobalFeatureDatabase(f"camera_{camera_id}_global_features.pkl")

        # DETECTION PARAMETERS (same as combined_filtering_detection.py)
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]

        # Update DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.1
            self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt

        # FILTERING SETTINGS (same as combined_filtering_detection.py)
        self.MIN_AREA = 10000   # Exclude very small noise
        self.MAX_AREA = 100000  # Exclude very large background objects
        self.CELL_SIZE = 40     # 40x40 pixel cells for better accuracy

        # Detection results storage
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.final_tracked_detections = []

        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.new_objects = 0
        self.existing_objects = 0

        logger.info(f"Complete warehouse tracker initialized for {self.camera_name}")
        logger.info(f"Confidence: {self.pallet_detector.confidence_threshold}")
        logger.info(f"Area filter: {self.MIN_AREA} - {self.MAX_AREA} pixels")
        logger.info(f"Grid cell size: {self.CELL_SIZE}x{self.CELL_SIZE} pixels")

    def _initialize_coordinate_mapper(self):
        """Initialize coordinate mapper with camera-specific calibration"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                self.coordinate_mapper_initialized = True
                logger.info(f"✅ Coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ Coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinate mapper for {self.camera_name}: {e}")
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

            # Test frame capture
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

    def apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply area-based filtering"""
        accepted = []

        for detection in detections:
            area = detection.get('area', 0)

            if self.MIN_AREA <= area <= self.MAX_AREA:
                accepted.append(detection)

        return accepted

    def apply_grid_cell_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply grid cell filtering"""
        if len(detections) <= 1:
            return detections

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

    def translate_to_physical_coordinates(self, detection: Dict, frame_width: int, frame_height: int) -> Dict:
        """Translate detection pixel coordinates to physical warehouse coordinates"""
        if not self.coordinate_mapper_initialized:
            detection['physical_x_ft'] = None
            detection['physical_y_ft'] = None
            detection['coordinate_status'] = 'MAPPER_NOT_AVAILABLE'
            return detection

        try:
            center = detection.get('center')
            if not center:
                center = self.calculate_center(detection['bbox'])
                detection['center'] = center

            center_x, center_y = center

            # Scale coordinates to calibration frame size (4K)
            scale_x = 3840 / frame_width
            scale_y = 2160 / frame_height

            scaled_center_x = center_x * scale_x
            scaled_center_y = center_y * scale_y

            # Convert to physical coordinates
            physical_x, physical_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

            if physical_x is not None and physical_y is not None:
                detection['physical_x_ft'] = round(physical_x, 2)
                detection['physical_y_ft'] = round(physical_y, 2)
                detection['coordinate_status'] = 'SUCCESS'
            else:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'CONVERSION_FAILED'

        except Exception as e:
            logger.error(f"Error translating coordinates: {e}")
            detection['physical_x_ft'] = None
            detection['physical_y_ft'] = None
            detection['coordinate_status'] = 'ERROR'

        return detection

    def extract_detection_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract image region for feature analysis"""
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

            region = frame[y1:y2, x1:x2]
            return region

        except Exception as e:
            logger.error(f"Error extracting detection region: {e}")
            return np.array([])

    def assign_global_ids(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Assign global IDs to filtered detections using SIFT"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            self.total_detections += 1

            # Extract image region for feature analysis
            image_region = self.extract_detection_region(frame, detection['bbox'])

            # Assign global ID using SIFT features
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

    def start_tracking(self):
        """Start the complete warehouse tracking system"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True

        # Create display window
        window_name = f"Complete Warehouse Tracking - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        logger.info("=== COMPLETE WAREHOUSE TRACKING SYSTEM ===")
        logger.info("Pipeline: Detection → Area Filter → Grid Filter → Physical Coords → SIFT Tracking")
        logger.info("Press 'q' or ESC to quit")
        logger.info("=" * 60)

        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break

                # Process frame
                processed_frame = self._process_frame(frame)

                # Display frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                break

        self.stop_tracking()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with complete pipeline"""
        self.frame_count += 1
        processed_frame = frame.copy()

        # Apply fisheye correction if enabled
        if Config.FISHEYE_CORRECTION_ENABLED:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")

        # Resize for display
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Complete detection and tracking pipeline
        try:
            # Stage 1: Raw detection
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)

            # Stage 2: Area filtering
            self.area_filtered_detections = self.apply_area_filter(self.raw_detections)

            # Stage 3: Grid cell filtering
            self.grid_filtered_detections = self.apply_grid_cell_filter(self.area_filtered_detections)

            # Stage 4: Physical coordinate translation
            frame_height, frame_width = processed_frame.shape[:2]
            for detection in self.grid_filtered_detections:
                self.translate_to_physical_coordinates(detection, frame_width, frame_height)

            # Stage 5: SIFT feature matching and global ID assignment
            self.final_tracked_detections = self.assign_global_ids(self.grid_filtered_detections, processed_frame)

        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.final_tracked_detections = []

        # Draw results
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)

        return processed_frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracked detections with comprehensive information"""
        result_frame = frame.copy()

        for detection in self.final_tracked_detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            area = detection.get('area', 0)

            # Global tracking info
            global_id = detection.get('global_id', -1)
            tracking_status = detection.get('tracking_status', 'unknown')
            similarity_score = detection.get('similarity_score', 0.0)

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
                status_text = "TRACKED"
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
                id_label = f"ID:{global_id} ({status_text})"
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

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw comprehensive system information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 300), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"COMPLETE WAREHOUSE TRACKING", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"DETECTION PIPELINE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"1. Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"2. Area Filtered: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"3. Grid Filtered: {len(self.grid_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"4. Final Tracked: {len(self.final_tracked_detections)}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 25
        cv2.putText(frame, f"TRACKING STATS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"New Objects: {self.new_objects}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Existing Objects: {self.existing_objects}", (20, y_offset), font, 0.4, (255, 165, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Database Objects: {len(self.global_db.features)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        # Coordinate mapping status
        y_offset += 25
        if self.coordinate_mapper_initialized:
            coord_status = "✅ ENABLED"
            coord_color = (0, 255, 0)
        else:
            coord_status = "❌ DISABLED"
            coord_color = (0, 0, 255)
        cv2.putText(frame, f"Physical Coords: {coord_status}", (20, y_offset), font, 0.4, coord_color, 1)

        return frame

    def stop_tracking(self):
        """Stop the tracking system"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped tracking for {self.camera_name}")
        logger.info(f"Session stats - New: {self.new_objects}, Existing: {self.existing_objects}, Total: {self.total_detections}")


def main():
    """Main function"""
    print("COMPLETE WAREHOUSE TRACKING SYSTEM")
    print("=" * 60)
    print("Integrates ALL functionalities:")
    print("1) Detection (Grounding DINO)")
    print("2) Area + Grid Cell Filtering")
    print("3) Physical Coordinate Translation")
    print("4) SIFT Feature Matching")
    print("5) Persistent Object IDs")
    print("6) Cross-Frame Tracking & Database")
    print("=" * 60)
    print("Camera: 8 (Column 3 - Bottom)")
    print("Physical Coordinates: Warehouse coordinate system (feet)")
    print("Global Database: Persistent across sessions")
    print("=" * 60)
    print("\nPipeline:")
    print("Detection → Area Filter → Grid Filter → Physical Coords → SIFT → Global IDs")
    print("\nColor Coding:")
    print("- Green: New objects")
    print("- Orange: Tracked existing objects")
    print("- Red: Failed tracking")
    print("- Cyan: Physical coordinate labels")
    print("=" * 60)

    tracker = CompleteWarehouseTracker(camera_id=8)

    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("\nShutting down tracker...")
    except Exception as e:
        logger.error(f"Error running tracker: {e}")
    finally:
        tracker.stop_tracking()


if __name__ == "__main__":
    main()
