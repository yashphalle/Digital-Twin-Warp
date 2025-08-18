#!/usr/bin/env python3
"""
Feature Database Module
CPU-based global feature database with CPU SIFT for object tracking
Extracted from main.py for modular architecture
"""

import cv2
import numpy as np
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CPUGlobalFeatureDatabase:
    """CPU-based global feature database with CPU SIFT (same as combined filtering)"""

    def __init__(self, database_file: str = "cpu_warehouse_global_features.pkl", camera_id: int = 1):
        # Update database file path to use data/features directory
        if not database_file.startswith("data/features/"):
            database_file = f"data/features/{os.path.basename(database_file)}"
        
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
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
            
            data = {
                'features': self.features,
                'next_id': self.next_global_id
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

            similarity = len(good_matches) / max(len(features1), len(features2))
            return similarity

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
            'bbox': detection_info.get('bbox'),
            'confidence': detection_info.get('confidence'),
            'area': detection_info.get('area'),
            'physical_x_ft': detection_info.get('physical_x_ft'),
            'physical_y_ft': detection_info.get('physical_y_ft'),
            'coordinate_status': detection_info.get('coordinate_status'),
            'camera_id': self.camera_id
        }

        # Add color information if available
        if 'dominant_color' in detection_info:
            feature_data['dominant_color'] = detection_info['dominant_color']
        if 'color_name' in detection_info:
            feature_data['color_name'] = detection_info['color_name']

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
            if 'physical_x_ft' in detection_info:
                feature_data['physical_x_ft'] = detection_info['physical_x_ft']
            if 'physical_y_ft' in detection_info:
                feature_data['physical_y_ft'] = detection_info['physical_y_ft']
            if 'coordinate_status' in detection_info:
                feature_data['coordinate_status'] = detection_info['coordinate_status']

            # Update color information if available
            if 'dominant_color' in detection_info:
                feature_data['dominant_color'] = detection_info['dominant_color']
            if 'color_name' in detection_info:
                feature_data['color_name'] = detection_info['color_name']

            self.save_database()

    def cleanup_disappeared_objects(self):
        """Remove objects that haven't been seen for too long"""
        to_remove = []
        for global_id, feature_data in self.features.items():
            feature_data['disappeared_frames'] += 1
            if feature_data['disappeared_frames'] > self.max_disappeared_frames:
                to_remove.append(global_id)

        for global_id in to_remove:
            del self.features[global_id]
            logger.info(f"🗑️ Removed disappeared object {global_id}")

        if to_remove:
            self.save_database()

    def mark_disappeared_objects(self, seen_ids: set):
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
