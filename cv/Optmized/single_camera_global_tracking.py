#!/usr/bin/env python3
"""
Single Camera Global Tracking Test
Tests global ID assignment and feature persistence for one camera
Foundation for multi-camera warehouse tracking system
"""

import cv2
import numpy as np
import logging
import sys
import os
import pickle
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalFeatureDatabase:
    """Global feature database for single camera testing"""
    
    def __init__(self, database_file: str = "global_features.pkl"):
        self.database_file = database_file
        self.features = {}  # Global ID -> Feature data
        self.next_global_id = 1000  # Start with 1000 for global IDs
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
        self.similarity_threshold = 0.3  # Minimum similarity for re-identification
        self.min_matches = 10  # Minimum good matches required
        
        logger.info(f"Global feature database initialized with {len(self.features)} objects")
    
    def extract_features(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT features from image region"""
        if image_region is None or image_region.size == 0:
            return None
        
        # Convert to grayscale
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        # Extract SIFT features
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
            # Find matches using FLANN
            matches = self.flann.knnMatch(features1, features2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity score
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
            'detection_info': detection_info,
            'confidence_history': [detection_info.get('confidence', 0.0)]
        }
        
        self.features[global_id] = feature_data
        self.save_database()
        
        logger.info(f"🆕 NEW GLOBAL ID: {global_id} - Added to database")
        return global_id
    
    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Update existing object in database"""
        if global_id in self.features:
            feature_data = self.features[global_id]
            
            # Update features (could implement feature fusion here)
            feature_data['features'] = features
            feature_data['last_seen'] = datetime.now().isoformat()
            feature_data['times_seen'] += 1
            feature_data['confidence_history'].append(detection_info.get('confidence', 0.0))
            
            # Keep only recent confidence history
            if len(feature_data['confidence_history']) > 50:
                feature_data['confidence_history'] = feature_data['confidence_history'][-50:]
            
            self.save_database()
            logger.info(f"🔄 UPDATED GLOBAL ID: {global_id} - Times seen: {feature_data['times_seen']}")
    
    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """
        Assign global ID to detection
        Returns: (global_id, status, similarity_score)
        Status: 'new', 'existing', 'failed'
        """
        # Extract features
        features = self.extract_features(image_region)
        if features is None:
            return -1, 'failed', 0.0
        
        # Search for matching object
        match_id, similarity = self.find_matching_object(features)
        
        if match_id is not None:
            # Existing object found
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            # New object
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0
    
    def save_database(self):
        """Save feature database to file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump({
                    'features': self.features,
                    'next_global_id': self.next_global_id
                }, f)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def load_database(self):
        """Load feature database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.features = data.get('features', {})
                    self.next_global_id = data.get('next_global_id', 1000)
                logger.info(f"Loaded database with {len(self.features)} objects")
            else:
                logger.info("No existing database found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.features = {}
            self.next_global_id = 1000
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        if not self.features:
            return {'total_objects': 0}
        
        total_objects = len(self.features)
        times_seen_list = [data['times_seen'] for data in self.features.values()]
        avg_times_seen = np.mean(times_seen_list) if times_seen_list else 0
        
        return {
            'total_objects': total_objects,
            'next_id': self.next_global_id,
            'avg_times_seen': avg_times_seen,
            'most_seen': max(times_seen_list) if times_seen_list else 0
        }


class SingleCameraGlobalTracker:
    """Single camera tracker with global ID assignment"""
    
    def __init__(self, camera_id: int = 11):
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
        
        # Set detection parameters
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]
        
        # Update DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.1
            self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
        
        # Filtering settings (from combined filtering)
        self.MIN_AREA = 10000
        self.MAX_AREA = 100000
        self.CELL_SIZE = 40
        
        # Global tracking
        self.global_db = GlobalFeatureDatabase(f"camera_{camera_id}_global_features.pkl")
        
        # Detection results
        self.raw_detections = []
        self.filtered_detections = []
        self.global_tracked_detections = []
        
        # Statistics
        self.session_stats = {
            'new_objects': 0,
            'existing_objects': 0,
            'failed_features': 0,
            'total_detections': 0
        }

        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.fps_update_interval = 10  # Update FPS every 10 frames (faster updates)
        self.frame_times = []  # Store recent frame times for smoother FPS calculation
        self.last_frame_time = time.time()  # For immediate FPS calculation
        
        logger.info(f"Initialized global tracker for {self.camera_name}")
        logger.info(f"Global database: {self.global_db.get_database_stats()}")
    
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

    def extract_detection_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract image region for feature extraction"""
        x1, y1, x2, y2 = bbox

        # Add padding for better feature extraction
        padding = 10
        height, width = frame.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        return frame[y1:y2, x1:x2]

    def apply_combined_filtering(self, detections: List[Dict]) -> List[Dict]:
        """Apply area and grid cell filtering (simplified version)"""
        # Area filtering
        area_filtered = []
        for detection in detections:
            area = detection.get('area', 0)
            if self.MIN_AREA <= area <= self.MAX_AREA:
                area_filtered.append(detection)

        # Simple grid cell filtering (basic version for testing)
        if len(area_filtered) <= 1:
            return area_filtered

        # Sort by confidence and apply simple distance filtering
        sorted_dets = sorted(area_filtered, key=lambda x: x['confidence'], reverse=True)
        filtered = []

        for detection in sorted_dets:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Check if too close to existing detections
            too_close = False
            for accepted in filtered:
                acc_bbox = accepted['bbox']
                acc_center = ((acc_bbox[0] + acc_bbox[2]) // 2, (acc_bbox[1] + acc_bbox[3]) // 2)

                distance = ((center[0] - acc_center[0])**2 + (center[1] - acc_center[1])**2)**0.5
                if distance < self.CELL_SIZE:
                    too_close = True
                    break

            if not too_close:
                filtered.append(detection)

        return filtered

    def assign_global_ids(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Assign global IDs to filtered detections"""
        global_detections = []

        for detection in detections:
            self.session_stats['total_detections'] += 1

            # Extract image region for feature analysis
            image_region = self.extract_detection_region(frame, detection['bbox'])

            # Assign global ID
            global_id, status, similarity = self.global_db.assign_global_id(
                image_region, detection
            )

            # Update statistics
            if status == 'new':
                self.session_stats['new_objects'] += 1
            elif status == 'existing':
                self.session_stats['existing_objects'] += 1
            else:
                self.session_stats['failed_features'] += 1

            # Add global tracking info to detection
            detection['global_id'] = global_id
            detection['tracking_status'] = status
            detection['similarity_score'] = similarity

            global_detections.append(detection)

            # Log detection info
            if global_id > 0:
                print(f"🎯 Detection: Global ID {global_id} ({status}) - "
                      f"Confidence: {detection['confidence']:.3f}, "
                      f"Similarity: {similarity:.3f}")

        return global_detections

    def start_tracking(self):
        """Start global tracking test"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True

        # Create display window
        window_name = f"Global Tracking Test - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        logger.info("=== GLOBAL TRACKING TEST ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'n'/'p' - Next/Previous prompt")
        logger.info("  's' - Save database")
        logger.info("  'r' - Reset database")
        logger.info("  'i' - Show database info")
        logger.info("=" * 50)

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
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('s'):  # Save database
                    self.global_db.save_database()
                    logger.info("Database saved manually")
                elif key == ord('r'):  # Reset database
                    self._reset_database()
                elif key == ord('i'):  # Show database info
                    self._print_database_info()

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                break

        self.stop_tracking()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with global tracking"""
        # Calculate FPS using rolling average
        current_time = time.time()

        # For immediate FPS (first few frames)
        if self.fps_counter < 5:
            frame_time_diff = current_time - self.last_frame_time
            if frame_time_diff > 0:
                self.current_fps = 1.0 / frame_time_diff
        else:
            # Use rolling average for stable FPS
            self.frame_times.append(current_time)

            # Keep only recent frame times (last 20 frames for faster updates)
            if len(self.frame_times) > 20:
                self.frame_times.pop(0)

            # Calculate FPS from recent frame times
            if len(self.frame_times) >= 2:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                if time_diff > 0:
                    self.current_fps = (len(self.frame_times) - 1) / time_diff

        self.last_frame_time = current_time
        self.fps_counter += 1

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

        # Run detection pipeline
        try:
            # Stage 1: Raw detection
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)

            # Stage 2: Combined filtering
            self.filtered_detections = self.apply_combined_filtering(self.raw_detections)

            # Stage 3: Global ID assignment
            self.global_tracked_detections = self.assign_global_ids(self.filtered_detections, processed_frame)

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.raw_detections = []
            self.filtered_detections = []
            self.global_tracked_detections = []

        # Draw results
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)
        processed_frame = self._draw_fps_display(processed_frame)

        return processed_frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections with highly visible global ID information"""
        result_frame = frame.copy()

        for detection in self.global_tracked_detections:
            bbox = detection['bbox']
            global_id = detection.get('global_id', -1)
            status = detection.get('tracking_status', 'unknown')
            similarity = detection.get('similarity_score', 0.0)
            confidence = detection['confidence']

            x1, y1, x2, y2 = bbox

            # Enhanced color scheme for better visibility
            if status == 'new':
                box_color = (0, 255, 0)      # Bright green for new objects
                bg_color = (0, 200, 0)       # Darker green background
                text_color = (255, 255, 255) # White text
                status_text = "NEW"
            elif status == 'existing':
                box_color = (0, 255, 255)    # Bright cyan for existing objects
                bg_color = (0, 180, 180)     # Darker cyan background
                text_color = (0, 0, 0)       # Black text for better contrast
                status_text = "TRACKED"
            else:
                box_color = (0, 0, 255)      # Red for failed feature extraction
                bg_color = (0, 0, 180)       # Darker red background
                text_color = (255, 255, 255) # White text
                status_text = "FAILED"

            # Draw thick bounding box with rounded corners effect
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), box_color, 4)

            # Draw corner markers for better visibility
            corner_size = 20
            corner_thickness = 6
            # Top-left corner
            cv2.line(result_frame, (x1, y1), (x1 + corner_size, y1), box_color, corner_thickness)
            cv2.line(result_frame, (x1, y1), (x1, y1 + corner_size), box_color, corner_thickness)
            # Top-right corner
            cv2.line(result_frame, (x2, y1), (x2 - corner_size, y1), box_color, corner_thickness)
            cv2.line(result_frame, (x2, y1), (x2, y1 + corner_size), box_color, corner_thickness)
            # Bottom-left corner
            cv2.line(result_frame, (x1, y2), (x1 + corner_size, y2), box_color, corner_thickness)
            cv2.line(result_frame, (x1, y2), (x1, y2 - corner_size), box_color, corner_thickness)
            # Bottom-right corner
            cv2.line(result_frame, (x2, y2), (x2 - corner_size, y2), box_color, corner_thickness)
            cv2.line(result_frame, (x2, y2), (x2, y2 - corner_size), box_color, corner_thickness)

            # Draw global ID and status with enhanced visibility
            if global_id > 0:
                # Prepare labels
                main_label = f"ID: {global_id}"
                status_label = f"{status_text}"
                conf_label = f"Conf: {confidence:.2f}"
                sim_label = f"Sim: {similarity:.2f}" if status == 'existing' else f"Area: {detection.get('area', 0):.0f}"

                # Calculate label dimensions
                font = cv2.FONT_HERSHEY_DUPLEX
                main_font_scale = 0.8
                sub_font_scale = 0.6
                thickness = 2

                # Get text sizes
                (main_w, main_h), _ = cv2.getTextSize(main_label, font, main_font_scale, thickness)
                (status_w, status_h), _ = cv2.getTextSize(status_label, font, sub_font_scale, thickness)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, sub_font_scale, 1)
                (sim_w, sim_h), _ = cv2.getTextSize(sim_label, font, sub_font_scale, 1)

                # Calculate label box dimensions
                label_width = max(main_w, status_w, conf_w, sim_w) + 20
                label_height = main_h + status_h + conf_h + sim_h + 40

                # Position label box (above detection if space, otherwise below)
                if y1 - label_height - 10 > 0:
                    # Above the detection
                    label_x = x1
                    label_y = y1 - label_height - 10
                else:
                    # Below the detection
                    label_x = x1
                    label_y = y2 + 10

                # Ensure label doesn't go off screen
                frame_height, frame_width = result_frame.shape[:2]
                if label_x + label_width > frame_width:
                    label_x = frame_width - label_width - 10
                if label_y + label_height > frame_height:
                    label_y = frame_height - label_height - 10

                # Draw label background with border
                cv2.rectangle(result_frame,
                             (label_x - 5, label_y - 5),
                             (label_x + label_width, label_y + label_height),
                             (0, 0, 0), -1)  # Black background
                cv2.rectangle(result_frame,
                             (label_x - 5, label_y - 5),
                             (label_x + label_width, label_y + label_height),
                             bg_color, 3)  # Colored border

                # Draw text labels
                text_y = label_y + main_h + 10

                # Main ID label (larger)
                cv2.putText(result_frame, main_label,
                           (label_x + 10, text_y),
                           font, main_font_scale, text_color, thickness)

                text_y += status_h + 8

                # Status label
                cv2.putText(result_frame, status_label,
                           (label_x + 10, text_y),
                           font, sub_font_scale, text_color, thickness)

                text_y += conf_h + 6

                # Confidence label
                cv2.putText(result_frame, conf_label,
                           (label_x + 10, text_y),
                           font, sub_font_scale, text_color, 1)

                text_y += sim_h + 6

                # Similarity/Area label
                cv2.putText(result_frame, sim_label,
                           (label_x + 10, text_y),
                           font, sub_font_scale, text_color, 1)

                # Draw center point for reference
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_frame, (center_x, center_y), 8, box_color, -1)
                cv2.circle(result_frame, (center_x, center_y), 8, (255, 255, 255), 2)
                cv2.putText(result_frame, str(global_id),
                           (center_x - 10, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return result_frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced tracking information overlay"""
        frame_height, frame_width = frame.shape[:2]

        # Create semi-transparent background
        overlay = frame.copy()
        panel_width = 400
        panel_height = 280

        # Position panel in top-right corner
        panel_x = frame_width - panel_width - 20
        panel_y = 20

        # Draw background with border
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 255), 3)  # Cyan border

        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Enhanced font settings
        font = cv2.FONT_HERSHEY_DUPLEX
        title_font_scale = 0.8
        text_font_scale = 0.6
        small_font_scale = 0.5

        # Colors
        title_color = (0, 255, 255)  # Cyan
        text_color = (255, 255, 255)  # White
        new_color = (0, 255, 0)      # Green
        existing_color = (0, 255, 255)  # Cyan
        failed_color = (0, 100, 255)    # Orange-red

        y_offset = panel_y + 35

        # Title with FPS
        cv2.putText(frame, "GLOBAL TRACKING",
                   (panel_x + 20, y_offset),
                   font, title_font_scale, title_color, 2)

        # FPS display in top-right of panel
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, font, text_font_scale, 1)[0]
        fps_x = panel_x + panel_width - fps_size[0] - 20
        cv2.putText(frame, fps_text,
                   (fps_x, y_offset),
                   font, text_font_scale, title_color, 2)

        y_offset += 35

        # Database stats
        db_stats = self.global_db.get_database_stats()
        cv2.putText(frame, f"Database Objects: {db_stats['total_objects']}",
                   (panel_x + 20, y_offset),
                   font, text_font_scale, text_color, 1)

        y_offset += 25
        cv2.putText(frame, f"Next Global ID: {db_stats['next_id']}",
                   (panel_x + 20, y_offset),
                   font, small_font_scale, text_color, 1)

        y_offset += 30

        # Session statistics with colored indicators
        cv2.putText(frame, "SESSION STATS:",
                   (panel_x + 20, y_offset),
                   font, text_font_scale, text_color, 1)

        y_offset += 25

        # New objects
        new_count = self.session_stats['new_objects']
        cv2.circle(frame, (panel_x + 30, y_offset - 5), 8, new_color, -1)
        cv2.putText(frame, f"New Objects: {new_count}",
                   (panel_x + 50, y_offset),
                   font, small_font_scale, new_color, 1)

        y_offset += 20

        # Existing objects
        existing_count = self.session_stats['existing_objects']
        cv2.circle(frame, (panel_x + 30, y_offset - 5), 8, existing_color, -1)
        cv2.putText(frame, f"Re-identified: {existing_count}",
                   (panel_x + 50, y_offset),
                   font, small_font_scale, existing_color, 1)

        y_offset += 20

        # Failed features
        failed_count = self.session_stats['failed_features']
        cv2.circle(frame, (panel_x + 30, y_offset - 5), 8, failed_color, -1)
        cv2.putText(frame, f"Failed Features: {failed_count}",
                   (panel_x + 50, y_offset),
                   font, small_font_scale, failed_color, 1)

        y_offset += 30

        # Current frame info
        cv2.putText(frame, f"Current Frame:",
                   (panel_x + 20, y_offset),
                   font, text_font_scale, text_color, 1)

        y_offset += 20
        cv2.putText(frame, f"  Detections: {len(self.global_tracked_detections)}",
                   (panel_x + 20, y_offset),
                   font, small_font_scale, text_color, 1)

        y_offset += 15
        cv2.putText(frame, f"  Raw: {len(self.raw_detections)}",
                   (panel_x + 20, y_offset),
                   font, small_font_scale, text_color, 1)

        y_offset += 15
        cv2.putText(frame, f"  Filtered: {len(self.filtered_detections)}",
                   (panel_x + 20, y_offset),
                   font, small_font_scale, text_color, 1)

        y_offset += 25

        # Controls
        cv2.putText(frame, "CONTROLS:",
                   (panel_x + 20, y_offset),
                   font, small_font_scale, title_color, 1)

        y_offset += 15
        cv2.putText(frame, "s=Save r=Reset i=Info",
                   (panel_x + 20, y_offset),
                   font, 0.4, text_color, 1)

        # Draw legend in bottom-left corner
        legend_x = 20
        legend_y = frame_height - 120

        # Legend background
        cv2.rectangle(frame, (legend_x, legend_y),
                     (legend_x + 200, legend_y + 100),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x, legend_y),
                     (legend_x + 200, legend_y + 100),
                     (255, 255, 255), 2)

        legend_y += 20
        cv2.putText(frame, "LEGEND:",
                   (legend_x + 10, legend_y),
                   font, 0.5, (255, 255, 255), 1)

        legend_y += 20
        cv2.rectangle(frame, (legend_x + 10, legend_y - 10),
                     (legend_x + 30, legend_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, "NEW",
                   (legend_x + 40, legend_y),
                   font, 0.4, (255, 255, 255), 1)

        legend_y += 20
        cv2.rectangle(frame, (legend_x + 10, legend_y - 10),
                     (legend_x + 30, legend_y + 5), (0, 255, 255), -1)
        cv2.putText(frame, "TRACKED",
                   (legend_x + 40, legend_y),
                   font, 0.4, (255, 255, 255), 1)

        legend_y += 20
        cv2.rectangle(frame, (legend_x + 10, legend_y - 10),
                     (legend_x + 30, legend_y + 5), (0, 0, 255), -1)
        cv2.putText(frame, "FAILED",
                   (legend_x + 40, legend_y),
                   font, 0.4, (255, 255, 255), 1)

        return frame

    def _draw_fps_display(self, frame: np.ndarray) -> np.ndarray:
        """Draw prominent FPS display in top-left corner"""
        # FPS display settings
        fps_text = f"FPS: {self.current_fps:.1f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 3

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)

        # Position in top-left corner
        x = 20
        y = 50

        # Draw background rectangle
        padding = 15
        cv2.rectangle(frame,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     (0, 0, 0), -1)  # Black background

        # Draw border based on FPS performance
        if self.current_fps >= 25:
            border_color = (0, 255, 0)  # Green for good FPS
        elif self.current_fps >= 15:
            border_color = (0, 255, 255)  # Yellow for moderate FPS
        else:
            border_color = (0, 0, 255)  # Red for low FPS

        cv2.rectangle(frame,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     border_color, 3)

        # Draw FPS text
        cv2.putText(frame, fps_text, (x, y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _reset_database(self):
        """Reset the global database"""
        self.global_db.features = {}
        self.global_db.next_global_id = 1000
        self.global_db.save_database()

        # Reset session stats
        self.session_stats = {
            'new_objects': 0,
            'existing_objects': 0,
            'failed_features': 0,
            'total_detections': 0
        }

        logger.info("🔄 Database reset - All objects cleared")

    def _print_database_info(self):
        """Print detailed database information"""
        stats = self.global_db.get_database_stats()

        print("\n" + "=" * 50)
        print("GLOBAL DATABASE INFORMATION")
        print("=" * 50)
        print(f"Total Objects: {stats['total_objects']}")
        print(f"Next Global ID: {stats['next_id']}")

        if stats['total_objects'] > 0:
            print(f"Average Times Seen: {stats['avg_times_seen']:.1f}")
            print(f"Most Seen Object: {stats['most_seen']} times")

            print("\nObject Details:")
            for global_id, data in list(self.global_db.features.items())[:10]:  # Show first 10
                print(f"  ID {global_id}: Seen {data['times_seen']} times, "
                      f"Last: {data['last_seen'][:19]}")

        print("\nSession Statistics:")
        print(f"  New Objects: {self.session_stats['new_objects']}")
        print(f"  Existing Objects: {self.session_stats['existing_objects']}")
        print(f"  Failed Features: {self.session_stats['failed_features']}")
        print(f"  Total Detections: {self.session_stats['total_detections']}")
        print("=" * 50)

    def stop_tracking(self):
        """Stop the tracking"""
        self.running = False

        # Print final statistics
        self._print_database_info()

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped tracking for {self.camera_name}")


def main():
    """Main function"""
    print("SINGLE CAMERA GLOBAL TRACKING TEST")
    print("=" * 50)
    print("Tests global ID assignment and feature persistence")
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("=" * 50)
    print("\nWhat this tests:")
    print("1. Global ID assignment to detected objects")
    print("2. Feature extraction and matching (SIFT)")
    print("3. Object re-identification when they return")
    print("4. Feature database persistence")
    print("5. Foundation for multi-camera tracking")
    print("=" * 50)

    tracker = SingleCameraGlobalTracker(camera_id=11)

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
