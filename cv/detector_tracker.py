"""
Combined Detection and SIFT Tracking System
Integrates Grounding DINO detection with SIFT-based persistent object tracking
Adapted from the working SIFT tracking system
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from config import Config
import logging
import json

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class CoordinateMapper:
    """Maps pixel coordinates to real-world floor coordinates"""
    def __init__(self, floor_width=5.0, floor_length=4.0):
        self.floor_width = floor_width  # meters
        self.floor_length = floor_length  # meters
        self.image_corners = None
        self.matrix = None
        self.inverse_matrix = None
        self.is_calibrated = False

        # Try to load calibration
        self.load_calibration()

    def load_calibration(self, filename="warehouse_calibration.json"):
        """Load calibration from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Get actual dimensions from calibration file
            warehouse_dims = data.get('warehouse_dimensions', {})
            self.floor_width = warehouse_dims.get('width_meters', self.floor_width)
            self.floor_length = warehouse_dims.get('length_meters', self.floor_length)

            image_corners = np.array(data['image_corners'], dtype=np.float32)
            self.set_floor_rectangle(image_corners)

            logger.info(f"Coordinate calibration loaded from: {filename}")
            logger.info(f"Warehouse dimensions: {self.floor_width:.2f}m x {self.floor_length:.2f}m")
            return True
        except Exception as e:
            logger.debug(f"No calibration file found: {e}")
            return False

    def set_floor_rectangle(self, image_corners):
        """Set the floor rectangle corners in image coordinates"""
        self.image_corners = np.array(image_corners, dtype=np.float32)

        # Real-world corners (floor coordinates in meters) - using actual loaded dimensions
        real_corners = np.array([
            [0, 0],                                    # Top-left
            [self.floor_width, 0],                     # Top-right
            [self.floor_width, self.floor_length],     # Bottom-right
            [0, self.floor_length]                     # Bottom-left
        ], dtype=np.float32)

        # Calculate transformation matrices
        self.matrix = cv2.getPerspectiveTransform(real_corners, self.image_corners)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.image_corners, real_corners)
        self.is_calibrated = True

        logger.info(f"Floor rectangle set: {self.floor_width:.3f}m x {self.floor_length:.3f}m")

    def pixel_to_real(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates"""
        if not self.is_calibrated:
            return None, None

        try:
            real_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(real_point, self.inverse_matrix)
            real_x, real_y = pixel_point[0][0]

            return float(real_x), float(real_y)
        except Exception as e:
            logger.error(f"Coordinate conversion error: {e}")
            return None, None

    def real_to_pixel(self, real_x, real_y):
        """Convert real-world coordinates to pixel coordinates"""
        if not self.is_calibrated:
            return None, None

        try:
            real_point = np.array([[[real_x, real_y]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(real_point, self.matrix)
            pixel_x, pixel_y = pixel_point[0][0]

            return int(pixel_x), int(pixel_y)
        except Exception as e:
            logger.error(f"Coordinate conversion error: {e}")
            return None, None

class TrackedObject:
    """Represents a tracked object with visual features"""
    def __init__(self, obj_id: int, detection: Dict, sift_keypoints, sift_descriptors, box_region: np.ndarray):
        self.id = obj_id
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.last_center = detection['center']
        self.last_bbox = detection['bbox']
        self.last_confidence = detection['confidence']
        
        # Visual features for matching
        self.sift_keypoints = sift_keypoints
        self.sift_descriptors = sift_descriptors
        self.box_region = box_region.copy() if box_region is not None else None
        
        # Tracking statistics
        self.times_seen = 1
        self.disappeared_count = 0
        self.match_scores = []  # History of match scores
        
        # Grid position tracking
        self.current_grid_position = None
        self.grid_history = []
        
    def update(self, detection: Dict, match_score: float, sift_keypoints=None, sift_descriptors=None, box_region=None):
        """Update object with new detection"""
        self.last_seen = datetime.now()
        self.last_center = detection['center']
        self.last_bbox = detection['bbox']
        self.last_confidence = detection['confidence']
        self.times_seen += 1
        self.disappeared_count = 0
        self.match_scores.append(match_score)
        
        # Keep only recent match scores
        if len(self.match_scores) > Config.MAX_MATCH_HISTORY:
            self.match_scores.pop(0)
            
        # Update visual features if provided
        if sift_keypoints is not None and sift_descriptors is not None:
            self.sift_keypoints = sift_keypoints
            self.sift_descriptors = sift_descriptors
            if box_region is not None:
                self.box_region = box_region.copy()
    
    def mark_disappeared(self):
        """Mark object as not seen in current frame"""
        self.disappeared_count += 1
    
    def get_age_seconds(self) -> float:
        """Get age of object in seconds"""
        return (datetime.now() - self.first_seen).total_seconds()
    
    def get_average_match_score(self) -> float:
        """Get average match score"""
        return np.mean(self.match_scores) if self.match_scores else 0.0

class SIFTTracker:
    """SIFT-based visual object tracker"""
    def __init__(self):
        # SIFT detector with optimized parameters from config
        self.sift = cv2.SIFT_create(
            nfeatures=Config.SIFT_N_FEATURES,
            nOctaveLayers=Config.SIFT_N_OCTAVE_LAYERS,
            contrastThreshold=Config.SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=Config.SIFT_EDGE_THRESHOLD,
            sigma=Config.SIFT_SIGMA
        )
        
        # FLANN matcher for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=Config.FLANN_TREES)
        search_params = dict(checks=Config.FLANN_CHECKS)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Matching parameters from config
        self.min_match_count = Config.MIN_MATCH_COUNT
        self.good_match_ratio = Config.GOOD_MATCH_RATIO
        self.match_score_threshold = Config.MATCH_SCORE_THRESHOLD
        
    def extract_features(self, box_region: np.ndarray):
        """Extract SIFT features from box region"""
        if box_region is None or box_region.size == 0:
            return None, None
            
        # Convert to grayscale for SIFT
        if len(box_region.shape) == 3:
            gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = box_region
            
        # Apply histogram equalization for better contrast (if enabled)
        if Config.HISTOGRAM_EQUALIZATION:
            gray = cv2.equalizeHist(gray)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def calculate_match_score(self, des1, des2) -> float:
        """Calculate match score between two sets of descriptors"""
        if des1 is None or des2 is None:
            return 0.0
        
        if len(des1) < 2 or len(des2) < 2:
            return 0.0
        
        try:
            # Find matches using FLANN
            matches = self.flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.good_match_ratio * n.distance:
                        good_matches.append(m)
            
            # Calculate normalized match score
            if len(good_matches) >= self.min_match_count:
                score = len(good_matches) / min(len(des1), len(des2))
                return min(score, 1.0)  # Cap at 1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in SIFT matching: {e}")
            return 0.0

class DetectorTracker:
    """Combined detection and tracking system"""
    def __init__(self, force_gpu: bool = True):
        logger.info("Initializing DetectorTracker...")
        
        # Initialize GPU
        self.setup_gpu(force_gpu)
        
        # Initialize SIFT tracker
        self.sift_tracker = SIFTTracker()
        logger.info("SIFT tracker initialized")

        # Initialize coordinate mapper
        self.coordinate_mapper = CoordinateMapper(
            floor_width=Config.WAREHOUSE_FLOOR_WIDTH,
            floor_length=Config.WAREHOUSE_FLOOR_LENGTH
        )
        logger.info(f"Coordinate mapper initialized - Calibrated: {self.coordinate_mapper.is_calibrated}")

        # Initialize detection model
        self.model_id = Config.MODEL_ID
        logger.info(f"Loading detection model on {self.device}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
            logger.info("Detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            raise
        
        if self.device == "cuda":
            self.model.eval()
            torch.backends.cudnn.benchmark = True
        
        # Detection parameters
        self.prompt = Config.DETECTION_PROMPT
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.max_disappeared_frames = Config.MAX_DISAPPEARED_FRAMES
        # Removed spatial_threshold - using pure visual matching for moving objects
        
        # Performance tracking
        self.frame_count = 0
        self.detection_times = []
        self.tracking_times = []
        
        # Statistics
        self.stats = {
            'new_objects_created': 0,
            'objects_updated': 0,
            'objects_removed': 0,
            'total_detections': 0
        }

        # Clean up any existing low-confidence objects from previous runs
        self.cleanup_low_confidence_objects()
        
    def setup_gpu(self, force_gpu: bool = True):
        """Setup GPU configuration"""
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            torch.cuda.empty_cache()
            self.device = "cuda"
            logger.info("Using CUDA GPU")
        else:
            if force_gpu:
                logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(Config.GPU_MEMORY_FRACTION)
    
    def detect_boxes(self, frame: np.ndarray) -> Dict:
        """Detect boxes in frame using Grounding DINO"""
        detection_start = time.time()
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs['input_ids'],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )[0]

            # GPU memory management
            if self.device == "cuda" and self.frame_count % Config.MODEL_CACHE_FRAMES == 0:
                torch.cuda.empty_cache()
            
            # Track detection time
            detection_time = time.time() - detection_start
            self.detection_times.append(detection_time)
            if len(self.detection_times) > Config.FPS_CALCULATION_FRAMES:
                self.detection_times.pop(0)
            
            return results
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {'scores': torch.tensor([]), 'boxes': torch.tensor([])}
    
    def extract_box_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract box region from frame with padding"""
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Add padding and ensure bounds
        height, width = frame.shape[:2]
        padding = Config.BOX_PADDING
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(width, xmax + padding)
        ymax = min(height, ymax + padding)
        
        return frame[ymin:ymax, xmin:xmax]
    
    def match_detection_to_objects(self, detection: Dict, frame: np.ndarray) -> Tuple[Optional[int], float]:
        """Match new detection to existing tracked objects using SIFT (no spatial filtering)"""
        best_match_id = None
        best_score = 0.0

        # Extract visual features from detection
        box_region = self.extract_box_region(frame, detection['bbox'])
        if box_region.size == 0:
            return None, 0.0

        _, sift_des = self.sift_tracker.extract_features(box_region)

        if sift_des is None:
            return None, 0.0

        # Test against all tracked objects (removed spatial filtering for moving objects)
        for obj_id, tracked_obj in self.tracked_objects.items():
            # Calculate SIFT match score - rely purely on visual similarity
            match_score = self.sift_tracker.calculate_match_score(sift_des, tracked_obj.sift_descriptors)

            if match_score > self.sift_tracker.match_score_threshold and match_score > best_score:
                best_match_id = obj_id
                best_score = match_score

        return best_match_id, best_score
    
    def update_tracking(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Update tracking with new detections"""
        tracking_start = time.time()
        
        matched_objects = set()
        current_boxes = []
        
        # Process each detection
        for detection in detections:
            self.stats['total_detections'] += 1
            
            # Try to match with existing objects
            matched_id, match_score = self.match_detection_to_objects(detection, frame)
            
            if matched_id is not None:
                # Update existing object
                self.tracked_objects[matched_id].update(detection, match_score)
                matched_objects.add(matched_id)
                self.stats['objects_updated'] += 1
                
                box_info = {
                    'id': matched_id,
                    'center': detection['center'],
                    'real_center': detection.get('real_center'),
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'match_score': match_score,
                    'age': self.tracked_objects[matched_id].get_age_seconds(),
                    'times_seen': self.tracked_objects[matched_id].times_seen,
                    'status': 'updated'
                }
            else:
                # Create new tracked object
                box_region = self.extract_box_region(frame, detection['bbox'])
                sift_kp, sift_des = self.sift_tracker.extract_features(box_region)
                
                if sift_des is not None:
                    if Config.SHOW_ID_ASSIGNMENT_DEBUG:
                        print(f"🆕 NEW ID:{self.next_id} - No existing match found (active objects: {len(self.tracked_objects)})")

                    new_obj = TrackedObject(self.next_id, detection, sift_kp, sift_des, box_region)
                    self.tracked_objects[self.next_id] = new_obj
                    matched_objects.add(self.next_id)
                    self.stats['new_objects_created'] += 1
                    
                    box_info = {
                        'id': self.next_id,
                        'center': detection['center'],
                        'real_center': detection.get('real_center'),
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'match_score': 1.0,
                        'age': 0,
                        'times_seen': 1,
                        'status': 'new'
                    }
                    
                    self.next_id += 1
                else:
                    # Skip this detection if we can't extract features
                    continue
            
            current_boxes.append(box_info)
        
        # Mark unmatched objects as disappeared
        for obj_id, tracked_obj in self.tracked_objects.items():
            if obj_id not in matched_objects:
                tracked_obj.mark_disappeared()
        
        # Remove objects that have been missing too long OR have consistently low confidence
        objects_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            should_remove = False

            # Remove if disappeared too long
            if tracked_obj.disappeared_count > self.max_disappeared_frames:
                should_remove = True
                if Config.SHOW_ID_ASSIGNMENT_DEBUG:
                    print(f"🗑️ REMOVING ID:{obj_id} - disappeared for {tracked_obj.disappeared_count} frames")

            # 🎯 NEW: Remove if confidence is consistently below threshold
            elif tracked_obj.last_confidence < self.confidence_threshold:
                should_remove = True
                if Config.SHOW_ID_ASSIGNMENT_DEBUG:
                    print(f"🗑️ REMOVING ID:{obj_id} - low confidence {tracked_obj.last_confidence:.3f} < {self.confidence_threshold}")

            if should_remove:
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            self.stats['objects_removed'] += 1
        
        # Track timing
        tracking_time = time.time() - tracking_start
        self.tracking_times.append(tracking_time)
        if len(self.tracking_times) > Config.FPS_CALCULATION_FRAMES:
            self.tracking_times.pop(0)
        
        return current_boxes
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """Process a single frame - detect and track objects"""
        self.frame_count += 1

        # Detect objects
        detection_results = self.detect_boxes(frame)

        # Convert detection results to our format
        detections = []
        if len(detection_results['scores']) > 0:
            sorted_indices = torch.argsort(detection_results['scores'], descending=True)

            for idx in sorted_indices:
                score = detection_results['scores'][idx].item()
                box = detection_results['boxes'][idx].tolist()

                xmin, ymin, xmax, ymax = map(int, box)
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                # Get real-world coordinates
                real_x, real_y = None, None
                if self.coordinate_mapper.is_calibrated:
                    real_x, real_y = self.coordinate_mapper.pixel_to_real(center_x, center_y)

                detection = {
                    'center': (center_x, center_y),
                    'real_center': (real_x, real_y) if real_x is not None else None,
                    'bbox': (xmin, ymin, xmax, ymax),
                    'confidence': score
                }
                detections.append(detection)

        # Update tracking
        tracked_boxes = self.update_tracking(detections, frame)

        # Prepare performance stats
        perf_stats = self.get_performance_stats()

        return tracked_boxes, perf_stats
    
    def draw_tracked_objects(self, frame: np.ndarray, tracked_boxes: List[Dict]) -> np.ndarray:
        """Draw tracked objects on frame"""
        annotated_frame = frame.copy()
        
        if not tracked_boxes:
            cv2.putText(annotated_frame, "No objects tracked", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        for box_info in tracked_boxes:
            xmin, ymin, xmax, ymax = box_info['bbox']
            center_x, center_y = box_info['center']
            
            # Color coding based on object status and age
            age = box_info['age']
            status = box_info.get('status', 'unknown')
            
            if status == 'new' or age < Config.NEW_OBJECT_THRESHOLD:
                color = Config.COLOR_NEW_OBJECT
            elif age > Config.ESTABLISHED_OBJECT_THRESHOLD:
                color = Config.COLOR_ESTABLISHED_OBJECT
            else:
                color = Config.COLOR_TRACKING_OBJECT
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, Config.BBOX_THICKNESS)
            cv2.circle(annotated_frame, (center_x, center_y), Config.CENTER_DOT_RADIUS, Config.CENTER_DOT_COLOR, -1)
            cv2.circle(annotated_frame, (center_x, center_y), Config.CENTER_BORDER_RADIUS, Config.CENTER_BORDER_COLOR, 2)
            
            # Enhanced label with tracking info
            if Config.SHOW_OBJECT_IDS:
                label = f"ID:{box_info['id']}"
                if age > 0:
                    label += f" Duration:{age:.0f}s"
                if Config.SHOW_MATCH_SCORES and 'match_score' in box_info:
                    label += f" S:{box_info['match_score']:.2f}"
                if Config.SHOW_DETECTION_CONFIDENCE and 'confidence' in box_info:
                    label += f" C:{box_info['confidence']:.2f}"

                cv2.putText(annotated_frame, label, (xmin, ymin-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Real-world coordinates label (if available)
                real_center = box_info.get('real_center')
                if real_center and real_center[0] is not None:
                    real_x, real_y = real_center
                    real_label = f"Real: {real_x:.2f}m, {real_y:.2f}m"
                    cv2.putText(annotated_frame, real_label, (xmin, ymin-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated_frame

    def draw_calibrated_zone_overlay(self, frame):
        """Draw calibrated zone with physical coordinates overlay"""
        if not self.coordinate_mapper.is_calibrated or not Config.SHOW_CALIBRATED_ZONE:
            return frame

        overlay_frame = frame.copy()

        # Draw the calibrated zone boundary
        corners = self.coordinate_mapper.image_corners.astype(int)

        # Draw zone boundary
        cv2.polylines(overlay_frame, [corners], True, (0, 255, 255), 3)  # Yellow boundary

        # Fill zone with semi-transparent overlay
        zone_overlay = np.zeros_like(frame)
        cv2.fillPoly(zone_overlay, [corners], (0, 255, 255))  # Yellow fill
        cv2.addWeighted(overlay_frame, 0.95, zone_overlay, 0.05, 0, overlay_frame)

        # Draw corner markers with physical coordinates
        width = self.coordinate_mapper.floor_width
        length = self.coordinate_mapper.floor_length
        corner_labels = ["(0,0)", f"({width:.2f},0)",
                        f"({width:.2f},{length:.2f})",
                        f"(0,{length:.2f})"]

        for i, (corner, label) in enumerate(zip(corners, corner_labels)):
            x, y = corner

            # Draw corner marker
            cv2.circle(overlay_frame, (x, y), 8, (0, 255, 255), -1)  # Yellow dot
            cv2.circle(overlay_frame, (x, y), 10, (255, 255, 255), 2)  # White border

            # Add coordinate label
            label_text = f"{label}m"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Position label to avoid going off screen
            label_x = max(5, min(x - text_size[0]//2, frame.shape[1] - text_size[0] - 5))
            label_y = max(20, y - 15) if i < 2 else min(frame.shape[0] - 5, y + 25)

            # Draw label background
            cv2.rectangle(overlay_frame,
                         (label_x - 3, label_y - 15),
                         (label_x + text_size[0] + 3, label_y + 5),
                         (0, 0, 0), -1)

            # Draw label text
            cv2.putText(overlay_frame, label_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw grid overlay
        if Config.SHOW_COORDINATE_GRID:
            self._draw_coordinate_grid(overlay_frame)

        # Calibration info panel removed - keeping only zone boundary and grid

        return overlay_frame

    def draw_all_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw ALL detections (including filtered ones) for debugging"""
        debug_frame = frame.copy()

        if not hasattr(self, 'all_detections') or not self.all_detections:
            cv2.putText(debug_frame, "No detections found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return debug_frame

        high_conf_count = 0
        low_conf_count = 0

        for detection in self.all_detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            above_threshold = detection['above_threshold']

            # Choose color based on confidence
            if above_threshold:
                color = Config.COLOR_HIGH_CONFIDENCE  # Green for high confidence
                high_conf_count += 1
                status = "STORED"
            else:
                color = Config.COLOR_LOW_CONFIDENCE   # Red for low confidence
                low_conf_count += 1
                status = "FILTERED"

            # Draw bounding box
            cv2.rectangle(debug_frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw center dot
            cv2.circle(debug_frame, (center_x, center_y), 5, color, -1)

            # Draw confidence label
            conf_label = f"C:{confidence:.3f} {status}"
            cv2.putText(debug_frame, conf_label, (xmin, ymin-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw summary
        summary = f"Detections: {high_conf_count} STORED (Green), {low_conf_count} FILTERED (Red)"
        cv2.putText(debug_frame, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw threshold line
        threshold_text = f"Threshold: {self.confidence_threshold:.3f}"
        cv2.putText(debug_frame, threshold_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return debug_frame

    def cleanup_low_confidence_objects(self):
        """Clean up existing objects with low confidence (from previous runs)"""
        objects_to_remove = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.last_confidence < self.confidence_threshold:
                objects_to_remove.append(obj_id)

        removed_count = len(objects_to_remove)
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]

        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} existing low-confidence objects")
            if Config.SHOW_ID_ASSIGNMENT_DEBUG:
                print(f"🧹 CLEANUP: Removed {removed_count} existing low-confidence objects")

        return removed_count

    def get_tracking_stats(self):
        """Get comprehensive tracking statistics"""
        active_objects = len(self.tracked_objects)
        total_created = self.next_id - 1

        # Calculate average match score
        all_scores = []
        for obj in self.tracked_objects.values():
            if hasattr(obj, 'match_scores') and obj.match_scores:
                all_scores.extend(obj.match_scores)

        avg_match_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # Calculate ID assignment rate (rough estimate)
        runtime_minutes = max(1, (time.time() - getattr(self, 'start_time', time.time())) / 60)
        id_rate = total_created / runtime_minutes

        return {
            'next_id': self.next_id,
            'active_objects': active_objects,
            'total_created': total_created,
            'objects_lost': total_created - active_objects,
            'avg_match_score': avg_match_score,
            'id_rate': id_rate,
            'id_efficiency': active_objects / max(1, total_created)  # How many IDs are still active
        }

    def _draw_coordinate_grid(self, frame):
        """Draw coordinate grid inside calibrated zone"""
        if not self.coordinate_mapper.is_calibrated:
            return

        # Dynamic grid spacing based on warehouse size
        max_dimension = max(self.coordinate_mapper.floor_width, self.coordinate_mapper.floor_length)
        if max_dimension <= 3.0:
            grid_spacing = 0.5  # 0.5m for small warehouses
        elif max_dimension <= 6.0:
            grid_spacing = 1.0  # 1m for medium warehouses
        else:
            grid_spacing = 2.0  # 2m for large warehouses

        # Draw vertical grid lines
        for x_real in np.arange(0, self.coordinate_mapper.floor_width + grid_spacing, grid_spacing):
            if x_real > 0 and x_real < self.coordinate_mapper.floor_width:  # Skip boundaries
                start_pixel = self.coordinate_mapper.real_to_pixel(x_real, 0)
                end_pixel = self.coordinate_mapper.real_to_pixel(x_real, self.coordinate_mapper.floor_length)

                if start_pixel[0] is not None and end_pixel[0] is not None:
                    cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

                    # Add distance label
                    mid_pixel = self.coordinate_mapper.real_to_pixel(x_real, self.coordinate_mapper.floor_length / 2)
                    if mid_pixel[0] is not None:
                        label = f'{x_real:.1f}m' if grid_spacing < 1.0 else f'{x_real:.0f}m'
                        cv2.putText(frame, label,
                                   (mid_pixel[0] - 10, mid_pixel[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Draw horizontal grid lines
        for y_real in np.arange(0, self.coordinate_mapper.floor_length + grid_spacing, grid_spacing):
            if y_real > 0 and y_real < self.coordinate_mapper.floor_length:  # Skip boundaries
                start_pixel = self.coordinate_mapper.real_to_pixel(0, y_real)
                end_pixel = self.coordinate_mapper.real_to_pixel(self.coordinate_mapper.floor_width, y_real)

                if start_pixel[0] is not None and end_pixel[0] is not None:
                    cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

                    # Add distance label
                    mid_pixel = self.coordinate_mapper.real_to_pixel(self.coordinate_mapper.floor_width / 2, y_real)
                    if mid_pixel[0] is not None:
                        label = f'{y_real:.1f}m' if grid_spacing < 1.0 else f'{y_real:.0f}m'
                        cv2.putText(frame, label,
                                   (mid_pixel[0], mid_pixel[1] - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)


    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'frame_count': self.frame_count,
            'active_objects': len(self.tracked_objects),
            'next_id': self.next_id,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_tracking_time': np.mean(self.tracking_times) if self.tracking_times else 0,
            'detection_fps': 1.0 / np.mean(self.detection_times) if self.detection_times else 0,
            'tracking_fps': 1.0 / np.mean(self.tracking_times) if self.tracking_times else 0,
            **self.stats
        }
        
        # GPU stats if available
        if self.device == "cuda":
            stats['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return stats
    
    def get_tracked_objects_info(self) -> List[Dict]:
        """Get detailed information about tracked objects"""
        objects_info = []
        
        for obj_id, obj in self.tracked_objects.items():
            info = {
                'id': obj_id,
                'age_seconds': obj.get_age_seconds(),
                'times_seen': obj.times_seen,
                'last_confidence': obj.last_confidence,
                'avg_match_score': obj.get_average_match_score(),
                'disappeared_count': obj.disappeared_count,
                'last_center': obj.last_center,
                'last_bbox': obj.last_bbox
            }
            objects_info.append(info)
        
        return objects_info
    
    def cleanup(self):
        """Cleanup resources"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("DetectorTracker resources cleaned up")

# Test function
def test_detector_tracker():
    """Test the detector tracker"""
    print("Testing DetectorTracker...")
    
    try:
        # Initialize detector tracker
        detector = DetectorTracker()
        
        # Create dummy frame for testing
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        tracked_boxes, perf_stats = detector.process_frame(test_frame)
        
        print(f"Processed frame successfully")
        print(f"Tracked boxes: {len(tracked_boxes)}")
        print(f"Performance stats: {perf_stats}")
        
        detector.cleanup()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_detector_tracker()
