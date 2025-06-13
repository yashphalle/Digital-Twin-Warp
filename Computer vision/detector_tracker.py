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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from config import Config
import logging

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

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
        if len(self.match_scores) > 10:
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
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
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
            
        # Apply histogram equalization for better contrast
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
        padding = 5
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(width, xmax + padding)
        ymax = min(height, ymax + padding)
        
        return frame[ymin:ymax, xmin:xmax]
    
    def match_detection_to_objects(self, detection: Dict, frame: np.ndarray) -> Tuple[Optional[int], float]:
        """Match new detection to existing tracked objects using SIFT"""
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
                    new_obj = TrackedObject(self.next_id, detection, sift_kp, sift_des, box_region)
                    self.tracked_objects[self.next_id] = new_obj
                    matched_objects.add(self.next_id)
                    self.stats['new_objects_created'] += 1
                    
                    box_info = {
                        'id': self.next_id,
                        'center': detection['center'],
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
        
        # Remove objects that have been missing too long
        objects_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.disappeared_count > self.max_disappeared_frames:
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
                
                detection = {
                    'center': (center_x, center_y),
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
            
            if status == 'new':
                color = Config.COLOR_NEW_OBJECT
            elif age > 60:
                color = Config.COLOR_ESTABLISHED_OBJECT
            else:
                color = Config.COLOR_TRACKING_OBJECT
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 255, 255), 2)
            
            # Enhanced label with tracking info
            if Config.SHOW_OBJECT_IDS:
                label = f"ID:{box_info['id']}"
                if age > 0:
                    label += f" Duration:{age:.0f}s"
                if Config.SHOW_MATCH_SCORES and 'match_score' in box_info:
                    label += f" S:{box_info['match_score']:.2f}"
                
                cv2.putText(annotated_frame, label, (xmin, ymin-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
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
