
#!/usr/bin/env python3
"""
BoT-SORT GPU Batch Processor with Smart Persistence Integration

This module integrates BoT-SORT tracking with Smart Persistence for cross-camera tracking
and Redis-based track storage. It extends the fast tracking processor to add:

1. BoT-SORT tracking with warehouse-optimized configuration
2. Smart Persistence for cross-camera feature matching
3. Redis storage for track continuity
4. Background persistence threads for performance
5. Consecutive detection filtering to reduce false positives

Key Features:
- Real-time BoT-SORT tracking with custom configuration
- Cross-camera track matching using appearance features
- Redis persistence every 5 seconds
- Non-blocking background operations
- Consecutive detection threshold (20 detections) before database insertion
- Compatible with existing GPU batch processing pipeline

Consecutive Detection Logic:
- Objects must be detected 20+ times consecutively before database insertion
- Reduces false positives and temporary detections
- Tracks show as "pending" (yellow) until threshold is reached
- Configurable via CONSECUTIVE_DETECTION_THRESHOLD constant
"""

import sys
import os
import time
import logging
import cv2
import numpy as np
import threading
import queue
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from pymongo import MongoClient
from pymongo.operations import UpdateOne
import redis
import pickle
from collections import deque

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker_parallel import ParallelCameraWorker
from GPU.pipelines.ring_buffer import RingBuffer
from GPU.pipelines.gpu_processor_detection_only import GPUBatchProcessorDetectionOnly
from GPU.pipelines.gpu_processor_fast_tracking import GPUBatchProcessorFastTracking

SIMILARITY_THRESHOLD = 0.4 
class GPUBatchProcessorBoTSORT(GPUBatchProcessorFastTracking):
    """BoT-SORT processor with Smart Persistence extending the fast tracking processor"""

    def __init__(self, model_path, device, active_cameras, confidence=0.5, use_fp16=False,
                 botsort_config_path=None):
        # Initialize parent class
        self.botsort_config_path = botsort_config_path or create_botsort_config_file()

        super().__init__(model_path, device, active_cameras, confidence, use_fp16)

        # Store BoT-SORT config path
        #self.botsort_config_path = botsort_config_path or create_botsort_config_file()

        # Add missing attributes for Smart Persistence integration
        self.conf_threshold = confidence  # Make sure this attribute exists
        self.device = device  # Ensure device is available

        # Initialize Smart Persistence System
        self.smart_persistence = SmartPersistenceManager(active_cameras)
        self.persistent_id_map={}

        logger.info(f"🎯 BoT-SORT processor initialized with config: {self.botsort_config_path}")
        logger.info(f"🎯 ReID enabled: {WAREHOUSE_BOTSORT_CONFIG['with_reid']}")
        logger.info(f"🎯 Track buffer: {WAREHOUSE_BOTSORT_CONFIG['track_buffer']} frames")
        logger.info(f"🎯 Appearance threshold: {WAREHOUSE_BOTSORT_CONFIG['appearance_thresh']}")
        logger.info(f"🚀 Smart Persistence enabled for cross-camera tracking")

    def _create_model_instance(self, model_path, device):
        """Create YOLO model instance with BoT-SORT tracker"""
        from ultralytics import YOLO

        model = YOLO(model_path)
        model.to(device)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        model.track(dummy_frame, persist=True, tracker=self.botsort_config_path, verbose=False)

        try:
            # Access the initialized tracker object
            tracker = model.predictor.trackers[0]

            # The 'encoder' is the object responsible for feature extraction
            encoder_model = tracker.encoder

            print("\n" + "="*50)
            print("✅ RE-ID MODEL VERIFICATION")
            print(f"   Tracker Config ('model'): {tracker.args.model}")
            print(f"   Feature Extractor Class Name: {type(encoder_model).__name__}")
            print(f"   Feature Extractor in Use: {encoder_model}")
            print("="*50 + "\n")

        except Exception as e:
            print(f"\n⚠️ Could not verify Re-ID model: {e}\n")

        logger.info(f"📦 Created BoT-SORT model instance: {model_path} on {device}")
        return model

    # In GPUBatchProcessorBoTSORT class

    def process_batch(self, ring_buffer) -> tuple:
        """
        FINAL VERSION 2: Uses a more robust trigger for cross-camera checks.
        (Corrected to include statistics updates for GUI)
        """
        start_time = time.time()
        batch_frames = ring_buffer.get_batch(max_age=1)
        if not batch_frames: return {}, {}

        active_frames = {c: f for c, f in batch_frames.items() if c in self.active_cameras}
        if not active_frames: return {}, {}

        # --- FIX: Increment batch and frame counters ---
        self.total_batches += 1
        self.frames_processed += len(active_frames)
        # --- END OF FIX ---

        # Step 1: Run Inference
        all_results_raw = {}
        for cam_id, frame in active_frames.items():
            try:
                camera_model = self.models.get(cam_id)
                if camera_model:
                    results = camera_model.track(frame, device=self.device, conf=self.conf_threshold, tracker=self.botsort_config_path, persist=True, verbose=False)
                    all_results_raw[cam_id] = results
            except Exception as e:
                logger.error(f"Inference failed for camera {cam_id}: {e}")
                all_results_raw[cam_id] = None

        # Step 2: Process results and handle ID logic
        detections_by_camera = {}
        for cam_id, results_raw in all_results_raw.items():
            if results_raw is None:
                detections_by_camera[cam_id] = []
                continue

            base_objects = self._extract_tracked_objects_from_results(results_raw, cam_id)
            id_manager = self.id_managers.get(cam_id)
            final_detections = []
            active_yolo_ids = {obj['track_id'] for obj in base_objects}

            for base_obj in base_objects:
                yolo_track_id = base_obj['track_id']
                global_id = id_manager.get_global_id(yolo_track_id)
                track_age = id_manager.get_track_age(global_id)

                # --- FINAL ROBUST TRIGGER LOGIC ---
                # Check if this global_id has been assigned a persistent ID in our map yet.
                if global_id not in self.persistent_id_map:

                    # NEW, SMARTER CHECK: Only query Redis if we have a feature vector to compare.
                    if base_obj.get('feature_vector') is not None:
                        logger.info(f"⚡️ New track (Global ID: {global_id}) has a feature. Referring to Redis database...")
                        match = self.smart_persistence.check_cross_camera_matches(base_obj, cam_id)
                        persistent_id = match['persistent_id'] if match else global_id
                    else:
                        # The new track does not have a feature vector yet. We can't check it.
                        # We will treat it as a new entity for now and check again on a later frame.
                        logger.info(f"⏳ New track (Global ID: {global_id}) detected, waiting for feature vector before checking Redis.")
                        persistent_id = global_id

                    # Store the mapping for future frames.
                    self.persistent_id_map[global_id] = persistent_id
                else:
                    # This is an existing track, retrieve its persistent_id from our map.
                    persistent_id = self.persistent_id_map.get(global_id, global_id)
                # --- END OF FINAL LOGIC ---

                # Build the final, clean dictionary
                final_detection_obj = {
                    'camera_id': cam_id,
                    'global_id': global_id,
                    'persistent_id': persistent_id,
                    'track_age': track_age,
                    'feature_vector': base_obj['feature_vector'],
                    'bbox': base_obj['bbox'],
                    'confidence': base_obj['confidence'],
                    'class': self.models[cam_id].names.get(base_obj['class'], 'unknown'),
                    'track_id': yolo_track_id
                }
                final_detections.append(final_detection_obj)

            detections_by_camera[cam_id] = final_detections
            id_manager.cleanup_lost_tracks(list(active_yolo_ids))

        # Step 3: Trigger background persistence
        for cam_id, detections in detections_by_camera.items():
            if detections:
                self.smart_persistence.background_persistence(detections, cam_id)

        # --- FIX: Record the total time taken for this batch ---
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        # --- END OF FIX ---

        return active_frames, detections_by_camera

    
    def _run_inference_with_tracking(self, camera_model, frame, cam_id):
        """Run inference with BoT-SORT tracking + Smart Persistence"""
        try:
            # PHASE 1: Normal BoT-SORT tracking (full speed, unchanged)
            results = camera_model.track(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                tracker=self.botsort_config_path,  # Use our custom BoT-SORT config
                persist=True,  # Maintain tracks across calls
                verbose=False
            )
            # PHASE 2: Smart Persistence Integration
            if results and self.smart_persistence:
                # Extract tracked objects from results
                tracked_objects = self._extract_tracked_objects_from_results(results, cam_id)

                # Background persistence to Redis
                if tracked_objects:
                    self.smart_persistence.background_persistence(tracked_objects, cam_id)

                # Cross-camera matching for unmatched detections
                # Get original detections that weren't matched to tracks
                unmatched_detections = self._get_unmatched_detections(results, tracked_objects, cam_id)
                if unmatched_detections:
                    logger.info(f"🔍 Found {len(unmatched_detections)} unmatched detections for Camera {cam_id}")
                    cross_camera_matches = self.smart_persistence.check_cross_camera_matches(unmatched_detections, cam_id)
                    if cross_camera_matches:
                        logger.info(f"🔄 Found {len(cross_camera_matches)} cross-camera matches for Camera {cam_id}")
                    else:
                        logger.debug(f"🔍 No cross-camera matches found for Camera {cam_id}")
                else:
                    logger.debug(f"🔍 No unmatched detections for Camera {cam_id}")

            return results

        except Exception as e:
            logger.error(f"❌ BoT-SORT inference error for camera {cam_id}: {e}")
            return None

    def _get_unmatched_detections(self, results, tracked_objects, cam_id):
        """Get detections that weren't matched to existing tracks (potential new objects)"""
        unmatched_detections = []

        if not results or len(results) == 0:
            return unmatched_detections

        try:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes

                # Get track IDs that are already being tracked
                tracked_ids = set()
                for track in tracked_objects:
                    track_id = track.get('track_id')
                    if track_id is not None:
                        # Convert numpy arrays to Python int/float for hashability
                        if hasattr(track_id, 'item'):
                            track_id = track_id.item()
                        tracked_ids.add(track_id)

                # Find detections without track IDs (unmatched detections)
                for i in range(len(boxes)):
                    bbox = boxes.xywh[i].cpu().numpy() if hasattr(boxes, 'xywh') else None
                    confidence = boxes.conf[i].cpu().numpy() if hasattr(boxes, 'conf') else 0.0
                    class_id = boxes.cls[i].cpu().numpy() if hasattr(boxes, 'cls') else 0
                    track_id = boxes.id[i].cpu().numpy() if hasattr(boxes, 'id') and boxes.id is not None else None

                    # Convert track_id to hashable type for comparison
                    if track_id is not None and hasattr(track_id, 'item'):
                        track_id = track_id.item()

                    # If no track ID or track ID not in our tracked objects, it's unmatched
                    if track_id is None or track_id not in tracked_ids:
                        if bbox is not None and confidence > 0.3:  # Only consider confident detections
                            x_center, y_center, width, height = bbox
                            x1 = int(x_center - width/2)
                            y1 = int(y_center - height/2)
                            x2 = int(x_center + width/2)
                            y2 = int(y_center + height/2)

                            unmatched_detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class': int(class_id),
                                'camera_id': cam_id
                            }
                            unmatched_detections.append(unmatched_detection)

        except Exception as e:
            logger.error(f"Error extracting unmatched detections: {e}")

        return unmatched_detections

    # In GPUBatchProcessorBoTSORT class

    # In GPUBatchProcessorBoTSORT class

    def _extract_tracked_objects_from_results(self, results, cam_id):
        """
        FINAL CORRECTED VERSION: Extracts ONLY tracked objects that have an available 
        feature vector, ensuring data integrity for the persistence stage.
        """
        tracked_objects = []
        result = results[0]

        tracker = None
        try:
            camera_model = self.models[cam_id]
            if hasattr(camera_model, 'predictor') and hasattr(camera_model.predictor, 'trackers'):
                if camera_model.predictor.trackers:
                    tracker = camera_model.predictor.trackers[0]
        except Exception as e:
            logger.warning(f"Camera {cam_id}: Could not access tracker object: {e}")

        if not hasattr(result, 'boxes') or result.boxes is None or tracker is None:
            return []

        boxes = result.boxes
        if not hasattr(boxes, 'id') or boxes.id is None:
            return []

        # Step 1: Build the feature map from ONLY the activated tracks that have features.
        feature_map = {}
        if hasattr(tracker, 'tracked_stracks'):
            for strack in tracker.tracked_stracks:
                # A track is 'activated' once it's been seen for a few frames.
                # Only these tracks are guaranteed to have reliable features.
                if strack.is_activated and strack.features:
                    feature_map[strack.track_id] = strack.features[-1]

        # Step 2: Loop through the detection results and process ONLY the IDs found in our map.
        for i in range(len(boxes)):
            yolo_track_id = int(boxes.id[i].cpu().item())

            # --- THIS IS THE FINAL, CRITICAL FIX ---
            # Only proceed if this track ID exists in our feature map.
            # This explicitly filters out new tracks that aren't fully activated yet.
            if yolo_track_id not in feature_map:
                continue
            # --- END OF FIX ---

            # If we are here, we are GUARANTEED to have a feature vector.
            feature_vector = feature_map[yolo_track_id]

            bbox = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())

            tracked_obj = {
                'camera_id': cam_id,
                'bbox': bbox.tolist(),
                'confidence': float(conf),
                'class': cls,
                'track_id': yolo_track_id,
                'feature_vector': feature_vector  # This is now guaranteed to exist
            }
            tracked_objects.append(tracked_obj)

        return tracked_objects
# Import tested coordinate mapper from final folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cv', 'final', 'modules'))
from coordinate_mapper import CoordinateMapper

# Import fast color extractor
from color_extractor import FastColorExtractor, extract_color_for_detection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
CONSECUTIVE_DETECTION_THRESHOLD = 20  # Number of consecutive detections required before database insertion

# BoT-SORT Configuration for Warehouse Tracking
WAREHOUSE_BOTSORT_CONFIG = {
    'tracker_type': 'botsort',

    # TUNABLE detection thresholds (adjust these for better tracking)
    'track_high_thresh': 0.2,      # Lower = easier to continue existing tracks (try 0.2-0.5)
    'track_low_thresh': 0.05,      # Lower = better recovery from occlusion (try 0.05-0.2)
    'new_track_thresh': 0.6,       # Higher = harder to create new tracks (try 0.5-0.8)

    # TUNABLE persistence (adjust for your warehouse conditions)
    'track_buffer': 200,            # Frames to keep lost tracks (try 30-120)
    'max_age': 250,                # Max frames before deletion (try 60-150)

    # TUNABLE matching (adjust for tracking stability)
    'match_thresh': 0.5,          # IoU threshold for matching (try 0.5-0.8)
    'proximity_thresh': 0.4,      # Spatial constraint for ReID (try 0.3-0.6)

    # TUNABLE ReID (adjust for appearance matching)
    'with_reid': True,
    'model': 'yolov8n-cls.pt',              # Use native YOLO features
    'appearance_thresh': 0.4,    # Lower = more lenient appearance matching (try 0.1-0.4)

    # TUNABLE stability features
    'gmc_method': 'sparseOptFlow', # Camera motion compensation
    'fuse_score': True,           # Combine confidence + IoU
    'min_hits': 10,                # Frames needed to confirm track (try 1-3)
}

def create_botsort_config_file():
    """Create warehouse-optimized BoT-SORT configuration file"""
    import yaml
    import os

    # Create configs directory if it doesn't exist
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, 'warehouse_botsort.yaml')

    # Write BoT-SORT configuration
    with open(config_path, 'w') as f:
        yaml.dump(WAREHOUSE_BOTSORT_CONFIG, f, default_flow_style=False)

    logger.info(f"✅ Created BoT-SORT config: {config_path}")
    return config_path

class SmartPersistenceManager:
    """Smart Persistence Manager for BoT-SORT with Feature-Based Cross-Camera Tracking"""

    def __init__(self, active_cameras: List[int]):
        self.active_cameras = active_cameras

        # Redis setup for fast feature-based access
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False,
                socket_keepalive=True,
                health_check_interval=30
            )
            # Test Redis connection
            self.redis_client.ping()
            logger.info("✅ Connected to Redis for Smart Persistence")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

        # Camera neighbor mapping for cross-camera tracking (your warehouse layout)
        self.camera_neighbors = {
            1: [2, 5],
            2: [1, 3, 6],
            3: [2, 4, 7],
            4: [3],
            5: [1, 6, 8],
            6: [5, 7, 2, 9],
            7: [6, 3, 10],
            8: [5, 9],
            9: [8, 10, 6],
            10: [9, 11, 7],
            11: [10]
        }

        # Background persistence settings
        self.redis_save_interval = 5.0  # 5 seconds
        self.last_redis_saves = {cam_id: time.time() for cam_id in active_cameras}

        logger.info(f"🚀 Smart Persistence Manager initialized for cameras: {active_cameras}")

    def extract_botsort_features(self, results, camera_id):
        """Extract BoT-SORT appearance features from tracking results"""
        features_data = {}

        if not results or len(results) == 0:
            return features_data

        try:
            # Access the tracker from results to get BOTrack objects
            # This is where we get the actual BoT-SORT features
            result = results[0]

            # Try to access tracker internals (this may vary based on ultralytics version)
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes

                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()

                    for i, track_id in enumerate(track_ids):
                        if track_id is not None:
                            # For now, we'll use a placeholder for features
                            # In a full implementation, you'd extract from the actual BOTrack object
                            features_data[int(track_id)] = {
                                'track_id': int(track_id),
                                'camera_id': camera_id,
                                'bbox': boxes.xywh[i].cpu().numpy() if hasattr(boxes, 'xywh') else None,
                                'confidence': boxes.conf[i].cpu().numpy() if hasattr(boxes, 'conf') else 0.0,
                                'features_placeholder': True  # Will be replaced with real features
                            }

        except Exception:
            pass

        return features_data


    # def save_tracks_to_redis(self, tracked_objects, camera_id):
    #     """Save tracks with their persistent_id and feature vector to Redis."""
    #     if not self.redis_client or not tracked_objects:
    #         return

    #     try:
    #         pipe = self.redis_client.pipeline()
    #         for track in tracked_objects:
    #             # We now use the persistent_id as the key for Redis
    #             persistent_id = track.get('persistent_id')
    #             feature_vector = track.get('feature_vector')

    #             if persistent_id and feature_vector is not None:
    #                 redis_key = f"track:{persistent_id}"  # The key is the TRUE ID

    #                 track_data = {
    #                     'persistent_id': persistent_id,
    #                     'last_global_id': track.get('global_id'), # Store its last known location-based ID
    #                     'camera_id': camera_id,
    #                     'last_bbox': track['bbox'],
    #                     'last_seen': time.time(),
    #                     'feature_vector': feature_vector
    #                 }

    #                 pipe.set(redis_key, pickle.dumps(track_data))
    #                 pipe.expire(redis_key, 60) # Remember for 60 seconds

    #         pipe.execute()

    #     except Exception as e:
    #         logger.error(f"Error saving tracks to Redis: {e}")

    def _calculate_feature_similarity(self, new_features, stored_features):
        """Calculate similarity based on the cosine similarity of appearance feature vectors."""
        try:
            # --- Cosine Similarity Calculation ---
            dot_product = np.dot(new_features, stored_features)
            norm_new = np.linalg.norm(new_features)
            norm_stored = np.linalg.norm(stored_features)

            if norm_new == 0 or norm_stored == 0:
                return 0.0

            return dot_product / (norm_new * norm_stored)

        except Exception:
            return 0.0

    def check_cross_camera_matches(self, unmatched_detection, camera_id):
        """
        CORRECTED DIAGNOSTIC VERSION: Checks for matches in the CURRENT camera first,
        then in neighboring cameras, to handle re-identification after long occlusions.
        """
        if not self.redis_client or unmatched_detection.get('feature_vector') is None:
            return None

        best_match = None
        best_similarity = 0.0
        new_features = unmatched_detection['feature_vector']
        
        # --- FIX: Create a list of cameras to check, starting with the current one ---
        cameras_to_check = [camera_id] + self.camera_neighbors.get(camera_id, [])
        # --- END FIX ---
        
        logger.info(f"--- [MATCH INSPECTOR] Cam {camera_id}: Checking for match ---")
        logger.info(f"  - Cameras to check in Redis (own first, then neighbors): {cameras_to_check}")
        
        try:
            track_keys = self.redis_client.keys('track:*')
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            logger.info("------------------------------------------------------------------")
            return None
            
        logger.info(f"  - Found {len(track_keys)} total tracks in Redis to check against.")
        
        found_candidates = 0
        for key in track_keys:
            try:
                stored_track_data_pkl = self.redis_client.get(key)
                if not stored_track_data_pkl: continue
                
                stored_track = pickle.loads(stored_track_data_pkl)

                # --- FIX: Check against the full list (current camera + neighbors) ---
                if stored_track.get('camera_id') not in cameras_to_check:
                    continue
                # --- END FIX ---
                
                found_candidates += 1
                p_id = stored_track.get('persistent_id')
                logger.info(f"  -> Evaluating candidate: P_ID {p_id} from Cam {stored_track.get('camera_id')}")

                # Check 2: Was it seen recently (within 30 seconds)?
                time_diff = time.time() - stored_track.get('last_seen', 0)
                time_ok = time_diff <= 30.0
                logger.info(f"       - Time since last seen: {time_diff:.2f}s. Passes time check? -> {time_ok}")
                if not time_ok:
                    continue
                
                # Check 3: Calculate visual similarity
                stored_features = stored_track.get('feature_vector')
                if stored_features is None:
                    logger.warning(f"       - ❗️ FAILED: Candidate has no feature vector.")
                    continue
                
                similarity = self._calculate_feature_similarity(new_features, stored_features)
                similarity_ok = similarity > SIMILARITY_THRESHOLD
                logger.info(f"       - Calculated similarity: {similarity:.4f}. Passes threshold ({SIMILARITY_THRESHOLD})? -> {similarity_ok}")

                if similarity_ok and similarity > best_similarity:
                    logger.info(f"       - ✅ NEW BEST MATCH FOUND!")
                    best_similarity = similarity
                    best_match = stored_track
            
            except Exception:
                continue
            
        if found_candidates == 0:
            logger.info("  - INFO: No recently seen tracks from any relevant cameras were found.")

        if not best_match:
            logger.info("  - FINAL VERDICT: No suitable match found above the threshold.")
        else:
            logger.info(f"  - FINAL VERDICT: Match found with P_ID {best_match.get('persistent_id')} and similarity {best_similarity:.4f}")
            
        logger.info("------------------------------------------------------------------")
        
        return best_match

    

    def background_persistence(self, tracked_objects, camera_id):
        """Background persistence to Redis with enhanced logging."""
        # print("--- [SAVE TRIGGER CHECK] ---") # ADDED PRINT

        # Check 1: Is the Redis client connected?
        # if not self.redis_client:
        #     # print("  - ❗️ FAILED: Redis client is not connected. Cannot save.") # ADDED PRINT
        #     # print("--------------------------\n")
        #     return
        # else:
        #     print("  - ✅ Redis client is connected.") # ADDED PRINT

        # Check 2: Is it time to save? (every 5 seconds)
        current_time = time.time()
        last_save_time = self.last_redis_saves.get(camera_id, 0) # Use .get for safety
        time_since_last_save = current_time - last_save_time
        # print(f"  - Time since last save for Cam {camera_id}: {time_since_last_save:.2f} seconds.") # ADDED PRINT

        if time_since_last_save > self.redis_save_interval:
            #print(f"  - ✅ TIMER FIRED! Starting background save for {len(tracked_objects)} tracks.") # ADDED PRINT

            threading.Thread(
                target=self.save_tracks_to_redis,
                args=(tracked_objects, camera_id),
                daemon=True
            ).start()
            self.last_redis_saves[camera_id] = current_time
        

    # In SmartPersistenceManager class

    def save_tracks_to_redis(self, tracked_objects, camera_id):
        """
        FINAL DIAGNOSTIC VERSION: This function will print the exact contents of the
        objects it receives before attempting to save them.
        """
        # --- START OF PRE-FLIGHT CHECK ---
        

        if tracked_objects:
           
            first_obj = tracked_objects[0]

            # Print all available keys in the dictionary
            
            # Specifically check for the two critical keys
            pid_present = 'persistent_id' in first_obj
            fv_present = 'feature_vector' in first_obj
            

            if fv_present:
                # If the key exists, let's check if the value is None
                fv_value = first_obj['feature_vector']
                
            
        if not self.redis_client or not tracked_objects:
            return

        try:
            pipe = self.redis_client.pipeline()
            count = 0
            for track in tracked_objects:
                persistent_id = track.get('persistent_id')
                feature_vector = track.get('feature_vector')

                if persistent_id and feature_vector is not None:
                    redis_key = f"track:{persistent_id}"
                    track_data = {
                        'persistent_id': persistent_id,
                        'last_global_id': track.get('global_id'),
                        'camera_id': track.get('camera_id'),
                        'last_bbox': track['bbox'],
                        'last_seen': time.time(),
                        'feature_vector': feature_vector
                    }
                    pipe.set(redis_key, pickle.dumps(track_data))
                    # pipe.expire(redis_key, 60)
                    count += 1

            pipe.execute()
            # This is the line from your log that says "0"
            print(f"  - ✅ SUCCESS: Attempted to save {count} tracks to Redis.") 
            print("----------------------------\n")

        except Exception as e:
            print(f"  - ❌ ERROR: Exception during Redis save: {e}")
            print("----------------------------\n")

def enrich_detection_fast(detection, coordinate_mappers, color_info, frame_width=1600, frame_height=900):
    """
    MODIFIED: Fast enrichment that correctly uses the 'global_id' and 'persistent_id' from the detection object.
    """
    # --- CORRECTED ID ASSIGNMENT ---
    # Read the correct ID fields passed from the main processing loop.
    # The old 'track_id' (e.g., 1, 2, 3) is now correctly ignored.
    global_id = detection['global_id']
    persistent_id = detection.get('persistent_id', global_id) # Fallback to global_id if not present
    # --- END CORRECTION ---

    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    track_age = detection['track_age']
    camera_id = detection['camera_id']

    # Determine tracking status based on consecutive detection threshold
    if track_age < CONSECUTIVE_DETECTION_THRESHOLD:
        tracking_status = 'pending'
    elif track_age == CONSECUTIVE_DETECTION_THRESHOLD:
        tracking_status = 'new'
    else:
        tracking_status = 'existing'

    # (The rest of the function for coordinate mapping remains the same)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    physical_x_ft = None
    physical_y_ft = None
    coordinate_status = 'unmapped'

    if camera_id in coordinate_mappers:
        mapper = coordinate_mappers[camera_id]
        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height
        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y
        physical_x_ft, physical_y_ft = mapper.pixel_to_real(scaled_center_x, scaled_center_y)

        if physical_x_ft is not None and physical_y_ft is not None:
            coordinate_status = 'mapped'
        else:
            physical_x_ft = camera_id * 20.0
            physical_y_ft = 50.0
            coordinate_status = 'fallback'
    else:
        physical_x_ft = camera_id * 20.0
        physical_y_ft = 50.0
        coordinate_status = 'no_mapper'

    enriched = {
        # --- CORRECTED IDENTITY FIELDS ---
        'persistent_id': persistent_id,  # The true, unchanging ID (e.g., 8001)
        'global_id': global_id,          # The current, location-based ID (e.g., 9009)
        # --- END CORRECTION ---
        'camera_id': detection['camera_id'],
        'warp_id': None,
        'tracking_status': tracking_status,
        'class': detection['class'],
        'age_seconds': track_age,
        'consecutive_detections': track_age,
        'physical_x_ft': physical_x_ft,
        'physical_y_ft': physical_y_ft,
        'coordinate_status': coordinate_status,
        'real_center': [physical_x_ft, physical_y_ft],
        'bbox': bbox,
        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        'physical_corners': None,
        'area': (x2 - x1) * (y2 - y1),
        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
        'shape_type': 'quadrangle',
        'confidence': detection['confidence'],
        'similarity_score': 1.0,
        'timestamp': datetime.now(pytz.timezone('US/Pacific')),
        'color_rgb': color_info['rgb'],
        'color_hsv': color_info['hsv'],
        'color_hex': color_info['hex'],
        'color_name': color_info['name'],
        'color_confidence': color_info['confidence'],
        'extraction_method': color_info['extraction_method']
    }

    # (The rest of the function for calculating physical corners remains the same)
    if camera_id in coordinate_mappers and physical_x_ft is not None and physical_y_ft is not None:
        mapper = coordinate_mappers[camera_id]
        corners_pixel = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        physical_corners = []
        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height

        for corner in corners_pixel:
            pixel_x, pixel_y = corner
            scaled_x = pixel_x * scale_x
            scaled_y = pixel_y * scale_y
            phys_x, phys_y = mapper.pixel_to_real(scaled_x, scaled_y)
            if phys_x is not None and phys_y is not None:
                physical_corners.append([round(phys_x, 2), round(phys_y, 2)])
            else:
                physical_corners.append([None, None])

        if not any(corner == [None, None] for corner in physical_corners):
            enriched['physical_corners'] = physical_corners

    # The logging can also be updated for better debugging
    if tracking_status == 'new':
        logger.info(f"🆕 NEW TRACK READY FOR DB: Camera {enriched['camera_id']}, global_id={enriched['global_id']}, persistent_id={enriched['persistent_id']}, consecutive_detections={track_age}")
    elif tracking_status == 'pending' and track_age % 10 == 0:
        logger.info(f"⏳ TRACK BUILDING: Camera {enriched['camera_id']}, global_id={enriched['global_id']}, persistent_id={enriched['persistent_id']}, consecutive_detections={track_age}/{CONSECUTIVE_DETECTION_THRESHOLD}")

    return enriched

class OptimizedWarehouseDatabaseWriter(threading.Thread):
    """High-performance async database writer with 2-second batching"""

    def __init__(self):
        super().__init__(daemon=True)

        # High-capacity queue for data volume
        self.detection_queue = queue.Queue(maxsize=20000)

        # Batch settings (2-second batches)
        self.batch_timeout = 2.0
        self.max_batch_size = 1000
        self.running = False

        # MongoDB connection (existing Atlas)
        try:
            self.mongo_client = MongoClient("mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/")
            self.db = self.mongo_client.WARP  # Correct database name
            self.collection = self.db.detections  # Correct collection name

            # Test connection with a ping
            self.mongo_client.admin.command('ping')
            logger.info("✅ Connected to MongoDB Atlas")

            # Test write capability
            test_doc = {"test": "connection", "timestamp": datetime.now(pytz.timezone('US/Pacific'))}
            test_result = self.collection.insert_one(test_doc)
            if test_result.inserted_id:
                logger.info("✅ Database write test successful")
                # Clean up test document
                self.collection.delete_one({"_id": test_result.inserted_id})
            else:
                logger.error("❌ Database write test failed")

        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            self.mongo_client = None

        # REAL coordinate mappers for each camera (using tested code from final folder)
        self.coordinate_mappers = {}
        self._initialize_coordinate_mappers()

        # Fast color extractor for new tracks
        self.color_extractor = FastColorExtractor()

        # Performance tracking
        self.stats = {
            'total_queued': 0,
            'total_new_inserts': 0,
            'total_updates': 0,
            'total_errors': 0,
            'last_batch_size': 0,
            'queue_depth': 0
        }

    def _initialize_coordinate_mappers(self):
        """Initialize REAL coordinate mappers for ALL 11 cameras using tested code from final folder"""
        all_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All warehouse cameras

        for camera_id in all_cameras:
            try:
                # Create coordinate mapper (same as tested final code)
                mapper = CoordinateMapper(camera_id=camera_id)

                # Load calibration file (same path structure as final code)
                calibration_file = f"../configs/warehouse_calibration_camera_{camera_id}.json"
                mapper.load_calibration(calibration_file)

                if mapper.is_calibrated:
                    self.coordinate_mappers[camera_id] = mapper
                    logger.info(f"✅ REAL coordinate mapper loaded for Camera {camera_id}")
                else:
                    logger.warning(f"⚠️ Coordinate mapper not calibrated for Camera {camera_id}")

            except Exception as e:
                logger.error(f"❌ Failed to load coordinate mapper for Camera {camera_id}: {e}")

        logger.info(f"📍 Initialized {len(self.coordinate_mappers)} real coordinate mappers")

    def queue_detection(self, enriched_detection):
        """Thread-safe queuing with overflow handling"""
        if not self.mongo_client:
            return  # Skip if no database connection

        try:
            self.detection_queue.put(enriched_detection, block=False)
            self.stats['total_queued'] += 1
        except queue.Full:
            # Drop oldest to make room (prevents blocking)
            try:
                self.detection_queue.get(block=False)
                self.detection_queue.put(enriched_detection, block=False)
                logger.warning("⚠️ Database queue full, dropped oldest detection")
            except queue.Empty:
                pass  # Queue emptied by writer thread

    def start(self):
        """Start the database writer thread"""
        if self.mongo_client:
            self.running = True
            super().start()
            logger.info("🚀 Database writer started (20K queue, 2-sec batches)")
        else:
            logger.warning("⚠️ Database writer not started - no connection")

    def stop(self):
        """Stop the database writer thread"""
        self.running = False
        if self.is_alive():
            self.join(timeout=5)

        # Final database statistics
        logger.info("📊 === FINAL DATABASE STATISTICS ===")
        logger.info(f"📊 Total queued: {self.stats['total_queued']}")
        logger.info(f"📊 Total new inserts: {self.stats['total_new_inserts']}")
        logger.info(f"📊 Total updates: {self.stats['total_updates']}")
        logger.info(f"📊 Total errors: {self.stats['total_errors']}")
        logger.info(f"📊 Final queue depth: {self.stats['queue_depth']}")

        if self.mongo_client:
            try:
                # Get actual document count from database
                doc_count = self.collection.count_documents({})
                logger.info(f"📊 Total documents in WARP/detections: {doc_count}")

                # Get count with physical coordinates (what frontend will see)
                coord_count = self.collection.count_documents({
                    "physical_x_ft": {"$exists": True, "$ne": None},
                    "physical_y_ft": {"$exists": True, "$ne": None}
                })
                logger.info(f"📊 Documents with coordinates (frontend visible): {coord_count}")

            except Exception as e:
                logger.error(f"❌ Could not get document count: {e}")

        logger.info("📊 ==================================")

    def run(self):
        """2-second batch writer loop"""
        if not self.mongo_client:
            return

        batch = []
        last_write = time.time()

        logger.info("📝 Database writer thread started")

        while self.running:
            try:
                # Collect detections
                detection = self.detection_queue.get(timeout=0.1)
                batch.append(detection)

                # Write every 2 seconds or when batch full
                if (time.time() - last_write >= self.batch_timeout or
                    len(batch) >= self.max_batch_size):

                    self._process_batch(batch)
                    batch = []
                    last_write = time.time()

            except queue.Empty:
                # Timeout - write partial batch if 2 seconds passed
                if batch and time.time() - last_write >= self.batch_timeout:
                    self._process_batch(batch)
                    batch = []
                    last_write = time.time()
            except Exception as e:
                logger.error(f"❌ Database writer error: {e}")
                self.stats['total_errors'] += 1

        # Write final batch
        if batch:
            self._process_batch(batch)

        logger.info("📝 Database writer thread finished")

    # In OptimizedWarehouseDatabaseWriter class

    def _process_batch(self, detections):
        """
        MODIFIED: Smart INSERT/UPDATE batch processing that uses the 'persistent_id' as the true key.
        This prevents duplicate entries after cross-camera handoffs.
        """
        if not detections or not self.mongo_client:
            return

        # We no longer need separate lists for new_inserts and updates.
        # A single, powerful 'upsert' operation will handle both cases.
        update_operations = []
        pending_count = 0
        new_doc_count = 0
        updated_doc_count = 0
        current_time = datetime.now(pytz.timezone('US/Pacific'))

        for detection in detections:
            # The 'tracking_status' (pending, new, existing) from enrich_detection_fast is still used
            tracking_status = detection['tracking_status']

            if tracking_status == 'pending':
                pending_count += 1
                continue  # Skip database operations for pending tracks

            # For both 'new' and 'existing' tracks, we will use a single UpdateOne with upsert=True.
            # This is the key to the new logic.
            persistent_id = detection.get('persistent_id')
            if not persistent_id:
                logger.warning("Received a detection without a persistent_id. Skipping.")
                continue
            
            # Determine if this operation will likely be an insert or an update for logging purposes
            if tracking_status == 'new':
                new_doc_count += 1
            elif tracking_status == 'existing':
                updated_doc_count += 1

            # This operation will:
            # 1. FIND a document using the unchanging 'persistent_id'.
            # 2. IF it exists, UPDATE its location ('global_id', 'camera_id', bbox, etc.).
            # 3. IF it DOES NOT exist, CREATE ('upsert') a new document using all the provided data.
            update_operations.append(UpdateOne(
                {'persistent_id': persistent_id},  # --- KEY CHANGE: Use persistent_id as the filter
                {
                    '$set': {
                        # Update fields that change with every detection
                        'last_seen': current_time,
                        'bbox': detection['bbox'],
                        'corners': detection['corners'],
                        'physical_corners': detection.get('physical_corners'),
                        'real_center': detection.get('real_center'),
                        'confidence': detection['confidence'],
                        'area': detection['area'],
                        'center': detection['center'],
                        'physical_x_ft': detection.get('physical_x_ft'),
                        'physical_y_ft': detection.get('physical_y_ft'),
                        'coordinate_status': detection.get('coordinate_status'),
                        'similarity_score': detection.get('similarity_score', 1.0),
                        # --- KEY CHANGE: Also update the location-based IDs ---
                        'global_id': detection.get('global_id'),
                        'camera_id': detection.get('camera_id'),
                    },
                    '$setOnInsert': {
                        # These fields are only set when a NEW document is created
                        'persistent_id': persistent_id,
                        'first_seen': current_time,
                        'class': detection['class'],
                        'color_rgb': detection.get('color_rgb'),
                        'color_hsv': detection.get('color_hsv'),
                        'color_hex': detection.get('color_hex'),
                        'color_name': detection.get('color_name'),
                        'color_confidence': detection.get('color_confidence'),
                    },
                    '$inc': {'times_seen': 1}
                },
                upsert=True  # This is the magic that creates the doc if it doesn't exist
            ))

        # Execute batch operations
        try:
            if update_operations:
                result = self.collection.bulk_write(update_operations, ordered=False)
                self.stats['total_new_inserts'] += result.upserted_count
                self.stats['total_updates'] += result.modified_count

            self.stats['last_batch_size'] = len(detections)
            self.stats['queue_depth'] = self.detection_queue.qsize()

            logger.info(f"📊 DB Batch: {new_doc_count} new, {updated_doc_count} updates, {pending_count} pending, "
                        f"Queue: {self.stats['queue_depth']}/20000 "
                        f"[Total Inserts: {self.stats['total_new_inserts']}, Total Updates: {self.stats['total_updates']}]")

        except Exception as e:
            logger.error(f"❌ Database batch error: {e}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            self.stats['total_errors'] += 1

    def _get_update_fields(self, detection):
        """Get fields to update for existing tracks (excludes color fields to preserve them)"""
        current_time = datetime.now(pytz.timezone('US/Pacific'))

        return {
            'last_seen': current_time,
            'bbox': detection['bbox'],
            'corners': detection['corners'],
            'physical_corners': detection.get('physical_corners'),
            'shape_type': detection.get('shape_type'),
            'real_center': detection.get('real_center'),
            'confidence': detection['confidence'],
            'area': detection['area'],
            'center': detection['center'],
            'physical_x_ft': detection.get('physical_x_ft'),
            'physical_y_ft': detection.get('physical_y_ft'),
            'coordinate_status': detection.get('coordinate_status'),
            'similarity_score': detection.get('similarity_score', 1.0)
            # NOTE: Deliberately excludes color fields to preserve original colors
        }

class SimpleGUIDisplay:
    """Simple GUI display for detection speed testing"""

    def __init__(self, camera_ids: List[int]):
        self.camera_ids = camera_ids
        self.running = False
        self.display_thread = None
        self.latest_frames = {}
        self.latest_detections = {}
        self.stats = {}
        self.tracking_enabled = False  # Will be set by test function

        # Calculate grid layout
        self.num_cameras = len(camera_ids)
        if self.num_cameras <= 4:
            self.grid_cols = 2
            self.grid_rows = 2
        elif self.num_cameras <= 6:
            self.grid_cols = 3
            self.grid_rows = 2
        elif self.num_cameras <= 9:
            self.grid_cols = 3
            self.grid_rows = 3
        else:
            self.grid_cols = 4
            self.grid_rows = 3

        # Display settings
        self.cell_width = 320
        self.cell_height = 240
        self.display_width = self.grid_cols * self.cell_width
        self.display_height = self.grid_rows * self.cell_height + 100  # Extra space for stats

        logger.info(f"GUI Display initialized for {self.num_cameras} cameras ({self.grid_cols}x{self.grid_rows} grid)")

    def update_frame(self, camera_id: int, frame: np.ndarray, detections: List[Dict] = None):
        """Update frame and detections for a camera"""
        if frame is not None:
            self.latest_frames[camera_id] = frame.copy()
        if detections is not None:
            self.latest_detections[camera_id] = detections

    def update_stats(self, stats: Dict):
        """Update performance statistics"""
        self.stats = stats.copy()

    def get_tracking_color(self, detection: Dict) -> tuple:
        """Get color based on tracking state"""
        track_id = detection.get('track_id')
        track_age = detection.get('track_age', 0)

        # Calculate tracking status based on consecutive detection logic
        if track_id is None:
            tracking_status = 'detection'
        elif track_age < CONSECUTIVE_DETECTION_THRESHOLD:
            tracking_status = 'pending'
        else:
            tracking_status = 'database'  # 20+ consecutive detections



        if tracking_status == 'detection':
            return (0, 255, 0)      # Green - New detection (no tracking yet)
        elif tracking_status == 'pending':
            return (0, 255, 255)    # Yellow - Tracked but building consecutive detections
        else:
            return (0, 165, 255)    # Orange - Database entry (20+ consecutive detections)

    def create_tracking_label(self, detection: Dict) -> str:
        """
        MODIFIED: Creates a label with the correct Global and Persistent tracking information.
        """
        # --- CORRECTED ID ASSIGNMENT ---
        # Read from the correct fields provided by the main processor.
        global_id = detection.get('global_id')
        persistent_id = detection.get('persistent_id', global_id) # Use global_id as a fallback if persistent_id isn't there yet
        # --- END CORRECTION ---
    
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'object')
        track_age = detection.get('track_age', 0)
        
        # Fallback for detections that might not have an ID yet
        if global_id is None:
            return f"{class_name}: {confidence:.2f}"
    
        # Determine the status for display (pending vs confirmed for database)
        if track_age < CONSECUTIVE_DETECTION_THRESHOLD:
            # Pending mode: "ID:9001 (P:8001) forklift: 0.85 (3/20)"
            return f"ID:{global_id} (P:{persistent_id}) {class_name}: {confidence:.2f} ({track_age}/{CONSECUTIVE_DETECTION_THRESHOLD})"
        else:
            # Database mode: "ID:9001 (P:8001) forklift: 0.85 (25f)"
            return f"ID:{global_id} (P:{persistent_id}) {class_name}: {confidence:.2f} ({track_age}f)"

    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict],
                                original_frame_size: tuple = None) -> np.ndarray:
        """Draw detection/tracking bounding boxes on frame with proper scaling and colors"""
        if not detections:
            return frame

        result_frame = frame.copy()
        current_height, current_width = frame.shape[:2]

        for detection in detections:
            # Get bounding box coordinates
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]

                # Scale bounding box if frame was resized
                if original_frame_size is not None:
                    orig_height, orig_width = original_frame_size
                    scale_x = current_width / orig_width
                    scale_y = current_height / orig_height

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                else:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, current_width - 1))
                y1 = max(0, min(y1, current_height - 1))
                x2 = max(0, min(x2, current_width - 1))
                y2 = max(0, min(y2, current_height - 1))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Get tracking color and label
                color = self.get_tracking_color(detection)
                label = self.create_tracking_label(detection)

                # Draw bounding box with tracking color
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_frame, (center_x, center_y), 3, color, -1)

                # Draw track ID badge for tracked objects
                track_id = detection.get('track_id')
                if track_id is not None:
                    # Large track ID in top-left corner of box
                    id_text = str(track_id)
                    cv2.putText(result_frame, id_text, (x1 - 20, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw enhanced label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

                # Position label above box, or below if too close to top
                label_y = y1 - 5 if y1 > label_size[1] + 10 else y2 + label_size[1] + 5
                label_x = x1

                # Draw label background
                cv2.rectangle(result_frame, (label_x, label_y - label_size[1] - 2),
                            (label_x + label_size[0], label_y + 2), color, -1)
                cv2.putText(result_frame, label, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return result_frame

    def create_camera_cell(self, camera_id: int, tracking_enabled: bool = False) -> np.ndarray:
        """Create display cell for a single camera with tracking status"""
        cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)

        # Camera header with tracking status
        if tracking_enabled:
            header_text = f"Camera {camera_id} - TRACKING"
            header_color = (0, 165, 255)  # Orange
        else:
            header_text = f"Camera {camera_id} - DETECTION"
            header_color = (0, 255, 0)    # Green

        cv2.putText(cell, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, header_color, 2)

        # Get frame and detections
        frame = self.latest_frames.get(camera_id)
        detections = self.latest_detections.get(camera_id, [])

        if frame is not None:
            # Store original frame size for bbox scaling
            original_frame_size = frame.shape[:2]  # (height, width)

            # Resize frame to fit cell (leave space for header)
            frame_height = self.cell_height - 30
            frame_resized = cv2.resize(frame, (self.cell_width, frame_height))

            # Draw detections with proper scaling
            frame_with_detections = self.draw_detections_on_frame(
                frame_resized, detections, original_frame_size)

            # Place frame in cell
            cell[30:, :] = frame_with_detections

            # Add tracking/detection count
            if tracking_enabled:
                tracked_count = sum(1 for d in detections if d.get('track_id') is not None)
                status_text = f"Tracks: {tracked_count}"
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = f"Detections: {len(detections)}"
                status_color = (0, 255, 255)  # Cyan

            cv2.putText(cell, status_text, (self.cell_width - 120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        else:
            # No frame available
            cv2.putText(cell, "NO SIGNAL", (self.cell_width//2 - 50, self.cell_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return cell

    def create_display_frame(self, tracking_enabled: bool = False) -> np.ndarray:
        """Create complete display frame with all cameras"""
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Place camera cells in grid
        for i, camera_id in enumerate(self.camera_ids):
            if i >= self.grid_cols * self.grid_rows:
                break  # Skip if too many cameras for grid

            row = i // self.grid_cols
            col = i % self.grid_cols

            x = col * self.cell_width
            y = row * self.cell_height

            cell = self.create_camera_cell(camera_id, tracking_enabled)
            display[y:y+self.cell_height, x:x+self.cell_width] = cell

        # Add statistics at bottom with tracking info
        stats_y = self.grid_rows * self.cell_height + 20
        if self.stats:
            if tracking_enabled:
                stats_text = [
                    f"Total Frames: {self.stats.get('total_frames', 0)}",
                    f"Active Tracks: {self.stats.get('active_tracks', 0)}",
                    f"Total Tracks Created: {self.stats.get('total_tracks_created', 0)}",
                    f"Avg FPS: {self.stats.get('avg_fps', 0.0):.1f}",
                    f"Avg Inference: {self.stats.get('avg_inference_time_ms', 0.0):.1f}ms"
                ]
            else:
                stats_text = [
                    f"Total Frames: {self.stats.get('total_frames', 0)}",
                    f"Total Detections: {self.stats.get('total_detections', 0)}",
                    f"Avg FPS: {self.stats.get('avg_fps', 0.0):.1f}",
                    f"Avg Inference: {self.stats.get('avg_inference_time_ms', 0.0):.1f}ms"
                ]

            # Display stats in two rows if tracking enabled
            for i, text in enumerate(stats_text):
                if tracking_enabled and i >= 3:
                    # Second row for tracking stats
                    x_pos = 20 + ((i - 3) * 200)
                    y_pos = stats_y + 25
                else:
                    # First row
                    x_pos = 20 + (i * 200)
                    y_pos = stats_y

                if x_pos < self.display_width - 150:
                    cv2.putText(display, text, (x_pos, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def display_loop(self):
        """Main display loop running in separate thread"""
        window_title = "Tracking Speed Test" if self.tracking_enabled else "Detection Speed Test"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, self.display_width, self.display_height)

        while self.running:
            try:
                display_frame = self.create_display_frame(self.tracking_enabled)
                cv2.imshow(window_title, display_frame)

                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    logger.info("GUI quit requested")
                    self.running = False
                    break
                elif key == ord('f'):  # 'f' for fullscreen
                    cv2.setWindowProperty("Detection Speed Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('w'):  # 'w' for windowed
                    cv2.setWindowProperty("Detection Speed Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(0.1)

        cv2.destroyAllWindows()
        logger.info("GUI display stopped")

    def start(self):
        """Start the display in a separate thread"""
        if not self.running:
            self.running = True
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()
            logger.info("GUI display started")

    def stop(self):
        """Stop the display"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        cv2.destroyAllWindows()

def test_detection_speed(camera_ids=[8, 9], duration=None, confidence=0.5, enable_gui=True, enable_tracking=False, enable_database=True, continuous=False):
    """
    Test detection or tracking speed with PARALLEL camera reading and database integration

    Args:
        camera_ids: List of camera IDs to test
        duration: Test duration in seconds (None for continuous mode)
        confidence: Detection confidence threshold
        enable_gui: Enable GUI display (default: True)
        enable_tracking: Enable tracking mode (default: False for detection only)
        enable_database: Enable database writes (default: True)
        continuous: Run continuously until manually stopped (default: False)
    """
    # Determine run mode
    if continuous or duration is None:
        run_mode = "CONTINUOUS"
        duration = None  # Ensure duration is None for continuous mode
    else:
        run_mode = f"{duration}s"

    mode_name = "BoT-SORT TRACKING" if enable_tracking else "DETECTION"
    logger.info(f"=== {mode_name} SPEED TEST (PARALLEL) ===")
    logger.info(f"Cameras: {camera_ids}")
    logger.info(f"Duration: {run_mode}")
    logger.info(f"Confidence: {confidence}")
    logger.info(f"GUI Enabled: {enable_gui}")
    logger.info(f"Tracking Enabled: {enable_tracking}")
    if enable_tracking:
        logger.info(f"Tracker: BoT-SORT with ReID")
        logger.info(f"ReID Enabled: {WAREHOUSE_BOTSORT_CONFIG['with_reid']}")
        logger.info(f"Track Buffer: {WAREHOUSE_BOTSORT_CONFIG['track_buffer']} frames")
        logger.info(f"Appearance Threshold: {WAREHOUSE_BOTSORT_CONFIG['appearance_thresh']}")
    logger.info(f"Database Enabled: {enable_database}")

    # Initialize components
    ring_buffer = RingBuffer(num_cameras=11, buffer_size=30)

    # Initialize database writer
    db_writer = None
    if enable_database:
        db_writer = OptimizedWarehouseDatabaseWriter()
        db_writer.start()
        logger.info("✅ Database writer initialized")

    # Initialize GPU processor based on mode
    if enable_tracking:
        # Use BoT-SORT processor instead of ByteTracker
        gpu_processor = GPUBatchProcessorBoTSORT(
            model_path='custom_yolo.pt',
            device='cuda:0',
            active_cameras=camera_ids,
            confidence=confidence,
            use_fp16=False
        )
        logger.info("🎯 BoT-SORT tracking processor initialized")
        # Safe access to models attribute
        if hasattr(gpu_processor, 'models') and gpu_processor.models:
            logger.info(f"🔧 Using {len(gpu_processor.models)} separate BoT-SORT model instances: custom_yolo.pt")
        else:
            logger.info("🔧 Using custom_yolo.pt model with BoT-SORT")
    else:
        gpu_processor = GPUBatchProcessorDetectionOnly(
            model_path='custom_yolo.pt',
            device='cuda:0',
            active_cameras=camera_ids,
            confidence=confidence,
            use_fp16=False
        )
        logger.info("? Detection-only processor initialized")

    # Initialize GUI display if enabled
    gui_display = None
    if enable_gui:
        gui_display = SimpleGUIDisplay(camera_ids)
        gui_display.tracking_enabled = enable_tracking  # Set tracking mode
        gui_display.start()
        logger.info(f"? GUI display started ({'tracking' if enable_tracking else 'detection'} mode)")
    
    # CHANGE: Create and start parallel camera workers
    camera_workers = []
    for cam_id in camera_ids:
        worker = ParallelCameraWorker(cam_id, ring_buffer, frame_skip=5)
        if worker.connect():
            worker.start()  # Start the thread!
            camera_workers.append(worker)
            logger.info(f"? Camera {cam_id} thread started")
    
    if not camera_workers:
        logger.error("No cameras connected")
        return
    
    # Wait for cameras to start producing frames
    logger.info("Waiting for cameras to stabilize...")
    time.sleep(3)
    
    # Warmup
    logger.info("Warming up GPU...")
    for _ in range(5):
        gpu_processor.process_batch(ring_buffer)
        time.sleep(0.1)
    
    # Reset stats
    gpu_processor.total_batches = 0
    gpu_processor.total_inference_time = 0
    gpu_processor.frames_processed = 0
    gpu_processor.detection_count = 0
    
    # Start actual test
    if duration is None:
        logger.info(f"\n🚀 Starting CONTINUOUS speed test (press 'q' in GUI or Ctrl+C to stop)...")
    else:
        logger.info(f"\n?? Starting {duration} second speed test...")
    start_time = time.time()

    try:
        # CHANGE: Main loop just processes batches (no camera reading!)
        while True:
            # Check duration for timed tests
            if duration is not None and time.time() - start_time >= duration:
                logger.info(f"⏰ Test duration of {duration} seconds completed")
                break
            # Check if GUI requested quit
            if gui_display and not gui_display.running:
                logger.info("🛑 GUI quit requested, stopping test")
                break

            # Just process batches - cameras feed themselves!
            # ?? FIX: Get synchronized frames and detections from GPU processor
            # This ensures the SAME frames used for processing are used for display
            processed_frames, detections_by_camera = gpu_processor.process_batch(ring_buffer)

            # NEW: Queue detections for database with REAL coordinates + color analysis (only in tracking mode)
            if enable_database and db_writer and enable_tracking:
                for cam_id, detections in detections_by_camera.items():
                    # Get current fisheye-corrected frame for color extraction
                    frame_result = ring_buffer.get_latest(cam_id)
                    current_frame = frame_result[0] if frame_result is not None else None

                    for detection in detections:
                        # Extract color (only for new tracks: age=1)
                        color_info = extract_color_for_detection(current_frame, detection, db_writer.color_extractor)

                        # Use REAL coordinate mappers + color info from database writer
                        enriched = enrich_detection_fast(detection, db_writer.coordinate_mappers, color_info)
                        db_writer.queue_detection(enriched)

            # Update GUI with synchronized frames and detections
            if gui_display:
                # ? Use the SAME frames that were processed for detections
                # This eliminates the flashing issue in tracking mode
                for cam_id in camera_ids:
                    frame = processed_frames.get(cam_id)
                    detections = detections_by_camera.get(cam_id, [])

                    # Minimal logging - focus on FPS performance
                    # Removed verbose detection logging to focus on performance metrics

                    if frame is not None:
                        gui_display.update_frame(cam_id, frame, detections)

                # Update performance stats
                stats = gpu_processor.get_stats()
                gui_display.update_stats(stats)

            # Small delay to prevent CPU spinning
            time.sleep(0.02)  # 20ms

    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user (Ctrl+C)")

    # Results
    elapsed = time.time() - start_time
    stats = gpu_processor.get_stats()

    logger.info("\n=== RESULTS ===")
    if duration is None:
        logger.info(f"Continuous test ran for: {elapsed:.1f}s")
    else:
        logger.info(f"Test duration: {elapsed:.1f}s")
    logger.info(f"Total frames processed: {stats['total_frames']}")

    if enable_tracking:
        logger.info(f"Total tracks created: {stats.get('total_tracks_created', 0)}")
        logger.info(f"Active tracks: {stats.get('active_tracks', 0)}")
    else:
        logger.info(f"Total detections: {stats.get('total_detections', 0)}")
        if stats.get('total_detections', 0) > 0:
            logger.info(f"Average detections per frame: {stats['total_detections']/max(stats['total_frames'],1):.1f}")

    logger.info(f"\n?? PERFORMANCE:")
    logger.info(f"  - Average inference time: {stats['avg_inference_time_ms']:.1f}ms per batch")
    logger.info(f"  - Average per frame: {stats['avg_ms_per_frame']:.1f}ms")
    logger.info(f"  - Average FPS: {stats['avg_fps']:.1f}")
    logger.info(f"  - Throughput: {stats['total_frames']/elapsed:.1f} frames/second")
    
    # Stop GUI display
    if gui_display:
        logger.info("Stopping GUI display...")
        gui_display.stop()

    # CHANGE: Stop parallel workers
    logger.info("\nStopping camera threads...")
    for worker in camera_workers:
        worker.stop()
    for worker in camera_workers:
        worker.join(timeout=1)

    # Stop database writer
    if db_writer:
        logger.info("Stopping database writer...")
        db_writer.stop()

    # Cleanup
    gpu_processor.cleanup()

def run_continuous_tracking(camera_ids=[8, 9, 10], confidence=0.5, enable_gui=True):
    """
    Convenience function to run continuous BoT-SORT tracking with database integration

    Args:
        camera_ids: List of camera IDs to track
        confidence: Detection confidence threshold
        enable_gui: Enable GUI display
    """
    logger.info("🚀 Starting CONTINUOUS BoT-SORT warehouse tracking system...")
    logger.info("🎯 Using BoT-SORT with ReID for enhanced tracking accuracy")
    logger.info("💡 Press 'q' in GUI window or Ctrl+C to stop")

    test_detection_speed(
        camera_ids=camera_ids,
        duration=None,
        confidence=confidence,
        enable_gui=enable_gui,
        enable_tracking=True,
        enable_database=True,
        continuous=True
    )

def run_timed_test(camera_ids=[8, 9, 10], duration=60, confidence=0.5, enable_gui=True):
    """
    Convenience function to run timed test with database integration

    Args:
        camera_ids: List of camera IDs to track
        duration: Test duration in seconds
        confidence: Detection confidence threshold
        enable_gui: Enable GUI display
    """
    logger.info(f"🚀 Starting {duration}-second warehouse tracking test...")

    test_detection_speed(
        camera_ids=camera_ids,
        duration=duration,
        confidence=confidence,
        enable_gui=enable_gui,
        enable_tracking=True,
        enable_database=True,
        continuous=False
    )

if __name__ == "__main__":
    """
    🎯 BoT-SORT WAREHOUSE TRACKING SYSTEM - CONTINUOUS MODE

    Enhanced tracking with BoT-SORT + ReID:
    - Appearance-based re-identification
    - Camera motion compensation
    - Extended track persistence (60+ seconds)
    - Reduced duplicate database entries
    - Multi-cue association (motion + appearance)

    Options for running:
    1. Continuous mode (runs until stopped)
    2. Timed mode (runs for specific duration)
    3. GUI quit (press 'q' in GUI window)
    4. Keyboard interrupt (Ctrl+C)

    Features:
    - All warehouse cameras supported
    - BoT-SORT tracking with ReID (maintains persistence across occlusions)
    - Real-time database integration
    - Performance monitoring
    - Warehouse-optimized parameters
    """

    # 🚀 CHOOSE YOUR RUNNING MODE:

    # Option 1: CONTINUOUS MODE (recommended for production)
    run_continuous_tracking(
        camera_ids=[1,2,3],  # Add more cameras: [8, 9, 10, 11] or [1,2,3,4,5,6,7,8,9,10,11]
        confidence=0.85,
        enable_gui=True
    )

    # Option 2: TIMED TEST MODE (uncomment to use instead)
    # run_timed_test(
    #     camera_ids=[8, 9, 10],
    #     duration=60,           # Run for 60 seconds
    #     confidence=0.5,
    #     enable_gui=True
    # )

    # Option 3: ADVANCED CUSTOM MODE (uncomment to use instead)
    # test_detection_speed(
    #     camera_ids=[8, 9, 10],
    #     duration=None,          # None = continuous, or specify seconds
    #     confidence=0.5,
    #     enable_gui=True,
    #     enable_tracking=True,
    #     enable_database=True,
    #     continuous=True
    # )