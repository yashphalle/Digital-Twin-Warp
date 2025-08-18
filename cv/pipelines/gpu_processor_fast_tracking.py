# cv/GPU/pipelines/gpu_processor_fast_tracking.py

import os
import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)



class FastGlobalIDManager:
    """
    Upgraded global ID manager with grace period for cleanup and support for external assignments.
    """
    GRACE_PERIOD_FRAMES = 90  # How many frames to remember a lost track (should match BoT-SORT's buffer)

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.id_base = camera_id * 1000  # Camera 8 -> 8000
        self.next_id = self.id_base + 1      # Start from 8001
        
        # Mappings
        self.yolo_to_global = {}  # Map: yolo_track_id -> global_id
        self.global_to_yolo = {}  # Map: global_id -> yolo_track_id (for reverse lookup)

        # State Tracking
        self.track_ages = {}          # Map: global_id -> age in frames
        self.last_seen_frame = {}     # Map: global_id -> frame_index
        self.current_frame_index = 0

        logger.info(f"Upgraded GlobalIDManager initialized for Camera {camera_id}")

    def get_global_id(self, yolo_track_id: int) -> int:
        """Get global ID for a YOLO track ID, creating a new one if it's the first time."""
        self.current_frame_index += 1

        if yolo_track_id not in self.yolo_to_global:
            # New track for this manager - assign a new global ID
            global_id = self.next_id
            self.yolo_to_global[yolo_track_id] = global_id
            self.global_to_yolo[global_id] = yolo_track_id
            self.track_ages[global_id] = 0
            self.next_id += 1
            logger.info(f"Camera {self.camera_id}: New track {yolo_track_id} -> Global ID {global_id}")
        
        global_id = self.yolo_to_global[yolo_track_id]
        self.track_ages[global_id] += 1  # Increment age
        self.last_seen_frame[global_id] = self.current_frame_index # Update last seen time
        return global_id

    def assign_external_global_id(self, new_yolo_id: int, existing_global_id: int, original_track_age: int = 20):
        """
        Forces a mapping for a cross-camera handoff.
        Called when Redis ReID finds a match.
        """
        logger.info(f"HANDOFF on Cam {self.camera_id}: Mapping new YOLO ID {new_yolo_id} to existing Global ID {existing_global_id}")
        self.yolo_to_global[new_yolo_id] = existing_global_id
        self.global_to_yolo[existing_global_id] = new_yolo_id
        self.track_ages[existing_global_id] = original_track_age  # Inherit age to avoid age<1 filters
        self.last_seen_frame[existing_global_id] = self.current_frame_index

    def get_track_age(self, global_id: int) -> int:
        """Get track age in frames."""
        return self.track_ages.get(global_id, 0)

    def cleanup_lost_tracks(self, active_yolo_ids: List[int]):
        """
        Removes mappings for tracks that have been lost for longer than the grace period.
        """
        lost_global_ids = []
        for global_id, last_seen in self.last_seen_frame.items():
            if self.current_frame_index - last_seen > self.GRACE_PERIOD_FRAMES:
                lost_global_ids.append(global_id)
        
        for global_id in lost_global_ids:
            if global_id in self.global_to_yolo:
                yolo_id = self.global_to_yolo.pop(global_id)
                self.yolo_to_global.pop(yolo_id, None)
            
            self.track_ages.pop(global_id, None)
            self.last_seen_frame.pop(global_id, None)
            logger.debug(f"Camera {self.camera_id}: Purged expired track (Global ID {global_id})")

    # get_stats method can remain the same
    def get_stats(self) -> Dict:
        return {
            'active_tracks': len(self.yolo_to_global),
            'next_id': self.next_id,
            'total_tracks_created': self.next_id - self.id_base - 1
        }

class GPUBatchProcessorFastTracking:
    """
    GPU batch processor for YOLOv8 tracking with speed optimization
    Uses camera-prefixed global IDs and 10-second track persistence
    """
    
    def __init__(self, 
                 model_path: str = 'custom_yolo.pt',
                 device: str = 'cuda:0',
                 active_cameras: List[int] = None,
                 confidence: float = 0.5,
                 use_fp16: bool = True):
        """
        Initialize GPU batch processor for fast tracking
        
        Args:
            model_path: Path to custom YOLOv8 model
            device: CUDA device to use
            active_cameras: List of camera IDs to process
            confidence: Detection confidence threshold
            use_fp16: Use half precision for faster inference
        """
        self.device = device
        self.confidence = confidence
        self.active_cameras = active_cameras or []
        
        # Initialize separate YOLOv8 model instances for each camera
        # This ensures each camera maintains its own tracking state
        logger.info(f"🔄 Loading YOLOv8 models for {len(self.active_cameras)} cameras: {model_path}")

        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Model file not found: {model_path}")
            logger.info("📁 Current working directory: " + os.getcwd())
            logger.info("📁 Looking for model in current directory...")

        # Create separate model instance for each camera
        self.models = {}
        for cam_id in self.active_cameras:
            self.models[cam_id] = self._create_model_instance(model_path, device)
            logger.info(f"✅ Model loaded for Camera {cam_id}")

        logger.info(f"🔍 Debug: models attribute created with {len(self.models)} instances")

        # Log model info (using first model as reference)
        first_model = list(self.models.values())[0]
        logger.info(f"✅ {len(self.models)} separate model instances loaded successfully")
        logger.info(f"📊 Model classes: {len(first_model.names)} classes")
        logger.info(f"🏷️ Class names: {list(first_model.names.values())[:5]}...")  # Show first 5 classes
        
        # Configure tracking for speed and 10-second persistence
        # Use only valid YOLO tracking parameters
        self.tracking_config = {
            "tracker": "bytetrack.yaml",  # Use ByteTrack algorithm
            "persist": True,              # Maintain tracks across calls
            "verbose": False              # Reduce logging
        }
        
        # Initialize global ID managers for each camera
        self.id_managers = {}
        for cam_id in self.active_cameras:
            self.id_managers[cam_id] = FastGlobalIDManager(cam_id)
        
        # Performance tracking
        self.total_batches = 0
        self.total_inference_time = 0
        self.frames_processed = 0
        self.total_tracks_created = 0
        self.total_tracks_active = 0
        
        # FP16 optimization
        if use_fp16 and torch.cuda.is_available():
            logger.info("Enabling FP16 optimization")
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"✅ Fast Tracking Processor initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Active cameras: {active_cameras}")
        logger.info(f"   Confidence: {confidence}")
        logger.info(f"   Track persistence: 10 seconds (150 frames)")
        
    # In GPUBatchProcessorBoTSORT class

    def process_batch(self, ring_buffer) -> tuple:
        """
        FINAL DIAGNOSTIC VERSION: Adds two checkpoints to trace the 'persistent_id' key.
        """
        logger.info("process_batch called")
        start_time = time.time()
        batch_frames = ring_buffer.get_batch(max_age=1)
        if not batch_frames: return {}, {}

        active_frames = {c: f for c, f in batch_frames.items() if c in self.active_cameras}
        if not active_frames: return {}, {}

        # Run Inference
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

        # Process results
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

                if track_age == 1:
                    match = self.smart_persistence.check_cross_camera_matches(base_obj, cam_id)
                    persistent_id = match['persistent_id'] if match else global_id
                    self.persistent_id_map[global_id] = persistent_id
                else:
                    persistent_id = self.persistent_id_map.get(global_id, global_id)

                # Build the final dictionary
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

                # --- CHECKPOINT 1: Right after creation ---
                if 'persistent_id' in final_detection_obj:
                    print(f"DEBUG (Checkpoint 1 - Cam {cam_id}): Object created. 'persistent_id' IS PRESENT. Value: {final_detection_obj['persistent_id']}")
                else:
                    print(f"DEBUG (Checkpoint 1 - Cam {cam_id}): ❗️❗️ OBJECT CREATED BUT 'persistent_id' IS MISSING!")
                # --- END CHECKPOINT 1 ---

                final_detections.append(final_detection_obj)

            detections_by_camera[cam_id] = final_detections
            id_manager.cleanup_lost_tracks(list(active_yolo_ids))

        # Trigger background persistence
        for cam_id, detections in detections_by_camera.items():
            if detections:
                # --- CHECKPOINT 2: Right before saving ---
                first_obj_before_save = detections[0]
                print(f"\nDEBUG (Checkpoint 2 - Cam {cam_id}): About to call background_persistence for {len(detections)} objects.")
                if 'persistent_id' in first_obj_before_save:
                    print(f"  - ✅ 'persistent_id' IS STILL PRESENT in the list.")
                else:
                    print(f"  - ❗️❗️ 'persistent_id' HAS DISAPPEARED from the list before saving!")
                # --- END CHECKPOINT 2 ---

                self.smart_persistence.background_persistence(detections, cam_id)

        return active_frames, detections_by_camera

    def get_stats(self) -> Dict:
        """
        MODIFIED: Get comprehensive tracking statistics with corrected calculations
        for both single-frame and high-throughput batch processing.
        """
        if self.total_batches == 0 or self.total_inference_time == 0:
            return {
                'total_batches': 0,
                'total_frames': 0,
                'total_tracks_created': 0,
                'active_tracks': 0,
                'avg_inference_time_ms': 0.0,
                'avg_ms_per_frame': 0.0,
                'avg_fps': 0.0,
                'throughput': 0.0
            }

        # Aggregate stats from all ID managers
        total_tracks_created = sum(mgr.get_stats()['total_tracks_created'] for mgr in self.id_managers.values())
        active_tracks = sum(mgr.get_stats()['active_tracks'] for mgr in self.id_managers.values())

        # --- CORRECTED CALCULATIONS ---
        # Avg time to process one entire batch of frames
        avg_inference_time_ms = (self.total_inference_time / self.total_batches) * 1000

        # Avg time it takes to process a single frame within a batch
        avg_ms_per_frame = (self.total_inference_time / self.frames_processed) * 1000

        # Processing FPS: How many frames the GPU can process per second (inverse of time per frame)
        avg_fps = 1000 / avg_ms_per_frame if avg_ms_per_frame > 0 else 0

        # Throughput: How many frames are completed and leave the system per second (end-to-end)
        throughput = self.frames_processed / self.total_inference_time
        # --- END CORRECTION ---

        return {
            'total_batches': self.total_batches,
            'total_frames': self.frames_processed,
            'total_tracks_created': total_tracks_created,
            'active_tracks': active_tracks,
            'avg_inference_time_ms': avg_inference_time_ms,
            'avg_ms_per_frame': avg_ms_per_frame,
            'avg_fps': avg_fps,
            'throughput': throughput
        }

    
    def cleanup(self):
        """Cleanup GPU resources"""
        # Clear multiple model instances
        if hasattr(self, 'models'):
            num_models = len(self.models)
            for model in self.models.values():
                del model
            self.models.clear()
            logger.info(f"🗑️ Cleared {num_models} model instances")

        # Legacy cleanup for single model
        if hasattr(self, 'model'):
            del self.model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Fast tracking processor cleaned up")

    # In GPUBatchProcessorFastTracking class

    def _create_model_instance(self, model_path, device):
        """Default model instance creation."""
        model = YOLO(model_path)
        model.to(device)
        return model
