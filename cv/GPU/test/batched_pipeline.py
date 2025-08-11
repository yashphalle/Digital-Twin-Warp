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
from GPU.pipelines.batch_ring_buffer import RingBuffer, BatchedRingBuffer
from GPU.pipelines.gpu_processor_fast_tracking import GPUBatchProcessorFastTracking

SIMILARITY_THRESHOLD = 0.5
class GPUBatchProcessorBoTSORT(GPUBatchProcessorFastTracking):
    """BoT-SORT processor with Smart Persistence extending the fast tracking processor"""

    def __init__(self, model_path, device, active_cameras, confidence=0.5, use_fp16=False,
                 botsort_config_path=None):
        self.botsort_config_path = botsort_config_path or create_botsort_config_file()
        super().__init__(model_path, device, active_cameras, confidence, use_fp16)
        self.conf_threshold = confidence
        self.device = device
        self.smart_persistence = SmartPersistenceManager(active_cameras)
        self.persistent_id_map={}
        logger.info(f"🎯 BoT-SORT processor initialized with config: {self.botsort_config_path}")
        logger.info(f"🎯 ReID enabled: {WAREHOUSE_BOTSORT_CONFIG['with_reid']}")
        logger.info(f"🎯 Track buffer: {WAREHOUSE_BOTSORT_CONFIG['track_buffer']} frames")
        logger.info(f"🎯 Appearance threshold: {WAREHOUSE_BOTSORT_CONFIG['appearance_thresh']}")
        logger.info(f"🚀 Smart Persistence enabled for cross-camera tracking")

    def _create_model_instance(self, model_path, device):
        """Create YOLO model instance and VERIFY the Re-ID model being used."""
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        model.track(dummy_frame, persist=True, tracker=self.botsort_config_path, verbose=False)
        try:
            tracker = model.predictor.trackers[0]
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

    def process_batch(self, frame_batch: Dict[int, List[np.ndarray]]) -> tuple:
        """
        MODIFIED FOR HIGH THROUGHPUT: Processes a batch of frame batches.
        """
        start_time = time.time()
        
        all_detections = {}
        processed_frames_for_gui = {}

        for cam_id, frames in frame_batch.items():
            if not frames:
                continue

            camera_model = self.models.get(cam_id)
            if not camera_model:
                continue

            results_list = camera_model.track(frames, device=self.device, conf=self.conf_threshold, tracker=self.botsort_config_path, persist=True, verbose=False)

            self.total_batches += 1
            self.frames_processed += len(frames)

            cam_detections = []
            last_results_raw = results_list[-1]
            
            base_objects = self._extract_tracked_objects_from_results([last_results_raw], cam_id)
            id_manager = self.id_managers.get(cam_id)
            
            for base_obj in base_objects:
                yolo_track_id = base_obj['track_id']
                global_id = id_manager.get_global_id(yolo_track_id)
                track_age = id_manager.get_track_age(global_id)

                if global_id not in self.persistent_id_map:
                    if base_obj.get('feature_vector') is not None:
                        match = self.smart_persistence.check_cross_camera_matches(base_obj, cam_id)
                        persistent_id = match['persistent_id'] if match else global_id
                    else:
                        persistent_id = global_id
                    self.persistent_id_map[global_id] = persistent_id
                else:
                    persistent_id = self.persistent_id_map.get(global_id, global_id)

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
                cam_detections.append(final_detection_obj)

            all_detections[cam_id] = cam_detections
            processed_frames_for_gui[cam_id] = frames[-1]

        inference_time = time.time() - start_time
        self.total_inference_time += inference_time

        for cam_id, detections in all_detections.items():
            if detections:
                self.smart_persistence.background_persistence(detections, cam_id)

        return processed_frames_for_gui, all_detections

    def _extract_tracked_objects_from_results(self, results, cam_id):
        tracked_objects = []
        if not results or not results[0].boxes:
            return []
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
        feature_map = {}
        if hasattr(tracker, 'tracked_stracks'):
            for strack in tracker.tracked_stracks:
                if strack.is_activated and strack.features:
                    feature_map[strack.track_id] = strack.features[-1]
        for i in range(len(boxes)):
            yolo_track_id = int(boxes.id[i].cpu().item())
            if yolo_track_id not in feature_map:
                continue
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
                'feature_vector': feature_vector
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
CONSECUTIVE_DETECTION_THRESHOLD = 20

# BoT-SORT Configuration for Warehouse Tracking
WAREHOUSE_BOTSORT_CONFIG = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.2,
    'track_low_thresh': 0.05,
    'new_track_thresh': 0.6,
    'track_buffer': 200,
    'max_age': 250,
    'match_thresh': 0.5,
    'proximity_thresh': 0.4,
    'with_reid': True,
    'model': 'yolov8n-cls.pt',
    'appearance_thresh': 0.3,
    'gmc_method': 'sparseOptFlow',
    'fuse_score': True,
    'min_hits': 10,
}

def create_botsort_config_file():
    """Create warehouse-optimized BoT-SORT configuration file"""
    import yaml
    import os
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'warehouse_botsort.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(WAREHOUSE_BOTSORT_CONFIG, f, default_flow_style=False)
    logger.info(f"✅ Created BoT-SORT config: {config_path}")
    return config_path

class SmartPersistenceManager:
    """Smart Persistence Manager for BoT-SORT with Feature-Based Cross-Camera Tracking"""

    def __init__(self, active_cameras: List[int]):
        self.active_cameras = active_cameras
        try:
            self.redis_client = redis.Redis(
                host='localhost', port=6379, decode_responses=False,
                socket_keepalive=True, health_check_interval=30
            )
            self.redis_client.ping()
            logger.info("✅ Connected to Redis for Smart Persistence")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
        self.camera_neighbors = {
            1: [2, 5], 2: [1, 3, 6], 3: [2, 4, 7], 4: [3], 5: [1, 6, 8],
            6: [5, 7, 2, 9], 7: [6, 3, 10], 8: [5, 9], 9: [8, 10, 6],
            10: [9, 11, 7], 11: [10]
        }
        self.redis_save_interval = 5.0
        self.last_redis_saves = {cam_id: time.time() for cam_id in active_cameras}
        logger.info(f"🚀 Smart Persistence Manager initialized for cameras: {active_cameras}")

    def _calculate_feature_similarity(self, new_features, stored_features):
        try:
            dot_product = np.dot(new_features, stored_features)
            norm_new = np.linalg.norm(new_features)
            norm_stored = np.linalg.norm(stored_features)
            if norm_new == 0 or norm_stored == 0:
                return 0.0
            return dot_product / (norm_new * norm_stored)
        except Exception:
            return 0.0

    def check_cross_camera_matches(self, unmatched_detection, camera_id):
        if not self.redis_client or unmatched_detection.get('feature_vector') is None:
            return None
        best_match = None
        best_similarity = 0.0
        new_features = unmatched_detection['feature_vector']
        cameras_to_check = [camera_id] + self.camera_neighbors.get(camera_id, [])
        try:
            track_keys = self.redis_client.keys('track:*')
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return None
        for key in track_keys:
            try:
                stored_track_data_pkl = self.redis_client.get(key)
                if not stored_track_data_pkl: continue
                stored_track = pickle.loads(stored_track_data_pkl)
                if stored_track.get('camera_id') not in cameras_to_check:
                    continue
                time_diff = time.time() - stored_track.get('last_seen', 0)
                if time_diff > 30.0:
                    continue
                stored_features = stored_track.get('feature_vector')
                if stored_features is None:
                    continue
                similarity = self._calculate_feature_similarity(new_features, stored_features)
                if similarity > SIMILARITY_THRESHOLD and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = stored_track
            except Exception:
                continue
        return best_match

    def background_persistence(self, tracked_objects, camera_id):
        current_time = time.time()
        last_save_time = self.last_redis_saves.get(camera_id, 0)
        if current_time - last_save_time > self.redis_save_interval:
            threading.Thread(
                target=self.save_tracks_to_redis,
                args=(tracked_objects, camera_id),
                daemon=True
            ).start()
            self.last_redis_saves[camera_id] = current_time

    def save_tracks_to_redis(self, tracked_objects, camera_id):
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
                    count += 1
            pipe.execute()
        except Exception as e:
            logger.error(f"❌ ERROR: Exception during Redis save: {e}")

class OptimizedWarehouseDatabaseWriter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.detection_queue = queue.Queue(maxsize=20000)
        self.batch_timeout = 2.0
        self.max_batch_size = 1000
        self.running = False
        try:
            self.mongo_client = MongoClient("mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/")
            self.db = self.mongo_client.WARP
            self.collection = self.db.detections
            self.mongo_client.admin.command('ping')
            logger.info("✅ Connected to MongoDB Atlas")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
        self.coordinate_mappers = {}
        self._initialize_coordinate_mappers()
        self.color_extractor = FastColorExtractor()
        self.stats = {'total_queued': 0, 'total_new_inserts': 0, 'total_updates': 0, 'total_errors': 0, 'last_batch_size': 0, 'queue_depth': 0}

    def _initialize_coordinate_mappers(self):
        all_cameras = list(range(1, 12))
        for camera_id in all_cameras:
            try:
                mapper = CoordinateMapper(camera_id=camera_id)
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
        if not self.mongo_client:
            return
        try:
            self.detection_queue.put(enriched_detection, block=False)
            self.stats['total_queued'] += 1
        except queue.Full:
            try:
                self.detection_queue.get(block=False)
                self.detection_queue.put(enriched_detection, block=False)
                logger.warning("⚠️ Database queue full, dropped oldest detection")
            except queue.Empty:
                pass

    def start(self):
        if self.mongo_client:
            self.running = True
            super().start()
            logger.info("🚀 Database writer started (20K queue, 2-sec batches)")
        else:
            logger.warning("⚠️ Database writer not started - no connection")

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=5)
        logger.info("📊 === FINAL DATABASE STATISTICS ===")
        # ... (logging stats) ...

    def run(self):
        if not self.mongo_client:
            return
        batch = []
        last_write = time.time()
        logger.info("📝 Database writer thread started")
        while self.running:
            try:
                detection = self.detection_queue.get(timeout=0.1)
                batch.append(detection)
                if (time.time() - last_write >= self.batch_timeout or len(batch) >= self.max_batch_size):
                    self._process_batch(batch)
                    batch = []
                    last_write = time.time()
            except queue.Empty:
                if batch and time.time() - last_write >= self.batch_timeout:
                    self._process_batch(batch)
                    batch = []
                    last_write = time.time()
            except Exception as e:
                logger.error(f"❌ Database writer error: {e}")
                self.stats['total_errors'] += 1
        if batch:
            self._process_batch(batch)
        logger.info("📝 Database writer thread finished")

    def _process_batch(self, detections):
        if not detections or not self.mongo_client:
            return
        update_operations = []
        pending_count, new_doc_count, updated_doc_count = 0, 0, 0
        current_time = datetime.now(pytz.timezone('US/Pacific'))
        for detection in detections:
            tracking_status = detection['tracking_status']
            if tracking_status == 'pending':
                pending_count += 1
                continue
            persistent_id = detection.get('persistent_id')
            if not persistent_id:
                continue
            if tracking_status == 'new': new_doc_count += 1
            elif tracking_status == 'existing': updated_doc_count += 1
            update_operations.append(UpdateOne(
                {'persistent_id': persistent_id},
                {
                    '$set': {
                        'last_seen': current_time, 'bbox': detection['bbox'], 'corners': detection['corners'],
                        'physical_corners': detection.get('physical_corners'), 'real_center': detection.get('real_center'),
                        'confidence': detection['confidence'], 'area': detection['area'], 'center': detection['center'],
                        'physical_x_ft': detection.get('physical_x_ft'), 'physical_y_ft': detection.get('physical_y_ft'),
                        'coordinate_status': detection.get('coordinate_status'), 'similarity_score': detection.get('similarity_score', 1.0),
                        'global_id': detection.get('global_id'), 'camera_id': detection.get('camera_id'),
                    },
                    '$setOnInsert': {
                        'persistent_id': persistent_id, 'first_seen': current_time, 'class': detection['class'],
                        'color_rgb': detection.get('color_rgb'), 'color_hsv': detection.get('color_hsv'),
                        'color_hex': detection.get('color_hex'), 'color_name': detection.get('color_name'),
                        'color_confidence': detection.get('color_confidence'),
                    },
                    '$inc': {'times_seen': 1}
                },
                upsert=True
            ))
        try:
            if update_operations:
                result = self.collection.bulk_write(update_operations, ordered=False)
                self.stats['total_new_inserts'] += result.upserted_count
                self.stats['total_updates'] += result.modified_count
            self.stats['last_batch_size'] = len(detections)
            self.stats['queue_depth'] = self.detection_queue.qsize()
            logger.info(f"📊 DB Batch: {new_doc_count} new, {updated_doc_count} updates, {pending_count} pending, Queue: {self.stats['queue_depth']}/20000")
        except Exception as e:
            logger.error(f"❌ Database batch error: {e}")
            self.stats['total_errors'] += 1

def enrich_detection_fast(detection, coordinate_mappers, color_info, frame_width=1600, frame_height=900):
    global_id = detection['global_id']
    persistent_id = detection.get('persistent_id', global_id)
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    track_age = detection['track_age']
    camera_id = detection['camera_id']
    if track_age < CONSECUTIVE_DETECTION_THRESHOLD: tracking_status = 'pending'
    elif track_age == CONSECUTIVE_DETECTION_THRESHOLD: tracking_status = 'new'
    else: tracking_status = 'existing'
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    physical_x_ft, physical_y_ft, coordinate_status = None, None, 'unmapped'
    if camera_id in coordinate_mappers:
        mapper = coordinate_mappers[camera_id]
        scale_x, scale_y = 3840 / frame_width, 2160 / frame_height
        scaled_center_x, scaled_center_y = center_x * scale_x, center_y * scale_y
        physical_x_ft, physical_y_ft = mapper.pixel_to_real(scaled_center_x, scaled_center_y)
        if physical_x_ft is not None: coordinate_status = 'mapped'
        else: physical_x_ft, physical_y_ft, coordinate_status = camera_id * 20.0, 50.0, 'fallback'
    else: physical_x_ft, physical_y_ft, coordinate_status = camera_id * 20.0, 50.0, 'no_mapper'
    enriched = {
        'persistent_id': persistent_id, 'global_id': global_id, 'camera_id': camera_id, 'warp_id': None,
        'tracking_status': tracking_status, 'class': detection['class'], 'age_seconds': track_age,
        'consecutive_detections': track_age, 'physical_x_ft': physical_x_ft, 'physical_y_ft': physical_y_ft,
        'coordinate_status': coordinate_status, 'real_center': [physical_x_ft, physical_y_ft], 'bbox': bbox,
        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 'physical_corners': None, 'area': (x2 - x1) * (y2 - y1),
        'center': [center_x, center_y], 'shape_type': 'quadrangle', 'confidence': detection['confidence'],
        'similarity_score': 1.0, 'timestamp': datetime.now(pytz.timezone('US/Pacific')), **color_info
    }
    return enriched

class SimpleGUIDisplay:
    def __init__(self, camera_ids: List[int]):
        self.camera_ids = camera_ids
        self.running = False
        self.display_thread = None
        self.latest_frames = {}
        self.latest_detections = {}
        self.stats = {}
        self.tracking_enabled = False
        self.num_cameras = len(camera_ids)
        if self.num_cameras <= 4: self.grid_cols, self.grid_rows = 2, 2
        elif self.num_cameras <= 6: self.grid_cols, self.grid_rows = 3, 2
        elif self.num_cameras <= 9: self.grid_cols, self.grid_rows = 3, 3
        else: self.grid_cols, self.grid_rows = 4, 3
        self.cell_width, self.cell_height = 320, 240
        self.display_width = self.grid_cols * self.cell_width
        self.display_height = self.grid_rows * self.cell_height + 100
        logger.info(f"GUI Display initialized for {self.num_cameras} cameras ({self.grid_cols}x{self.grid_rows} grid)")

    def update_frame(self, camera_id: int, frame: np.ndarray, detections: List[Dict] = None):
        if frame is not None: self.latest_frames[camera_id] = frame.copy()
        if detections is not None: self.latest_detections[camera_id] = detections

    def update_stats(self, stats: Dict):
        self.stats = stats.copy()

    def get_tracking_color(self, detection: Dict) -> tuple:
        track_age = detection.get('track_age', 0)
        if track_age < CONSECUTIVE_DETECTION_THRESHOLD: return (0, 255, 255)
        else: return (0, 165, 255)

    def create_tracking_label(self, detection: Dict) -> str:
        persistent_id = detection.get('persistent_id')
        class_name = detection.get('class', 'Object').upper()
        if persistent_id is not None: return f"{class_name}: {persistent_id}"
        else: return f"{class_name}"

    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict], original_frame_size: tuple = None) -> np.ndarray:
        if not detections: return frame
        result_frame = frame.copy()
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = self.get_tracking_color(detection)
                label = self.create_tracking_label(detection)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 10 if y1 > 30 else y2 + 20
                cv2.rectangle(result_frame, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
                cv2.putText(result_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return result_frame

    def create_camera_cell(self, camera_id: int, tracking_enabled: bool = False) -> np.ndarray:
        cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)
        header_text = f"Camera {camera_id} - {'TRACKING' if tracking_enabled else 'DETECTION'}"
        header_color = (0, 165, 255) if tracking_enabled else (0, 255, 0)
        cv2.putText(cell, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, header_color, 2)
        frame = self.latest_frames.get(camera_id)
        detections = self.latest_detections.get(camera_id, [])
        if frame is not None:
            frame_resized = cv2.resize(frame, (self.cell_width, self.cell_height - 30))
            frame_with_detections = self.draw_detections_on_frame(frame_resized, detections, frame.shape[:2])
            cell[30:, :] = frame_with_detections
        else:
            cv2.putText(cell, "NO SIGNAL", (self.cell_width//2 - 50, self.cell_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return cell

    def create_display_frame(self, tracking_enabled: bool = False) -> np.ndarray:
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        for i, camera_id in enumerate(self.camera_ids):
            if i >= self.grid_cols * self.grid_rows: break
            row, col = i // self.grid_cols, i % self.grid_cols
            x, y = col * self.cell_width, row * self.cell_height
            cell = self.create_camera_cell(camera_id, tracking_enabled)
            display[y:y+self.cell_height, x:x+self.cell_width] = cell
        stats_y = self.grid_rows * self.cell_height + 20
        if self.stats:
            stats_text = [f"Total Frames: {self.stats.get('total_frames', 0)}", f"Active Tracks: {self.stats.get('active_tracks', 0)}", f"Total Tracks Created: {self.stats.get('total_tracks_created', 0)}", f"Avg FPS: {self.stats.get('avg_fps', 0.0):.1f}", f"Avg Inference: {self.stats.get('avg_inference_time_ms', 0.0):.1f}ms"]
            for i, text in enumerate(stats_text):
                x_pos = 20 + (i % 3 * 250)
                y_pos = stats_y + (i // 3 * 25)
                if x_pos < self.display_width - 150:
                    cv2.putText(display, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display

    def display_loop(self):
        window_title = "Tracking Speed Test"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, self.display_width, self.display_height)
        while self.running:
            try:
                display_frame = self.create_display_frame(self.tracking_enabled)
                cv2.imshow(window_title, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                    break
            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(0.1)
        cv2.destroyAllWindows()

    def start(self):
        if not self.running:
            self.running = True
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()

    def stop(self):
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        cv2.destroyAllWindows()

def test_detection_speed(camera_ids=[8, 9], duration=None, confidence=0.5, enable_gui=True, enable_tracking=False, enable_database=True, continuous=False):
    run_mode = "CONTINUOUS" if continuous or duration is None else f"{duration}s"
    duration = None if continuous or duration is None else duration
    mode_name = "BoT-SORT TRACKING" if enable_tracking else "DETECTION"
    logger.info(f"=== {mode_name} SPEED TEST (PARALLEL) ===")
    
    HIGH_THROUGHPUT_BATCH_SIZE = 10
    ring_buffer = BatchedRingBuffer(num_cameras=11, buffer_size=60)
    logger.info("🚀 Using BatchedRingBuffer for high-throughput processing.")

    db_writer = None
    if enable_database:
        db_writer = OptimizedWarehouseDatabaseWriter()
        db_writer.start()

    if enable_tracking:
        gpu_processor = GPUBatchProcessorBoTSORT(
            model_path='custom_yolo.pt', device='cuda:0', active_cameras=camera_ids,
            confidence=confidence, use_fp16=False
        )
    else:
        from GPU.pipelines.gpu_processor_detecction_only_batch import GPUBatchProcessorDetectionOnly as GPUBatchProcessorDetectionOnlyBatched
        gpu_processor = GPUBatchProcessorDetectionOnlyBatched(
            model_path='custom_yolo.pt', device='cuda:0', active_cameras=camera_ids,
            confidence=confidence, use_fp16=False
        )

    gui_display = None
    if enable_gui:
        gui_display = SimpleGUIDisplay(camera_ids)
        gui_display.tracking_enabled = enable_tracking
        gui_display.start()

    camera_workers = []
    for cam_id in camera_ids:
        worker = ParallelCameraWorker(cam_id, ring_buffer, frame_skip=2)
        if worker.connect():
            worker.start()
            camera_workers.append(worker)

    if not camera_workers:
        logger.error("No cameras connected")
        return

    time.sleep(3)
    logger.info("Warming up GPU...")
    for _ in range(5):
        frame_batches = ring_buffer.get_latest_batch(HIGH_THROUGHPUT_BATCH_SIZE)
        if frame_batches: gpu_processor.process_batch(frame_batches)
        time.sleep(0.1)
    
    gpu_processor.total_batches, gpu_processor.total_inference_time, gpu_processor.frames_processed = 0, 0, 0
    start_time = time.time()

    try:
        while True:
            if duration is not None and time.time() - start_time >= duration: break
            if gui_display and not gui_display.running: break

            frame_batches = ring_buffer.get_latest_batch(HIGH_THROUGHPUT_BATCH_SIZE)
            if not frame_batches:
                time.sleep(0.01)
                continue

            processed_frames, detections_by_camera = gpu_processor.process_batch(frame_batches)

            if enable_database and db_writer and enable_tracking:
                for cam_id, detections in detections_by_camera.items():
                    latest_frame_for_color = processed_frames.get(cam_id)
                    for detection in detections:
                        color_info = extract_color_for_detection(latest_frame_for_color, detection, db_writer.color_extractor)
                        enriched = enrich_detection_fast(detection, db_writer.coordinate_mappers, color_info)
                        db_writer.queue_detection(enriched)

            if gui_display:
                for cam_id, frame in processed_frames.items():
                    gui_display.update_frame(cam_id, frame, detections_by_camera.get(cam_id, []))
                stats = gpu_processor.get_stats()
                gui_display.update_stats(stats)

    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user (Ctrl+C)")

    elapsed = time.time() - start_time
    stats = gpu_processor.get_stats()
    logger.info("\n=== RESULTS ===")
    logger.info(f"Test ran for: {elapsed:.1f}s")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Throughput: {stats['throughput']:.1f} frames/second")
    if gui_display: gui_display.stop()
    for worker in camera_workers: worker.stop()
    if db_writer: db_writer.stop()
    gpu_processor.cleanup()

def run_continuous_tracking(camera_ids=[8, 9, 10], confidence=0.5, enable_gui=True):
    test_detection_speed(
        camera_ids=camera_ids, duration=None, confidence=confidence,
        enable_gui=enable_gui, enable_tracking=True, enable_database=True, continuous=True
    )

if __name__ == "__main__":
    test_detection_speed(
        camera_ids=[1,2,3],
        duration=None,
        confidence=0.85,
        enable_gui=True,
        enable_tracking=True,  # <-- THIS IS THE KEY CHANGE
        enable_database=True,   # You can set this to False if you don't want to save detections
        continuous=True
    )

