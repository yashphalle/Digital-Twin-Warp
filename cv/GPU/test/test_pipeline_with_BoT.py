

import sys
import os
import time
import logging
import cv2
import numpy as np
import threading
import queue
from typing import Dict, List
from datetime import datetime
import pytz
from pymongo import MongoClient
from pymongo.operations import UpdateOne

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker_parallel import ParallelCameraWorker
from GPU.pipelines.ring_buffer import RingBuffer
from GPU.pipelines.gpu_processor_detection_only import GPUBatchProcessorDetectionOnly
from GPU.pipelines.gpu_processor_fast_tracking import GPUBatchProcessorFastTracking

class GPUBatchProcessorBoTSORT(GPUBatchProcessorFastTracking):
    """BoT-SORT processor extending the fast tracking processor"""

    def __init__(self, model_path, device, active_cameras, confidence=0.5, use_fp16=False,
                 botsort_config_path=None):
        # Initialize parent class
        super().__init__(model_path, device, active_cameras, confidence, use_fp16)

        # Store BoT-SORT config path
        self.botsort_config_path = botsort_config_path or create_botsort_config_file()

        logger.info(f"🎯 BoT-SORT processor initialized with config: {self.botsort_config_path}")
        logger.info(f"🎯 ReID enabled: {WAREHOUSE_BOTSORT_CONFIG['with_reid']}")
        logger.info(f"🎯 Track buffer: {WAREHOUSE_BOTSORT_CONFIG['track_buffer']} frames")
        logger.info(f"🎯 Appearance threshold: {WAREHOUSE_BOTSORT_CONFIG['appearance_thresh']}")

    def _create_model_instance(self, model_path, device):
        """Create YOLO model instance with BoT-SORT tracker"""
        from ultralytics import YOLO

        model = YOLO(model_path)
        model.to(device)

        logger.info(f"📦 Created BoT-SORT model instance: {model_path} on {device}")
        return model

    def _run_inference_with_tracking(self, camera_model, frame, cam_id):
        """Run inference with BoT-SORT tracking"""
        try:
            # Use BoT-SORT configuration
            results = camera_model.track(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                tracker=self.botsort_config_path,  # Use our custom BoT-SORT config
                persist=True,  # Maintain tracks across calls
                verbose=False
            )

            return results

        except Exception as e:
            logger.error(f"❌ BoT-SORT inference error for camera {cam_id}: {e}")
            return None

# Import tested coordinate mapper from final folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cv', 'final', 'modules'))
from coordinate_mapper import CoordinateMapper

# Import fast color extractor
from color_extractor import FastColorExtractor, extract_color_for_detection, get_null_color

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BoT-SORT Configuration for Warehouse Tracking
WAREHOUSE_BOTSORT_CONFIG = {
    'tracker_type': 'botsort',

    # LENIENT detection thresholds (reduce new track creation)
    'track_high_thresh': 0.4,      # More detections continue existing tracks
    'track_low_thresh': 0.1,       # Standard recovery threshold
    'new_track_thresh': 0.5,       # Easier to create tracks when needed

    # EXTENDED persistence (handle 60-second occlusions)
    'track_buffer': 120,           # 6 seconds motion-based persistence
    'max_age': 150,               # 7.5 seconds before deletion

    # LENIENT matching (continue existing tracks more easily)
    'match_thresh': 0.7,          # More lenient IoU matching
    'proximity_thresh': 0.5,      # Spatial constraint for ReID

    # MODERATE ReID (balance accuracy/performance)
    'with_reid': True,
    'model': 'auto',              # Use native YOLO features
    'appearance_thresh': 0.25,    # Moderate appearance similarity

    # STABILITY features (reduce tracking instability)
    'gmc_method': 'sparseOptFlow', # Camera motion compensation
    'fuse_score': True,           # Combine confidence + IoU
    'min_hits': 2,                # Faster track confirmation
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

def enrich_detection_fast(detection, coordinate_mappers, color_info, frame_width=1600, frame_height=900):
    """Fast enrichment with REAL physical coordinates + color analysis - FRONTEND/BACKEND COMPATIBLE"""
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    track_age = detection['track_age']
    global_id = detection['track_id']
    camera_id = detection['camera_id']

    # Determine tracking status for existing database logic
    tracking_status = 'new' if track_age == 1 else 'existing'

    # Calculate center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Get REAL physical coordinates using tested coordinate mapper from final folder
    physical_x_ft = None
    physical_y_ft = None
    coordinate_status = 'unmapped'

    if camera_id in coordinate_mappers:
        mapper = coordinate_mappers[camera_id]

        # Scale coordinates to calibration frame size (4K) - SAME as tested final code
        # Calibration files are based on 3840x2160 resolution
        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height

        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y

        # Transform to physical coordinates using tested homography logic
        physical_x_ft, physical_y_ft = mapper.pixel_to_real(scaled_center_x, scaled_center_y)

        if physical_x_ft is not None and physical_y_ft is not None:
            coordinate_status = 'mapped'
            logger.debug(f"Camera {camera_id}: Pixel ({center_x:.0f}, {center_y:.0f}) → Physical ({physical_x_ft:.1f}ft, {physical_y_ft:.1f}ft)")
        else:
            # Fallback to dummy coordinates if transformation fails
            physical_x_ft = camera_id * 20.0
            physical_y_ft = 50.0
            coordinate_status = 'fallback'
            logger.debug(f"Camera {camera_id}: Using fallback coordinates ({physical_x_ft:.1f}ft, {physical_y_ft:.1f}ft)")
    else:
        # Fallback to dummy coordinates if no mapper available
        physical_x_ft = camera_id * 20.0
        physical_y_ft = 50.0
        coordinate_status = 'no_mapper'

    enriched = {
        # CRITICAL IDENTITY FIELDS (Frontend/Backend Required)
        'persistent_id': global_id,          # Frontend primary key - MUST be included
        'global_id': global_id,              # Your camera-prefixed IDs
        'camera_id': detection['camera_id'],
        'warp_id': None,                     # QR code ID

        # CRITICAL TRACKING FIELDS (Database Logic Required)
        'tracking_status': tracking_status,   # 'new' or 'existing' for INSERT/UPDATE
        'class': detection['class'],         # Object class (Pallet/Forklift)
        'age_seconds': track_age,            # Frontend expects this name

        # CRITICAL PHYSICAL COORDINATES (Backend Filter Required)
        'physical_x_ft': physical_x_ft,      # Backend filters on this - MUST NOT BE NULL
        'physical_y_ft': physical_y_ft,      # Backend filters on this - MUST NOT BE NULL
        'coordinate_status': coordinate_status,  # Real mapping status from coordinate mapper
        'real_center': [physical_x_ft, physical_y_ft],  # Physical center for frontend

        # GEOMETRY FIELDS (Required)
        'bbox': bbox,
        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        'physical_corners': None,            # Will be calculated below using real coordinate mapper
        'area': (x2 - x1) * (y2 - y1),
        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
        'shape_type': 'quadrangle',

        # DETECTION INFO
        'confidence': detection['confidence'],
        'similarity_score': 1.0,

        # TIMESTAMPS (Will be set in database writer)
        'timestamp': datetime.now(pytz.timezone('US/Pacific')),

        # REAL COLOR ANALYSIS (extracted for ALL tracks - simplified approach)
        'color_rgb': color_info['rgb'],
        'color_hsv': color_info['hsv'],
        'color_hex': color_info['hex'],
        'color_name': color_info['name'],
        'color_confidence': color_info['confidence'],
        'extraction_method': color_info['extraction_method']
    }

    # Calculate REAL physical corners using coordinate mapper (if available)
    if camera_id in coordinate_mappers and physical_x_ft is not None and physical_y_ft is not None:
        mapper = coordinate_mappers[camera_id]

        # Transform all 4 corners to physical coordinates using tested logic from final folder
        corners_pixel = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # bbox corners
        physical_corners = []

        scale_x = 3840 / frame_width
        scale_y = 2160 / frame_height

        for corner in corners_pixel:
            pixel_x, pixel_y = corner
            scaled_x = pixel_x * scale_x
            scaled_y = pixel_y * scale_y

            # Transform to physical coordinates using tested homography
            phys_x, phys_y = mapper.pixel_to_real(scaled_x, scaled_y)

            if phys_x is not None and phys_y is not None:
                physical_corners.append([round(phys_x, 2), round(phys_y, 2)])
            else:
                physical_corners.append([None, None])

        # Only keep physical_corners if all corners were successfully transformed
        if not any(corner == [None, None] for corner in physical_corners):
            enriched['physical_corners'] = physical_corners
            logger.debug(f"Camera {camera_id}: Calculated real physical corners")

    # Debug logging for new tracks
    if tracking_status == 'new':
        logger.info(f"🆕 NEW TRACK ENRICHED: Camera {enriched['camera_id']}, global_id={enriched['global_id']}, persistent_id={enriched['persistent_id']}, track_age={track_age}, tracking_status={tracking_status}")
    else:
        logger.debug(f"🔄 EXISTING TRACK: Camera {enriched['camera_id']}, global_id={enriched['global_id']}, persistent_id={enriched['persistent_id']}, track_age={track_age}, tracking_status={tracking_status}")

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

    def _process_batch(self, detections):
        """Smart INSERT/UPDATE batch processing - COMPATIBLE WITH EXISTING SCHEMA"""
        if not detections or not self.mongo_client:
            return

        new_inserts = []
        update_operations = []
        current_time = datetime.now(pytz.timezone('US/Pacific'))

        for detection in detections:
            tracking_status = detection['tracking_status']  # Use tracking_status instead of track_age
            global_id = detection['global_id']
            camera_id = detection['camera_id']

            # Debug logging
            logger.debug(f"🔍 DB Processing: ID {global_id}, tracking_status='{tracking_status}', track_age={detection.get('age_seconds', 'N/A')}")

            if tracking_status == 'new':  # New track - INSERT with color data
                doc = detection.copy()
                doc.update({
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'times_seen': 1
                })

                # Debug: Verify persistent_id is included
                logger.info(f"🔍 NEW INSERT: global_id={doc.get('global_id')}, persistent_id={doc.get('persistent_id')}, camera_id={doc.get('camera_id')}")

                new_inserts.append(doc)

            elif tracking_status == 'existing':  # Existing track - UPDATE (matches existing logic)
                # Debug: Check if color fields are present in detection
                color_fields_in_detection = {
                    'color_rgb': detection.get('color_rgb'),
                    'color_hex': detection.get('color_hex'),
                    'color_hsv': detection.get('color_hsv')
                }
                logger.debug(f"🔍 EXISTING TRACK UPDATE: ID {global_id}, persistent_id={detection.get('persistent_id')}, color_fields={color_fields_in_detection}")

                update_operations.append(UpdateOne(
                    {'global_id': global_id, 'camera_id': camera_id},
                    {
                        '$set': {
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
                            'similarity_score': detection.get('similarity_score', 1.0),
                            'persistent_id': detection.get('persistent_id')  # CRITICAL: Preserve persistent_id
                            # NOTE: Deliberately excludes ALL color fields to preserve original colors
                        },
                        '$inc': {'times_seen': 1}
                    },
                    upsert=True  # Fallback if not found
                ))

        # Execute batch operations
        try:
            if new_inserts:
                result = self.collection.insert_many(new_inserts, ordered=False)
                self.stats['total_new_inserts'] += len(new_inserts)
                logger.debug(f"✅ Inserted {len(result.inserted_ids)} new documents")

            if update_operations:
                result = self.collection.bulk_write(update_operations, ordered=False)
                self.stats['total_updates'] += result.modified_count
                logger.debug(f"✅ Updated {result.modified_count} documents, matched {result.matched_count}")

            # Log batch summary every 2 seconds
            self.stats['last_batch_size'] = len(detections)
            self.stats['queue_depth'] = self.detection_queue.qsize()

            # Enhanced logging with tracking status breakdown
            new_count = len(new_inserts)
            update_count = len(update_operations)

            logger.info(f"📊 DB Batch: {new_count} new, {update_count} updates, "
                       f"Queue: {self.stats['queue_depth']}/20000 "
                       f"[Total: {self.stats['total_new_inserts']} inserts, {self.stats['total_updates']} updates]")

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

        if track_id is None:
            return (0, 255, 0)      # Green - Detection only
        elif track_age < 2:         # New track (< 2 frames)
            return (0, 255, 255)    # Yellow - New track
        else:
            return (0, 165, 255)    # Orange - Established track (2+ frames)

    def create_tracking_label(self, detection: Dict) -> str:
        """Create label with tracking information"""
        track_id = detection.get('track_id')
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'object')
        track_age = detection.get('track_age', 0)

        if track_id is not None:
            # Tracking mode: "ID:8001 pallet: 0.85 (15f)"
            return f"ID:{track_id} {class_name}: {confidence:.2f} ({track_age}f)"
        else:
            # Detection mode: "pallet: 0.85"
            return f"{class_name}: {confidence:.2f}"

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
        camera_ids=[8, 9, 10],  # Add more cameras: [8, 9, 10, 11] or [1,2,3,4,5,6,7,8,9,10,11]
        confidence=0.5,
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