#!/usr/bin/env python3
"""
FIXED PARALLEL CPU-BASED COMPLETE WAREHOUSE TRACKING SYSTEM
Fixed GUI display and camera capture issues
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from dataclasses import dataclass
from collections import deque

# sklearn import - will fallback to simple mean if not available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available, using simple color extraction")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations (keeping all existing imports)
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from warehouse_database_handler import WarehouseDatabaseHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# THREAD-SAFE PERFORMANCE COUNTERS
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for performance monitoring"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def get_value(self):
        with self._lock:
            return self._value
    
    def add(self, value):
        with self._lock:
            self._value += value

@dataclass
class FrameData:
    """Data structure for frame processing pipeline"""
    camera_id: int
    frame_number: int
    timestamp: float
    raw_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    detections: List[Dict] = None

# ============================================================================
# SIMPLIFIED PARALLEL SYSTEM (FIXING ISSUES)
# ============================================================================

class SimplifiedParallelTracker:
    """Simplified parallel tracker that fixes GUI and camera issues"""
    
    def __init__(self, active_cameras: List[int], enable_gui: bool = True):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.running = False
        
        # Get warehouse configuration
        self.warehouse_config = get_warehouse_config()
        
        # Build camera configurations
        self.camera_configs = {}
        for cam_id in active_cameras:
            if str(cam_id) in self.warehouse_config.camera_zones:
                camera_zone = self.warehouse_config.camera_zones[str(cam_id)]
                self.camera_configs[cam_id] = camera_zone.rtsp_url
            else:
                self.camera_configs[cam_id] = Config.RTSP_CAMERA_URLS.get(cam_id, "")
        
        # Initialize components for each camera
        self.camera_components = {}
        self.camera_captures = {}
        self.latest_results = {}
        
        # Performance stats
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'fps_start_time': time.time()
        }
        
        # Thread management
        self.camera_threads = {}
        self.processing_threads = {}
        
        logger.info(f"🔧 Simplified Parallel Tracker initialized for cameras: {active_cameras}")
    
    def _initialize_camera_components(self, camera_id: int):
        """Initialize components for a single camera"""
        try:
            components = {}
            
            # Fisheye corrector
            components['fisheye_corrector'] = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
            
            # Pallet detector
            components['pallet_detector'] = self._create_pallet_detector()
            
            # Coordinate mapper
            components['coordinate_mapper'] = self._create_coordinate_mapper(camera_id)
            
            # Global feature database
            components['global_db'] = self._create_global_db(camera_id)
            
            # Color extractor
            components['color_extractor'] = self._create_color_extractor()
            
            # Database handler
            components['db_handler'] = self._create_db_handler()
            
            self.camera_components[camera_id] = components
            logger.info(f"✅ Components initialized for Camera {camera_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components for Camera {camera_id}: {e}")
            return False
    
    def _create_pallet_detector(self):
        """Create pallet detector"""
        class CPUSimplePalletDetector:
            def __init__(self):
                self.confidence_threshold = 0.08
                self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
                self.current_prompt_index = 0
                self.current_prompt = self.sample_prompts[0]
                
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"🚀 Using GPU for Grounding DINO: {torch.cuda.get_device_name()}")
                else:
                    self.device = torch.device("cpu")
                    logger.info("⚠️ GPU not available, using CPU for Grounding DINO")
                
                self._initialize_grounding_dino()
            
            def _initialize_grounding_dino(self):
                try:
                    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                    
                    model_id = "IDEA-Research/grounding-dino-tiny"
                    self.processor = AutoProcessor.from_pretrained(model_id)
                    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    
                    logger.info(f"✅ Grounding DINO-Tiny loaded on {self.device}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Grounding DINO: {e}")
                    self.processor = None
                    self.model = None
            
            def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
                if self.model is None or self.processor is None:
                    return []
                
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    inputs = self.processor(
                        images=pil_image,
                        text=self.current_prompt,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    if self.device.type == 'cuda':
                        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                            outputs = self.model(**inputs)
                    else:
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                    
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        box_threshold=self.confidence_threshold,
                        text_threshold=self.confidence_threshold,
                        target_sizes=[pil_image.size[::-1]]
                    )
                    
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    detections = []
                    if results and len(results) > 0:
                        boxes = results[0]["boxes"].cpu().numpy()
                        scores = results[0]["scores"].cpu().numpy()
                        
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = map(int, box)
                            area = (x2 - x1) * (y2 - y1)
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(score),
                                'area': area,
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            }
                            detections.append(detection)
                    
                    return detections
                    
                except Exception as e:
                    logger.error(f"Detection failed: {e}")
                    return []
        
        return CPUSimplePalletDetector()
    
    def _create_coordinate_mapper(self, camera_id: int):
        """Create coordinate mapper"""
        class CoordinateMapper:
            def __init__(self, camera_id):
                self.camera_id = camera_id
                self.is_calibrated = False
                self.homography_matrix = None
                self.load_calibration()
            
            def load_calibration(self):
                try:
                    filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
                    with open(filename, 'r') as file:
                        calibration_data = json.load(file)
                    
                    image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
                    real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
                    
                    if len(image_corners) == 4 and len(real_world_corners) == 4:
                        self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
                        self.is_calibrated = True
                        logger.info(f"✅ Coordinate calibration loaded for Camera {self.camera_id}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Calibration failed for Camera {self.camera_id}: {e}")
                    self.is_calibrated = False
            
            def pixel_to_real(self, pixel_x, pixel_y):
                if not self.is_calibrated:
                    return None, None
                
                try:
                    points = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
                    transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
                    result = transformed_points.reshape(-1, 2)
                    if len(result) > 0:
                        return float(result[0][0]), float(result[0][1])
                    return None, None
                except Exception as e:
                    logger.error(f"Coordinate transformation failed: {e}")
                    return None, None
        
        return CoordinateMapper(camera_id)
    
    def _create_global_db(self, camera_id: int):
        """Create simplified global database"""
        class SimpleGlobalDB:
            def __init__(self, camera_id):
                self.camera_id = camera_id
                self.next_id = camera_id * 1000 + 1
                self.objects = {}
            
            def assign_id(self):
                new_id = self.next_id
                self.next_id += 1
                return new_id
        
        return SimpleGlobalDB(camera_id)
    
    def _create_color_extractor(self):
        """Create color extractor"""
        class SimpleColorExtractor:
            def extract_dominant_color(self, image_region):
                if image_region is None or image_region.size == 0:
                    return {'rgb': [128, 128, 128], 'hex': '#808080', 'color_name': 'gray'}
                
                try:
                    # Simple mean color
                    mean_color = np.mean(image_region.reshape(-1, 3), axis=0)
                    mean_color = np.clip(mean_color, 0, 255).astype(int)
                    
                    return {
                        'rgb': [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])],  # BGR to RGB
                        'hex': f"#{mean_color[2]:02x}{mean_color[1]:02x}{mean_color[0]:02x}",
                        'color_name': 'detected_color'
                    }
                except Exception as e:
                    return {'rgb': [128, 128, 128], 'hex': '#808080', 'color_name': 'gray'}
        
        return SimpleColorExtractor()
    
    def _create_db_handler(self):
        """Create database handler"""
        try:
            return WarehouseDatabaseHandler(
                mongodb_url="mongodb://localhost:27017/",
                database_name="warehouse_tracking",
                collection_name="detections",
                batch_save_size=10,
                enable_mongodb=True
            )
        except Exception as e:
            logger.warning(f"Database handler creation failed: {e}")
            return None
    
    def _connect_camera(self, camera_id: int) -> bool:
        """Connect to a single camera"""
        rtsp_url = self.camera_configs.get(camera_id, "")
        if not rtsp_url:
            logger.error(f"No RTSP URL for camera {camera_id}")
            return False
        
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # Increased timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}: {rtsp_url}")
                return False
            
            # Test frame capture
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from camera {camera_id}")
                cap.release()
                return False
            
            self.camera_captures[camera_id] = cap
            logger.info(f"✅ Camera {camera_id} connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to camera {camera_id}: {e}")
            return False
    
    def _camera_processing_thread(self, camera_id: int):
        """Processing thread for a single camera"""
        components = self.camera_components.get(camera_id)
        cap = self.camera_captures.get(camera_id)
        
        if not components or not cap:
            logger.error(f"Missing components or capture for camera {camera_id}")
            return
        
        frame_count = 0
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Apply fisheye correction
                processed_frame = components['fisheye_corrector'].correct(frame)
                
                # Resize if needed
                height, width = processed_frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                
                # Detect pallets
                detections = components['pallet_detector'].detect_pallets(processed_frame)
                
                # Process detections
                processed_detections = []
                for detection in detections:
                    # Add physical coordinates
                    center = detection['center']
                    physical_x, physical_y = components['coordinate_mapper'].pixel_to_real(center[0], center[1])
                    
                    # Add color information
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    image_region = processed_frame[y1:y2, x1:x2]
                    color_info = components['color_extractor'].extract_dominant_color(image_region)
                    
                    # Add global ID
                    global_id = components['global_db'].assign_id()
                    
                    # Combine all information
                    detection.update({
                        'physical_x_ft': physical_x,
                        'physical_y_ft': physical_y,
                        'global_id': global_id,
                        'tracking_status': 'new',
                        **color_info
                    })
                    
                    processed_detections.append(detection)
                
                # Save to database
                if components['db_handler']:
                    for detection in processed_detections:
                        try:
                            components['db_handler'].save_detection_to_db(camera_id, detection)
                        except Exception as e:
                            logger.error(f"Database save failed: {e}")
                
                # Store results for GUI display
                self.latest_results[camera_id] = {
                    'frame': processed_frame,
                    'detections': processed_detections,
                    'frame_count': frame_count
                }
                
                # Update performance stats
                self.performance_stats['total_frames'] += 1
                self.performance_stats['total_detections'] += len(processed_detections)
                
                # Control frame rate
                time.sleep(0.05)  # ~20 FPS max per camera
                
            except Exception as e:
                logger.error(f"Error in camera {camera_id} processing: {e}")
                time.sleep(1)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict], camera_id: int, frame_count: int) -> np.ndarray:
        """Draw detections on frame"""
        result_frame = frame.copy()
        
        # Draw info overlay
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)
        
        # Draw system info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        cv2.putText(result_frame, f"🚀 PARALLEL TRACKING - Camera {camera_id}", (20, y_offset), font, 0.5, (0, 255, 0), 2)
        
        y_offset += 20
        cv2.putText(result_frame, f"Frame: {frame_count}", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(result_frame, f"Detections: {len(detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        elapsed = time.time() - self.performance_stats['fps_start_time']
        if elapsed > 0:
            total_fps = self.performance_stats['total_frames'] / elapsed
            cv2.putText(result_frame, f"Total FPS: {total_fps:.1f}", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(result_frame, f"Total Detections: {self.performance_stats['total_detections']}", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center = detection['center']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result_frame, center, 5, (0, 255, 0), -1)
            
            # Draw labels
            global_id = detection.get('global_id', -1)
            label = f"ID:{global_id}"
            cv2.putText(result_frame, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
            
            # Draw confidence
            conf_label = f"Conf:{confidence:.2f}"
            cv2.putText(result_frame, conf_label, (x1, y2 + 20), font, 0.4, (255, 255, 255), 1)
            
            # Draw physical coordinates
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')
            if physical_x is not None and physical_y is not None:
                coord_label = f"({physical_x:.1f}, {physical_y:.1f})ft"
                cv2.putText(result_frame, coord_label, (x1, y2 + 40), font, 0.4, (0, 255, 255), 1)
        
        return result_frame
    
    def start(self):
        """Start the tracking system"""
        logger.info("🚀 Starting Simplified Parallel Tracking System")
        
        # Initialize components for each camera
        for camera_id in self.active_cameras:
            if not self._initialize_camera_components(camera_id):
                logger.error(f"Failed to initialize camera {camera_id}")
                continue
            
            if not self._connect_camera(camera_id):
                logger.error(f"Failed to connect camera {camera_id}")
                continue
        
        if not self.camera_captures:
            logger.error("No cameras connected!")
            return
        
        # Start processing threads
        self.running = True
        for camera_id in self.camera_captures.keys():
            thread = threading.Thread(target=self._camera_processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.camera_threads[camera_id] = thread
            logger.info(f"🔄 Started processing thread for camera {camera_id}")
        
        # Start GUI if enabled
        if self.enable_gui:
            self._run_gui()
    
    def _run_gui(self):
        """Run GUI display"""
        logger.info("🖥️ Starting GUI display...")
        
        # Performance monitoring
        last_stats_time = time.time()
        
        while self.running:
            try:
                # Display frames
                for camera_id in self.active_cameras:
                    if camera_id in self.latest_results:
                        result = self.latest_results[camera_id]
                        frame = result['frame']
                        detections = result['detections']
                        frame_count = result['frame_count']
                        
                        # Draw detections
                        display_frame = self._draw_detections(frame, detections, camera_id, frame_count)
                        
                        # Show frame
                        cv2.imshow(f"Parallel Camera {camera_id}", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Performance logging
                current_time = time.time()
                if current_time - last_stats_time >= 10:  # Every 10 seconds
                    elapsed = current_time - self.performance_stats['fps_start_time']
                    if elapsed > 0:
                        total_fps = self.performance_stats['total_frames'] / elapsed
                        logger.info(f"📊 Performance: {total_fps:.1f} FPS total, {self.performance_stats['total_detections']} detections")
                    last_stats_time = current_time
                
            except Exception as e:
                logger.error(f"GUI error: {e}")
                break
        
        self.stop()
    
    def stop(self):
        """Stop the tracking system"""
        logger.info("🛑 Stopping Simplified Parallel Tracking System")
        
        self.running = False
        
        # Close camera captures
        for cap in self.camera_captures.values():
            cap.release()
        
        # Close GUI windows
        cv2.destroyAllWindows()
        
        logger.info("✅ System stopped")

# ============================================================================
# CONFIGURATION (PRESERVED FROM ORIGINAL)
# ============================================================================

# 📹 CAMERA CONFIGURATION
ALL_CAMERAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All available cameras

# Support for camera group splitting
CAMERA_GROUP = os.getenv('CAMERA_GROUP', 'SINGLE8')  # Changed default to SINGLE8 for testing

if CAMERA_GROUP == 'GROUP1':
    ACTIVE_CAMERAS = [1, 2, 3, 4, 5, 6]
    print("🔥 RUNNING CAMERA GROUP 1: Cameras 1-6")
elif CAMERA_GROUP == 'GROUP2':
    ACTIVE_CAMERAS = [7, 8, 9, 10, 11]
    print("🔥 RUNNING CAMERA GROUP 2: Cameras 7-11")
elif CAMERA_GROUP == 'SINGLE8':
    ACTIVE_CAMERAS = [8]
    print("🔥 RUNNING SINGLE CAMERA 8 FOR TESTING")
elif CAMERA_GROUP.startswith('SINGLE'):
    try:
        cam_num = int(CAMERA_GROUP.replace('SINGLE', ''))
        if 1 <= cam_num <= 11:
            ACTIVE_CAMERAS = [cam_num]
            print(f"🔥 RUNNING SINGLE CAMERA {cam_num} FOR TESTING")
        else:
            print(f"❌ Invalid camera number: {cam_num}. Using Camera 8.")
            ACTIVE_CAMERAS = [8]
    except ValueError:
        print("❌ Invalid CAMERA_GROUP format. Using Camera 8.")
        ACTIVE_CAMERAS = [8]
else:
    ACTIVE_CAMERAS = [8]
    print("🔥 RUNNING DEFAULT CAMERA: 8")

# 🖥️ GUI CONFIGURATION
ENABLE_GUI = True

def main():
    """Main function for simplified parallel warehouse tracking"""
    print("🚀 SIMPLIFIED PARALLEL WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("CONFIGURATION:")
    print(f"📹 Active Cameras: {ACTIVE_CAMERAS}")
    print(f"🖥️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'DISABLED'}")
    print("=" * 80)
    print("FEATURES:")
    print("1) 🚀 Parallel camera processing")
    print("2) 🚀 Grounding DINO Tiny detection")
    print("3) 🚀 Real-time GUI display")
    print("4) 🚀 Database integration")
    print("5) 🚀 Color extraction")
    print("6) 🚀 Physical coordinate mapping")
    print("=" * 80)
    if ENABLE_GUI:
        print("GUI Controls:")
        print("- Press 'q' to quit")
        print("- Green boxes: Detected objects")
        print("- Cyan text: Physical coordinates")
    print("=" * 80)

    # Initialize simplified parallel tracker
    tracker = SimplifiedParallelTracker(
        active_cameras=ACTIVE_CAMERAS,
        enable_gui=ENABLE_GUI
    )

    try:
        # Start the system
        tracker.start()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        tracker.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        tracker.stop()

if __name__ == "__main__":
    main()