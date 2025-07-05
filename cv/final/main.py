#!/usr/bin/env python3
"""
CPU-Based Complete Warehouse Tracking System (Copy of GPU script)
Modified to use CPU-based detection like combined filtering script
Uses same detection method as combined_filtering_detection.py
1) Detection (CPU - post_process_grounded_object_detection)
2) Area + Grid Cell Filtering (CPU)  
3) Physical Coordinate Translation (CPU)
4) CPU SIFT Feature Matching
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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from warehouse_database_handler import WarehouseDatabaseHandler

# Import utility modules (unused imports removed during modularization)

# Import feature database module
from modules.feature_database import CPUGlobalFeatureDatabase

# Import filtering module
from modules.filtering import DetectionFiltering

# Import color extractor module
from modules.color_extractor import ObjectColorExtractor

# Import coordinate mapper module - CRITICAL COMPONENT
from modules.coordinate_mapper import CoordinateMapper

# Import detector module - MOST CRITICAL COMPONENT
from modules.detector import CPUSimplePalletDetector

# Import GUI display module
from modules.gui_display import CPUDisplayManager

# Import frame processor module
from modules.frame_processor import CPUFrameProcessor

# Import camera manager module
from modules.camera_manager import CPUCameraManager

# Import multi-camera system module
from modules.multi_camera_system import MultiCameraCPUSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Removed custom fisheye corrector - using imported OptimizedFisheyeCorrector

















class CPUCompleteWarehouseTracker:
    """CPU-based complete warehouse tracking system (same logic as combined filtering)"""

    def __init__(self, camera_id: int = 8):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()

        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            camera_name = self.camera_zone.camera_name
            rtsp_url = self.camera_zone.rtsp_url
        else:
            camera_name = f"Camera {camera_id}"
            rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

        # Initialize camera manager
        self.camera_manager = CPUCameraManager(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            camera_name=camera_name
        )

        # Backward compatibility properties
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url

        # CPU-based detection components (same as combined filtering)
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = CPUSimplePalletDetector()

        # CPU coordinate mapping
        self.coordinate_mapper = CoordinateMapper(camera_id=camera_id)
        self.coordinate_mapper_initialized = False
        self._initialize_coordinate_mapper()

        # CPU global feature database with camera-specific ID ranges
        self.global_db = CPUGlobalFeatureDatabase(f"cpu_camera_{camera_id}_global_features.pkl", camera_id)

        # Initialize color extraction module
        self.color_extractor = ObjectColorExtractor()

        # Test color extraction with a simple red image
        test_image = np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8)  # Red image
        test_color = self.color_extractor.extract_dominant_color(test_image)
        logger.info(f"🧪 Color extractor test: {test_color}")

        # Database handler for MongoDB integration (same as GPU script)
        self.db_handler = WarehouseDatabaseHandler(
            mongodb_url="mongodb://localhost:27017/",
            database_name="warehouse_tracking",
            collection_name="detections",
            batch_save_size=10,
            enable_mongodb=True
        )

        # Detection parameters (same as combined filtering)
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]

        # Initialize detection filtering module
        self.filtering = DetectionFiltering(
            camera_id=self.camera_id,
            min_area=10000,
            max_area=100000,
            max_physical_size_ft=15.0,
            cell_size=40
        )

        # Initialize GUI display manager
        self.display_manager = CPUDisplayManager(
            camera_name=self.camera_name,
            camera_id=self.camera_id
        )

        # Initialize frame processor
        self.frame_processor = CPUFrameProcessor(camera_id=self.camera_id)

        # Inject all components into frame processor
        self.frame_processor.inject_components(
            fisheye_corrector=self.fisheye_corrector,
            pallet_detector=self.pallet_detector,
            filtering=self.filtering,
            coordinate_mapper=self.coordinate_mapper,
            coordinate_mapper_initialized=self.coordinate_mapper_initialized,
            global_db=self.global_db,
            color_extractor=self.color_extractor,
            db_handler=self.db_handler,
            display_manager=self.display_manager
        )

        # Add backward compatibility properties for camera access
        self.cap = None
        self.connected = False
        self.running = False

        # Frame processing settings
        self.FRAME_SKIP = 20  # Process every 20th frame for real-time performance

        # Performance tracking
        self.frame_count = 0
        self.total_detections = 0

        logger.info(f"Hybrid warehouse tracker initialized for {self.camera_name}")
        logger.info(f"🚀 Detection: GPU-accelerated Grounding DINO")
        logger.info(f"🔧 Processing: CPU-based SIFT, Coordinates, Database")

    def _initialize_coordinate_mapper(self):
        """Initialize CPU coordinate mapper"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                self.coordinate_mapper_initialized = True
                logger.info(f"✅ CPU coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ CPU coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize CPU coordinate mapper for {self.camera_name}: {e}")
            self.coordinate_mapper_initialized = False

    def connect_camera(self) -> bool:
        """Connect to the camera using camera manager"""
        success = self.camera_manager.connect_camera()

        # Update backward compatibility properties
        self.connected = self.camera_manager.connected
        self.running = self.camera_manager.running
        self.cap = self.camera_manager.cap

        return success



    def _get_latest_frame_cpu(self):
        """Get the latest frame using camera manager"""
        return self.camera_manager.get_latest_frame()





    def start_detection(self):
        """Start CPU-based detection"""
        logger.info("=== CPU-BASED WAREHOUSE TRACKING ===")
        logger.info("🚀 All operations CPU-based for compatibility")
        logger.info("Pipeline: CPU Detection → CPU Filtering → CPU Coords → CPU SIFT")
        logger.info("Press 'q' or ESC to quit")
        logger.info("=" * 60)

        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True
        frame_count = 0
        processed_frame_count = 0

        try:
            while self.running:
                frame_count += 1
                self.frame_count = frame_count

                # Frame skipping logic - process every FRAME_SKIP frames
                if frame_count % self.FRAME_SKIP != 0:
                    # Skip this frame - just read and discard
                    ret, frame = self.camera_manager.read_frame()
                    if not ret:
                        logger.warning("Failed to read frame")
                        continue
                    continue

                # Read frame normally (frame skipping handles the timing)
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue

                processed_frame_count += 1

                # Process frame using frame processor
                processed_frame = self.frame_processor.process_frame(frame)

                # Display result
                cv2.imshow(f"CPU Tracking - {self.camera_name}", processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")

                # Log progress every 10 processed frames (every 200 total frames with skip=20)
                if processed_frame_count % 10 == 0:
                    logger.info(f"Processed {processed_frame_count} frames (skipped {frame_count - processed_frame_count}): {len(self.frame_processor.final_tracked_detections)} objects tracked")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False

        # Cleanup camera manager
        if hasattr(self, 'camera_manager'):
            self.camera_manager.cleanup_camera()

        # Backward compatibility - cleanup cap if it exists
        if self.cap:
            self.cap.release()

        # Cleanup database handler (same as GPU script)
        if hasattr(self, 'db_handler'):
            self.db_handler.cleanup()

        cv2.destroyAllWindows()
        logger.info("CPU tracking system shutdown complete")



# ============================================================================
# CONFIGURATION
# ============================================================================

# 📹 SIMPLE CAMERA CONFIGURATION - EDIT THESE LISTS:
# =======================================================

# 🎯 DETECTION CAMERAS: Add camera numbers you want to run detection on
ACTIVE_CAMERAS = [1,2,3,4,5,6,7,8,9,10,11]  # Cameras that will detect objects

# 🖥️ GUI CAMERAS: Add camera numbers you want to see windows for
GUI_CAMERAS = [1,2,3,4,5,6,7,8,9,10,11]  # Cameras that will show GUI windows (subset of ACTIVE_CAMERAS)

# 🎛️ GUI CONFIGURATION
ENABLE_GUI = True  # Set to False for headless mode
ENABLE_CONSOLE_LOGGING = True  # Print logs to console

print(f"🔥 CPU RUNNING CAMERAS: {ACTIVE_CAMERAS}")
print(f"🖥️ GUI WINDOWS FOR: {GUI_CAMERAS if ENABLE_GUI else 'NONE (HEADLESS)'}")

def main():
    """Main function for CPU-based 11-camera warehouse tracking"""
    print("🚀 CPU-BASED 11-CAMERA WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("CONFIGURATION:")
    print(f"📹 Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    print(f"🖥️ GUI Cameras: {GUI_CAMERAS if ENABLE_GUI else 'DISABLED'} ({len(GUI_CAMERAS) if ENABLE_GUI else 0} windows)")
    print(f"🎛️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    print("=" * 80)
    print("CPU-BASED PROCESSING - All operations CPU-based:")
    print("1) 🚀 CPU Detection (Grounding DINO + post_process_grounded_object_detection)")
    print("2) 🚀 CPU Area + Grid Cell Filtering")
    print("3) 🚀 CPU Physical Coordinate Translation")
    print("4) 🚀 CPU SIFT Feature Matching")
    print("5) 🚀 CPU Persistent Object IDs")
    print("6) 🚀 CPU Cross-Frame Tracking & Database")
    print("=" * 80)
    if ENABLE_GUI:
        print("\nGUI Mode:")
        print("- Green: New objects")
        print("- Orange: CPU-tracked existing objects")
        print("- Red: Failed tracking")
        print("- Cyan: Physical coordinate labels")
        print("- Press 'q' to quit")
        print("- Press 'n'/'p' to change detection prompts")
    else:
        print("\nHEADLESS Mode: No GUI windows, console logging only")
    print("=" * 80)

    # Initialize multi-camera system
    multi_camera_system = MultiCameraCPUSystem(
        active_cameras=ACTIVE_CAMERAS,
        gui_cameras=GUI_CAMERAS,
        enable_gui=ENABLE_GUI
    )

    # Set the tracker class for camera initialization
    multi_camera_system.set_tracker_class(CPUCompleteWarehouseTracker)

    # Run the system
    multi_camera_system.run()

if __name__ == "__main__":
    main()
