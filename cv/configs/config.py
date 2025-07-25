"""
Comprehensive Configuration Settings for Warehouse Tracking System
All adjustable parameters organized by category for easy tuning
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Try multiple possible locations for .env file
possible_env_paths = [
    # Current working directory
    '.env',
    # Project root (two levels up from this config file)
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'),
    # One level up from cv directory
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

# Fallback: try to load from any .env file found
if not env_loaded:
    load_dotenv()  # This will look for .env in current directory and parent directories

class Config:
    # ==================== CAMERA SETTINGS ====================
    # Hardcoded single camera setup
    CAMERA_1_ID = 1  # ZED camera ID (hardcoded)
    CAMERA_2_ID = None  # No second camera

    # Single camera mode
    SINGLE_CAMERA_MODE = True
    USE_ZED_LEFT_VIEW_ONLY = True
    
    # ZED Camera resolution and FPS
    # Note: ZED cameras output stereo images (side-by-side)
    # Full stereo frame: 1344x376, Left view: 672x376
    FRAME_WIDTH = 1344  # Full stereo width
    FRAME_HEIGHT = 376  # Height
    CAMERA_FPS = 30
    
    # Use only left view from each stereo camera
    USE_STEREO_LEFT_ONLY = True
    EFFECTIVE_WIDTH = 672  # Width of left view (FRAME_WIDTH // 2)
    EFFECTIVE_HEIGHT = 376  # Same as FRAME_HEIGHT
    
    # ==================== STITCHING SETTINGS ====================
    # Overlap configuration
    OVERLAP_ENABLED = True
    OVERLAP_PERCENTAGE = 0.2  # 20% overlap between cameras
    STITCH_MODE = "side_by_side"  # Options: "side_by_side", "blend", "overlap"
    
    # Calibration settings
    AUTO_CALIBRATE = True
    CALIBRATION_FRAMES = 50  # Number of frames for auto-calibration
    
    # ==================== DETECTION SETTINGS ====================
    # Model configuration
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    DETECTION_PROMPT = "box. cardboard box. package."

    # 🎯 DETECTION CONFIDENCE TUNING (Key for intermittent detection)
    CONFIDENCE_THRESHOLD = 0.20        # Main detection threshold - Optimized for box detection
    BOX_THRESHOLD = 0.20               # Same as confidence threshold
    TEXT_THRESHOLD = 0.20              # Text matching threshold

    # 📉 LOWER THESE FOR MORE SENSITIVE DETECTION:
    # CONFIDENCE_THRESHOLD = 0.15      # More sensitive (more detections, some false positives)
    # CONFIDENCE_THRESHOLD = 0.12      # Very sensitive (catches weak detections)
    # CONFIDENCE_THRESHOLD = 0.25      # Less sensitive (fewer false positives, may miss objects)

    # Alternative prompts for pallet detection:
    # DETECTION_PROMPT = "pallet. wooden pallet. plastic pallet. shipping pallet."  # More specific
    # DETECTION_PROMPT = "pallet. skid."  # Simpler
    # DETECTION_PROMPT = "pallet. wooden pallet. shipping skid. warehouse pallet."  # Broader

    # GPU settings
    FORCE_GPU = True
    GPU_MEMORY_FRACTION = 0.8

    # 🚀 STRATEGY 1: SEQUENTIAL PROCESSING FOR 11 CAMERAS ON 6GB GPU
    DETECTION_RESOLUTION = (1920, 1080)  # 1080p for good quality/performance balance
    SKIP_DETECTION_FRAMES = 2  # Process every 3rd frame
    USE_MIXED_PRECISION = True  # FP16 for 2x memory efficiency
    AGGRESSIVE_MEMORY_CLEANUP = True  # Clear cache after each camera

    # Sequential processing settings
    SEQUENTIAL_CAMERA_PROCESSING = True  # Process cameras one at a time
    CAMERA_PROCESSING_BATCH_SIZE = 1  # Process 1 camera at a time
    GPU_MEMORY_BUFFER = 1.5  # Reserve 1.5GB for safety

    # Advanced GPU optimization
    ENABLE_GPU_MEMORY_FRACTION = True
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory
    TORCH_CUDNN_BENCHMARK = True  # Optimize for consistent input sizes

    # GPU debugging (will be logged at startup)
    @classmethod
    def log_gpu_status(cls):
        """Log GPU status for debugging"""
        import torch
        import logging
        logger = logging.getLogger(__name__)

        logger.info("🔍 GPU STATUS CHECK:")
        logger.info(f"   PyTorch Version: {torch.__version__}")
        logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")

        if torch.cuda.is_available():
            logger.info(f"   GPU Count: {torch.cuda.device_count()}")
            logger.info(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU Memory: {memory_total:.1f}GB")
        else:
            logger.warning("   ⚠️ CUDA NOT AVAILABLE - Will use CPU")
            if torch.version.cuda is None:
                logger.error("   ❌ PyTorch compiled without CUDA support!")
                logger.error("   💡 Run: python gpu_diagnostics.py for detailed analysis")

    # Performance optimization
    DETECTION_BATCH_SIZE = 1
    MODEL_CACHE_FRAMES = 100  # Clear GPU cache every N frames
    
    # ==================== SIFT TRACKING SETTINGS ====================
    # SIFT detector parameters (affects feature detection quality)
    SIFT_N_FEATURES = 500              # Max features per object (higher = more accurate, slower)
    SIFT_N_OCTAVE_LAYERS = 3           # Scale space layers (3-4 recommended)
    SIFT_CONTRAST_THRESHOLD = 0.04     # Feature contrast threshold (lower = more features)
    SIFT_EDGE_THRESHOLD = 10           # Edge response threshold (higher = fewer edge features)
    SIFT_SIGMA = 1.6                   # Gaussian blur sigma (1.6 is standard)

    # FLANN matcher parameters (affects matching speed)
    FLANN_TREES = 5                    # Number of trees for FLANN index
    FLANN_CHECKS = 50                  # Number of checks for FLANN search

    # 🏃 MATCHING PARAMETERS (Critical for moving objects) - UPDATED to match working stitched version
    MIN_MATCH_COUNT = 10               # Minimum matches required for valid tracking
    GOOD_MATCH_RATIO = 0.7             # Lowe's ratio test threshold (more lenient like stitched version)
    MATCH_SCORE_THRESHOLD = 0.2        # Minimum match score for object association (higher like stitched version)

    # 📦 FOR MOVING BOXES - LOWER THESE VALUES:
    # MIN_MATCH_COUNT = 6              # Fewer matches required (better for moving objects)
    # GOOD_MATCH_RATIO = 0.7           # More lenient matching (accepts more matches)
    # MATCH_SCORE_THRESHOLD = 0.05     # Lower threshold (accepts weaker matches)

    # Object lifecycle parameters
    MAX_DISAPPEARED_FRAMES = 30        # Frames before object is considered lost
    MAX_MATCH_HISTORY = 10             # Number of recent match scores to keep

    # Visual feature extraction
    BOX_PADDING = 5                    # Pixels to pad around detected boxes
    HISTOGRAM_EQUALIZATION = True      # Apply histogram equalization for better contrast

    # Cross-camera tracking
    ENABLE_CROSS_CAMERA_TRACKING = True
    CROSS_CAMERA_MATCH_THRESHOLD = 0.6
    
    # ==================== DATABASE SETTINGS ====================
    # Database mode configuration (load from environment with fallback)
    USE_LOCAL_DATABASE = os.getenv('USE_LOCAL_DATABASE', 'false').lower() == 'true'

    # MongoDB connection strings (load from environment with fallbacks)
    LOCAL_MONGO_URI = os.getenv('MONGODB_LOCAL_URI', "mongodb://localhost:27017/")
    ONLINE_MONGO_URI = os.getenv('MONGODB_ONLINE_URI', "mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/WARP?retryWrites=true&w=majority")

    # Current MongoDB connection (will be set based on USE_LOCAL_DATABASE)
    MONGO_URI = LOCAL_MONGO_URI if USE_LOCAL_DATABASE else ONLINE_MONGO_URI

    # Database and collection names (load from environment with fallbacks)
    DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', "WARP")
    COLLECTION_NAME = os.getenv('MONGODB_COLLECTION_NAME', "detections")
    
    # Database behavior
    AUTO_CREATE_INDEXES = True
    CONNECTION_TIMEOUT = 5000  # milliseconds

    # 💾 DATABASE FILTERING (Critical for preventing low-confidence object storage)
    FILTER_DATABASE_BY_CONFIDENCE = True   # Only store objects above confidence threshold
    DATABASE_MIN_CONFIDENCE = None         # Use CONFIDENCE_THRESHOLD if None

    # Data retention
    CLEANUP_OLD_DATA_HOURS = 24
    MAX_TRACKING_HISTORY = 1000
    
    # ==================== COORDINATE MAPPING SETTINGS ====================
    # Warehouse physical dimensions (meters)
    WAREHOUSE_FLOOR_WIDTH = 10.0   # meters (warehouse width)
    WAREHOUSE_FLOOR_LENGTH = 8.0   # meters (warehouse length)

    # Calibration file for coordinate mapping
    COORDINATE_CALIBRATION_FILE = "warehouse_calibration.json"

    # ==================== GRID MAPPING SETTINGS ====================
    # Warehouse grid configuration
    GRID_ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    GRID_COLUMNS = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Camera zone mapping
    CAMERA_ZONES = {
        1: {'rows': ['A', 'B', 'C', 'D'], 'cols': [1, 2, 3, 4]},  # Top-left
        2: {'rows': ['A', 'B', 'C', 'D'], 'cols': [5, 6, 7, 8]},  # Top-right
        3: {'rows': ['E', 'F', 'G', 'H'], 'cols': [1, 2, 3, 4]},  # Bottom-left
        4: {'rows': ['E', 'F', 'G', 'H'], 'cols': [5, 6, 7, 8]}   # Bottom-right
    }
    
    # Dual camera zone mapping (for 2 cameras covering all zones)
    DUAL_CAMERA_ZONES = {
        1: [1, 3],  # Camera 1 covers zones 1 and 3 (left side)
        2: [2, 4]   # Camera 2 covers zones 2 and 4 (right side)
    }
    
    # ==================== PERFORMANCE SETTINGS ====================
    # Frame processing
    FRAME_BUFFER_SIZE = 30
    PROCESSING_THREADS = 2
    
    # Performance monitoring
    FPS_CALCULATION_FRAMES = 30
    PERFORMANCE_LOG_INTERVAL = 100  # frames
    
    # Memory management
    ENABLE_MEMORY_OPTIMIZATION = True
    GARBAGE_COLLECT_INTERVAL = 1000  # frames
    
    # ==================== DISPLAY SETTINGS ====================
    # Object visualization
    SHOW_BOUNDING_BOXES = True
    SHOW_OBJECT_IDS = True
    SHOW_MATCH_SCORES = True
    SHOW_DETECTION_CONFIDENCE = True   # Show detection confidence for tuning
    SHOW_REAL_COORDINATES = True
    SHOW_GRID_OVERLAY = False  # Disabled grid overlay (per user request)

    # 🎯 DEBUG VISUALIZATION (Show ALL detections including filtered ones)
    SHOW_ALL_DETECTIONS = True         # Show both accepted and filtered detections
    SHOW_FILTERED_DETECTIONS = True    # Show low-confidence detections in red
    SHOW_DATABASE_STORED_ONLY = False  # If True, only show objects stored in database

    # ID assignment debugging
    SHOW_ID_ASSIGNMENT_DEBUG = True    # Show why new IDs are created
    SHOW_TRACKING_STATS = True         # Show tracking statistics

    # Calibrated zone overlay
    SHOW_CALIBRATED_ZONE = False  # Disabled per user request (no grid system)
    SHOW_COORDINATE_GRID = False  # Disabled per user request (no grid system)
    SHOW_CALIBRATION_INFO = False  # Black info panel disabled

    # Object age thresholds (seconds)
    NEW_OBJECT_THRESHOLD = 5           # Objects newer than this are "new"
    ESTABLISHED_OBJECT_THRESHOLD = 60  # Objects older than this are "established"

    # Color coding for object ages (BGR format)
    COLOR_NEW_OBJECT = (0, 255, 255)      # Yellow - New objects
    COLOR_TRACKING_OBJECT = (255, 255, 0)  # Cyan - Tracking objects
    COLOR_ESTABLISHED_OBJECT = (0, 255, 0) # Green - Established objects

    # 🎯 DEBUG COLORS (for showing all detections)
    COLOR_HIGH_CONFIDENCE = (0, 255, 0)    # Green - Above threshold (stored in DB)
    COLOR_LOW_CONFIDENCE = (0, 0, 255)     # Red - Below threshold (filtered)
    COLOR_DATABASE_STORED = (0, 255, 0)    # Green - Actually stored in database
    COLOR_DATABASE_FILTERED = (0, 0, 255)  # Red - Filtered from database

    # Bounding box and marker settings
    BBOX_THICKNESS = 2                 # Bounding box line thickness
    CENTER_DOT_RADIUS = 8              # Center dot radius
    CENTER_DOT_COLOR = (0, 0, 255)     # Center dot color (red)
    CENTER_BORDER_RADIUS = 10          # Center dot border radius
    CENTER_BORDER_COLOR = (255, 255, 255)  # Center dot border color (white)

    # Text and label settings
    LABEL_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX (imported in detector_tracker.py)
    LABEL_FONT_SCALE = 0.5
    LABEL_THICKNESS = 2
    REAL_COORD_FONT_SCALE = 1.2  # Increased from 0.8 for better visibility
    REAL_COORD_THICKNESS = 2  # Increased from 1 for better visibility
    REAL_COORD_COLOR = (255, 255, 0)   # Yellow for real coordinates

    # Zone overlay settings
    ZONE_BOUNDARY_COLOR = (0, 255, 255)    # Yellow zone boundary
    ZONE_BOUNDARY_THICKNESS = 3
    ZONE_FILL_ALPHA = 0.05                 # Zone fill transparency
    ZONE_CORNER_RADIUS = 8                 # Corner marker radius
    ZONE_CORNER_BORDER = 10                # Corner marker border

    # Grid settings
    GRID_COLOR = (100, 100, 100)          # Gray grid lines
    GRID_THICKNESS = 1
    GRID_LABEL_FONT_SCALE = 0.3
    GRID_LABEL_COLOR = (150, 150, 150)

    # Dynamic grid spacing thresholds
    SMALL_WAREHOUSE_THRESHOLD = 3.0       # Use 0.5m grid if max dimension <= 3m
    MEDIUM_WAREHOUSE_THRESHOLD = 6.0      # Use 1.0m grid if max dimension <= 6m
    GRID_SPACING_SMALL = 0.5              # Grid spacing for small warehouses
    GRID_SPACING_MEDIUM = 1.0             # Grid spacing for medium warehouses
    GRID_SPACING_LARGE = 2.0              # Grid spacing for large warehouses

    # Info overlay
    SHOW_INFO_OVERLAY = True
    INFO_OVERLAY_HEIGHT = 180
    
    # ==================== TRAINING DATA COLLECTION SETTINGS ====================
    # Training data collection paths
    TRAINING_DATA_ROOT = "training/data"
    RAW_COLLECTION_DIR = "training/data/raw_collection"
    PROCESSED_TRAINING_DIR = "training/data/processed"

    # Training data quality settings
    RAW_IMAGE_QUALITY = 95      # JPEG quality for raw 4K images
    CORRECTED_IMAGE_QUALITY = 90 # JPEG quality for corrected 2K images

    # Collection timing
    COLLECTION_FRAME_SKIP = 6000  # Frames to skip between saves (5 min at 20fps)

    # ==================== LOGGING SETTINGS ====================
    # Log levels
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE = True
    LOG_FILE_PATH = "warehouse_tracking.log"
    
    # Console output
    VERBOSE_OUTPUT = True
    SHOW_DETECTION_STATS = True
    SHOW_TRACKING_STATS = True
    
    # ==================== ADVANCED TUNING PARAMETERS ====================
    # Detection fine-tuning
    DETECTION_RETRY_COUNT = 1          # Retry detection if no objects found
    MIN_BOX_AREA = 100                 # Minimum bounding box area (pixels²)
    MAX_BOX_AREA = 50000               # Maximum bounding box area (pixels²)
    MIN_BOX_WIDTH = 10                 # Minimum bounding box width
    MIN_BOX_HEIGHT = 10                # Minimum bounding box height

    # Tracking fine-tuning
    FEATURE_EXTRACTION_TIMEOUT = 1.0   # Seconds to timeout feature extraction
    MATCH_CALCULATION_TIMEOUT = 0.5    # Seconds to timeout match calculation
    TRACKING_CONFIDENCE_DECAY = 0.95   # Confidence decay per frame for lost objects

    # Memory management
    MAX_DETECTION_HISTORY = 30         # Frames of detection history to keep
    MAX_TRACKING_HISTORY_PER_OBJECT = 100  # Max tracking points per object
    CLEANUP_INTERVAL_FRAMES = 1000     # Frames between memory cleanup

    # Performance optimization
    SKIP_DETECTION_FRAMES = 0          # Skip detection every N frames (0 = no skip)
    PARALLEL_PROCESSING = True         # Enable parallel processing where possible
    BATCH_PROCESSING_SIZE = 1          # Batch size for processing multiple detections

    # ==================== ALERT SETTINGS ====================
    # Notification thresholds
    ALERT_ON_NEW_OBJECT = True
    ALERT_ON_LOST_OBJECT = True
    ALERT_ON_LONG_TRACKING = True      # Objects tracked > 1 hour
    LONG_TRACKING_THRESHOLD = 3600     # Seconds (1 hour)

    # Performance alerts
    ALERT_LOW_FPS_THRESHOLD = 10
    ALERT_HIGH_GPU_USAGE = 90          # percentage
    ALERT_HIGH_MEMORY_USAGE = 85       # percentage
    
    # ==================== MULTI-CAMERA RTSP SETTINGS ====================
    # Full 11-camera warehouse configuration - NOW MANAGED BY warehouse_config.py
    ENABLE_MULTI_CAMERA_SYSTEM = True

    # Import warehouse configuration
    try:
        from configs.warehouse_config import get_warehouse_config, get_active_cameras
        _warehouse_config = get_warehouse_config()
        ACTIVE_CAMERAS = get_active_cameras()  # Dynamic from warehouse config
    except ImportError:
        # Fallback for Phase 1
        ACTIVE_CAMERAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # ALL CAMERAS
    
    # ==================== RTSP URL CONFIGURATION ====================

    # Flag to switch between LOCAL and REMOTE camera URLs
    USE_LOCAL_CAMERAS = False  # Set to True for local URLs (192.168.x.x), False for remote URLs (104.181.138.5)

    # LOCAL RTSP Camera URLs (192.168.x.x network) - UPDATED FROM ACTUAL CAMERA LIST
    LOCAL_RTSP_CAMERA_URLS = {
        # Updated with actual camera IPs from camera management system
        1: "rtsp://admin:wearewarp!@192.168.0.81:8000/Streaming/channels/1",   # Cam 1 Front
        2: "rtsp://admin:wearewarp!@192.168.0.85:8000/Streaming/channels/1",   # Cam 2 Front
        3: "rtsp://admin:wearewarp!@192.168.0.83:8000/Streaming/channels/1",   # Cam 3 Front
        4: "rtsp://admin:wearewarp!@192.168.0.84:8000/Streaming/channels/1",   # Cam 4 Front

        # Row 2 (Middle) - 3 cameras
        5: "rtsp://admin:wearewarp!@192.168.0.77:8000/Streaming/channels/1",   # Cam 5 Mid
        6: "rtsp://admin:wearewarp!@192.168.0.106:8000/Streaming/channels/1",  # Cam 6 Mid
        7: "rtsp://admin:wearewarp!@192.168.0.78:8000/Streaming/channels/1",   # Cam 7 Mid

        # Row 3 (Back) - 4 cameras
        8: "rtsp://admin:wearewarp!@192.168.0.79:8000/Streaming/channels/1",   # Cam 8 Back
        9: "rtsp://admin:wearewarp!@192.168.0.80:8000/Streaming/channels/1",   # Cam 9 Back
        10: "rtsp://admin:wearewarp!@192.168.0.82:8000/Streaming/channels/1",  # Cam 10 Back
        11: "rtsp://admin:wearewarp!@192.168.0.64:8000/Streaming/channels/1"   # Cam 11 Back
    }

    # REMOTE RTSP Camera URLs (104.181.138.5 network)
    REMOTE_RTSP_CAMERA_URLS = {
        # Row 1 (Front) - 4 cameras
        1: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/101",  # Cam 1 REMOTE
        2: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/201",  # Cam 2 REMOTE
        3: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/301",  # Cam 3 REMOTE
        4: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/401",  # Cam 4 REMOTE

        # Row 2 (Middle) - 3 cameras
        5: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/501",  # Cam 5 REMOTE
        6: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/601",  # Cam 6 REMOTE
        7: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/701",  # Cam 7 REMOTE

        # Row 3 (Back) - 4 cameras
        8: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/801",  # Cam 8 REMOTE
        9: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/901",  # Cam 9 REMOTE
        10: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/1001", # Cam 10 REMOTE
        11: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/1101"  # Cam 11 REMOTE
    }

    # Active RTSP URLs (automatically selected based on USE_LOCAL_CAMERAS flag)
    RTSP_CAMERA_URLS = LOCAL_RTSP_CAMERA_URLS if USE_LOCAL_CAMERAS else REMOTE_RTSP_CAMERA_URLS

    @classmethod
    def get_camera_url(cls, camera_id: int) -> str:
        """Get RTSP URL for specific camera based on current configuration"""
        urls = cls.LOCAL_RTSP_CAMERA_URLS if cls.USE_LOCAL_CAMERAS else cls.REMOTE_RTSP_CAMERA_URLS
        return urls.get(camera_id, "")

    @classmethod
    def switch_to_local_cameras(cls):
        """Switch to local camera URLs (192.168.x.x)"""
        cls.USE_LOCAL_CAMERAS = True
        cls.RTSP_CAMERA_URLS = cls.LOCAL_RTSP_CAMERA_URLS
        print("🏠 Switched to LOCAL camera URLs (192.168.x.x)")

    @classmethod
    def switch_to_remote_cameras(cls):
        """Switch to remote camera URLs (104.181.138.5)"""
        cls.USE_LOCAL_CAMERAS = False
        cls.RTSP_CAMERA_URLS = cls.REMOTE_RTSP_CAMERA_URLS
        print("🌐 Switched to REMOTE camera URLs (104.181.138.5)")

    @classmethod
    def switch_to_local_database(cls):
        """Switch to local MongoDB database (localhost:27017)"""
        cls.USE_LOCAL_DATABASE = True
        cls.MONGO_URI = cls.LOCAL_MONGO_URI
        print("🏠 Switched to LOCAL database (localhost:27017)")

    @classmethod
    def switch_to_online_database(cls):
        """Switch to online MongoDB database (MongoDB Atlas)"""
        cls.USE_LOCAL_DATABASE = False
        cls.MONGO_URI = cls.ONLINE_MONGO_URI
        print("🌐 Switched to ONLINE database (MongoDB Atlas)")

    @classmethod
    def get_camera_info(cls, camera_id: int) -> dict:
        """Get camera information including current URL"""
        return {
            'camera_id': camera_id,
            'current_url': cls.get_camera_url(camera_id),
            'local_url': cls.LOCAL_RTSP_CAMERA_URLS.get(camera_id, ""),
            'remote_url': cls.REMOTE_RTSP_CAMERA_URLS.get(camera_id, ""),
            'using_local': cls.USE_LOCAL_CAMERAS
        }

    @classmethod
    def get_database_info(cls) -> dict:
        """Get database information including current connection"""
        return {
            'current_uri': cls.MONGO_URI,
            'local_uri': cls.LOCAL_MONGO_URI,
            'online_uri': cls.ONLINE_MONGO_URI,
            'using_local': cls.USE_LOCAL_DATABASE,
            'database_name': cls.DATABASE_NAME,
            'collection_name': cls.COLLECTION_NAME
        }
    
    # Camera names for identification - NEW LAYOUT (3-Column, Origin Top-Right)
    CAMERA_NAMES = {
        # Column 1 (Rightmost) - Cameras 1-4 (top to bottom)
        1: "Camera 1 - Column 1 Top",
        2: "Camera 2 - Column 1 Mid-Top", 
        3: "Camera 3 - Column 1 Mid-Bottom",
        4: "Camera 4 - Column 1 Bottom",
        
        # Column 2 (Middle) - Cameras 5-7 + Office (top to bottom)
        5: "Camera 5 - Column 2 Top",
        6: "Camera 6 - Column 2 Mid-Top",
        7: "Camera 7 - Column 2 Mid-Bottom",
        
        # Column 3 (Leftmost) - Cameras 8-11 (top to bottom)
        8: "Camera 8 - Column 3 Top",  # Currently active for testing
        9: "Camera 9 - Column 3 Mid-Top",
        10: "Camera 10 - Column 3 Mid-Bottom",
        11: "Camera 11 - Column 3 Bottom"
    }
    
    # Full warehouse dimensions (from user diagram)
    FULL_WAREHOUSE_WIDTH_FT = 180.0   # Total warehouse width in feet
    FULL_WAREHOUSE_LENGTH_FT = 90.0   # Total warehouse length in feet
    FULL_WAREHOUSE_WIDTH_M = 180.0 * 0.3048   # Convert to meters (54.864m)
    FULL_WAREHOUSE_LENGTH_M = 90.0 * 0.3048   # Convert to meters (27.432m)
    
    # Camera coverage zones - NEW COORDINATE SYSTEM (Origin: Top-Right, 3-Column Layout)
    # Warehouse: 180ft x 90ft, Origin (0,0) at top-right, (180,90) at bottom-left
    # Column widths: 60ft each, Camera zones: 22.5ft height each
    CAMERA_COVERAGE_ZONES = {
        # Column 1 (Rightmost, x: 0-60ft) - Cameras 1-4 (top to bottom)
        1: {"x_start": 0, "x_end": 60, "y_start": 0, "y_end": 22.5, "center_x": 30, "center_y": 11.25, "column": 1},
        2: {"x_start": 0, "x_end": 60, "y_start": 22.5, "y_end": 45, "center_x": 30, "center_y": 33.75, "column": 1},
        3: {"x_start": 0, "x_end": 60, "y_start": 45, "y_end": 67.5, "center_x": 30, "center_y": 56.25, "column": 1},
        4: {"x_start": 0, "x_end": 60, "y_start": 67.5, "y_end": 90, "center_x": 30, "center_y": 78.75, "column": 1},
        
        # Column 2 (Middle, x: 60-120ft) - Cameras 5-7 + Office (top to bottom)
        5: {"x_start": 60, "x_end": 120, "y_start": 0, "y_end": 22.5, "center_x": 90, "center_y": 11.25, "column": 2},
        6: {"x_start": 60, "x_end": 120, "y_start": 22.5, "y_end": 45, "center_x": 90, "center_y": 33.75, "column": 2},
        7: {"x_start": 60, "x_end": 120, "y_start": 45, "y_end": 67.5, "center_x": 90, "center_y": 56.25, "column": 2},
        "office": {"x_start": 60, "x_end": 120, "y_start": 67.5, "y_end": 90, "center_x": 90, "center_y": 78.75, "column": 2, "trackable": False},
        
        # Column 3 (Leftmost, x: 120-180ft) - Camera 8 (top-left area)
        8: {"x_start": 120, "x_end": 180, "y_start": 0, "y_end": 25, "center_x": 150, "center_y": 12.5, "column": 3},  # FIXED: Camera 8 in top-left area
        9: {"x_start": 120, "x_end": 180, "y_start": 25, "y_end": 50, "center_x": 150, "center_y": 37.5, "column": 3},
        10: {"x_start": 120, "x_end": 180, "y_start": 50, "y_end": 75, "center_x": 150, "center_y": 62.5, "column": 3},
        11: {"x_start": 120, "x_end": 180, "y_start": 75, "y_end": 90, "center_x": 150, "center_y": 82.5, "column": 3}
    }
    
    # RTSP connection settings
    RTSP_USERNAME = "admin"
    RTSP_PASSWORD = "wearewarp!"
    RTSP_PORT = 554
    RTSP_TIMEOUT = 10
    RTSP_BUFFER_SIZE = 1
    MAX_RECONNECTION_ATTEMPTS = 5
    RECONNECTION_DELAY = 5
    
    # Fisheye lens settings (2.8mm)
    FISHEYE_LENS_MM = 2.8
    FISHEYE_CORRECTION_ENABLED = True
    
    # Frame processing for RTSP cameras
    RTSP_FRAME_WIDTH = 3840   # 4K resolution from cameras
    RTSP_FRAME_HEIGHT = 2160
    RTSP_PROCESSING_WIDTH = 1920  # Scale down for processing
    RTSP_PROCESSING_HEIGHT = 1080
    RTSP_TARGET_FPS = 20

    @classmethod
    def get_camera_zone(cls, row: str, col: int) -> int:
        """Get camera zone for given grid position"""
        row_index = ord(row.upper()) - ord('A')
        col_index = col - 1
        
        if row_index < 4 and col_index < 4:
            return 1
        elif row_index < 4 and col_index >= 4:
            return 2
        elif row_index >= 4 and col_index < 4:
            return 3
        else:
            return 4
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Validate camera IDs
        if cls.CAMERA_1_ID == cls.CAMERA_2_ID:
            errors.append("Camera IDs must be different")
        
        # Validate thresholds
        if not 0 < cls.CONFIDENCE_THRESHOLD < 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if not 0 < cls.OVERLAP_PERCENTAGE < 1:
            errors.append("Overlap percentage must be between 0 and 1")
        
        # Validate SIFT parameters
        if cls.SIFT_N_FEATURES < 100:
            errors.append("SIFT features should be at least 100")
        
        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
        
        return True

    @classmethod
    def get_tuning_guide(cls):
        """Get parameter tuning guide"""
        return """
        🎯 PARAMETER TUNING GUIDE
        ========================

        📊 DETECTION TUNING:
        • CONFIDENCE_THRESHOLD (0.1-0.5): Lower = more detections, higher = fewer false positives
        • DETECTION_PROMPT: Customize for your objects ("box", "package", "container")
        • BOX_THRESHOLD: Same as confidence, affects detection sensitivity

        🔍 SIFT TRACKING TUNING:
        • SIFT_N_FEATURES (100-1000): More features = better tracking, slower processing
        • MIN_MATCH_COUNT (5-20): Higher = more reliable tracking, may lose objects
        • GOOD_MATCH_RATIO (0.5-0.8): Lower = stricter matching, fewer false matches
        • MATCH_SCORE_THRESHOLD (0.05-0.3): Higher = more confident matches only

        ⏱️ PERFORMANCE TUNING:
        • MODEL_CACHE_FRAMES (50-200): Lower = more memory cleanup, may affect performance
        • MAX_DISAPPEARED_FRAMES (10-60): Higher = objects persist longer when occluded
        • FRAME_BUFFER_SIZE (10-50): Larger buffer = smoother but more memory

        🎨 VISUALIZATION TUNING:
        • NEW_OBJECT_THRESHOLD (1-10): Seconds before object is no longer "new"
        • ESTABLISHED_OBJECT_THRESHOLD (30-120): Seconds before object is "established"
        • GRID_SPACING_*: Adjust grid density for your warehouse size

        💾 DATABASE TUNING:
        • CONNECTION_TIMEOUT (1000-10000): Higher for slower networks
        • CLEANUP_OLD_DATA_HOURS (1-168): How long to keep tracking data
        • MAX_TRACKING_HISTORY (100-5000): Objects to keep in memory

        🚨 TROUBLESHOOTING:
        • Too many false positives → Increase CONFIDENCE_THRESHOLD
        • Missing objects → Decrease CONFIDENCE_THRESHOLD, increase SIFT_N_FEATURES
        • Objects losing tracking → Decrease MIN_MATCH_COUNT, increase MAX_DISAPPEARED_FRAMES
        • Slow performance → Decrease SIFT_N_FEATURES, increase MODEL_CACHE_FRAMES
        • Memory issues → Decrease FRAME_BUFFER_SIZE, MAX_TRACKING_HISTORY
        """

    @classmethod
    def get_preset_configs(cls):
        """Get preset configurations for different scenarios"""
        return {
            "high_accuracy": {
                "CONFIDENCE_THRESHOLD": 0.35,
                "SIFT_N_FEATURES": 800,
                "MIN_MATCH_COUNT": 15,
                "GOOD_MATCH_RATIO": 0.6,
                "MATCH_SCORE_THRESHOLD": 0.15
            },
            "high_performance": {
                "CONFIDENCE_THRESHOLD": 0.25,
                "SIFT_N_FEATURES": 300,
                "MIN_MATCH_COUNT": 8,
                "GOOD_MATCH_RATIO": 0.7,
                "MODEL_CACHE_FRAMES": 50
            },
            "balanced": {
                "CONFIDENCE_THRESHOLD": 0.20,
                "SIFT_N_FEATURES": 500,
                "MIN_MATCH_COUNT": 10,
                "GOOD_MATCH_RATIO": 0.5,
                "MATCH_SCORE_THRESHOLD": 0.1
            },
            "small_objects": {
                "CONFIDENCE_THRESHOLD": 0.15,
                "MIN_BOX_AREA": 50,
                "SIFT_N_FEATURES": 600,
                "SIFT_CONTRAST_THRESHOLD": 0.03
            },
            "moving_objects": {
                "CONFIDENCE_THRESHOLD": 0.15,      # Lower detection threshold
                "MAX_DISAPPEARED_FRAMES": 15,      # Shorter persistence
                "MIN_MATCH_COUNT": 6,              # Fewer matches required
                "GOOD_MATCH_RATIO": 0.7,           # More lenient matching
                "MATCH_SCORE_THRESHOLD": 0.05,     # Accept weaker matches
                "SIFT_N_FEATURES": 600             # More features for better tracking
            },
            "sensitive_detection": {
                "CONFIDENCE_THRESHOLD": 0.12,      # Very sensitive detection
                "BOX_THRESHOLD": 0.12,
                "TEXT_THRESHOLD": 0.12,
                "MIN_BOX_AREA": 50,                # Detect smaller objects
                "SIFT_CONTRAST_THRESHOLD": 0.03    # More features
            }
        }

# Validate configuration on import
Config.validate_config()