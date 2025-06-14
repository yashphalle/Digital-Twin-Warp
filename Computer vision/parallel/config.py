"""
Configuration settings for the warehouse tracking system
"""

class Config:
    # ==================== CAMERA SETTINGS ====================
    # Camera IDs for dual ZED 2i setup
    CAMERA_1_ID = 1  # Primary camera (left side)
    CAMERA_2_ID = 2  # Secondary camera (right side)
    
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
    CONFIDENCE_THRESHOLD = 0.20
    
    # GPU settings
    FORCE_GPU = True
    GPU_MEMORY_FRACTION = 0.8
    
    # Performance optimization
    DETECTION_BATCH_SIZE = 1
    MODEL_CACHE_FRAMES = 100  # Clear GPU cache every N frames
    
    # ==================== SIFT TRACKING SETTINGS ====================
    # SIFT detector parameters
    SIFT_N_FEATURES = 500
    SIFT_N_OCTAVE_LAYERS = 3
    SIFT_CONTRAST_THRESHOLD = 0.04
    SIFT_EDGE_THRESHOLD = 10
    SIFT_SIGMA = 1.6
    
    # Matching parameters (optimized for moving objects without spatial filtering)
    MIN_MATCH_COUNT = 10  # Increased for more reliable matching without spatial filter
    GOOD_MATCH_RATIO = 0.5 # Slightly more strict ratio test
    MATCH_SCORE_THRESHOLD = 0.1  # Higher threshold for more confident matches
    
    # Tracking parameters
    MAX_DISAPPEARED_FRAMES = 30
    
    # Cross-camera tracking
    ENABLE_CROSS_CAMERA_TRACKING = True
    CROSS_CAMERA_MATCH_THRESHOLD = 0.6
    
    # ==================== DATABASE SETTINGS ====================
    # MongoDB connection
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "warehouse_tracking"
    COLLECTION_NAME = "tracked_objects"
    
    # Database behavior
    AUTO_CREATE_INDEXES = True
    CONNECTION_TIMEOUT = 5000  # milliseconds
    
    # Data retention
    CLEANUP_OLD_DATA_HOURS = 24
    MAX_TRACKING_HISTORY = 1000
    
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
    # Display settings
    SHOW_BOUNDING_BOXES = True
    SHOW_OBJECT_IDS = True
    SHOW_MATCH_SCORES = True
    SHOW_GRID_OVERLAY = False  # Disabled grid overlay
    
    # Color coding for object ages
    COLOR_NEW_OBJECT = (0, 255, 255)      # Yellow - New (< 5 seconds)
    COLOR_TRACKING_OBJECT = (255, 255, 0)  # Cyan - Tracking (5-60 seconds)
    COLOR_ESTABLISHED_OBJECT = (0, 255, 0) # Green - Established (> 60 seconds)
    
    # Info overlay
    SHOW_INFO_OVERLAY = True
    INFO_OVERLAY_HEIGHT = 180
    
    # ==================== LOGGING SETTINGS ====================
    # Log levels
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE = True
    LOG_FILE_PATH = "warehouse_tracking.log"
    
    # Console output
    VERBOSE_OUTPUT = True
    SHOW_DETECTION_STATS = True
    SHOW_TRACKING_STATS = True
    
    # ==================== ALERT SETTINGS ====================
    # Notification thresholds
    ALERT_ON_NEW_OBJECT = True
    ALERT_ON_LOST_OBJECT = True
    ALERT_ON_LONG_TRACKING = True  # Objects tracked > 1 hour
    
    # Performance alerts
    ALERT_LOW_FPS_THRESHOLD = 10
    ALERT_HIGH_GPU_USAGE = 90  # percentage
    
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

# Validate configuration on import
Config.validate_config()