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
from configs.config import Config
import logging
import json

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class CoordinateMapper:
    """Maps pixel coordinates to real-world coordinates using homography"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        """Initialize coordinate mapper with floor dimensions in FEET"""
        self.floor_width_ft = floor_width  # Floor width in feet
        self.floor_length_ft = floor_length  # Floor length in feet
        self.camera_id = camera_id  # Add camera ID for offset calculation
        self.homography_matrix = None
        self.is_calibrated = False
        
        # Column 3 cameras (8,9,10,11) use direct global coordinate mapping
        # Calibration files map directly to warehouse coordinates - no offset transformation needed
        # This will be expanded for other columns when they are activated
        
        logger.info(f"Coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")
            logger.info(f"Camera {camera_id}: Using direct global coordinate mapping")

    def load_calibration(self, filename="configs/warehouse_calibration.json"):
        """Load calibration from JSON file"""
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            # Extract warehouse dimensions from real_world_corners for global coordinate mapping
            # Calculate actual coverage area from the real_world_corners
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)

            # Calculate width and height from corners (assuming rectangular area)
            # Corners are: top-left, top-right, bottom-right, bottom-left
            width_ft = abs(real_world_corners[1][0] - real_world_corners[0][0])  # top-right X - top-left X
            height_ft = abs(real_world_corners[2][1] - real_world_corners[1][1])  # bottom-right Y - top-right Y

            self.floor_width_ft = width_ft
            self.floor_length_ft = height_ft
            
            # Extract image corners
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            
            # Check units and convert if necessary
            units = calibration_data.get('calibration_info', {}).get('units', 'feet')
            if units == 'meters':
                # Convert meters to feet
                real_world_corners = real_world_corners * 3.28084
                logger.info("Converted real-world corners from meters to feet")
            
            # Calculate homography from image points to real-world points
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            self.is_calibrated = True
            
            logger.info(f"Coordinate calibration loaded from: {filename}")
            logger.info(f"Camera local area: {self.floor_width_ft:.1f}ft x {self.floor_length_ft:.1f}ft")
            
            # Log the coordinate transformation for Column 3 cameras
            if self.camera_id in [8, 9, 10, 11]:
                zone_info = Config.CAMERA_COVERAGE_ZONES.get(self.camera_id, {})
                x_range = f"{zone_info.get('x_start', 0)}-{zone_info.get('x_end', 0)}ft"
                y_range = f"{zone_info.get('y_start', 0)}-{zone_info.get('y_end', 0)}ft"
                logger.info(f"Camera {self.camera_id} coordinate mapping: Direct mapping to global ({x_range}, {y_range})")
            
            logger.info(f"Coordinate mapper initialized - Calibrated: {self.is_calibrated}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def set_floor_rectangle(self, image_corners):
        """Set floor rectangle corners for homography calculation (DEPRECATED - use load_calibration)"""
        logger.warning("set_floor_rectangle is deprecated, use load_calibration instead")

    def pixel_to_real(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates in FEET with warehouse offset"""
        if not self.is_calibrated or self.homography_matrix is None:
            return None, None
        
        try:
            # Apply homography transformation
            pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
            
            # Extract global coordinates (in feet) - calibration files now use direct global mapping
            global_x = float(real_point[0][0][0])
            global_y = float(real_point[0][0][1])

            # For Column 3 cameras (8,9,10,11), coordinates are already global
            # No offset transformation needed since calibration files map directly to global coordinates
            logger.debug(f"Camera {self.camera_id}: Pixel ({pixel_x}, {pixel_y}) → Global ({global_x:.1f}ft, {global_y:.1f}ft)")

            return global_x, global_y
        
        except Exception as e:
            logger.error(f"Error in pixel_to_real conversion: {e}")
            return None, None

    def real_to_pixel(self, real_x, real_y):
        """Convert real-world coordinates (in feet) to pixel coordinates"""
        if not self.is_calibrated or self.homography_matrix is None:
            return None, None
        
        try:
            # For Column 3 cameras (8,9,10,11), coordinates are already global
            # Apply inverse homography transformation directly (no offset removal needed)
            real_point = np.array([[[real_x, real_y]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(real_point, np.linalg.inv(self.homography_matrix))
            
            pixel_x = int(pixel_point[0][0][0])
            pixel_y = int(pixel_point[0][0][1])
            
            return pixel_x, pixel_y
        
        except Exception as e:
            logger.error(f"Error in real_to_pixel conversion: {e}")
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
        logger.info(f"🔍 Loading detection model on {self.device}...")
        logger.info(f"🔍 Model ID: {self.model_id}")

        try:
            # Load processor
            logger.info("🔍 Loading AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info("✅ AutoProcessor loaded successfully")

            # Load model with detailed logging
            logger.info("🔍 Loading AutoModelForZeroShotObjectDetection...")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            logger.info("✅ Model loaded from pretrained")

            # Move model to device with verification
            logger.info(f"🔍 Moving model to device: {self.device}")
            self.model = self.model.to(self.device)

            # Verify model device placement
            model_device = next(self.model.parameters()).device
            logger.info(f"🔍 Model device after .to(): {model_device}")

            # Fix device comparison (cuda and cuda:0 are the same)
            if (str(model_device).startswith('cuda') and self.device.startswith('cuda')) or str(model_device) == self.device:
                logger.info(f"✅ Model successfully placed on {model_device}")
            else:
                logger.error(f"❌ MODEL DEVICE MISMATCH! Expected: {self.device}, Got: {model_device}")
                logger.error("💡 This indicates a GPU allocation problem")

            # Check model memory usage if on GPU
            if self.device == "cuda":
                memory_after_model = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"🔍 GPU Memory after model loading: {memory_after_model:.2f}GB")

                # Estimate model size
                model_params = sum(p.numel() for p in self.model.parameters())
                model_size_gb = model_params * 4 / 1024**3  # Assuming float32
                logger.info(f"🔍 Estimated model size: {model_size_gb:.2f}GB ({model_params:,} parameters)")

            logger.info("✅ Detection model loaded and placed successfully")

            # 🚀 IMMEDIATE GPU INFERENCE TEST
            if self.device == "cuda":
                logger.info("🔍 Testing GPU inference with dummy data...")
                try:
                    # Create dummy image and text
                    dummy_image = Image.new('RGB', (640, 480), color='red')
                    dummy_inputs = self.processor(images=dummy_image, text="test", return_tensors="pt")
                    dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}

                    # Run inference
                    with torch.no_grad():
                        if self.use_mixed_precision:
                            with torch.amp.autocast('cuda', dtype=torch.float16):
                                dummy_outputs = self.model(**dummy_inputs)
                        else:
                            with torch.amp.autocast('cuda'):
                                dummy_outputs = self.model(**dummy_inputs)

                    # Check GPU memory after test
                    memory_after_test = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"✅ GPU INFERENCE TEST SUCCESSFUL! Memory used: {memory_after_test:.2f}GB")
                    logger.info("🚀 GPU IS ACTIVELY PROCESSING - System ready for camera feeds")

                    # Cleanup
                    del dummy_inputs, dummy_outputs
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"❌ GPU inference test failed: {e}")
                    logger.error("💡 GPU may not be working properly for inference")

        except Exception as e:
            logger.error(f"❌ Failed to load detection model: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            logger.error(f"🔍 Error details: {str(e)}")

            # Additional GPU-specific error handling
            if self.device == "cuda" and "CUDA" in str(e):
                logger.error("💡 CUDA-related error detected. Possible causes:")
                logger.error("   1. Insufficient GPU memory")
                logger.error("   2. CUDA driver/runtime mismatch")
                logger.error("   3. Model too large for GPU")
                logger.error("💡 Try setting smaller GPU_MEMORY_FRACTION in config")

            raise
        
        if self.device == "cuda":
            self.model.eval()
            torch.backends.cudnn.benchmark = True
        
        # Detection parameters
        self.prompt = Config.DETECTION_PROMPT
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

        # 🚀 Performance optimization settings
        self.detection_resolution = Config.DETECTION_RESOLUTION
        self.skip_detection_frames = Config.SKIP_DETECTION_FRAMES
        self.frame_skip_counter = 0
        self.use_mixed_precision = Config.USE_MIXED_PRECISION
        self.aggressive_memory_cleanup = Config.AGGRESSIVE_MEMORY_CLEANUP

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
        
    def set_camera_id(self, camera_id: int):
        """Set camera ID for coordinate mapping and load coverage zone info"""
        self.coordinate_mapper.camera_id = camera_id
        
        # Set camera coverage zone info from Config
        if camera_id in Config.CAMERA_COVERAGE_ZONES:
            zone_config = Config.CAMERA_COVERAGE_ZONES[camera_id]
            self.coordinate_mapper.camera_coverage_zone = {
                'x_start': zone_config['x_start'],
                'x_end': zone_config['x_end'], 
                'y_start': zone_config['y_start'],
                'y_end': zone_config['y_end']
            }
            logger.info(f"Camera {camera_id} coverage zone set: {self.coordinate_mapper.camera_coverage_zone}")
        else:
            self.coordinate_mapper.camera_coverage_zone = None
            
        # Load calibration after setting camera ID
        calibration_file = f"configs/warehouse_calibration_camera_{camera_id}.json"
        self.coordinate_mapper.load_calibration(calibration_file)

    def setup_gpu(self, force_gpu: bool = True):
        """Setup GPU configuration with comprehensive debugging"""
        logger.info("🔍 GPU DEBUGGING - Starting comprehensive GPU diagnostics...")

        # 1. Basic CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"🔍 CUDA Available: {cuda_available}")

        if cuda_available:
            # 2. CUDA device information
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(0)

            logger.info(f"🔍 CUDA Device Count: {device_count}")
            logger.info(f"🔍 Current CUDA Device: {current_device}")
            logger.info(f"🔍 Device Name: {device_name}")

            # 3. Memory information
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

            logger.info(f"🔍 GPU Memory - Allocated: {memory_allocated:.2f}GB")
            logger.info(f"🔍 GPU Memory - Reserved: {memory_reserved:.2f}GB")
            logger.info(f"🔍 GPU Memory - Total: {memory_total:.2f}GB")

            # 4. CUDA version information
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"🔍 CUDA Version: {cuda_version}")
            logger.info(f"🔍 cuDNN Version: {cudnn_version}")

            # 5. PyTorch build info
            logger.info(f"🔍 PyTorch Version: {torch.__version__}")
            logger.info(f"🔍 PyTorch CUDA Compiled: {torch.version.cuda is not None}")

            torch.cuda.empty_cache()
            self.device = "cuda"
            logger.info("✅ Using CUDA GPU")
        else:
            # 6. Detailed CPU fallback reasons
            logger.error("❌ CUDA not available - investigating reasons...")
            logger.error(f"🔍 PyTorch Version: {torch.__version__}")
            logger.error(f"🔍 CUDA Compiled: {torch.version.cuda}")

            # Check if this is a CPU-only PyTorch installation
            if torch.version.cuda is None:
                logger.error("❌ CRITICAL: PyTorch was compiled WITHOUT CUDA support!")
                logger.error("💡 SOLUTION: Install CUDA-enabled PyTorch:")
                logger.error("   pip uninstall torch torchvision")
                logger.error("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

            if force_gpu:
                logger.warning("⚠️ Forced to use CPU despite GPU request")
            self.device = "cpu"

        if self.device == "cuda":
            # 7. Set memory fraction and test allocation
            try:
                torch.cuda.set_per_process_memory_fraction(Config.GPU_MEMORY_FRACTION)
                logger.info(f"🔍 GPU Memory Fraction Set: {Config.GPU_MEMORY_FRACTION}")

                # Test GPU allocation
                test_tensor = torch.randn(1000, 1000).to(self.device)
                logger.info("✅ GPU Test Allocation Successful")
                del test_tensor
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"❌ GPU Setup Error: {e}")
                logger.error("💡 Falling back to CPU due to GPU setup failure")
                self.device = "cpu"
    
    def detect_boxes(self, frame: np.ndarray) -> Dict:
        """Detect boxes in frame using Grounding DINO with GPU debugging and performance optimization"""
        detection_start = time.time()

        # 🚀 FRAME SKIPPING FOR PERFORMANCE
        self.frame_skip_counter += 1
        if self.frame_skip_counter <= self.skip_detection_frames:
            # Skip this frame - return empty detection
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'processing_time': time.time() - detection_start,
                'skipped': True
            }

        # Reset counter - process this frame
        self.frame_skip_counter = 0

        # Debug frame count for periodic GPU monitoring
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1

        # Detailed GPU debugging every 100 frames
        debug_this_frame = (self._debug_frame_count % 100 == 1)

        # 🚀 REAL-TIME GPU USAGE MONITORING (every 10 frames)
        monitor_gpu = (self._debug_frame_count % 10 == 1) and self.device == "cuda"

        try:
            # 🚀 RESOLUTION OPTIMIZATION - Resize frame for detection
            original_height, original_width = frame.shape[:2]
            target_width, target_height = self.detection_resolution

            # Only resize if frame is larger than target resolution
            if original_width > target_width or original_height > target_height:
                # Calculate aspect ratio preserving resize
                scale = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                if debug_this_frame:
                    logger.info(f"🔍 Frame resized: {original_width}x{original_height} → {new_width}x{new_height}")
            else:
                resized_frame = frame
                scale = 1.0

            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt")

            if debug_this_frame:
                logger.info(f"🔍 Frame {self._debug_frame_count} - Input tensor shapes:")
                for k, v in inputs.items():
                    if hasattr(v, 'shape'):
                        logger.info(f"   {k}: {v.shape} on {v.device}")

            # Move inputs to device with verification
            if self.device == "cuda":
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

                if debug_this_frame:
                    logger.info("🔍 After moving to CUDA:")
                    for k, v in inputs.items():
                        if hasattr(v, 'device'):
                            logger.info(f"   {k}: on {v.device}")

                    # Check GPU memory before inference
                    memory_before = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🔍 GPU Memory before inference: {memory_before:.2f}GB")
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Model inference with device verification
            with torch.no_grad():
                if self.device == "cuda":
                    # 🔥 CRITICAL: Check GPU memory before inference
                    memory_available = torch.cuda.get_device_properties(0).total_memory
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_free = memory_available - memory_allocated
                    memory_free_gb = memory_free / 1024**3

                    if memory_free_gb < Config.GPU_MEMORY_BUFFER:
                        logger.warning(f"⚠️ LOW GPU MEMORY: {memory_free_gb:.2f}GB free, clearing cache...")
                        torch.cuda.empty_cache()
                        # Re-check after cleanup
                        memory_allocated = torch.cuda.memory_allocated(0)
                        memory_free = memory_available - memory_allocated
                        memory_free_gb = memory_free / 1024**3
                        logger.info(f"🧹 After cleanup: {memory_free_gb:.2f}GB free")

                    # Verify model is still on GPU
                    model_device = next(self.model.parameters()).device
                    if debug_this_frame:
                        logger.info(f"🔍 Model device at inference: {model_device}")

                    if not str(model_device).startswith("cuda"):
                        logger.error(f"❌ MODEL MOVED TO CPU! Device: {model_device}")
                        logger.error("💡 This indicates GPU memory overflow - CRITICAL ISSUE!")
                        logger.error(f"💡 Available memory: {memory_free_gb:.2f}GB")
                        logger.error("💡 Try reducing ACTIVE_CAMERAS or increasing SKIP_DETECTION_FRAMES")

                        # Try to move model back to GPU
                        try:
                            logger.info("🔄 Attempting to move model back to GPU...")
                            self.model = self.model.to('cuda')
                            torch.cuda.empty_cache()
                            logger.info("✅ Model moved back to GPU")
                        except Exception as e:
                            logger.error(f"❌ Failed to move model back to GPU: {e}")
                            logger.error("💡 System will continue on CPU (slower performance)")
                            self.device = "cpu"  # Switch to CPU mode

                    # 🚀 MIXED PRECISION INFERENCE for 2x memory efficiency
                    # 🚀 MONITOR GPU BEFORE INFERENCE
                    if monitor_gpu:
                        memory_before_inference = torch.cuda.memory_allocated(0) / 1024**3
                        logger.info(f"🚀 FRAME {self._debug_frame_count}: GPU ACTIVE - Memory before inference: {memory_before_inference:.2f}GB")

                    if self.use_mixed_precision:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            outputs = self.model(**inputs)
                    else:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(**inputs)

                    # 🚀 MONITOR GPU AFTER INFERENCE
                    if monitor_gpu:
                        memory_after_inference = torch.cuda.memory_allocated(0) / 1024**3
                        logger.info(f"🚀 FRAME {self._debug_frame_count}: GPU INFERENCE COMPLETE - Memory after: {memory_after_inference:.2f}GB")

                    if debug_this_frame:
                        memory_after = torch.cuda.memory_allocated(0) / 1024**3
                        logger.info(f"🔍 GPU Memory after inference: {memory_after:.2f}GB")
                else:
                    if debug_this_frame:
                        logger.info("🔍 Running inference on CPU")
                    outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs['input_ids'],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )[0]

            # 🚀 STRATEGY 5: AGGRESSIVE GPU MEMORY MANAGEMENT for 11 cameras
            if self.device == "cuda":
                if self.aggressive_memory_cleanup:
                    # Clear cache after every camera processing
                    torch.cuda.empty_cache()
                    # Force garbage collection
                    import gc
                    gc.collect()
                elif self.frame_count % Config.MODEL_CACHE_FRAMES == 0:
                    # Standard cleanup
                    torch.cuda.empty_cache()

                # Log GPU utilization periodically
                if debug_this_frame:
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_utilization = (memory_used / memory_total) * 100
                    logger.info(f"🔍 GPU Utilization: {gpu_utilization:.1f}% ({memory_used:.2f}GB / {memory_total:.1f}GB)")

            # Track detection time
            detection_time = time.time() - detection_start
            self.detection_times.append(detection_time)
            if len(self.detection_times) > Config.FPS_CALCULATION_FRAMES:
                self.detection_times.pop(0)

            # Scale bounding boxes back to original resolution if frame was resized
            if 'scale' in locals() and scale != 1.0:
                for i, box in enumerate(results['boxes']):
                    if len(box) == 4:
                        results['boxes'][i] = [coord / scale for coord in box]

            results['processing_time'] = detection_time
            results['skipped'] = False
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

        # Calculate scaling factor for coordinate conversion
        # Calibration was done on 4K (3840x2160), but detection may be on resized frames
        frame_height, frame_width = frame.shape[:2]
        scale_x = 3840 / frame_width  # Scale factor for X coordinates
        scale_y = 2160 / frame_height  # Scale factor for Y coordinates
        
        # Log scaling info occasionally
        if self.frame_count % 100 == 1:
            logger.info(f"Frame size: {frame_width}x{frame_height}, Scale factors: X={scale_x:.2f}, Y={scale_y:.2f}")

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

                # Scale coordinates back to calibration frame size (4K) for accurate coordinate mapping
                scaled_center_x = center_x * scale_x
                scaled_center_y = center_y * scale_y

                # Get real-world coordinates using scaled coordinates
                real_x, real_y = None, None
                if self.coordinate_mapper.is_calibrated:
                    real_x, real_y = self.coordinate_mapper.pixel_to_real(scaled_center_x, scaled_center_y)

                detection = {
                    'center': (center_x, center_y),  # Keep original frame coordinates for display
                    'real_center': (real_x, real_y) if real_x is not None else None,
                    'bbox': (xmin, ymin, xmax, ymax),  # Keep original frame coordinates for display
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

                cv2.putText(annotated_frame, label, (xmin, ymin-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Real-world coordinates label (if available) - PROMINENT DISPLAY
                real_center = box_info.get('real_center')
                if real_center and real_center[0] is not None:
                    real_x_ft, real_y_ft = real_center
                    
                    # Large, prominent real coordinate display
                    real_label = f"Position: ({real_x_ft:.1f}ft, {real_y_ft:.1f}ft)"
                    cv2.putText(annotated_frame, real_label, (xmin, ymin-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                               
                    # Add background for better visibility
                    text_size = cv2.getTextSize(real_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (xmin-2, ymin-35), (xmin + text_size[0] + 2, ymin-15), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, real_label, (xmin, ymin-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Show pixel coordinates only if real coordinates unavailable
                    pixel_label = f"Pixel: ({center_x}, {center_y}) - No Real Coords"
                    cv2.putText(annotated_frame, pixel_label, (xmin, ymin-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Camera coverage zone info at bottom of screen
        if hasattr(self.coordinate_mapper, 'camera_coverage_zone') and self.coordinate_mapper.camera_coverage_zone:
            zone = self.coordinate_mapper.camera_coverage_zone
            zone_label = f"Camera {self.coordinate_mapper.camera_id} Zone: X({zone['x_start']}-{zone['x_end']}ft) Y({zone['y_start']}-{zone['y_end']}ft)"
            cv2.putText(annotated_frame, zone_label, (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        width = self.coordinate_mapper.floor_width_ft
        length = self.coordinate_mapper.floor_length_ft
        corner_labels = ["(0,0)", f"({width:.0f},0)",
                        f"({width:.0f},{length:.0f})",
                        f"(0,{length:.0f})"]

        for i, (corner, label) in enumerate(zip(corners, corner_labels)):
            x, y = corner

            # Draw corner marker
            cv2.circle(overlay_frame, (x, y), 8, (0, 255, 255), -1)  # Yellow dot
            cv2.circle(overlay_frame, (x, y), 10, (255, 255, 255), 2)  # White border

            # Add coordinate label in FEET
            label_text = f"{label}ft"
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

        # Dynamic grid spacing based on warehouse size - Now in FEET
        max_dimension = max(self.coordinate_mapper.floor_width_ft, self.coordinate_mapper.floor_length_ft)
        if max_dimension <= 10.0:
            grid_spacing = 2.0  # 2ft for small areas
        elif max_dimension <= 30.0:
            grid_spacing = 5.0  # 5ft for medium areas
        else:
            grid_spacing = 10.0  # 10ft for large areas

        # Draw vertical grid lines
        for x_real in np.arange(0, self.coordinate_mapper.floor_width_ft + grid_spacing, grid_spacing):
            if x_real > 0 and x_real < self.coordinate_mapper.floor_width_ft:  # Skip boundaries
                start_pixel = self.coordinate_mapper.real_to_pixel(x_real, 0)
                end_pixel = self.coordinate_mapper.real_to_pixel(x_real, self.coordinate_mapper.floor_length_ft)

                if start_pixel[0] is not None and end_pixel[0] is not None:
                    cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

                    # Add distance label in FEET
                    mid_pixel = self.coordinate_mapper.real_to_pixel(x_real, self.coordinate_mapper.floor_length_ft / 2)
                    if mid_pixel[0] is not None:
                        label = f'{x_real:.0f}ft'
                        cv2.putText(frame, label,
                                   (mid_pixel[0] - 10, mid_pixel[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Draw horizontal grid lines
        for y_real in np.arange(0, self.coordinate_mapper.floor_length_ft + grid_spacing, grid_spacing):
            if y_real > 0 and y_real < self.coordinate_mapper.floor_length_ft:  # Skip boundaries
                start_pixel = self.coordinate_mapper.real_to_pixel(0, y_real)
                end_pixel = self.coordinate_mapper.real_to_pixel(self.coordinate_mapper.floor_width_ft, y_real)

                if start_pixel[0] is not None and end_pixel[0] is not None:
                    cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

                    # Add distance label in FEET
                    mid_pixel = self.coordinate_mapper.real_to_pixel(self.coordinate_mapper.floor_width_ft / 2, y_real)
                    if mid_pixel[0] is not None:
                        label = f'{y_real:.0f}ft'
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

    def draw_coordinate_system_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw coordinate system information overlay"""
        info_frame = frame.copy()
        
        # Coordinate system info panel
        panel_x, panel_y = 10, 10
        panel_width, panel_height = 300, 120
        
        # Draw semi-transparent background
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(info_frame, 0.7, overlay, 0.3, 0, info_frame)
        
        # Draw border
        cv2.rectangle(info_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 2)
        
        # Add title
        cv2.putText(info_frame, "Coordinate System (FEET)", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add coordinate system info
        y_offset = 50
        if self.coordinate_mapper.is_calibrated:
            cv2.putText(info_frame, f"Camera Coverage: {self.coordinate_mapper.floor_width_ft:.1f}ft x {self.coordinate_mapper.floor_length_ft:.1f}ft", 
                       (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            if hasattr(self.coordinate_mapper, 'camera_coverage_zone') and self.coordinate_mapper.camera_coverage_zone:
                zone = self.coordinate_mapper.camera_coverage_zone
                cv2.putText(info_frame, f"Global Zone: X({zone['x_start']}-{zone['x_end']}ft) Y({zone['y_start']}-{zone['y_end']}ft)", 
                           (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
            
            cv2.putText(info_frame, f"Camera ID: {self.coordinate_mapper.camera_id}", 
                       (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(info_frame, "Not Calibrated", (panel_x + 10, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return info_frame

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
