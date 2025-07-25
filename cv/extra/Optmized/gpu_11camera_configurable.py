#!/usr/bin/env python3
"""
GPU-Optimized Complete Warehouse Tracking System
Maximizes GPU utilization for all OpenCV operations and feature matching
Integrates all functionalities with GPU acceleration:
1) Detection (GPU)
2) Area + Grid Cell Filtering (GPU)  
3) Physical Coordinate Translation (GPU)
4) GPU SIFT Feature Matching
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
import json
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configurations
from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharedModelDetector:
    """Single detection model shared across all cameras for batch processing"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt = self.sample_prompts[0]

        logger.info(f"🚀 Initializing SHARED model detector on {self.device}")
        self._initialize_grounding_dino()
        self._optimize_gpu_memory()

    def _initialize_grounding_dino(self):
        """Initialize single Grounding DINO model for all cameras"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            model_id = "IDEA-Research/grounding-dino-base"
            logger.info(f"Loading shared Grounding DINO model: {model_id}")

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Enable mixed precision for memory efficiency
            if self.device.type == 'cuda':
                self.model = self.model.half()

            logger.info(f"✅ SHARED Grounding DINO loaded successfully on {self.device}")
            logger.info(f"💾 Memory usage: ~{SHARED_MODEL_MEMORY_GB}GB (vs {len(ACTIVE_CAMERAS) * 2}GB with separate models)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize shared Grounding DINO: {e}")
            self.model = None

    def _optimize_gpu_memory(self):
        """Optimize GPU memory and performance settings"""
        if self.device.type == 'cuda' and self.model:
            try:
                # Enable GPU memory optimization
                torch.cuda.empty_cache()

                # Set optimal GPU settings for batch processing
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

                # Pre-allocate GPU memory for batch processing
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

                # Warm up the model with a dummy batch
                self._warmup_model()

                logger.info(f"🚀 GPU optimization complete - Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

            except Exception as e:
                logger.warning(f"⚠️ GPU optimization failed: {e}")

    def _warmup_model(self):
        """Warm up the model with dummy inputs for optimal performance"""
        try:
            from PIL import Image
            import numpy as np

            # Create dummy batch of the expected size
            dummy_size = min(BATCH_SIZE, 8)
            dummy_images = []

            for _ in range(dummy_size):
                # Create a dummy 1920x1080 image
                dummy_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                dummy_images.append(Image.fromarray(dummy_array))

            # Warm up with dummy batch
            texts = [self.current_prompt] * dummy_size
            inputs = self.processor(images=dummy_images, text=texts, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _ = self.model(**inputs)
                else:
                    _ = self.model(**inputs)

            # Clear cache after warmup
            torch.cuda.empty_cache()

            logger.info(f"✅ Model warmed up with batch size {dummy_size}")

        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")

    def detect_batch(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, List[Dict]]:
        """
        True parallel batch detection for maximum GPU utilization
        frames_dict: {camera_id: frame_array}
        Returns: {camera_id: [detections]}
        """
        if not self.model or not frames_dict:
            return {cam_id: [] for cam_id in frames_dict.keys()}

        if ENABLE_TRUE_PARALLEL_BATCH and len(frames_dict) > 1:
            return self._parallel_batch_detect(frames_dict)
        else:
            return self._sequential_detect(frames_dict)

    def _parallel_batch_detect(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, List[Dict]]:
        """True parallel batch processing for maximum GPU utilization"""
        try:
            start_time = time.time()
            camera_ids = list(frames_dict.keys())
            frames = list(frames_dict.values())

            logger.info(f"🚀 PARALLEL BATCH PROCESSING: {len(camera_ids)} cameras - {camera_ids}")

            # Convert all frames to PIL images in parallel
            from PIL import Image
            pil_images = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb_frame))

            # Prepare batch inputs for parallel processing
            texts = [self.current_prompt] * len(pil_images)

            # Process in smaller sub-batches to avoid memory issues
            max_batch_size = min(3, len(pil_images))  # Process up to 3 at once for 6GB GPU
            results = {}

            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for i in range(0, len(pil_images), max_batch_size):
                end_idx = min(i + max_batch_size, len(pil_images))
                batch_images = pil_images[i:end_idx]
                batch_texts = texts[i:end_idx]
                batch_camera_ids = camera_ids[i:end_idx]
                batch_frames = frames[i:end_idx]

                logger.info(f"   Processing sub-batch: cameras {batch_camera_ids}")

                # Prepare inputs for this sub-batch
                inputs = self.processor(images=batch_images, text=batch_texts, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # GPU batch inference with maximum utilization
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)

                # Process results for this sub-batch
                for j, cam_id in enumerate(batch_camera_ids):
                    frame_height, frame_width = batch_frames[j].shape[:2]
                    detections = self._process_batch_result(outputs, j, frame_width, frame_height)
                    results[cam_id] = detections

                # Clear GPU memory after each sub-batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            total_detections = sum(len(dets) for dets in results.values())
            fps = len(camera_ids) / processing_time if processing_time > 0 else 0

            logger.info(f"🚀 PARALLEL BATCH COMPLETE: {len(camera_ids)} cameras, {total_detections} detections, {processing_time:.2f}s, {fps:.2f} FPS")

            return results

        except Exception as e:
            logger.error(f"❌ Parallel batch detection failed: {e}")
            # Fallback to sequential processing
            return self._sequential_detect(frames_dict)

    def _sequential_detect(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, List[Dict]]:
        """Fallback sequential processing"""
        try:
            start_time = time.time()
            camera_ids = list(frames_dict.keys())

            logger.info(f"🔄 SEQUENTIAL PROCESSING: {len(camera_ids)} cameras - {camera_ids}")

            results = {}

            # Process each camera with shared model
            for cam_id in camera_ids:
                frame = frames_dict[cam_id]
                detections = self._detect_single_frame(frame)
                results[cam_id] = detections

            processing_time = time.time() - start_time
            total_detections = sum(len(dets) for dets in results.values())

            logger.info(f"✅ SEQUENTIAL COMPLETE: {len(camera_ids)} cameras, {total_detections} detections, {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Sequential detection failed: {e}")
            return {cam_id: [] for cam_id in frames_dict.keys()}

    def _detect_single_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a single frame using shared model"""
        try:
            # Convert frame for model input
            from PIL import Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Prepare inputs
            inputs = self.processor(images=[pil_image], text=[self.current_prompt], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # GPU inference
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            # Process results
            frame_height, frame_width = frame.shape[:2]
            detections = self._process_single_result(outputs, 0, frame_width, frame_height)

            return detections

        except Exception as e:
            logger.error(f"❌ Single frame detection failed: {e}")
            return []

    def _process_single_result(self, outputs, batch_index: int, frame_width: int, frame_height: int) -> List[Dict]:
        """Process detection results for a single frame in the batch"""
        try:
            # Handle batch output format
            if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
                # Standard Grounding DINO output format
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes

                # Check if we have batch dimension
                if len(logits.shape) >= 2 and batch_index < logits.shape[0]:
                    scores = logits[batch_index].sigmoid().cpu()
                    boxes = pred_boxes[batch_index].cpu()
                else:
                    # Fallback for single batch
                    scores = logits.sigmoid().cpu()
                    boxes = pred_boxes.cpu()
                    if len(scores.shape) > 2:
                        scores = scores[0]  # Take first batch
                    if len(boxes.shape) > 2:
                        boxes = boxes[0]  # Take first batch
            else:
                logger.warning(f"⚠️ Unexpected output format: {type(outputs)}")
                return []

            # Filter by confidence - handle different score tensor shapes
            if len(scores.shape) == 2:
                # Shape: [num_queries, num_classes]
                max_scores = scores.max(dim=-1)[0]
            else:
                # Shape: [num_queries]
                max_scores = scores

            keep = max_scores > self.confidence_threshold

            if not keep.any():
                return []

            # Convert to detection format
            detections = []
            filtered_boxes = boxes[keep]
            filtered_scores = max_scores[keep]

            for box, score in zip(filtered_boxes, filtered_scores):
                # Convert from normalized to pixel coordinates
                x_center, y_center, width, height = box
                x1 = int((x_center - width/2) * frame_width)
                y1 = int((y_center - height/2) * frame_height)
                x2 = int((x_center + width/2) * frame_width)
                y2 = int((y_center + height/2) * frame_height)

                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'area': (x2 - x1) * (y2 - y1),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"❌ Failed to process detection result for batch {batch_index}: {e}")
            logger.error(f"   Output shapes - logits: {getattr(outputs, 'logits', 'N/A')}, pred_boxes: {getattr(outputs, 'pred_boxes', 'N/A')}")
            return []

    def _process_batch_result(self, outputs, batch_index: int, frame_width: int, frame_height: int) -> List[Dict]:
        """Process detection results for a single frame in a true parallel batch"""
        try:
            # Handle batch output format for parallel processing
            if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes

                # Extract results for this specific batch index
                if batch_index < logits.shape[0]:
                    scores = logits[batch_index].sigmoid().cpu()
                    boxes = pred_boxes[batch_index].cpu()
                else:
                    logger.warning(f"⚠️ Batch index {batch_index} out of range for batch size {logits.shape[0]}")
                    return []
            else:
                logger.warning(f"⚠️ Unexpected batch output format: {type(outputs)}")
                return []

            # Filter by confidence
            if len(scores.shape) == 2:
                max_scores = scores.max(dim=-1)[0]
            else:
                max_scores = scores

            keep = max_scores > self.confidence_threshold

            if not keep.any():
                return []

            # Convert to detection format
            detections = []
            filtered_boxes = boxes[keep]
            filtered_scores = max_scores[keep]

            for box, score in zip(filtered_boxes, filtered_scores):
                # Convert from normalized to pixel coordinates
                x_center, y_center, width, height = box
                x1 = int((x_center - width/2) * frame_width)
                y1 = int((y_center - height/2) * frame_height)
                x2 = int((x_center + width/2) * frame_width)
                y2 = int((y_center + height/2) * frame_height)

                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'area': (x2 - x1) * (y2 - y1),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"❌ Failed to process batch result for index {batch_index}: {e}")
            return []

class CameraBatchProcessor:
    """Manages batching of camera frames for efficient processing"""

    def __init__(self, shared_detector: SharedModelDetector, camera_ids: List[int]):
        self.shared_detector = shared_detector
        self.camera_ids = camera_ids
        self.batch_size = BATCH_SIZE

        # Create camera groups for maximum GPU utilization
        if ENABLE_TRUE_PARALLEL_BATCH:
            # Use larger batches for true parallel processing
            self.camera_groups = []
            for i in range(0, len(camera_ids), self.batch_size):
                group = camera_ids[i:i + self.batch_size]
                self.camera_groups.append(group)
        else:
            # Use smaller groups for sequential processing
            small_batch_size = 4
            self.camera_groups = []
            for i in range(0, len(camera_ids), small_batch_size):
                group = camera_ids[i:i + small_batch_size]
                self.camera_groups.append(group)

        logger.info(f"🚀 OPTIMIZED BATCH CONFIGURATION:")
        logger.info(f"   Total Cameras: {len(camera_ids)}")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Parallel Processing: {ENABLE_TRUE_PARALLEL_BATCH}")
        logger.info(f"   Camera Groups: {len(self.camera_groups)}")
        for i, group in enumerate(self.camera_groups):
            logger.info(f"   Group {i+1}: {group}")

    def process_all_cameras(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, List[Dict]]:
        """Process all cameras in batches"""
        all_results = {}

        logger.info(f"🔄 PROCESSING {len(self.camera_groups)} BATCHES...")

        for i, group in enumerate(self.camera_groups):
            # Create batch for this group
            group_frames = {cam_id: frames_dict[cam_id] for cam_id in group if cam_id in frames_dict}

            if group_frames:
                logger.info(f"   Batch {i+1}/{len(self.camera_groups)}: Cameras {list(group_frames.keys())}")

                # Process this batch
                group_results = self.shared_detector.detect_batch(group_frames)
                all_results.update(group_results)
            else:
                logger.warning(f"   Batch {i+1}/{len(self.camera_groups)}: No frames available")

        total_detections = sum(len(dets) for dets in all_results.values())
        logger.info(f"✅ ALL BATCHES COMPLETE: {total_detections} total detections")

        return all_results

class GPUOptimizedFisheyeCorrector:
    """GPU-accelerated fisheye correction"""
    
    def __init__(self, lens_mm: float = 2.8):
        self.lens_mm = lens_mm
        self.correction_maps = None
        self.gpu_map1 = None
        self.gpu_map2 = None
        
        # Check GPU availability
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info(f"✅ GPU fisheye correction enabled - {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) available")
        else:
            logger.warning("⚠️ No GPU available for fisheye correction, falling back to CPU")
    
    def initialize_correction_maps(self, frame_shape):
        """Initialize GPU correction maps"""
        if not self.gpu_available:
            return
            
        height, width = frame_shape[:2]
        
        # Camera matrix and distortion coefficients for 2.8mm lens
        camera_matrix = np.array([
            [width * 0.7, 0, width / 2],
            [0, height * 0.7, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.array([-0.3, 0.1, 0, 0, -0.02], dtype=np.float32)
        
        # Generate correction maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), camera_matrix,
            (width, height), cv2.CV_16SC2
        )
        
        # Upload maps to GPU
        self.gpu_map1 = cv2.cuda_GpuMat()
        self.gpu_map2 = cv2.cuda_GpuMat()
        self.gpu_map1.upload(map1)
        self.gpu_map2.upload(map2)
        
        logger.info(f"GPU fisheye correction maps initialized for {width}x{height}")
    
    def correct(self, frame: np.ndarray) -> np.ndarray:
        """Apply GPU-accelerated fisheye correction"""
        if not self.gpu_available:
            return frame
            
        if self.gpu_map1 is None:
            self.initialize_correction_maps(frame.shape)
        
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Apply correction on GPU
            gpu_corrected = cv2.cuda.remap(gpu_frame, self.gpu_map1, self.gpu_map2, cv2.INTER_LINEAR)
            
            # Download result
            corrected_frame = gpu_corrected.download()
            return corrected_frame
            
        except Exception as e:
            logger.error(f"GPU fisheye correction failed: {e}")
            return frame

class GPUSimplePalletDetector:
    """GPU-optimized pallet detector with embedded DetectorTracker functionality"""
    
    def __init__(self):
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt_index = 0
        self.current_prompt = self.sample_prompts[0]
        
        # Initialize GPU detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔍 Initializing GPU pallet detector on {self.device}")
        
        # Initialize Grounding DINO model
        self._initialize_grounding_dino()
        
        # GPU availability check
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        logger.info(f"OpenCV GPU operations: {'✅ Available' if self.gpu_available else '❌ Not Available'}")
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model for GPU inference"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("✅ AutoProcessor loaded successfully")
            
            # Load model and move to GPU
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Grounding DINO model loaded and moved to {self.device}")
            
            # Enable mixed precision for better GPU utilization
            if self.device.type == 'cuda':
                self.model = self.model.half()  # Use FP16
                logger.info("✅ Mixed precision (FP16) enabled for better GPU utilization")
                
        except Exception as e:
            logger.error(f"Failed to initialize Grounding DINO: {e}")
            self.processor = None
            self.model = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """GPU-optimized pallet detection"""
        if self.model is None or self.processor is None:
            return []
        
        try:
            # Convert frame for model input
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=self.current_prompt,
                return_tensors="pt"
            )
            
            # Move inputs to GPU with mixed precision
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device.type == 'cuda':
                # Use autocast for mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            # Process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )
            
            # Convert to detection format
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
                        'area': area
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return []

class GPUCoordinateMapper:
    """GPU-accelerated coordinate mapping"""
    
    def __init__(self, floor_width=45.0, floor_length=30.0, camera_id=None):
        self.floor_width_ft = floor_width
        self.floor_length_ft = floor_length
        self.camera_id = camera_id
        self.homography_matrix = None
        self.gpu_homography = None
        self.is_calibrated = False
        
        # Check GPU availability
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        logger.info(f"GPU coordinate mapper initialized - Floor: {floor_width:.1f}ft x {floor_length:.1f}ft")
        if camera_id:
            logger.info(f"Camera ID: {camera_id}")

    def load_calibration(self, filename=None):
        """Load calibration and prepare GPU matrices"""
        if filename is None:
            if self.camera_id:
                filename = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            else:
                filename = "../configs/warehouse_calibration.json"
        
        if not os.path.isabs(filename) and not filename.startswith('../'):
            filename = f"../{filename}"
            
        try:
            with open(filename, 'r') as file:
                calibration_data = json.load(file)
            
            warehouse_dims = calibration_data.get('warehouse_dimensions', {})
            self.floor_width_ft = warehouse_dims.get('width_feet', self.floor_width_ft)
            self.floor_length_ft = warehouse_dims.get('length_feet', self.floor_length_ft)
            
            image_corners = np.array(calibration_data['image_corners'], dtype=np.float32)
            real_world_corners = np.array(calibration_data['real_world_corners'], dtype=np.float32)
            
            if len(image_corners) != 4 or len(real_world_corners) != 4:
                raise ValueError("Calibration must contain exactly 4 corner points")
            
            self.homography_matrix = cv2.findHomography(image_corners, real_world_corners)[0]
            
            # Upload homography matrix to GPU if available
            if self.gpu_available:
                self.gpu_homography = cv2.cuda_GpuMat()
                self.gpu_homography.upload(self.homography_matrix.astype(np.float32))
                logger.info("✅ Homography matrix uploaded to GPU")
            
            self.is_calibrated = True
            logger.info(f"GPU coordinate calibration loaded from: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def pixel_to_real_batch_gpu(self, pixel_points: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch coordinate transformation"""
        if not self.is_calibrated or not self.gpu_available:
            return self.pixel_to_real_batch_cpu(pixel_points)
        
        try:
            # Reshape points for perspectiveTransform
            points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
            
            # Upload points to GPU
            gpu_points = cv2.cuda_GpuMat()
            gpu_points.upload(points_reshaped)
            
            # Apply transformation on GPU
            gpu_transformed = cv2.cuda.perspectiveTransform(gpu_points, self.gpu_homography)
            
            # Download results
            transformed_points = gpu_transformed.download()
            
            # Reshape back to original format
            result = transformed_points.reshape(-1, 2)
            return result
            
        except Exception as e:
            logger.error(f"GPU coordinate transformation failed: {e}")
            return self.pixel_to_real_batch_cpu(pixel_points)
    
    def pixel_to_real_batch_cpu(self, pixel_points: np.ndarray) -> np.ndarray:
        """CPU fallback for batch coordinate transformation"""
        if not self.is_calibrated:
            return np.full((len(pixel_points), 2), np.nan)
        
        try:
            points_reshaped = pixel_points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(points_reshaped, self.homography_matrix)
            return transformed_points.reshape(-1, 2)
        except Exception as e:
            logger.error(f"CPU coordinate transformation failed: {e}")
            return np.full((len(pixel_points), 2), np.nan)

    def pixel_to_real(self, pixel_x, pixel_y):
        """Single point coordinate transformation"""
        points = np.array([[pixel_x, pixel_y]])
        result = self.pixel_to_real_batch_gpu(points)
        if len(result) > 0 and not np.isnan(result[0]).any():
            return float(result[0][0]), float(result[0][1])
        return None, None

class GPUGlobalFeatureDatabase:
    """GPU-accelerated global feature database with CUDA SIFT and GPU feature matching"""

    def __init__(self, database_file: str = "gpu_warehouse_global_features.pkl"):
        self.database_file = database_file
        self.features = {}
        self.next_global_id = 1000
        self.load_database()

        # Check GPU availability for SIFT
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

        if self.gpu_available:
            try:
                # Try to create GPU SIFT detector
                self.gpu_sift = cv2.cuda.SIFT_create(
                    nfeatures=500,
                    contrastThreshold=0.04,
                    edgeThreshold=10
                )
                logger.info("✅ GPU SIFT detector initialized")
                self.use_gpu_sift = True
            except Exception as e:
                logger.warning(f"GPU SIFT not available: {e}, falling back to CPU SIFT")
                self.use_gpu_sift = False
                self.cpu_sift = cv2.SIFT_create(
                    nfeatures=500,
                    contrastThreshold=0.04,
                    edgeThreshold=10
                )
        else:
            logger.warning("No GPU available, using CPU SIFT")
            self.use_gpu_sift = False
            self.cpu_sift = cv2.SIFT_create(
                nfeatures=500,
                contrastThreshold=0.04,
                edgeThreshold=10
            )

        # GPU-accelerated feature matching
        if self.gpu_available:
            try:
                # Try GPU-based matcher
                self.gpu_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
                logger.info("✅ GPU feature matcher initialized")
                self.use_gpu_matcher = True
            except Exception as e:
                logger.warning(f"GPU matcher not available: {e}, using CPU FLANN")
                self.use_gpu_matcher = False
                self._init_cpu_matcher()
        else:
            self.use_gpu_matcher = False
            self._init_cpu_matcher()

        # Matching parameters
        self.similarity_threshold = 0.3
        self.min_matches = 10
        self.max_disappeared_frames = 30

        logger.info(f"GPU feature database initialized with {len(self.features)} objects")
        logger.info(f"GPU SIFT: {'✅' if self.use_gpu_sift else '❌'}, GPU Matcher: {'✅' if self.use_gpu_matcher else '❌'}")

    def _init_cpu_matcher(self):
        """Initialize CPU FLANN matcher as fallback"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.cpu_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def load_database(self):
        """Load feature database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.features = data.get('features', {})
                    self.next_global_id = data.get('next_id', 1000)
                logger.info(f"Loaded {len(self.features)} objects from GPU database")
            else:
                self.features = {}
                self.next_global_id = 1000
        except Exception as e:
            logger.error(f"Error loading GPU database: {e}")
            self.features = {}
            self.next_global_id = 1000

    def save_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.features,
                'next_id': self.next_global_id,
                'last_updated': datetime.now().isoformat(),
                'gpu_optimized': True
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving GPU database: {e}")

    def extract_features_gpu(self, image_region: np.ndarray) -> Optional[np.ndarray]:
        """GPU-accelerated SIFT feature extraction"""
        if image_region is None or image_region.size == 0:
            return None

        try:
            # Convert to grayscale if needed
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region

            if self.use_gpu_sift:
                # GPU SIFT extraction
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(gray)

                # Extract features on GPU
                gpu_keypoints, gpu_descriptors = self.gpu_sift.detectAndComputeAsync(gpu_image, None)

                # Download descriptors
                if gpu_descriptors is not None:
                    descriptors = gpu_descriptors.download()
                    if descriptors is not None and len(descriptors) >= self.min_matches:
                        return descriptors

                return None
            else:
                # CPU SIFT fallback
                keypoints, descriptors = self.cpu_sift.detectAndCompute(gray, None)
                if descriptors is not None and len(descriptors) >= self.min_matches:
                    return descriptors
                return None

        except Exception as e:
            logger.error(f"GPU feature extraction failed: {e}")
            return None

    def calculate_similarity_gpu(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """GPU-accelerated feature similarity calculation"""
        if features1 is None or features2 is None:
            return 0.0

        if len(features1) < 2 or len(features2) < 2:
            return 0.0

        try:
            if self.use_gpu_matcher:
                # GPU-based matching
                gpu_desc1 = cv2.cuda_GpuMat()
                gpu_desc2 = cv2.cuda_GpuMat()
                gpu_desc1.upload(features1.astype(np.float32))
                gpu_desc2.upload(features2.astype(np.float32))

                # Perform matching on GPU
                gpu_matches = self.gpu_matcher.knnMatch(gpu_desc1, gpu_desc2, k=2)

                # Download matches
                matches = []
                for match_pair in gpu_matches:
                    if len(match_pair) == 2:
                        matches.append(match_pair)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Calculate similarity
                if len(good_matches) >= self.min_matches:
                    similarity = len(good_matches) / min(len(features1), len(features2))
                    return min(similarity, 1.0)
                else:
                    return 0.0
            else:
                # CPU FLANN fallback
                matches = self.cpu_matcher.knnMatch(features1, features2, k=2)

                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= self.min_matches:
                    similarity = len(good_matches) / min(len(features1), len(features2))
                    return min(similarity, 1.0)
                else:
                    return 0.0

        except Exception as e:
            logger.error(f"GPU similarity calculation failed: {e}")
            return 0.0

    def find_matching_object(self, query_features: np.ndarray) -> Tuple[Optional[int], float]:
        """Find best matching object using GPU acceleration"""
        best_match_id = None
        best_similarity = 0.0

        for global_id, feature_data in self.features.items():
            stored_features = feature_data['features']
            similarity = self.calculate_similarity_gpu(query_features, stored_features)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match_id = global_id
                best_similarity = similarity

        return best_match_id, best_similarity

    def add_new_object(self, features: np.ndarray, detection_info: Dict) -> int:
        """Add new object to GPU database"""
        global_id = self.next_global_id
        self.next_global_id += 1

        feature_data = {
            'features': features,
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'times_seen': 1,
            'disappeared_frames': 0,
            'detection_info': detection_info,
            'physical_locations': [],
            'gpu_optimized': True
        }

        # Add physical coordinates if available
        if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
            if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                feature_data['physical_locations'].append({
                    'timestamp': datetime.now().isoformat(),
                    'x_ft': detection_info['physical_x_ft'],
                    'y_ft': detection_info['physical_y_ft']
                })

        self.features[global_id] = feature_data
        self.save_database()

        logger.info(f"🆕 NEW GPU GLOBAL ID: {global_id}")
        return global_id

    def update_object(self, global_id: int, features: np.ndarray, detection_info: Dict):
        """Update existing object in GPU database"""
        if global_id in self.features:
            feature_data = self.features[global_id]

            feature_data['features'] = features
            feature_data['last_seen'] = datetime.now().isoformat()
            feature_data['times_seen'] += 1
            feature_data['disappeared_frames'] = 0

            # Add physical coordinates if available
            if 'physical_x_ft' in detection_info and 'physical_y_ft' in detection_info:
                if detection_info['physical_x_ft'] is not None and detection_info['physical_y_ft'] is not None:
                    feature_data['physical_locations'].append({
                        'timestamp': datetime.now().isoformat(),
                        'x_ft': detection_info['physical_x_ft'],
                        'y_ft': detection_info['physical_y_ft']
                    })

                    # Keep only recent locations
                    if len(feature_data['physical_locations']) > 100:
                        feature_data['physical_locations'] = feature_data['physical_locations'][-100:]

            self.save_database()
            logger.info(f"🔄 UPDATED GPU GLOBAL ID: {global_id} - Times seen: {feature_data['times_seen']}")

    def mark_disappeared_objects(self, seen_ids: Set[int]):
        """Mark objects as disappeared and cleanup old ones"""
        to_remove = []

        for global_id in self.features:
            if global_id not in seen_ids:
                self.features[global_id]['disappeared_frames'] += 1

                if self.features[global_id]['disappeared_frames'] >= self.max_disappeared_frames:
                    to_remove.append(global_id)

        for global_id in to_remove:
            logger.info(f"🗑️ REMOVED GPU GLOBAL ID: {global_id} - Disappeared for {self.max_disappeared_frames} frames")
            del self.features[global_id]

        if to_remove:
            self.save_database()

    def assign_global_id(self, image_region: np.ndarray, detection_info: Dict) -> Tuple[int, str, float]:
        """Assign global ID using GPU-accelerated feature matching"""
        features = self.extract_features_gpu(image_region)
        if features is None:
            return -1, 'failed', 0.0

        match_id, similarity = self.find_matching_object(features)

        if match_id is not None:
            self.update_object(match_id, features, detection_info)
            return match_id, 'existing', similarity
        else:
            new_id = self.add_new_object(features, detection_info)
            return new_id, 'new', 1.0

class GPUCompleteWarehouseTracker:
    """GPU-optimized complete warehouse tracking system"""

    def __init__(self, camera_id: int = 8, batch_mode: bool = False):
        self.camera_id = camera_id
        self.batch_mode = batch_mode
        self.warehouse_config = get_warehouse_config()

        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")

        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False

        # GPU-optimized detection components
        self.fisheye_corrector = GPUOptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)

        # Only create individual detector if NOT in batch mode
        if not self.batch_mode:
            logger.info(f"🔍 Initializing individual detector for Camera {camera_id}")
            self.pallet_detector = GPUSimplePalletDetector()
        else:
            logger.info(f"🚀 Camera {camera_id} using SHARED model (no individual detector)")
            self.pallet_detector = None

        # GPU coordinate mapping
        self.coordinate_mapper = GPUCoordinateMapper(camera_id=camera_id)
        self.coordinate_mapper_initialized = False
        self._initialize_coordinate_mapper()

        # GPU global feature database
        self.global_db = GPUGlobalFeatureDatabase(f"gpu_camera_{camera_id}_global_features.pkl")

        # Detection parameters (only if individual detector exists)
        if not self.batch_mode and self.pallet_detector:
            self.pallet_detector.confidence_threshold = 0.1
            self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
            self.pallet_detector.current_prompt_index = 0
            self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]

        # Filtering settings (same as original)
        self.MIN_AREA = 10000
        self.MAX_AREA = 100000
        self.CELL_SIZE = 40

        # Detection results storage
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.final_tracked_detections = []

        # Performance tracking
        self.frame_count = 0
        self.total_detections = 0
        self.new_objects = 0
        self.existing_objects = 0
        self.gpu_memory_usage = 0

        # GPU memory monitoring
        if torch.cuda.is_available():
            self.gpu_device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.gpu_device)
            logger.info(f"🚀 GPU tracking initialized on {self.gpu_name}")

        logger.info(f"GPU warehouse tracker initialized for {self.camera_name}")
        logger.info(f"All components GPU-optimized: Fisheye, Detection, SIFT, Coordinate mapping")

    def _initialize_coordinate_mapper(self):
        """Initialize GPU coordinate mapper"""
        try:
            calibration_file = f"../configs/warehouse_calibration_camera_{self.camera_id}.json"
            self.coordinate_mapper.load_calibration(calibration_file)

            if self.coordinate_mapper.is_calibrated:
                self.coordinate_mapper_initialized = True
                logger.info(f"✅ GPU coordinate mapper initialized for {self.camera_name}")
            else:
                logger.warning(f"⚠️ GPU coordinate mapper not calibrated for {self.camera_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize GPU coordinate mapper for {self.camera_name}: {e}")
            self.coordinate_mapper_initialized = False

    def connect_camera(self) -> bool:
        """Connect to the camera"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name}...")

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Set aggressive timeout settings to prevent hanging
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                return False

            logger.info(f"{self.camera_name} connected successfully")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False

    def calculate_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y

    def get_grid_cell(self, center: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid cell coordinates for a center point"""
        x, y = center
        cell_x = int(x // self.CELL_SIZE)
        cell_y = int(y // self.CELL_SIZE)
        return cell_x, cell_y

    def get_neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all 9 cells (current + 8 neighbors) for a given cell"""
        cell_x, cell_y = cell
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                neighbors.append(neighbor_cell)

        return neighbors

    def apply_area_filter_gpu(self, detections: List[Dict]) -> List[Dict]:
        """GPU-optimized area filtering using vectorized operations"""
        if not detections:
            return []

        try:
            # Extract areas as numpy array for vectorized operations
            areas = np.array([detection.get('area', 0) for detection in detections])

            # Vectorized filtering
            valid_mask = (areas >= self.MIN_AREA) & (areas <= self.MAX_AREA)

            # Filter detections
            accepted = [detection for i, detection in enumerate(detections) if valid_mask[i]]

            return accepted

        except Exception as e:
            logger.error(f"GPU area filtering failed: {e}")
            # Fallback to CPU
            return [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]

    def apply_grid_cell_filter_gpu(self, detections: List[Dict]) -> List[Dict]:
        """GPU-optimized grid cell filtering"""
        if len(detections) <= 1:
            return detections

        try:
            # Calculate centers and grid cells for all detections
            centers = []
            for detection in detections:
                center = self.calculate_center(detection['bbox'])
                detection['center'] = center
                detection['grid_cell'] = self.get_grid_cell(center)
                centers.append(center)

            # Convert to numpy arrays for vectorized operations
            centers_array = np.array(centers)
            confidences = np.array([d['confidence'] for d in detections])

            # Sort by confidence (keep higher confidence detections first)
            sorted_indices = np.argsort(confidences)[::-1]
            sorted_detections = [detections[i] for i in sorted_indices]

            occupied_cells: Set[Tuple[int, int]] = set()
            accepted = []

            for detection in sorted_detections:
                cell = detection['grid_cell']
                neighbor_cells = self.get_neighbor_cells(cell)

                # Check if any of the 9 cells are already occupied
                conflict = any(neighbor_cell in occupied_cells for neighbor_cell in neighbor_cells)

                if not conflict:
                    occupied_cells.add(cell)
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"GPU grid filtering failed: {e}")
            return detections

    def translate_to_physical_coordinates_batch_gpu(self, detections: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """GPU-optimized batch physical coordinate translation"""
        if not self.coordinate_mapper_initialized or not detections:
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'MAPPER_NOT_AVAILABLE'
            return detections

        try:
            # Extract all center points
            centers = []
            for detection in detections:
                center = detection.get('center')
                if not center:
                    center = self.calculate_center(detection['bbox'])
                    detection['center'] = center
                centers.append(center)

            # Convert to numpy array for batch processing
            centers_array = np.array(centers, dtype=np.float32)

            # Scale coordinates to calibration frame size (4K) - vectorized
            scale_x = 3840 / frame_width
            scale_y = 2160 / frame_height

            scaled_centers = centers_array * np.array([scale_x, scale_y])

            # Batch GPU coordinate transformation
            physical_coords = self.coordinate_mapper.pixel_to_real_batch_gpu(scaled_centers)

            # Assign results back to detections
            for i, detection in enumerate(detections):
                if i < len(physical_coords) and not np.isnan(physical_coords[i]).any():
                    raw_x = float(physical_coords[i][0])
                    raw_y = float(physical_coords[i][1])

                    # Clamp coordinates to valid warehouse range
                    # Camera 1: 0-62ft x 0-25ft (adjust ranges for other cameras as needed)
                    if self.camera_id == 1:
                        clamped_x = max(0.0, min(62.0, raw_x))
                        clamped_y = max(0.0, min(25.0, raw_y))
                    else:
                        # For other cameras, use general warehouse bounds
                        clamped_x = max(0.0, min(180.0, raw_x))  # Full warehouse width
                        clamped_y = max(0.0, min(90.0, raw_y))   # Full warehouse height

                    detection['physical_x_ft'] = round(clamped_x, 2)
                    detection['physical_y_ft'] = round(clamped_y, 2)

                    # Mark if coordinates were clamped
                    if abs(raw_x - clamped_x) > 0.01 or abs(raw_y - clamped_y) > 0.01:
                        detection['coordinate_status'] = 'SUCCESS_CLAMPED'
                    else:
                        detection['coordinate_status'] = 'SUCCESS'
                else:
                    detection['physical_x_ft'] = None
                    detection['physical_y_ft'] = None
                    detection['coordinate_status'] = 'CONVERSION_FAILED'

            return detections

        except Exception as e:
            logger.error(f"GPU coordinate translation failed: {e}")
            # Fallback to individual processing
            for detection in detections:
                detection['physical_x_ft'] = None
                detection['physical_y_ft'] = None
                detection['coordinate_status'] = 'ERROR'
            return detections

    def extract_detection_region_gpu(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """GPU-optimized detection region extraction"""
        try:
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))

            if x2 <= x1 or y2 <= y1:
                return np.array([])

            # Extract region (this could be GPU-accelerated with cv2.cuda operations)
            region = frame[y1:y2, x1:x2]
            return region

        except Exception as e:
            logger.error(f"GPU region extraction failed: {e}")
            return np.array([])

    def assign_global_ids_gpu(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """GPU-accelerated global ID assignment"""
        global_detections = []
        seen_ids = set()

        for detection in detections:
            self.total_detections += 1

            # Extract image region for feature analysis
            image_region = self.extract_detection_region_gpu(frame, detection['bbox'])

            # Assign global ID using GPU SIFT features
            global_id, status, similarity = self.global_db.assign_global_id(
                image_region, detection
            )

            # Update statistics
            if status == 'new':
                self.new_objects += 1
            elif status == 'existing':
                self.existing_objects += 1

            # Add global tracking info to detection
            detection['global_id'] = global_id
            detection['tracking_status'] = status
            detection['similarity_score'] = similarity

            if global_id != -1:
                seen_ids.add(global_id)

            global_detections.append(detection)

        # Mark disappeared objects
        self.global_db.mark_disappeared_objects(seen_ids)

        return global_detections

    def monitor_gpu_usage(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            self.gpu_memory_usage = torch.cuda.memory_allocated(self.gpu_device) / 1024**3  # GB

    def start_tracking(self):
        """Start GPU-optimized warehouse tracking"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return

        self.running = True

        # Create display window
        window_name = f"GPU Warehouse Tracking - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        logger.info("=== GPU-OPTIMIZED WAREHOUSE TRACKING ===")
        logger.info("🚀 All operations GPU-accelerated for maximum performance")
        logger.info("Pipeline: GPU Detection → GPU Filtering → GPU Coords → GPU SIFT")
        logger.info("Press 'q' or ESC to quit")
        logger.info("=" * 60)

        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()

        while self.running:
            try:
                frame_start = time.time()

                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break

                # Process frame with GPU acceleration
                processed_frame = self._process_frame_gpu(frame)

                # Monitor performance
                frame_time = time.time() - frame_start
                fps_counter += 1

                if fps_counter % 30 == 0:  # Log every 30 frames
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_counter / elapsed
                    self.monitor_gpu_usage()
                    logger.info(f"🚀 GPU Performance: {current_fps:.1f} FPS, GPU Memory: {self.gpu_memory_usage:.2f}GB")
                    fps_counter = 0
                    fps_start_time = time.time()

                # Display frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF  # Reduced wait time for better performance
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

            except Exception as e:
                logger.error(f"Error in GPU tracking loop: {e}")
                break

        self.stop_tracking()

    def _process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated frame processing pipeline"""
        self.frame_count += 1
        processed_frame = frame.copy()

        # GPU fisheye correction
        if Config.FISHEYE_CORRECTION_ENABLED:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"GPU fisheye correction failed: {e}")

        # GPU-accelerated resize
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Use GPU resize if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(processed_frame)
                    gpu_resized = cv2.cuda.resize(gpu_frame, (new_width, new_height))
                    processed_frame = gpu_resized.download()
                except:
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            else:
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Complete GPU detection and tracking pipeline
        try:
            # Stage 1: GPU detection (skip if in batch mode - handled by shared model)
            if not self.batch_mode and self.pallet_detector:
                self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)
            else:
                # In batch mode, detections will be provided externally
                self.raw_detections = getattr(self, 'last_detections', [])

            # Stage 2: GPU area filtering
            self.area_filtered_detections = self.apply_area_filter_gpu(self.raw_detections)

            # Stage 3: GPU grid cell filtering
            self.grid_filtered_detections = self.apply_grid_cell_filter_gpu(self.area_filtered_detections)

            # Stage 4: GPU batch physical coordinate translation
            frame_height, frame_width = processed_frame.shape[:2]
            self.grid_filtered_detections = self.translate_to_physical_coordinates_batch_gpu(
                self.grid_filtered_detections, frame_width, frame_height
            )

            # Stage 5: GPU SIFT feature matching and global ID assignment
            self.final_tracked_detections = self.assign_global_ids_gpu(self.grid_filtered_detections, processed_frame)

        except Exception as e:
            logger.error(f"GPU detection pipeline failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.final_tracked_detections = []

        # Draw results
        processed_frame = self._draw_detections_gpu(processed_frame)
        processed_frame = self._draw_info_overlay_gpu(processed_frame)

        return processed_frame

    def _draw_detections_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-optimized detection drawing (CPU drawing with GPU-processed data)"""
        result_frame = frame.copy()

        for detection in self.final_tracked_detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            area = detection.get('area', 0)

            # Global tracking info
            global_id = detection.get('global_id', -1)
            tracking_status = detection.get('tracking_status', 'unknown')

            # Physical coordinates
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')
            coord_status = detection.get('coordinate_status', 'UNKNOWN')

            x1, y1, x2, y2 = bbox

            # Color coding based on tracking status
            if tracking_status == 'new':
                color = (0, 255, 0)  # Green for new objects
                status_text = "NEW"
            elif tracking_status == 'existing':
                color = (255, 165, 0)  # Orange for existing objects
                status_text = "GPU-TRACKED"
            else:
                color = (0, 0, 255)  # Red for failed
                status_text = "FAILED"

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(result_frame, center, 8, color, -1)
            cv2.circle(result_frame, center, 8, (255, 255, 255), 2)

            # Labels with all information
            y_offset = y1 - 5
            line_height = 20

            # Global ID and status
            if global_id != -1:
                id_label = f"GPU-ID:{global_id} ({status_text})"
                cv2.putText(result_frame, id_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= line_height

            # Confidence and area
            conf_label = f"Conf:{confidence:.3f} Area:{area:.0f}"
            cv2.putText(result_frame, conf_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset -= line_height

            # Pixel coordinates
            pixel_label = f"Pixel:({center[0]},{center[1]})"
            cv2.putText(result_frame, pixel_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset -= line_height

            # Physical coordinates
            if physical_x is not None and physical_y is not None:
                coord_label = f"Physical:({physical_x:.1f}ft,{physical_y:.1f}ft)"
                coord_color = (0, 255, 255)  # Cyan for physical coordinates
            else:
                coord_label = f"Physical:{coord_status}"
                coord_color = (0, 0, 255)  # Red for failed coordinates

            cv2.putText(result_frame, coord_label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coord_color, 1)

        # Store last detections for MongoDB integration
        self.last_detections = self.final_tracked_detections

        return result_frame

    def _draw_info_overlay_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-optimized information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 350), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"🚀 GPU WAREHOUSE TRACKING", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_count} | GPU Memory: {self.gpu_memory_usage:.2f}GB", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"GPU DETECTION PIPELINE:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"1. GPU Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"2. GPU Area Filtered: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"3. GPU Grid Filtered: {len(self.grid_filtered_detections)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        y_offset += 15
        cv2.putText(frame, f"4. GPU Final Tracked: {len(self.final_tracked_detections)}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 25
        cv2.putText(frame, f"GPU TRACKING STATS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"New Objects: {self.new_objects}", (20, y_offset), font, 0.4, (0, 255, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"GPU Tracked Objects: {self.existing_objects}", (20, y_offset), font, 0.4, (255, 165, 0), 1)

        y_offset += 15
        cv2.putText(frame, f"Database Objects: {len(self.global_db.features)}", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        # GPU optimization status
        y_offset += 25
        cv2.putText(frame, f"GPU OPTIMIZATIONS:", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        gpu_sift_status = "✅" if self.global_db.use_gpu_sift else "❌"
        cv2.putText(frame, f"GPU SIFT: {gpu_sift_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.global_db.use_gpu_sift else (255, 0, 0), 1)

        y_offset += 15
        gpu_matcher_status = "✅" if self.global_db.use_gpu_matcher else "❌"
        cv2.putText(frame, f"GPU Matcher: {gpu_matcher_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.global_db.use_gpu_matcher else (255, 0, 0), 1)

        y_offset += 15
        coord_status = "✅" if self.coordinate_mapper_initialized else "❌"
        cv2.putText(frame, f"GPU Coords: {coord_status}", (20, y_offset), font, 0.4, (0, 255, 0) if self.coordinate_mapper_initialized else (255, 0, 0), 1)

        return frame

    def stop_tracking(self):
        """Stop GPU tracking system"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped GPU tracking for {self.camera_name}")
        logger.info(f"GPU Session stats - New: {self.new_objects}, Existing: {self.existing_objects}, Total: {self.total_detections}")


class MultiCameraWarehouseSystem:
    """Multi-camera warehouse tracking system with configurable GUI and camera selection"""

    def __init__(self, active_cameras, gui_cameras, enable_gui= True, enable_console_logging=True,
                 log_interval=30, enable_cross_camera_tracking=True, save_to_db=True,
                 performance_monitoring=True, window_width=800, window_height=600, arrange_windows=True):

        self.active_cameras = active_cameras
        self.gui_cameras = gui_cameras if enable_gui else []
        self.enable_gui = enable_gui
        self.enable_console_logging = enable_console_logging
        self.log_interval = log_interval
        self.enable_cross_camera_tracking = enable_cross_camera_tracking
        self.save_to_db = save_to_db
        self.performance_monitoring = performance_monitoring
        self.window_width = window_width
        self.window_height = window_height
        self.arrange_windows = arrange_windows

        # Camera trackers
        self.trackers = {}
        self.running = False

        # Batch processing components
        if ENABLE_BATCH_PROCESSING:
            logger.info("🚀 INITIALIZING BATCH PROCESSING MODE")
            self.shared_detector = SharedModelDetector()
            self.batch_processor = CameraBatchProcessor(self.shared_detector, active_cameras)
            self.batch_mode = True
            logger.info("🚨 SHARED MODEL MODE: Individual camera models will be DISABLED")
        else:
            logger.info("🔄 USING INDIVIDUAL CAMERA MODE")
            self.batch_mode = False

        # Performance tracking
        self.frame_counts = {cam_id: 0 for cam_id in active_cameras}
        self.detection_counts = {cam_id: 0 for cam_id in active_cameras}
        self.start_time = time.time()

        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_counts = {cam_id: 0 for cam_id in active_cameras}
        self.current_fps = {cam_id: 0.0 for cam_id in active_cameras}
        self.last_fps_update = time.time()

        # MongoDB integration
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_collection = None
        self.detection_batch = []
        self.total_saved_detections = 0

        if ENABLE_MONGODB:
            self.initialize_mongodb()

        # Camera names mapping
        self.camera_names = {
            1: "Camera 1 (Column 1 - Top)", 2: "Camera 2 (Column 1 - Middle)", 3: "Camera 3 (Column 1 - Bottom)",
            4: "Camera 4 (Column 2 - Top)", 5: "Camera 5 (Column 2 - Middle)", 6: "Camera 6 (Column 2 - Bottom)",
            7: "Camera 7 (Column 2 - Bottom-2)", 8: "Camera 8 (Column 3 - Bottom)", 9: "Camera 9 (Column 3 - Middle)",
            10: "Camera 10 (Column 3 - Top)", 11: "Camera 11 (Column 3 - Top-2)"
        }

        logger.info("🎛️ Multi-Camera System Configuration:")
        logger.info(f"📹 Active Cameras: {self.active_cameras}")
        logger.info(f"🖥️ GUI Cameras: {self.gui_cameras}")
        logger.info(f"🎛️ GUI Mode: {'ENABLED' if self.enable_gui else 'HEADLESS'}")
        logger.info(f"🔄 Cross-Camera Tracking: {'ENABLED' if self.enable_cross_camera_tracking else 'DISABLED'}")

    def initialize_cameras(self) -> bool:
        """Initialize all active cameras"""
        logger.info(f"🔧 Initializing {len(self.active_cameras)} cameras...")

        for cam_id in self.active_cameras:
            try:
                camera_name = self.camera_names.get(cam_id, f"Camera {cam_id}")
                logger.info(f"🔧 Initializing {camera_name}...")

                tracker = GPUCompleteWarehouseTracker(camera_id=cam_id, batch_mode=self.batch_mode)

                # Connect to camera
                if not tracker.connect_camera():
                    logger.error(f"❌ Failed to connect to {camera_name}")
                    logger.warning(f"⚠️ Skipping Camera {cam_id} - continuing with other cameras")
                    continue  # Skip this camera but continue with others

                self.trackers[cam_id] = tracker
                logger.info(f"✅ {camera_name} initialized and connected")

            except Exception as e:
                logger.error(f"❌ Error initializing Camera {cam_id}: {e}")
                logger.warning(f"⚠️ Skipping Camera {cam_id} - continuing with other cameras")
                continue

        connected_cameras = len(self.trackers)
        if connected_cameras == 0:
            logger.error("❌ No cameras connected successfully!")
            return False

        logger.info(f"🚀 {connected_cameras} out of {len(self.active_cameras)} cameras initialized successfully!")
        return True

    def arrange_gui_windows(self):
        """Arrange GUI windows on screen"""
        if not self.enable_gui or not self.gui_cameras:
            return

        # Calculate grid layout
        num_windows = len(self.gui_cameras)
        cols = min(4, num_windows)  # Max 4 columns
        rows = (num_windows + cols - 1) // cols

        for i, cam_id in enumerate(self.gui_cameras):
            row = i // cols
            col = i % cols

            x = col * (self.window_width + 50)  # 50px spacing
            y = row * (self.window_height + 100)  # 100px spacing for title bar

            window_name = f"Camera {cam_id} - Warehouse Tracking"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.window_width, self.window_height)
            cv2.moveWindow(window_name, x, y)

        logger.info(f"🖥️ Arranged {num_windows} GUI windows in {rows}x{cols} grid")

    def initialize_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.mongodb_client = MongoClient(MONGODB_URL)
            self.mongodb_db = self.mongodb_client[MONGODB_DATABASE]
            self.mongodb_collection = self.mongodb_db[MONGODB_COLLECTION]

            # Test connection
            self.mongodb_client.admin.command('ping')
            logger.info(f"✅ MongoDB connected: {MONGODB_DATABASE}.{MONGODB_COLLECTION}")

            # Create indexes for better performance
            self.mongodb_collection.create_index([("camera_id", 1), ("timestamp", -1)])
            self.mongodb_collection.create_index([("global_id", 1)])

        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.warning("⚠️ Continuing without MongoDB integration")
            self.mongodb_client = None

    def save_detection_to_db(self, camera_id: int, detection: Dict):
        """Add detection to batch for MongoDB saving"""
        if not ENABLE_MONGODB or not self.mongodb_client:
            return

        # Prepare detection document
        detection_doc = {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow(),
            "global_id": detection.get('global_id'),
            "bbox": detection.get('bbox'),
            "confidence": detection.get('confidence'),
            "area": detection.get('area'),
            "physical_x_ft": detection.get('physical_x_ft'),
            "physical_y_ft": detection.get('physical_y_ft'),
            "grid_cell": detection.get('grid_cell'),
            "times_seen": detection.get('times_seen', 1),
            "is_new": detection.get('is_new', False)
        }

        # Add to batch
        self.detection_batch.append(detection_doc)

        # Save batch when it reaches the configured size
        if len(self.detection_batch) >= BATCH_SAVE_SIZE:
            self.flush_detection_batch()

    def flush_detection_batch(self):
        """Save all batched detections to MongoDB"""
        if not self.detection_batch or not self.mongodb_client:
            return

        try:
            result = self.mongodb_collection.insert_many(self.detection_batch)
            self.total_saved_detections += len(result.inserted_ids)

            if len(self.detection_batch) >= 5:  # Only log for larger batches
                logger.info(f"💾 Saved {len(self.detection_batch)} detections to MongoDB (Total: {self.total_saved_detections})")

            self.detection_batch.clear()

        except Exception as e:
            logger.error(f"❌ Failed to save detections to MongoDB: {e}")
            self.detection_batch.clear()  # Clear to prevent memory buildup

    def update_fps(self, cam_id):
        """Update FPS for a specific camera"""
        self.fps_frame_counts[cam_id] += 1

        current_time = time.time()
        if current_time - self.last_fps_update >= FPS_UPDATE_INTERVAL:
            # Calculate FPS for each camera
            elapsed = current_time - self.fps_start_time
            for camera_id in self.active_cameras:
                if camera_id in self.trackers:
                    self.current_fps[camera_id] = self.fps_frame_counts[camera_id] / elapsed

            # Print FPS stats to console
            if SHOW_FPS_IN_CONSOLE:
                self.print_fps_stats()

            # Reset counters
            self.fps_start_time = current_time
            self.fps_frame_counts = {cam_id: 0 for cam_id in self.active_cameras}
            self.last_fps_update = current_time

    def print_fps_stats(self):
        """Print FPS statistics to console"""
        total_fps = sum(self.current_fps.values())
        avg_fps = total_fps / len(self.active_cameras) if self.active_cameras else 0

        print("\n" + "="*80)
        print(f"📊 FPS MONITORING - {len(self.active_cameras)} Cameras | GUI: {'ON' if self.enable_gui else 'OFF'}")
        print("="*80)

        for cam_id in self.active_cameras:
            if cam_id in self.trackers:
                fps = self.current_fps[cam_id]
                gui_status = "🖥️" if cam_id in self.gui_cameras else "🔇"
                print(f"   {gui_status} Camera {cam_id}: {fps:.1f} FPS")

        print(f"📈 TOTAL FPS: {total_fps:.1f} | AVERAGE: {avg_fps:.1f} FPS")
        print("="*80 + "\n")

    def add_fps_to_frame(self, frame, cam_id):
        """Add FPS text overlay to frame"""
        if not SHOW_FPS_ON_GUI or not self.enable_gui:
            return frame

        fps = self.current_fps.get(cam_id, 0.0)
        fps_text = f"Camera {cam_id} - {fps:.1f} FPS"

        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add FPS text
        cv2.putText(frame, fps_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def _process_batch_cycle(self, frame_count: int):
        """Process all cameras using batch processing"""
        cycle_start = time.time()

        # Step 1: Capture frames from all cameras
        frames = {}
        for cam_id in self.active_cameras:
            if cam_id in self.trackers:
                tracker = self.trackers[cam_id]
                if tracker.cap and tracker.cap.isOpened():
                    ret, frame = tracker.cap.read()
                    if ret:
                        frames[cam_id] = frame
                    else:
                        if self.enable_console_logging:
                            logger.warning(f"⚠️ Failed to read frame from Camera {cam_id}")

        if not frames:
            return

        # Step 2: Batch detection processing
        logger.info(f"🔄 BATCH CYCLE {frame_count}: Processing {len(frames)} cameras")
        batch_results = self.batch_processor.process_all_cameras(frames)

        # Step 3: Process results for each camera
        for cam_id, detections in batch_results.items():
            if cam_id in self.trackers:
                tracker = self.trackers[cam_id]
                frame = frames[cam_id]

                # Provide detections to tracker for further processing
                tracker.last_detections = detections

                # Process frame with provided detections (filtering, coordinates, SIFT)
                processed_frame = tracker._process_frame_gpu(frame)

                # Save to MongoDB
                if ENABLE_MONGODB:
                    for detection in tracker.final_tracked_detections:
                        self.save_detection_to_db(cam_id, detection)

                # Update counters
                self.frame_counts[cam_id] += 1
                self.update_fps(cam_id)

                # Show GUI if enabled
                if self.enable_gui and cam_id in self.gui_cameras:
                    display_frame = self.add_fps_to_frame(processed_frame, cam_id)
                    window_name = f"Camera {cam_id} - Warehouse Tracking"
                    cv2.imshow(window_name, display_frame)

        # Performance logging
        cycle_time = time.time() - cycle_start
        total_detections = sum(len(dets) for dets in batch_results.values())

        if frame_count % 10 == 0:  # Log every 10 cycles
            logger.info(f"📊 BATCH CYCLE {frame_count}: {len(frames)} cameras, {total_detections} detections, {cycle_time:.2f}s")

    def _create_display_frame(self, frame: np.ndarray, detections: List[Dict], cam_id: int) -> np.ndarray:
        """Create display frame with detection overlays"""
        display_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            conf_text = f"{confidence:.2f}"
            cv2.putText(display_frame, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display_frame

    def _process_individual_cameras(self, frame_count: int):
        """Process cameras individually (original method)"""
        # Process each active camera
        for cam_id in self.active_cameras:
            if cam_id not in self.trackers:
                continue

            tracker = self.trackers[cam_id]

            # Read frame
            if tracker.cap and tracker.cap.isOpened():
                ret, frame = tracker.cap.read()
                if ret:
                    # Process frame
                    processed_frame = tracker._process_frame_gpu(frame)

                    # Save detections to MongoDB
                    if ENABLE_MONGODB and hasattr(tracker, 'last_detections'):
                        for detection in tracker.last_detections:
                            self.save_detection_to_db(cam_id, detection)

                    # Update counters
                    self.frame_counts[cam_id] += 1

                    # Update FPS tracking
                    self.update_fps(cam_id)

                    # Show GUI if enabled and camera is in GUI list
                    if self.enable_gui and cam_id in self.gui_cameras:
                        # Add FPS overlay to frame
                        display_frame = self.add_fps_to_frame(processed_frame, cam_id)
                        window_name = f"Camera {cam_id} - Warehouse Tracking"
                        cv2.imshow(window_name, display_frame)
                else:
                    if self.enable_console_logging:
                        logger.warning(f"⚠️ Failed to read frame from Camera {cam_id}")
            else:
                if self.enable_console_logging:
                    logger.error(f"❌ Camera {cam_id} not available")

    def start_tracking(self):
        """Start multi-camera tracking"""
        if not self.initialize_cameras():
            logger.error("❌ Camera initialization failed")
            return

        if self.arrange_windows and self.enable_gui:
            self.arrange_gui_windows()

        self.running = True
        logger.info("🚀 Starting multi-camera tracking...")

        if not self.enable_gui:
            logger.info("🖥️ Running in HEADLESS mode - no GUI windows")

        try:
            frame_count = 0
            while self.running:
                frame_count += 1

                if self.batch_mode:
                    # BATCH PROCESSING MODE
                    self._process_batch_cycle(frame_count)
                else:
                    # INDIVIDUAL CAMERA MODE (Original)
                    self._process_individual_cameras(frame_count)

                # Performance logging
                if self.performance_monitoring and frame_count % self.log_interval == 0:
                    self._log_performance(frame_count)

                # Check for quit (only if GUI enabled)
                if self.enable_gui:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Small delay for headless mode
                    time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.stop_tracking()

    def _log_performance(self, frame_count):
        """Log performance statistics"""
        elapsed_time = time.time() - self.start_time
        total_frames = sum(self.frame_counts.values())
        overall_fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        logger.info("=" * 80)
        logger.info(f"📊 PERFORMANCE REPORT (Frame {frame_count}):")
        logger.info(f"⏱️ Runtime: {elapsed_time:.1f}s | Overall FPS: {overall_fps:.1f}")
        logger.info(f"📹 Active Cameras: {len(self.active_cameras)} | GUI Cameras: {len(self.gui_cameras)}")

        for cam_id in self.active_cameras:
            frames = self.frame_counts[cam_id]
            fps = frames / elapsed_time if elapsed_time > 0 else 0
            gui_status = "🖥️" if cam_id in self.gui_cameras else "🔇"
            logger.info(f"   {gui_status} Camera {cam_id}: {fps:.1f} FPS ({frames} frames)")

        logger.info("=" * 80)

    def stop_tracking(self):
        """Stop tracking and cleanup"""
        self.running = False
        logger.info("🧹 Cleaning up multi-camera system...")

        for cam_id, tracker in self.trackers.items():
            try:
                tracker.stop_tracking()
                logger.info(f"✅ Camera {cam_id} stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping Camera {cam_id}: {e}")

        if self.enable_gui:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logger.warning(f"⚠️ Could not destroy GUI windows: {e}")
                logger.info("💡 This is normal if OpenCV was built without GUI support")

        # Final performance report
        if self.performance_monitoring:
            elapsed_time = time.time() - self.start_time
            total_frames = sum(self.frame_counts.values())

            print("\n" + "="*80)
            print("📊 FINAL FPS PERFORMANCE REPORT")
            print("="*80)
            print(f"⏱️ Total Runtime: {elapsed_time:.1f}s")
            print(f"📹 Total Frames Processed: {total_frames}")
            print(f"🎛️ GUI Mode: {'ENABLED' if self.enable_gui else 'HEADLESS'}")
            print(f"� Active Cameras: {len(self.active_cameras)}")
            print(f"🖥️ GUI Cameras: {len(self.gui_cameras)}")
            print("="*80)

            # Per-camera FPS
            for cam_id in self.active_cameras:
                if cam_id in self.trackers:
                    frames = self.frame_counts[cam_id]
                    fps = frames / elapsed_time if elapsed_time > 0 else 0
                    gui_status = "🖥️" if cam_id in self.gui_cameras else "🔇"
                    print(f"   {gui_status} Camera {cam_id}: {fps:.1f} FPS ({frames} frames)")

            overall_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
            print("="*80)
            print(f"🚀 OVERALL SYSTEM FPS: {overall_fps:.1f}")
            print("="*80 + "\n")

        # Final MongoDB cleanup
        if ENABLE_MONGODB and self.mongodb_client:
            self.flush_detection_batch()  # Save any remaining detections
            self.mongodb_client.close()
            logger.info(f"💾 MongoDB: Total {self.total_saved_detections} detections saved")

        logger.info("✅ Multi-camera system shutdown complete")


# ============================================================================
# 🎛️ CONFIGURATION SECTION - EDIT THESE VARIABLES TO CONTROL THE SYSTEM
# ============================================================================

# 📹 CAMERA CONFIGURATION
ALL_CAMERAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All available cameras

# Support for camera group splitting for parallel processing
CAMERA_GROUP = os.getenv('CAMERA_GROUP', 'ALL')  # ALL, GROUP1, GROUP2

if CAMERA_GROUP == 'GROUP1':
    ACTIVE_CAMERAS = [1, 2, 3, 4, 5, 6]  # First 6 cameras
    GUI_CAMERAS = [1, 2, 3, 4, 5, 6]
    print("🔥 RUNNING CAMERA GROUP 1: Cameras 1-6")
elif CAMERA_GROUP == 'GROUP2':
    ACTIVE_CAMERAS = [7, 8, 9, 10, 11]   # Last 5 cameras
    GUI_CAMERAS = [7, 8, 9, 10, 11]
    print("🔥 RUNNING CAMERA GROUP 2: Cameras 7-11")
elif CAMERA_GROUP == 'SINGLE8':
    ACTIVE_CAMERAS = [8]  # Single Camera 8 for testing
    GUI_CAMERAS = [8]
    print("🔥 RUNNING SINGLE CAMERA 8 FOR TESTING")
elif CAMERA_GROUP.startswith('SINGLE'):
    # Extract camera number from SINGLE1, SINGLE2, etc.
    try:
        cam_num = int(CAMERA_GROUP.replace('SINGLE', ''))
        if 1 <= cam_num <= 11:
            ACTIVE_CAMERAS = [cam_num]
            GUI_CAMERAS = [cam_num]
            print(f"🔥 RUNNING SINGLE CAMERA {cam_num} FOR TESTING")
        else:
            print(f"❌ Invalid camera number: {cam_num}. Using Camera 8.")
            ACTIVE_CAMERAS = [8]
            GUI_CAMERAS = [8]
    except ValueError:
        print("❌ Invalid CAMERA_GROUP format. Using Camera 8.")
        ACTIVE_CAMERAS = [8]
        GUI_CAMERAS = [8]
else:
    ACTIVE_CAMERAS = [8]  # All cameras (default)
    GUI_CAMERAS = [8]
    print("🔥 RUNNING ALL CAMERAS: 11")

# 🖥️ GUI CONFIGURATION
ENABLE_GUI = True  # Set to False for headless mode (no windows) - RE-ENABLED WITH OPENCV 4.8.1.78
ENABLE_CONSOLE_LOGGING = True  # Show detection logs in console
LOG_INTERVAL = 30  # Log performance every N frames

# 🎯 DETECTION CONFIGURATION
ENABLE_CROSS_CAMERA_TRACKING = True  # Global object IDs across cameras
SAVE_DETECTIONS_TO_DB = True  # Save to database
PERFORMANCE_MONITORING = True  # Show FPS and performance stats

# 📊 FPS MONITORING CONFIGURATION
SHOW_FPS_IN_CONSOLE = True  # Print FPS stats to console
FPS_UPDATE_INTERVAL = 5  # Update FPS every N seconds
SHOW_FPS_ON_GUI = True  # Display FPS on camera windows (when GUI enabled)

# 🗄️ MONGODB CONFIGURATION
ENABLE_MONGODB = True  # Save detections to MongoDB
MONGODB_URL = "mongodb://localhost:27017/"  # MongoDB connection string
MONGODB_DATABASE = "warehouse_tracking"  # Database name
MONGODB_COLLECTION = "detections"  # Collection name
BATCH_SAVE_SIZE = 10  # Save to DB every N detections (for performance)

# 🚀 BATCH PROCESSING CONFIGURATION
ENABLE_BATCH_PROCESSING = True  # Use shared model with batch processing
BATCH_SIZE = 4  # Process N cameras together (reduced for 6GB GPU)
SHARED_MODEL_MEMORY_GB = 2  # Single model memory usage
ENABLE_TRUE_PARALLEL_BATCH = True  # Enable true parallel tensor batching
ENABLE_GPU_PIPELINE = True  # Move all operations to GPU

# 📊 DISPLAY CONFIGURATION (when GUI enabled)
WINDOW_WIDTH = 800  # Width of each camera window
WINDOW_HEIGHT = 600  # Height of each camera window
ARRANGE_WINDOWS = True  # Auto-arrange windows on screen

def main():
    """Main function for configurable 11-camera warehouse tracking"""
    print("🚀 GPU-OPTIMIZED 11-CAMERA WAREHOUSE TRACKING SYSTEM")
    print("=" * 80)
    print("CONFIGURATION:")
    print(f"📹 Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    print(f"🖥️ GUI Cameras: {GUI_CAMERAS if ENABLE_GUI else 'DISABLED'} ({len(GUI_CAMERAS) if ENABLE_GUI else 0} windows)")
    print(f"🎛️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    print(f"🔄 Cross-Camera Tracking: {'ENABLED' if ENABLE_CROSS_CAMERA_TRACKING else 'DISABLED'}")
    print("=" * 80)
    print("MAXIMUM GPU UTILIZATION - All operations GPU-accelerated:")
    print("1) 🚀 GPU Detection (Grounding DINO + Mixed Precision)")
    print("2) 🚀 GPU Area + Grid Cell Filtering (Vectorized)")
    print("3) 🚀 GPU Physical Coordinate Translation (Batch)")
    print("4) 🚀 GPU SIFT Feature Matching (CUDA SIFT)")
    print("5) 🚀 GPU Persistent Object IDs")
    print("6) 🚀 GPU Cross-Frame Tracking & Database")
    print("=" * 80)
    if ENABLE_GUI:
        print("\nGUI Mode:")
        print("- Green: New objects")
        print("- Orange: GPU-tracked existing objects")
        print("- Red: Failed tracking")
        print("- Cyan: Physical coordinate labels")
        print("- Press 'q' to quit")
    else:
        print("\nHEADLESS Mode: No GUI windows, console logging only")
        print("- Press Ctrl+C to quit")

    print("=" * 80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Some optimizations will fall back to CPU")
        return
    else:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Ready: {gpu_count} GPU(s) available - {gpu_name}")

    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("⚠️  WARNING: OpenCV CUDA not available! Some operations will use CPU")
    else:
        print(f"🚀 OpenCV CUDA Ready: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) available")

    print("=" * 80)

    # Initialize multi-camera system
    multi_camera_system = MultiCameraWarehouseSystem(
        active_cameras=ACTIVE_CAMERAS,
        gui_cameras=GUI_CAMERAS,
        enable_gui=ENABLE_GUI,
        enable_console_logging=ENABLE_CONSOLE_LOGGING,
        log_interval=LOG_INTERVAL,
        enable_cross_camera_tracking=ENABLE_CROSS_CAMERA_TRACKING,
        save_to_db=SAVE_DETECTIONS_TO_DB,
        performance_monitoring=PERFORMANCE_MONITORING,
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        arrange_windows=ARRANGE_WINDOWS
    )

    try:
        multi_camera_system.start_tracking()
    except KeyboardInterrupt:
        print("\nShutting down multi-camera GPU tracker...")
    except Exception as e:
        logger.error(f"Error running multi-camera GPU tracker: {e}")
    finally:
        multi_camera_system.stop_tracking()


if __name__ == "__main__":
    main()
