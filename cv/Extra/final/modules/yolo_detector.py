#!/usr/bin/env python3
"""
YOLOv8 Pallet Detector
Replacement for Grounding DINO with batch processing support
"""

import logging
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

# YOLOv8PalletDetector - Uses model's native class names

class YOLOv8PalletDetector:
    """
    YOLOv8-based pallet detector with batch processing support
    Drop-in replacement for GroundingDINODetector
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda:0", conf_threshold: float = 0.5):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        # Initialize model
        self._load_model()
        
        # Performance tracking
        self.inference_times = []
        
        logger.info(f"✅ YOLOv8 Pallet Detector initialized")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Confidence: {conf_threshold}")
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            logger.info(f"🔄 Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move to specified device
            if torch.cuda.is_available() and "cuda" in self.device:
                self.model.to(self.device)
                logger.info(f"🚀 YOLOv8 model loaded on {self.device}")
            else:
                logger.warning(f"⚠️ CUDA not available, using CPU")
                self.device = "cpu"
                
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            raise
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect pallets in a single frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries compatible with existing system
        """
        start_time = time.time()

        # Debug frame information
        if frame is None:
            logger.error("❌ YOLOv8: Received None frame")
            return []

        try:
            # Optional: Enhance image for better detection
            enhanced_frame = self._enhance_frame_for_detection(frame)

            # Run YOLOv8 inference
            results = self.model(enhanced_frame, device=self.device, conf=self.conf_threshold, verbose=False)

            # Convert to standard format
            detections = self._convert_results_to_standard_format(results[0], frame.shape)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only recent times for FPS calculation
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]

            # Log performance
            fps = 1.0 / inference_time if inference_time > 0 else 0
            logger.info(f"🎯 YOLOv8 FPS: {fps:.2f} (Time: {inference_time:.3f}s)")

            return detections

        except Exception as e:
            logger.error(f"❌ YOLOv8 detection error: {e}")
            return []

    def detect_pallets_with_tracking(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track pallets using YOLOv8 built-in tracking

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries with tracking IDs
        """
        start_time = time.time()

        if frame is None:
            logger.error("❌ YOLOv8 Tracking: Received None frame")
            return []

        try:
            # Optional: Enhance image for better detection
            enhanced_frame = self._enhance_frame_for_detection(frame)

            # Run YOLOv8 inference with tracking
            results = self.model.track(
                enhanced_frame,
                device=self.device,
                conf=self.conf_threshold,
                tracker="bytetrack.yaml",  # Built-in ByteTrack tracker
                persist=True,  # Maintain track IDs across frames
                verbose=False
            )

            # Convert tracking results to standard format
            detections = self._convert_tracking_results_to_standard_format(results[0], frame.shape)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only recent times for FPS calculation
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]

            # Log performance
            fps = 1.0 / inference_time if inference_time > 0 else 0
            logger.info(f"🎯 YOLOv8 Tracking FPS: {fps:.2f} (Time: {inference_time:.3f}s)")

            return detections

        except Exception as e:
            logger.error(f"❌ YOLOv8 tracking error: {e}")
            # Fallback to regular detection
            return self.detect_pallets(frame)
    
    def detect_pallets_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect pallets in multiple frames (batch processing)
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        start_time = time.time()
        
        try:
            # Run batch inference
            results = self.model(frames, device=self.device, conf=self.conf_threshold, verbose=False)
            
            # Convert all results
            all_detections = []
            for i, result in enumerate(results):
                detections = self._convert_results_to_standard_format(result, frames[i].shape)
                all_detections.append(detections)
            
            # Log batch performance
            batch_time = time.time() - start_time
            fps_per_frame = len(frames) / batch_time if batch_time > 0 else 0
            logger.info(f"🎯 YOLOv8 Batch FPS: {fps_per_frame:.2f} ({len(frames)} frames in {batch_time:.3f}s)")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 batch detection error: {e}")
            return [[] for _ in frames]
    
    def _convert_results_to_standard_format(self, result, frame_shape) -> List[Dict]:
        """
        Convert YOLOv8 results to standard detection format
        Compatible with existing system expectations
        """
        detections = []
        
        if result.boxes is None:
            return detections

        frame_height, frame_width = frame_shape[:2]

        # Log total detections found
        total_detections = len(result.boxes)
        logger.info(f"🔍 YOLOv8: Found {total_detections} total detections before filtering")
        
        for box in result.boxes:
            # Extract box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate area and center
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Class information
            class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0

            # Get detection info - use model's native class names
            confidence = float(box.conf[0].cpu().numpy())

            # Get class name from model (not hardcoded COCO classes)
            if hasattr(self.model, 'names') and class_id in self.model.names:
                class_name = self.model.names[class_id]
            else:
                class_name = f"class_{class_id}"

            # Create detection dictionary (compatible with existing system)
            detection = {
                # Bounding box (existing system expects INTEGER coordinates)
                'bbox': [int(x1), int(y1), int(x2), int(y2)],

                # ✅ REQUIRED: 4-corner coordinates (same format as Grounding DINO)
                'corners': [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]],

                # Confidence score
                'confidence': float(box.conf[0].cpu().numpy()),

                # Geometric properties
                'area': float(area),

                # ✅ REQUIRED: Shape type (same as Grounding DINO)
                'shape_type': 'quadrangle',

                # YOLOv8-specific fields (for debugging/analysis)
                'width': float(width),
                'height': float(height),
                'center_x': float(center_x),
                'center_y': float(center_y),
                'class_id': class_id,
                'class_name': class_name,  # Use actual model class name

                # Normalized coordinates (for compatibility)
                'normalized_bbox': [
                    float(x1 / frame_width),
                    float(y1 / frame_height),
                    float(x2 / frame_width),
                    float(y2 / frame_height)
                ],

                # Additional metadata
                'detection_method': 'yolov8',
                'model_version': str(self.model.model.yaml.get('version', 'unknown'))
            }
            
            detections.append(detection)

        # Log detection summary
        logger.info(f"🔍 YOLOv8 SUMMARY: {total_detections} total → {len(detections)} accepted (filtered {total_detections - len(detections)})")
        if len(detections) > 0:
            class_counts = {}
            for det in detections:
                class_id = det['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            logger.info(f"🎯 Accepted classes: {class_counts}")

        return detections

    def _enhance_frame_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better detection (optional preprocessing)"""
        try:
            # Convert to RGB if needed (YOLOv8 expects RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assume BGR (OpenCV default) -> convert to RGB
                enhanced = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                enhanced = frame.copy()

            # Optional: Enhance contrast and brightness for warehouse conditions
            # enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

            return enhanced

        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}, using original frame")
            return frame

    def _convert_tracking_results_to_standard_format(self, result, frame_shape) -> List[Dict]:
        """
        Convert YOLOv8 tracking results to standard detection format with track IDs

        Args:
            result: YOLOv8 tracking result object
            frame_shape: Shape of the input frame (height, width, channels)

        Returns:
            List of detection dictionaries with tracking information
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Check if tracking IDs are available
        has_track_ids = hasattr(result.boxes, 'id') and result.boxes.id is not None

        for i, box in enumerate(result.boxes):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate geometric properties
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Get confidence and class info
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0

            # Get class name from model (not hardcoded COCO classes)
            if hasattr(self.model, 'names') and class_id in self.model.names:
                class_name = self.model.names[class_id]
            else:
                class_name = f"class_{class_id}"

            # Get tracking ID if available
            track_id = None
            if has_track_ids:
                track_id = int(box.id[0].cpu().numpy())

            # Create detection dictionary with tracking info
            detection = {
                # Bounding box (existing system expects INTEGER coordinates)
                'bbox': [int(x1), int(y1), int(x2), int(y2)],

                # 4-corner coordinates (same format as Grounding DINO)
                'corners': [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]],

                # Confidence score
                'confidence': confidence,

                # Geometric properties
                'area': float(area),
                'width': float(width),
                'height': float(height),
                'center_x': float(center_x),
                'center_y': float(center_y),

                # Shape type (required by existing system)
                'shape_type': 'quadrangle',

                # YOLOv8 class information
                'class_id': class_id,
                'class_name': class_name,  # Use actual model class name

                # TRACKING INFORMATION (NEW)
                'track_id': track_id,
                'has_track_id': has_track_ids,
                'tracking_method': 'yolov8_bytetrack',

                # Global ID fields (compatible with existing system)
                'global_id': track_id if track_id is not None else -1,
                'tracking_status': 'tracked' if track_id is not None else 'new',
                'similarity_score': 1.0 if track_id is not None else 0.0,

                # Detection metadata
                'detection_method': 'yolov8_tracking',
                'frame_shape': frame_shape
            }

            detections.append(detection)

        return detections

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'YOLOv8',
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.conf_threshold,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
