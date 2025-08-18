# cv/GPU/pipelines/gpu_processor_detection_only.py

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class FastGlobalIDManager:
    """Fast global ID management for camera-prefixed tracking IDs"""

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.id_base = camera_id * 1000  # Camera 8 -> 8000
        self.next_id = self.id_base + 1  # Start from 8001
        self.yolo_to_global = {}         # Map YOLO track ID to global ID
        self.track_ages = {}             # Track age in frames

        logger.info(f"GlobalIDManager initialized for Camera {camera_id}, IDs: {self.id_base + 1}+")

    def get_global_id(self, yolo_track_id: int) -> int:
        """Get global ID for YOLO track ID, create new if needed"""
        if yolo_track_id not in self.yolo_to_global:
            # New track - assign new global ID
            global_id = self.next_id
            self.yolo_to_global[yolo_track_id] = global_id
            self.track_ages[global_id] = 0
            self.next_id += 1
            logger.info(f"Camera {self.camera_id}: New track {yolo_track_id} -> Global ID {global_id}")

        global_id = self.yolo_to_global[yolo_track_id]
        self.track_ages[global_id] += 1  # Increment age
        return global_id

    def get_track_age(self, global_id: int) -> int:
        """Get track age in frames"""
        return self.track_ages.get(global_id, 0)

    def cleanup_lost_tracks(self, active_yolo_ids: List[int]):
        """Remove mappings for tracks that are no longer active"""
        # Find YOLO IDs that are no longer active
        lost_yolo_ids = []
        for yolo_id in list(self.yolo_to_global.keys()):
            if yolo_id not in active_yolo_ids:
                lost_yolo_ids.append(yolo_id)

        # Remove lost tracks
        for yolo_id in lost_yolo_ids:
            global_id = self.yolo_to_global.pop(yolo_id)
            self.track_ages.pop(global_id, None)
            logger.debug(f"Camera {self.camera_id}: Removed lost track {yolo_id} (Global ID {global_id})")

    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.yolo_to_global),
            'next_id': self.next_id,
            'total_tracks_created': self.next_id - self.id_base - 1
        }

class GPUBatchProcessorDetectionOnly:
    """
    GPU batch processor for YOLOv8 detection only (no tracking)
    Optimized for speed testing
    """
    
    def __init__(self, 
                 model_path: str = 'custom_yolo.pt',
                 device: str = 'cuda:0',
                 active_cameras: List[int] = None,
                 confidence: float = 0.5,
                 use_fp16: bool = True):
        """
        Initialize GPU batch processor for detection only
        
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
        
        # Load YOLOv8 model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            
            # Optimize model for speed
            self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers
            
            if use_fp16 and torch.cuda.is_available():
                self.model.half()  # Convert to FP16
                logger.info("✅ Model converted to FP16 for faster inference")
            
            logger.info(f"✅ Model loaded on {device}")
            
            # Log model info
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
        # Track processing stats
        self.total_batches = 0
        self.total_inference_time = 0
        self.frames_processed = 0
        self.detection_count = 0
        
    def process_batch(self, ring_buffer) -> tuple:
        """
        Get frames from ring buffer and run batch detection (no tracking)

        Args:
            ring_buffer: RingBuffer instance with camera frames

        Returns:
            Tuple of (processed_frames_dict, detections_dict)
            - processed_frames_dict: Dictionary mapping camera_id to processed frame
            - detections_dict: Dictionary mapping camera_id to list of detections
        """
        start_time = time.time()
        
        # Get latest frames from ring buffer
        batch_frames = ring_buffer.get_batch(max_age=1)  # 200ms max age
        
        # Filter to only active cameras if specified
        if self.active_cameras:
            batch_frames = {cam_id: frame 
                          for cam_id, frame in batch_frames.items() 
                          if cam_id in self.active_cameras}
        
        if not batch_frames:
            logger.warning("No frames available for processing")
            return {}, {}
            
        # Prepare batch for inference
        camera_ids = list(batch_frames.keys())
        frames = list(batch_frames.values())
        
        logger.debug(f"Processing batch of {len(frames)} frames from cameras: {camera_ids}")
        
        # Run YOLOv8 detection (no tracking)
        try:
            # Simple detection - much faster than tracking
            results = self.model(
                frames, 
                conf=self.confidence,
                verbose=False,
                stream=False  # Get all results at once
            )
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.total_batches += 1
            self.frames_processed += len(frames)
            
            # Log performance
            ms_per_frame = (inference_time * 1000) / len(frames)
            fps = len(frames) / inference_time
            logger.info(f"🚀 Batch detection: {len(frames)} frames in {inference_time*1000:.1f}ms "
                       f"({ms_per_frame:.1f}ms/frame, {fps:.1f} FPS)")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return {}, {}
            
        # Parse results into our format
        detections_by_camera = {}
        
        for cam_id, result in zip(camera_ids, results):
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Extract detection info
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Map class ID to name
                    class_name = self.model.names.get(cls, f'class_{cls}')
                    
                    detection = {
                        'camera_id': cam_id,
                        'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class': class_name,
                        'track_id': None,  # No tracking
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                    self.detection_count += 1
                    
                logger.debug(f"Camera {cam_id}: {len(detections)} detections")
            
            detections_by_camera[cam_id] = detections
            
        # Return both the processed frames and detections for perfect sync
        return batch_frames, detections_by_camera
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        avg_inference_time = 0
        avg_fps = 0
        
        if self.total_batches > 0:
            avg_inference_time = self.total_inference_time / self.total_batches
            avg_fps = self.frames_processed / self.total_inference_time if self.total_inference_time > 0 else 0
            
        return {
            'total_batches': self.total_batches,
            'total_frames': self.frames_processed,
            'total_detections': self.detection_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'avg_ms_per_frame': (avg_inference_time * 1000 / max(len(self.active_cameras), 1)) if self.total_batches > 0 else 0,
            'avg_fps': avg_fps
        }
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info("GPU resources cleaned up")