# cv/GPU/pipelines/gpu_processor.py

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class GPUBatchProcessor:
    """
    GPU batch processor for YOLOv8 inference on multiple camera frames
    Uses model.track() for built-in object tracking
    """
    
    def __init__(self, 
                 model_path: str = 'custom_yolo.pt',
                 device: str = 'cuda:0',
                 active_cameras: List[int] = None,
                 confidence: float = 0.5):
        """
        Initialize GPU batch processor
        
        Args:
            model_path: Path to custom YOLOv8 model
            device: CUDA device to use
            active_cameras: List of camera IDs to process
            confidence: Detection confidence threshold
        """
        self.device = device
        self.confidence = confidence
        self.active_cameras = active_cameras or []
        
        # Load YOLOv8 model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
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
        
    def process_batch(self, ring_buffer) -> Dict[int, List[Dict]]:
        """
        Get frames from ring buffer and run batch inference
        
        Args:
            ring_buffer: RingBuffer instance with camera frames
            
        Returns:
            Dictionary mapping camera_id to list of detections
        """
        start_time = time.time()
        
        # Get latest frames from ring buffer
        batch_frames = ring_buffer.get_batch(max_age=2)  # 200ms max age
        
        # Filter to only active cameras if specified
        if self.active_cameras:
            batch_frames = {cam_id: frame 
                          for cam_id, frame in batch_frames.items() 
                          if cam_id in self.active_cameras}
        
        if not batch_frames:
            logger.warning("No frames available for processing")
            return {}
            
        # Prepare batch for inference
        camera_ids = list(batch_frames.keys())
        frames = list(batch_frames.values())
        
        logger.debug(f"Processing batch of {len(frames)} frames from cameras: {camera_ids}")
        
        # Run YOLOv8 tracking (maintains object IDs across frames)
        try:
            # Use persist=True to maintain tracks across batches
            results = self.model.track(
                frames, 
                conf=self.confidence,
                persist=True,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.total_batches += 1
            self.frames_processed += len(frames)
            
            logger.info(f"Batch inference completed: {len(frames)} frames in {inference_time*1000:.1f}ms "
                       f"({inference_time*1000/len(frames):.1f}ms per frame)")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return {}
            
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
                    
                    # Get track ID if available
                    track_id = None
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())
                    
                    # Map class ID to name
                    class_name = self.model.names.get(cls, f'class_{cls}')
                    
                    detection = {
                        'camera_id': cam_id,
                        'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class': class_name,
                        'track_id': track_id,
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                    
                logger.debug(f"Camera {cam_id}: {len(detections)} detections")
            
            detections_by_camera[cam_id] = detections
            
        return detections_by_camera
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        avg_inference_time = 0
        if self.total_batches > 0:
            avg_inference_time = self.total_inference_time / self.total_batches
            
        return {
            'total_batches': self.total_batches,
            'total_frames': self.frames_processed,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'avg_fps': self.frames_processed / self.total_inference_time if self.total_inference_time > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info("GPU resources cleaned up")