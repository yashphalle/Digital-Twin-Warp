import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class GPUBatchProcessorDetectionOnly:
    """
    GPU batch processor for YOLOv8 detection only (no tracking)
    Optimized for speed testing and now supports high-throughput batching.
    """
    
    def __init__(self, 
                 model_path: str = 'custom_yolo.pt',
                 device: str = 'cuda:0',
                 active_cameras: List[int] = None,
                 confidence: float = 0.5,
                 use_fp16: bool = True):
        """
        Initialize GPU batch processor for detection only
        """
        self.device = device
        self.confidence = confidence
        self.active_cameras = active_cameras or []
        
        logger.info(f"Loading YOLOv8 model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            
            self.model.fuse()
            
            if use_fp16 and torch.cuda.is_available():
                self.model.half()
                logger.info("✅ Model converted to FP16 for faster inference")
            
            logger.info(f"✅ Model loaded on {device}")
            
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
        self.total_batches = 0
        self.total_inference_time = 0
        self.frames_processed = 0
        self.detection_count = 0
    
    def process_batch(self, frame_batch: Dict[int, List[np.ndarray]]) -> tuple:
        """
        MODIFIED: Get batches of frames and run detection.
        """
        start_time = time.time()
        
        if not frame_batch:
            return {}, {}
        
        detections_by_camera = {}
        processed_frames_for_gui = {}

        for cam_id, frames in frame_batch.items():
            if not frames or cam_id not in self.active_cameras:
                continue

            try:
                results = self.model(
                    frames, 
                    conf=self.confidence,
                    verbose=False,
                    stream=False
                )
                
                self.frames_processed += len(frames)
                
                # Parse results for the last frame in the batch for this camera
                last_result = results[-1]
                detections = []
                if last_result.boxes is not None and len(last_result.boxes) > 0:
                    boxes = last_result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        class_name = self.model.names.get(cls, f'class_{cls}')
                        
                        detection = {
                            'camera_id': cam_id,
                            'bbox': bbox.tolist(),
                            'confidence': float(conf),
                            'class': class_name,
                            'track_id': None,
                        }
                        detections.append(detection)
                        self.detection_count += 1
                
                detections_by_camera[cam_id] = detections
                processed_frames_for_gui[cam_id] = frames[-1]

            except Exception as e:
                logger.error(f"❌ Inference failed for camera {cam_id}: {e}")
                detections_by_camera[cam_id] = []
                processed_frames_for_gui[cam_id] = frames[-1]

        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.total_batches += 1
            
        return processed_frames_for_gui, detections_by_camera
    
    def get_stats(self) -> Dict:
        """Get processing statistics, corrected for batch processing."""
        if self.total_batches == 0 or self.total_inference_time == 0:
            return {
                'total_batches': 0,
                'total_frames': 0,
                'total_detections': 0,
                'avg_inference_time_ms': 0.0,
                'avg_ms_per_frame': 0.0,
                'avg_fps': 0.0,
                'throughput': 0.0
            }
        
        avg_inference_time_ms = (self.total_inference_time / self.total_batches) * 1000
        avg_ms_per_frame = (self.total_inference_time / self.frames_processed) * 1000
        avg_fps = 1000 / avg_ms_per_frame if avg_ms_per_frame > 0 else 0
        throughput = self.frames_processed / self.total_inference_time
            
        return {
            'total_batches': self.total_batches,
            'total_frames': self.frames_processed,
            'total_detections': self.detection_count,
            'avg_inference_time_ms': avg_inference_time_ms,
            'avg_ms_per_frame': avg_ms_per_frame,
            'avg_fps': avg_fps,
            'throughput': throughput
        }
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info("GPU resources cleaned up")
