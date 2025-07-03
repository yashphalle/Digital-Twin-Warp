#!/usr/bin/env python3
"""
Shared Model Architecture for 11-Camera System
Optimized for 6GB GPU - Single model serves all cameras
"""

import cv2
import numpy as np
import torch
import time
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

logger = logging.getLogger(__name__)

class SharedModelDetector:
    """Single detection model shared across all cameras"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_lock = threading.Lock()  # Prevent concurrent access
        
        # Batch processing configuration
        self.max_batch_size = 4  # Process 4 cameras at once
        self.batch_timeout = 0.1  # 100ms timeout for batching
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize single Grounding DINO model"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Enable mixed precision for memory efficiency
            if self.device.type == 'cuda':
                self.model = self.model.half()
                
            logger.info(f"✅ Shared model loaded on {self.device} (Memory: ~2GB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load shared model: {e}")
            
    def detect_batch(self, frames_dict: Dict[int, np.ndarray], prompt: str = "pallet") -> Dict[int, List[Dict]]:
        """
        Batch detection for multiple cameras
        frames_dict: {camera_id: frame}
        Returns: {camera_id: [detections]}
        """
        if not self.model:
            return {cam_id: [] for cam_id in frames_dict.keys()}
            
        with self.model_lock:
            try:
                # Prepare batch
                camera_ids = list(frames_dict.keys())
                frames = list(frames_dict.values())
                
                # Convert frames for model input
                from PIL import Image
                pil_images = []
                for frame in frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb_frame))
                
                # Batch processing
                inputs = self.processor(images=pil_images, text=[prompt] * len(pil_images), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # GPU inference
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                
                # Process results for each camera
                results = {}
                for i, cam_id in enumerate(camera_ids):
                    frame_height, frame_width = frames[i].shape[:2]
                    detections = self._process_single_result(outputs, i, frame_width, frame_height)
                    results[cam_id] = detections
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Batch detection failed: {e}")
                return {cam_id: [] for cam_id in frames_dict.keys()}
    
    def _process_single_result(self, outputs, batch_index: int, frame_width: int, frame_height: int) -> List[Dict]:
        """Process detection results for a single frame in the batch"""
        try:
            # Extract results for this batch index
            scores = outputs.logits[batch_index].sigmoid().cpu()
            boxes = outputs.pred_boxes[batch_index].cpu()
            
            # Filter by confidence
            confidence_threshold = 0.1
            keep = scores.max(dim=-1)[0] > confidence_threshold
            
            if not keep.any():
                return []
            
            # Convert to detection format
            detections = []
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep].max(dim=-1)[0]
            
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
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'area': (x2 - x1) * (y2 - y1)
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Failed to process detection result: {e}")
            return []

class CameraBatchProcessor:
    """Manages batching of camera frames for efficient processing"""
    
    def __init__(self, shared_detector: SharedModelDetector, camera_ids: List[int]):
        self.shared_detector = shared_detector
        self.camera_ids = camera_ids
        self.batch_size = 4  # Process 4 cameras at once
        
        # Create camera groups
        self.camera_groups = []
        for i in range(0, len(camera_ids), self.batch_size):
            group = camera_ids[i:i + self.batch_size]
            self.camera_groups.append(group)
            
        logger.info(f"📊 Created {len(self.camera_groups)} camera groups: {self.camera_groups}")
    
    def process_all_cameras(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, List[Dict]]:
        """Process all cameras in batches"""
        all_results = {}
        
        for group in self.camera_groups:
            # Create batch for this group
            group_frames = {cam_id: frames_dict[cam_id] for cam_id in group if cam_id in frames_dict}
            
            if group_frames:
                # Process this batch
                group_results = self.shared_detector.detect_batch(group_frames)
                all_results.update(group_results)
        
        return all_results

class OptimizedMultiCameraSystem:
    """Optimized multi-camera system using shared model architecture"""
    
    def __init__(self, camera_ids: List[int]):
        self.camera_ids = camera_ids
        
        # Initialize shared components
        self.shared_detector = SharedModelDetector()
        self.batch_processor = CameraBatchProcessor(self.shared_detector, camera_ids)
        
        # Camera connections (simplified)
        self.cameras = {}
        self.initialize_cameras()
        
        logger.info(f"🚀 Optimized system initialized for {len(camera_ids)} cameras")
        logger.info(f"💾 GPU Memory Usage: ~2GB (vs {len(camera_ids) * 2}GB with separate models)")
    
    def initialize_cameras(self):
        """Initialize camera connections"""
        from cv.configs.config import Config
        
        for cam_id in self.camera_ids:
            try:
                rtsp_url = Config.RTSP_CAMERA_URLS.get(cam_id)
                if rtsp_url:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cameras[cam_id] = cap
                    logger.info(f"✅ Camera {cam_id} connected")
            except Exception as e:
                logger.error(f"❌ Failed to connect Camera {cam_id}: {e}")
    
    def process_frame_cycle(self) -> Dict[int, List[Dict]]:
        """Process one frame from each camera"""
        frames = {}
        
        # Capture frames from all cameras
        for cam_id, cap in self.cameras.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = frame
        
        # Batch process all frames
        if frames:
            results = self.batch_processor.process_all_cameras(frames)
            return results
        
        return {}
    
    def run(self):
        """Main execution loop"""
        logger.info("🚀 Starting optimized multi-camera processing...")
        
        try:
            while True:
                start_time = time.time()
                
                # Process all cameras
                results = self.process_frame_cycle()
                
                # Calculate performance
                processing_time = time.time() - start_time
                fps = len(self.camera_ids) / processing_time if processing_time > 0 else 0
                
                # Log performance every 30 cycles
                if hasattr(self, 'cycle_count'):
                    self.cycle_count += 1
                else:
                    self.cycle_count = 1
                    
                if self.cycle_count % 30 == 0:
                    total_detections = sum(len(dets) for dets in results.values())
                    logger.info(f"📊 Cycle {self.cycle_count}: {fps:.1f} FPS, {total_detections} detections, {processing_time:.2f}s")
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping optimized system...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        for cam_id, cap in self.cameras.items():
            cap.release()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    # Test with all 11 cameras
    camera_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    system = OptimizedMultiCameraSystem(camera_ids)
    system.run()
