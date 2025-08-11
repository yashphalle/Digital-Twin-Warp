#!/usr/bin/env python3
"""
Grounding DINO Video Processor - Single Worker
Based on your production implementation for multi-threading compatibility
"""

import cv2
import numpy as np
import logging
import torch
import time
import threading
import argparse
from typing import List, Dict
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroundingDINODetector:
    """Production-ready Grounding DINO detector optimized for multi-threading"""
    
    def __init__(self, gpu_id=None, confidence_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.prompts = ["pallet wrapped in plastic", "stack of goods on pallet", "person", "car", "box"]
        self.current_prompt_index = 0
        self.current_prompt = self.prompts[0]
        
        # Device selection with explicit GPU assignment
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            device_name = torch.cuda.get_device_name(gpu_id)
            memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            logger.info(f"🚀 Using GPU {gpu_id}: {device_name}")
            logger.info(f"📊 GPU Memory: {memory_gb:.1f}GB")
        elif torch.cuda.is_available():
            # Auto-select best NVIDIA GPU
            nvidia_device = 0
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                if "NVIDIA" in device_name:
                    nvidia_device = i
                    break
            self.device = torch.device(f"cuda:{nvidia_device}")
            torch.cuda.set_device(nvidia_device)
            logger.info(f"🚀 Auto-selected GPU {nvidia_device}: {torch.cuda.get_device_name(nvidia_device)}")
        else:
            self.device = torch.device("cpu")
            logger.info("⚠️ GPU not available, using CPU")

        logger.info(f"🔍 Initializing detector on {self.device}")
        
        # Thread ID for unique temp files
        self.thread_id = threading.current_thread().ident
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Grounding DINO model using HuggingFace transformers"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            logger.info("📥 Loading AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("✅ AutoProcessor loaded successfully")
            
            logger.info("📥 Loading Grounding DINO model...")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Grounding DINO model loaded on {self.device}")

            # Log GPU memory usage
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"📊 GPU Memory allocated: {memory_allocated:.2f}GB")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Grounding DINO: {e}")
            self.processor = None
            self.model = None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame with precise timing"""
        if self.model is None or self.processor is None:
            return []
        
        try:
            # Convert frame to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=self.current_prompt,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Precise inference timing
            inference_start = time.time()

            # Run inference with mixed precision if GPU available
            if self.device.type == 'cuda':
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_fps = 1.0 / inference_time
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )

            # Clear GPU cache to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Convert to structured detection format
            detections = []
            if results and len(results) > 0:
                boxes = results[0]["boxes"].cpu().numpy()
                scores = results[0]["scores"].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        'confidence': float(score),
                        'area': area,
                        'shape_type': 'quadrangle'
                    }
                    detections.append(detection)
            
            return detections, inference_fps
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return [], 0.0

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{self.current_prompt.split('.')[0].strip()}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame

    def next_prompt(self):
        """Switch to next detection prompt"""
        self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)
        self.current_prompt = self.prompts[self.current_prompt_index]
        logger.info(f"🔄 Switched to prompt: '{self.current_prompt}'")

def process_video(video_path: str, gpu_id=None, no_gui=False, confidence=0.1):
    """Process single video with Grounding DINO detection"""
    
    # Initialize detector
    detector = GroundingDINODetector(gpu_id=gpu_id, confidence_threshold=confidence)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"📹 Video: {video_path}")
    logger.info(f"📊 Total frames: {total_frames}, Original FPS: {video_fps:.2f}")
    logger.info(f"🎯 Current prompt: '{detector.current_prompt}'")
    logger.info("🔄 Press 'n' for next prompt, 'q' to quit")
    
    # Processing variables
    frame_count = 0
    start_time = time.time()
    detection_count = 0
    total_inference_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_start = time.time()
        
        # Detect objects
        detections, inference_fps = detector.detect_objects(frame)
        total_inference_fps += inference_fps
        
        # Draw detections
        if detections:
            detection_count += len(detections)
            annotated_frame = detector.draw_detections(frame, detections)
            logger.info(f"🎯 Frame {frame_count}: Found {len(detections)} objects")
        else:
            annotated_frame = frame
        
        # Calculate and display FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            processing_fps = frame_count / elapsed
            avg_inference_fps = total_inference_fps / frame_count
            
            logger.info(f"📊 Frame {frame_count}/{total_frames}")
            logger.info(f"⚡ Processing FPS: {processing_fps:.2f}")
            logger.info(f"🧠 Avg Inference FPS: {avg_inference_fps:.2f}")
            logger.info(f"🎯 Total detections so far: {detection_count}")
        
        # Display frame (if GUI enabled)
        if not no_gui:
            cv2.imshow('Grounding DINO Detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                detector.next_prompt()
    
    # Final statistics
    total_time = time.time() - start_time
    final_fps = frame_count / total_time
    avg_inference_fps = total_inference_fps / frame_count if frame_count > 0 else 0
    
    logger.info("🏁 Processing Complete!")
    logger.info(f"📊 Processed {frame_count} frames in {total_time:.2f}s")
    logger.info(f"⚡ Average Processing FPS: {final_fps:.2f}")
    logger.info(f"🧠 Average Inference FPS: {avg_inference_fps:.2f}")
    logger.info(f"🎯 Total detections: {detection_count}")
    
    cap.release()
    if not no_gui:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Grounding DINO Video Processor - Single Worker')
    parser.add_argument('--video', '-v', required=True, help='Input video path')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID to use')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    parser.add_argument('--confidence', type=float, default=0.1, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Grounding DINO Video Processor")
    process_video(args.video, args.gpu_id, args.no_gui, args.confidence)

if __name__ == "__main__":
    main()