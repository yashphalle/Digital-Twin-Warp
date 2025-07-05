#!/usr/bin/env python3
"""
Detection Module - MOST CRITICAL COMPONENT
Hybrid pallet detector: GPU for Grounding DINO inference, CPU for post-processing
Extracted from main.py for modular architecture - HANDLE WITH EXTREME CARE
"""

import cv2
import numpy as np
import logging
import torch
from typing import List, Dict

logger = logging.getLogger(__name__)

class CPUSimplePalletDetector:
    """Hybrid pallet detector: GPU for Grounding DINO inference, CPU for post-processing"""
    
    def __init__(self):
        self.confidence_threshold = 0.1
        self.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.current_prompt_index = 0
        self.current_prompt = self.sample_prompts[0]
        
        # Initialize detection with GPU for Grounding DINO if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"🚀 Using GPU for Grounding DINO: {torch.cuda.get_device_name()}")
            logger.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device("cpu")
            logger.info("⚠️ GPU not available, using CPU for Grounding DINO")

        logger.info(f"🔍 Initializing pallet detector on {self.device}")
        
        # Initialize Grounding DINO model
        self._initialize_grounding_dino()
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model for GPU inference (CPU fallback if GPU unavailable)"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("✅ AutoProcessor loaded successfully")
            
            # Load model and move to selected device (GPU preferred)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Grounding DINO model loaded on {self.device}")

            # Log GPU memory usage if using GPU
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"📊 GPU Memory allocated: {memory_allocated:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to initialize Grounding DINO: {e}")
            self.processor = None
            self.model = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """CPU-based pallet detection using same method as combined filtering"""
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
            
            # Move inputs to selected device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # GPU/CPU inference with automatic mixed precision if GPU available
            if self.device.type == 'cuda':
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            # Process results using SAME METHOD as combined filtering
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )

            # Clear GPU cache if using GPU to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Convert to detection format (SAME as combined filtering)
            detections = []
            if results and len(results) > 0:
                boxes = results[0]["boxes"].cpu().numpy()
                scores = results[0]["scores"].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # 4 corners clockwise from top-left
                        'confidence': float(score),
                        'area': area,
                        'shape_type': 'quadrangle'
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"CPU detection failed: {e}")
            return []

    def next_prompt(self):
        """Switch to next prompt"""
        self.current_prompt_index = (self.current_prompt_index + 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
    
    def previous_prompt(self):
        """Switch to previous prompt"""
        self.current_prompt_index = (self.current_prompt_index - 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
