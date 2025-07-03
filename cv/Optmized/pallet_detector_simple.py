import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

class SimplePalletDetector:
    """
    Simple pallet detection using Grounding DINO only
    Interactive prompt and threshold tuning
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing simple pallet detector on {self.device}")
        
        # Grounding DINO model
        self.grounding_dino = None
        
        # Detection parameters - Lower threshold for better detection of wooden skids
        self.confidence_threshold = 0.15  # Lowered from 0.3 to catch 10-15% confidence detections
        self.current_prompt_index = 0
        
        # Updated prompts for loaded pallets and packages (15 options)
        self.sample_prompts = [
            "pallet with cargo",
            "loaded pallet",
            "pallet with boxes",
            "pallet wrapped in plastic",
            "stack of goods on a pallet",
            "shrink-wrapped freight on pallet",
            "cardboard box",
            "shipping box",
            "warehouse package",
            "wrapped package",
            "shrink-wrapped package",
            "brown cardboard box",
            "stacked boxes",
            "plastic wrap",
            "pallet"
        ]
        
        self.current_prompt = self.sample_prompts[0]
        
        # Performance tracking
        self.detection_times = []
        self.last_detections = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Grounding DINO model"""
        try:
            # Import DetectorTracker from the correct path
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from detector_tracker import DetectorTracker

            self.grounding_dino = DetectorTracker()
            logger.info("Grounding DINO (DetectorTracker) loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing Grounding DINO: {e}")
            logger.error("Detection will be disabled - Grounding DINO not available")
            self.grounding_dino = None
    
    def detect_pallets(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect pallets using current prompt and threshold
        """
        if self.grounding_dino is None:
            return []

        start_time = time.time()

        try:
            # Update the DetectorTracker's prompt
            self.grounding_dino.prompt = self.current_prompt

            # Update confidence threshold
            self.grounding_dino.confidence_threshold = self.confidence_threshold

            # Run detection using DetectorTracker's process_frame method
            tracked_objects, detection_info = self.grounding_dino.process_frame(frame)

            detections = []

            # Convert tracked objects to our format
            for obj in tracked_objects:
                bbox = obj.get('bbox', [])
                confidence = obj.get('confidence', 0.0)

                # Apply confidence threshold (double check)
                if len(bbox) == 4 and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = bbox

                    detection_dict = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'prompt': self.current_prompt,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1),
                        'track_id': obj.get('track_id', -1)
                    }
                    detections.append(detection_dict)

            # Track performance
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)

            self.last_detections = detections
            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def next_prompt(self):
        """Switch to next prompt in the list"""
        self.current_prompt_index = (self.current_prompt_index + 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
        logger.info(f"Switched to prompt: '{self.current_prompt}'")
    
    def previous_prompt(self):
        """Switch to previous prompt in the list"""
        self.current_prompt_index = (self.current_prompt_index - 1) % len(self.sample_prompts)
        self.current_prompt = self.sample_prompts[self.current_prompt_index]
        logger.info(f"Switched to prompt: '{self.current_prompt}'")
    
    def increase_threshold(self, step=0.05):
        """Increase confidence threshold"""
        self.confidence_threshold = min(0.95, self.confidence_threshold + step)
        logger.info(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def decrease_threshold(self, step=0.05):
        """Decrease confidence threshold"""
        self.confidence_threshold = max(0.05, self.confidence_threshold - step)
        logger.info(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def set_custom_prompt(self, prompt: str):
        """Set a custom prompt"""
        self.current_prompt = prompt
        logger.info(f"Custom prompt set: '{self.current_prompt}'")
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        for detection in self.last_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Pallet: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw tuning information overlay"""
        height, width = frame.shape[:2]
        
        # Overlay parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)  # Yellow
        thickness = 2
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Current settings
        y_offset = 30
        cv2.putText(frame, f"Current Prompt: '{self.current_prompt}'", (20, y_offset), font, font_scale, color, thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Confidence Threshold: {self.confidence_threshold:.2f}", (20, y_offset), font, font_scale, color, thickness)
        
        y_offset += 25
        pallet_count = len(self.last_detections)
        cv2.putText(frame, f"Pallets Detected: {pallet_count}", (20, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        y_offset += 25
        prompt_info = f"Prompt {self.current_prompt_index + 1}/{len(self.sample_prompts)}"
        cv2.putText(frame, prompt_info, (20, y_offset), font, font_scale, color, thickness)
        
        # Controls
        y_offset += 35
        cv2.putText(frame, "Controls:", (20, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, "N/P - Next/Previous Prompt", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        y_offset += 15
        cv2.putText(frame, "+/- - Increase/Decrease Threshold", (20, y_offset), font, 0.4, (255, 255, 255), 1)
        
        # Performance info
        if self.detection_times:
            avg_time = np.mean(self.detection_times[-10:])  # Last 10 detections
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"Detection FPS: {fps:.1f}", (400, 30), font, 0.5, color, 1)
        
        return frame
    
    def get_current_settings(self) -> Dict:
        """Get current detection settings"""
        return {
            'prompt': self.current_prompt,
            'confidence_threshold': self.confidence_threshold,
            'prompt_index': self.current_prompt_index,
            'detection_count': len(self.last_detections)
        }
    
    def save_settings(self, filename: str = "pallet_detection_settings.txt"):
        """Save current optimal settings"""
        settings = self.get_current_settings()
        
        try:
            with open(filename, 'w') as f:
                f.write(f"Optimal Pallet Detection Settings\n")
                f.write(f"================================\n")
                f.write(f"Prompt: {settings['prompt']}\n")
                f.write(f"Confidence Threshold: {settings['confidence_threshold']}\n")
                f.write(f"Detection Count: {settings['detection_count']}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Settings saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

# Import torch here to avoid issues if not available
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available, some features may not work")
