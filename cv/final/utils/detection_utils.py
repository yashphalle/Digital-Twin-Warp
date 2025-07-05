#!/usr/bin/env python3
"""
Detection Utilities
Helper functions for processing detection results
Handles bounding box processing, corner generation, and metadata
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def generate_detection_with_corners(box, score, prompt_used: str = None) -> Dict:
    """
    Generate detection dictionary with 4-corner coordinates
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        score: Confidence score
        prompt_used: Detection prompt used (optional)
        
    Returns:
        Detection dictionary with bbox, corners, and metadata
    """
    x1, y1, x2, y2 = map(int, box)
    area = (x2 - x1) * (y2 - y1)
    
    detection = {
        'bbox': [x1, y1, x2, y2],
        'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # 4 corners clockwise from top-left
        'confidence': float(score),
        'area': area,
        'shape_type': 'quadrangle'
    }
    
    # Add prompt information if provided
    if prompt_used:
        detection['prompt_used'] = prompt_used
    
    return detection

def calculate_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculate center point of bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def add_center_to_detection(detection: Dict) -> Dict:
    """
    Add center coordinates to detection if not already present
    
    Args:
        detection: Detection dictionary
        
    Returns:
        Detection dictionary with center added
    """
    if 'center' not in detection:
        detection['center'] = calculate_center(detection['bbox'])
    return detection

def validate_detection(detection: Dict) -> bool:
    """
    Validate detection dictionary has required fields
    
    Args:
        detection: Detection dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['bbox', 'confidence', 'area']
    
    for field in required_fields:
        if field not in detection:
            logger.warning(f"Detection missing required field: {field}")
            return False
    
    # Validate bbox format
    bbox = detection['bbox']
    if not isinstance(bbox, list) or len(bbox) != 4:
        logger.warning(f"Invalid bbox format: {bbox}")
        return False
    
    # Validate bbox coordinates
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bbox coordinates: {bbox}")
        return False
    
    return True

def process_detection_results(results, current_prompt: str = None) -> List[Dict]:
    """
    Process raw detection results into standardized detection format
    
    Args:
        results: Raw detection results from model
        current_prompt: Current detection prompt used
        
    Returns:
        List of processed detection dictionaries
    """
    detections = []
    
    if results and len(results) > 0:
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            detection = generate_detection_with_corners(box, score, current_prompt)
            
            # Validate detection before adding
            if validate_detection(detection):
                detections.append(detection)
            else:
                logger.warning(f"Skipping invalid detection: {detection}")
    
    logger.debug(f"Processed {len(detections)} valid detections")
    return detections

def enhance_detection_metadata(detection: Dict, **kwargs) -> Dict:
    """
    Add additional metadata to detection
    
    Args:
        detection: Detection dictionary
        **kwargs: Additional metadata to add
        
    Returns:
        Enhanced detection dictionary
    """
    # Add center if not present
    detection = add_center_to_detection(detection)
    
    # Add any additional metadata
    for key, value in kwargs.items():
        detection[key] = value
    
    return detection
