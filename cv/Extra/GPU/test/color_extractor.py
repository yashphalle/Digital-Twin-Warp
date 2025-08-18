#!/usr/bin/env python3
"""
Fast Color Extractor for Warehouse Objects
Lightweight color extraction optimized for real-time performance
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class FastColorExtractor:
    """Lightweight color extraction for warehouse objects - 2-5ms per detection"""
    
    def __init__(self):
        logger.info("🎨 Fast Color Extractor initialized")
    
    def extract_dominant_color_fast(self, frame, bbox):
        """Fast dominant color extraction using K-means clustering"""
        try:
            x1, y1, x2, y2 = bbox

            # Convert to integers (bbox coordinates might be floats)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                logger.debug("Invalid bbox for color extraction")
                return self._get_null_color()

            # Extract ROI from fisheye-corrected frame
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                logger.debug("Empty ROI for color extraction")
                return self._get_null_color()
            
            # Resize to small size for speed (32x32 = 1024 pixels)
            roi_small = cv2.resize(roi, (32, 32))
            
            # Reshape for K-means
            pixels = roi_small.reshape(-1, 3).astype(np.float32)
            
            # Fast K-means clustering (k=3, single init for speed)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=1, max_iter=10)
            kmeans.fit(pixels)
            
            # Get dominant color (largest cluster)
            labels = kmeans.labels_
            dominant_idx = np.argmax(np.bincount(labels))
            dominant_color_bgr = kmeans.cluster_centers_[dominant_idx].astype(int)
            
            # Convert BGR to RGB for consistency
            rgb_color = [int(dominant_color_bgr[2]), int(dominant_color_bgr[1]), int(dominant_color_bgr[0])]
            
            # Convert to HSV for color naming
            hsv_color = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            result = {
                'rgb': rgb_color,                    # ✅ Frontend Priority #1 - exact RGB values
                'hsv': hsv_color.tolist(),          # ✅ Keep for completeness
                'hex': f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}",  # ✅ Frontend Priority #2
                'name': None,                       # ✅ No discretization - let frontend handle
                'confidence': 0.75,                 # Fixed confidence for fast mode
                'extraction_method': 'kmeans_fast'
            }

            # 🔍 DEBUG: Log extracted colors to see what's happening
            # logger.info(f"🎨 Color extracted: RGB={rgb_color}, HEX={result['hex']}, ROI_size={roi.shape}")

            return result
            
        except Exception as e:
            logger.error(f"🚨 Color extraction failed: {e}")
            logger.error(f"🚨 This will cause frontend fallback to brown color #d97706")
            return self._get_null_color()
    
    # ✅ REMOVED: Problematic HSV to color name mapping
    # Frontend will handle color display using exact RGB values
    # No more discretization issues (brown→orange, etc.)
    
    def _get_null_color(self):
        """Return null color info for failed extractions"""
        logger.warning(f"🚨 Returning NULL color - Frontend will show brown fallback #d97706")
        return {
            'rgb': None,
            'hsv': None,
            'hex': None,
            'name': None,
            'confidence': None,
            'extraction_method': None
        }

def extract_color_for_detection(frame, detection, color_extractor):
    """Extract color for ALL tracks - simplified approach"""
    # Check if frame is available
    if frame is None:
        logger.warning(f"🚨 Frame is None for detection {detection.get('global_id', 'unknown')} - cannot extract color")
        return color_extractor._get_null_color()

    # Extract real color for ALL objects (Forklifts + Pallets) - ALL tracks get color extraction
    # Frontend will use exact RGB values to display true colors
    return color_extractor.extract_dominant_color_fast(frame, detection['bbox'])

def get_null_color():
    """Return null color info for existing tracks"""
    return {
        'rgb': None,
        'hsv': None,
        'hex': None,
        'name': None,
        'confidence': None,
        'extraction_method': None
    }
