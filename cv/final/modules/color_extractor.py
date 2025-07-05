#!/usr/bin/env python3
"""
Color Extractor Module
Object color extraction using K-means clustering and HSV analysis
Extracted from main.py for modular architecture
"""

import cv2
import numpy as np
import logging
from typing import Dict

# sklearn import - will fallback to simple mean if not available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ObjectColorExtractor:
    """Extract dominant colors from detected object regions"""

    def __init__(self):
        self.min_pixels = 100  # Minimum pixels needed for reliable color extraction
        logger.info("✅ Object color extractor initialized")

    def extract_dominant_color(self, image_region: np.ndarray) -> Dict:
        """Extract dominant color from object region and return HSV + RGB values"""
        if image_region is None or image_region.size == 0:
            return self._get_default_color()

        try:
            # Ensure we have enough pixels for reliable color extraction
            if image_region.size < self.min_pixels * 3:  # 3 channels
                return self._get_default_color()

            # Convert BGR to RGB for processing
            if len(image_region.shape) == 3:
                rgb_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            else:
                return self._get_default_color()

            # Reshape for color analysis
            pixels = rgb_region.reshape(-1, 3)

            # Remove very dark pixels (shadows) and very bright pixels (highlights)
            brightness = np.mean(pixels, axis=1)
            valid_pixels = pixels[(brightness > 30) & (brightness < 225)]

            if len(valid_pixels) < 50:  # Not enough valid pixels
                valid_pixels = pixels  # Use all pixels as fallback

            # Use K-means clustering if available, otherwise simple mean
            if SKLEARN_AVAILABLE:
                try:
                    n_clusters = min(3, len(valid_pixels) // 10)  # Adaptive cluster count
                    if n_clusters < 1:
                        n_clusters = 1

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(valid_pixels)

                    # Get the most frequent color cluster
                    colors = kmeans.cluster_centers_
                    labels = kmeans.labels_

                    unique, counts = np.unique(labels, return_counts=True)
                    dominant_idx = unique[np.argmax(counts)]
                    dominant_rgb = colors[dominant_idx]

                    # Calculate color confidence based on cluster dominance
                    color_confidence = np.max(counts) / len(labels)

                except Exception as e:
                    logger.warning(f"K-means clustering failed, using simple mean: {e}")
                    # Fallback: simple mean color
                    dominant_rgb = np.mean(valid_pixels, axis=0)
                    color_confidence = 0.5  # Medium confidence for fallback method
            else:
                # Simple mean color when sklearn not available
                dominant_rgb = np.mean(valid_pixels, axis=0)
                color_confidence = 0.6  # Good confidence for mean method

            # Ensure RGB values are in valid range
            dominant_rgb = np.clip(dominant_rgb, 0, 255).astype(int)

            # Convert RGB to HSV for better color representation
            rgb_normalized = dominant_rgb.reshape(1, 1, 3).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0, 0]

            return {
                'color_rgb': [int(dominant_rgb[0]), int(dominant_rgb[1]), int(dominant_rgb[2])],
                'color_hsv': [int(hsv[0]), int(hsv[1]), int(hsv[2])],
                'color_hex': f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}",
                'color_confidence': float(color_confidence),
                'color_name': self._get_color_name(hsv),
                'extraction_method': 'kmeans_clustering'
            }

        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return self._get_default_color()

    def _get_default_color(self) -> Dict:
        """Return default gray color when extraction fails"""
        return {
            'color_rgb': [128, 128, 128],
            'color_hsv': [0, 0, 128],
            'color_hex': "#808080",
            'color_confidence': 0.0,
            'color_name': 'gray',
            'extraction_method': 'default_fallback'
        }

    def _get_color_name(self, hsv: np.ndarray) -> str:
        """Get human-readable color name from HSV values"""
        h, s, v = hsv

        # Low saturation = grayscale
        if s < 30:
            if v < 50:
                return 'black'
            elif v > 200:
                return 'white'
            else:
                return 'gray'

        # Low value = dark colors
        if v < 50:
            return 'dark'

        # Categorize by hue
        if h < 10 or h > 170:
            return 'red'
        elif h < 25:
            return 'orange'
        elif h < 35:
            return 'yellow'
        elif h < 85:
            return 'green'
        elif h < 125:
            return 'blue'
        elif h < 150:
            return 'purple'
        else:
            return 'pink'
