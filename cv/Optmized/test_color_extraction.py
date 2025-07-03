#!/usr/bin/env python3
"""
Quick test script to verify color extraction is working
"""

import cv2
import numpy as np
import sys
import os

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Test sklearn availability
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    print("✅ sklearn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ sklearn not available")

# Import the color extractor
from cpu_11camera_configurable import ObjectColorExtractor

def test_color_extraction():
    """Test color extraction with known colors"""
    print("🧪 Testing Color Extraction...")
    
    extractor = ObjectColorExtractor()
    
    # Test 1: Pure red image
    red_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)  # BGR format
    red_result = extractor.extract_dominant_color(red_image)
    print(f"🔴 Red test: {red_result}")
    
    # Test 2: Pure blue image  
    blue_image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # BGR format
    blue_result = extractor.extract_dominant_color(blue_image)
    print(f"🔵 Blue test: {blue_result}")
    
    # Test 3: Pure green image
    green_image = np.full((100, 100, 3), [0, 255, 0], dtype=np.uint8)  # BGR format
    green_result = extractor.extract_dominant_color(green_image)
    print(f"🟢 Green test: {green_result}")
    
    # Test 4: Mixed color image
    mixed_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mixed_result = extractor.extract_dominant_color(mixed_image)
    print(f"🎨 Mixed test: {mixed_result}")
    
    # Test 5: Empty/invalid image
    empty_result = extractor.extract_dominant_color(None)
    print(f"⚫ Empty test: {empty_result}")
    
    print("✅ Color extraction test complete!")

if __name__ == "__main__":
    test_color_extraction()
