# cv/GPU/test/test_tracking_validation.py

"""
Quick validation test for tracking system
Tests both detection-only and tracking modes for comparison
"""

import sys
import os
import time
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from test_detection_speed import test_detection_speed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tracking_validation():
    """Run validation tests for tracking system"""
    
    logger.info("🎯 TRACKING SYSTEM VALIDATION")
    logger.info("=" * 50)
    
    # Test cameras (start with 2 for validation)
    test_cameras = [8, 9]
    test_duration = 30  # 30 seconds each
    
    print("\n🟢 PHASE 1: Detection-Only Mode (Baseline)")
    print("Expected: Green boxes, no persistent IDs")
    input("Press Enter to start detection-only test...")
    
    test_detection_speed(
        camera_ids=test_cameras,
        duration=test_duration,
        confidence=0.5,
        enable_gui=True,
        enable_tracking=False  # Detection only
    )
    
    print("\n" + "="*50)
    print("🟠 PHASE 2: Tracking Mode (New Implementation)")
    print("Expected: Orange boxes, persistent track IDs (8001, 8002, 9001, 9002...)")
    input("Press Enter to start tracking test...")
    
    test_detection_speed(
        camera_ids=test_cameras,
        duration=test_duration,
        confidence=0.5,
        enable_gui=True,
        enable_tracking=True   # Tracking enabled
    )
    
    print("\n" + "="*50)
    print("✅ VALIDATION COMPLETE")
    print("\nWhat to look for:")
    print("🟢 Detection Mode:")
    print("  - Green bounding boxes")
    print("  - Labels: 'pallet: 0.85'")
    print("  - No persistent IDs")
    print("  - Header: 'Camera X - DETECTION'")
    print("\n🟠 Tracking Mode:")
    print("  - Orange bounding boxes")
    print("  - Labels: 'ID:8001 pallet: 0.85 (15f)'")
    print("  - Persistent track IDs")
    print("  - Track age in frames")
    print("  - Header: 'Camera X - TRACKING'")
    print("  - Statistics show 'Active Tracks' instead of 'Detections'")

if __name__ == "__main__":
    run_tracking_validation()
