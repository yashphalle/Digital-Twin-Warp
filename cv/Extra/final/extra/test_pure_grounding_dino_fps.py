#!/usr/bin/env python3
"""
Pure Grounding DINO FPS Test Script
Tests ONLY the Grounding DINO model inference speed on GPU
NO GUI, NO fisheye correction, NO feature matching, NO other processing
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
import logging

# Add the modules path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pure_grounding_dino_fps():
    """Test pure Grounding DINO FPS with different input types"""
    
    print("🚀 PURE GROUNDING DINO FPS TEST")
    print("=" * 60)
    print("Testing ONLY Grounding DINO model inference speed")
    print("NO GUI | NO fisheye | NO feature matching | NO other processing")
    print("=" * 60)
    
    # Import detector
    try:
        from modules.detector import CPUSimplePalletDetector
        logger.info("✅ Successfully imported CPUSimplePalletDetector")
    except ImportError as e:
        logger.error(f"❌ Failed to import detector: {e}")
        return
    
    # Initialize detector (will use GPU if available)
    print("\n🔧 Initializing Grounding DINO detector...")
    detector = CPUSimplePalletDetector()
    
    # Check device being used
    device_info = f"Device: {detector.device}"
    if hasattr(detector, 'device') and detector.device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(detector.device)
        gpu_memory = torch.cuda.get_device_properties(detector.device).total_memory / 1024**3
        device_info += f" ({gpu_name}, {gpu_memory:.1f}GB)"
    
    print(f"🎯 {device_info}")
    print(f"🎯 Confidence Threshold: {detector.confidence_threshold}")
    print(f"🎯 Current Prompt: '{detector.current_prompt}'")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Random 1080p Frame',
            'frame_generator': lambda: np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
            'description': 'Random noise - tests pure model performance'
        },
        {
            'name': 'Black Frame',
            'frame_generator': lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
            'description': 'All black - minimal processing'
        },
        {
            'name': 'White Frame',
            'frame_generator': lambda: np.full((1080, 1920, 3), 255, dtype=np.uint8),
            'description': 'All white - tests different input'
        }
    ]
    
    # Run tests for each configuration
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   Description: {config['description']}")
        print("-" * 50)
        
        # Generate test frame
        test_frame = config['frame_generator']()
        
        # Warm-up runs (don't count these)
        print("   Warming up GPU...")
        for i in range(3):
            _ = detector.detect_pallets(test_frame)
        
        # Clear GPU cache if using GPU
        if hasattr(detector, 'device') and detector.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Actual test runs
        print("   Running test...")
        times = []
        detection_counts = []
        
        num_test_frames = 20
        for i in range(num_test_frames):
            # Time pure Grounding DINO inference
            start_time = time.time()
            detections = detector.detect_pallets(test_frame)
            end_time = time.time()
            
            inference_time = end_time - start_time
            fps = 1.0 / inference_time
            
            times.append(inference_time)
            detection_counts.append(len(detections))
            
            print(f"   Frame {i+1:2d}: {fps:6.2f} FPS | {inference_time:6.3f}s | {len(detections)} detections")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        avg_fps = 1.0 / avg_time
        max_fps = 1.0 / min_time
        min_fps = 1.0 / max_time
        
        avg_detections = sum(detection_counts) / len(detection_counts)
        
        print(f"\n   📊 Results for {config['name']}:")
        print(f"      Average FPS: {avg_fps:6.2f}")
        print(f"      Max FPS:     {max_fps:6.2f}")
        print(f"      Min FPS:     {min_fps:6.2f}")
        print(f"      Avg Time:    {avg_time:6.3f}s")
        print(f"      Avg Detections: {avg_detections:.1f}")
        
        # GPU memory info if available
        if hasattr(detector, 'device') and detector.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(detector.device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(detector.device) / 1024**3
            print(f"      GPU Memory Used: {memory_used:.2f}GB")
            print(f"      GPU Memory Cached: {memory_cached:.2f}GB")

def test_with_video_file():
    """Test with cam7.mp4 video file"""
    print("\n🎬 VIDEO FILE TEST - cam7.mp4")
    print("-" * 40)

    # Look for cam7.mp4 in common locations
    video_paths = [
        "cam7.mp4",
        "../cam7.mp4",
        "../../cam7.mp4",
        "../../../cam7.mp4",
        "training/cam7.mp4",
        "../training/cam7.mp4"
    ]

    video_path = None
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break

    if not video_path:
        print("   ⚠️ cam7.mp4 not found. Skipping video test.")
        print("   💡 Looking for cam7.mp4 in these locations:")
        for path in video_paths:
            print(f"      - {path}")
        return
    
    print(f"   📹 Found cam7.mp4: {video_path}")

    # Import detector
    from modules.detector import CPUSimplePalletDetector
    detector = CPUSimplePalletDetector()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ Failed to open cam7.mp4: {video_path}")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   📊 Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    print("   🎯 Testing pure Grounding DINO FPS with cam7.mp4 frames...")

    times = []
    detection_counts = []
    frame_count = 0
    max_frames = 50  # Test first 50 frames from cam7.mp4
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Test inference
        start_time = time.time()
        detections = detector.detect_pallets(frame)
        end_time = time.time()
        
        inference_time = end_time - start_time
        fps = 1.0 / inference_time
        
        times.append(inference_time)
        detection_counts.append(len(detections))
        frame_count += 1
        
        print(f"   Frame {frame_count:2d}: {fps:6.2f} FPS | {len(detections)} detections")
    
    cap.release()
    
    if times:
        avg_fps = 1.0 / (sum(times) / len(times))
        max_fps = 1.0 / min(times)
        min_fps = 1.0 / max(times)
        avg_detections = sum(detection_counts) / len(detection_counts)
        
        print(f"\n   📊 cam7.mp4 Test Results:")
        print(f"      Video: {width}x{height} @ {fps:.1f} FPS")
        print(f"      Frames Tested: {frame_count}/{total_frames}")
        print(f"      Pure Grounding DINO Average FPS: {avg_fps:6.2f}")
        print(f"      Pure Grounding DINO Max FPS:     {max_fps:6.2f}")
        print(f"      Pure Grounding DINO Min FPS:     {min_fps:6.2f}")
        print(f"      Average Detections per Frame: {avg_detections:.1f}")
        print(f"      🎯 This is PURE model inference speed (no other processing)")

def main():
    """Main test function"""
    try:
        # Test pure Grounding DINO FPS
        test_pure_grounding_dino_fps()
        
        # Test with video if available
        test_with_video_file()
        
        print("\n🎯 SUMMARY")
        print("=" * 60)
        print("✅ Pure Grounding DINO FPS test completed")
        print("💡 This shows the maximum FPS your GPU can achieve")
        print("💡 with ONLY Grounding DINO inference (no other processing)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
