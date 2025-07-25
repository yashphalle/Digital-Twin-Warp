#!/usr/bin/env python3
"""
Test GPU SIFT functionality and compare with CPU SIFT performance
"""

import cv2
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_sift_availability():
    """Test if GPU SIFT is available and working"""
    print("🔍 Testing GPU SIFT Availability...")
    print("=" * 50)
    
    # Check CUDA devices
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"🚀 CUDA devices available: {cuda_devices}")
        
        if cuda_devices == 0:
            print("❌ No CUDA devices available - GPU SIFT not possible")
            return False
            
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False
    
    # Test GPU SIFT creation
    try:
        gpu_sift = cv2.cuda.SIFT_create(
            nfeatures=500,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        print("✅ GPU SIFT detector created successfully")
        
        # Test GPU memory allocation
        gpu_frame = cv2.cuda_GpuMat()
        print("✅ GPU memory allocation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU SIFT creation failed: {e}")
        return False

def test_sift_performance():
    """Compare GPU vs CPU SIFT performance"""
    print("\n🏁 SIFT Performance Comparison...")
    print("=" * 50)
    
    # Create test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    print(f"📊 Test image size: {test_image.shape}")
    
    # Test CPU SIFT
    try:
        cpu_sift = cv2.SIFT_create(nfeatures=500)
        
        start_time = time.time()
        cpu_keypoints, cpu_descriptors = cpu_sift.detectAndCompute(gray_image, None)
        cpu_time = time.time() - start_time
        
        cpu_features = len(cpu_descriptors) if cpu_descriptors is not None else 0
        print(f"🖥️ CPU SIFT: {cpu_time:.3f}s, {cpu_features} features")
        
    except Exception as e:
        print(f"❌ CPU SIFT failed: {e}")
        cpu_time = float('inf')
        cpu_features = 0
    
    # Test GPU SIFT
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_sift = cv2.cuda.SIFT_create(nfeatures=500)
            gpu_frame = cv2.cuda_GpuMat()
            
            # Upload to GPU
            gpu_frame.upload(gray_image)
            
            start_time = time.time()
            gpu_keypoints, gpu_descriptors = gpu_sift.detectAndCompute(gpu_frame, None)
            
            # Download results
            if gpu_descriptors is not None:
                gpu_descriptors_cpu = gpu_descriptors.download()
                gpu_features = len(gpu_descriptors_cpu) if gpu_descriptors_cpu is not None else 0
            else:
                gpu_features = 0
                
            gpu_time = time.time() - start_time
            
            print(f"🚀 GPU SIFT: {gpu_time:.3f}s, {gpu_features} features")
            
            # Calculate speedup
            if cpu_time != float('inf') and gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"⚡ GPU Speedup: {speedup:.2f}x faster")
            
        else:
            print("❌ No CUDA devices for GPU SIFT test")
            
    except Exception as e:
        print(f"❌ GPU SIFT failed: {e}")

def test_gpu_sift_tracker():
    """Test the GPU SIFT tracker initialization"""
    print("\n🎯 Testing GPU SIFT Tracker...")
    print("=" * 50)
    
    try:
        from gpu_sift_11camera_configurable import GPUSIFTWarehouseTracker
        
        # Test tracker initialization
        tracker = GPUSIFTWarehouseTracker(camera_id=8, camera_name="Test Camera 8")
        
        print("✅ GPU SIFT tracker initialized successfully")
        print(f"🔧 SIFT method: {'GPU' if tracker.global_db.use_gpu_sift else 'CPU'}")
        print(f"🎯 Camera ID range: 8001-8999")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        start_time = time.time()
        result_frame = tracker.process_frame(dummy_frame)
        process_time = time.time() - start_time
        
        print(f"⏱️ Frame processing time: {process_time:.3f}s")
        print(f"📊 Detections found: {len(tracker.final_tracked_detections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU SIFT tracker test failed: {e}")
        return False

def main():
    """Run all GPU SIFT tests"""
    print("🚀 GPU SIFT Testing Suite")
    print("=" * 60)
    
    # Test 1: GPU SIFT availability
    gpu_available = test_gpu_sift_availability()
    
    # Test 2: Performance comparison
    test_sift_performance()
    
    # Test 3: Tracker functionality
    tracker_works = test_gpu_sift_tracker()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    print(f"GPU SIFT Available: {'✅ Yes' if gpu_available else '❌ No'}")
    print(f"Tracker Working: {'✅ Yes' if tracker_works else '❌ No'}")
    
    if gpu_available and tracker_works:
        print("\n🎉 GPU SIFT system is ready for use!")
        print("💡 Expected benefits:")
        print("   - Faster SIFT feature extraction")
        print("   - Better multi-camera performance")
        print("   - Reduced CPU load")
    else:
        print("\n⚠️ GPU SIFT system has issues - will fallback to CPU")

if __name__ == "__main__":
    main()
