#!/usr/bin/env python3
"""
GPU Detection Test Script
Tests which GPU PyTorch is actually using for computation
"""

import torch
import time

def test_gpu_usage():
    print("=== GPU Detection Test ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
    # List all available GPUs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f}GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Test NVIDIA GPU detection logic
    nvidia_device = 0  # Default fallback
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"Checking device {i}: {device_name}")
        if "NVIDIA" in device_name:
            nvidia_device = i
            print(f"  -> NVIDIA GPU found at device {i}")
            break
    
    print(f"\nSelected NVIDIA device: cuda:{nvidia_device}")
    
    # Set device and test computation
    device = torch.device(f"cuda:{nvidia_device}")
    torch.cuda.set_device(nvidia_device)
    
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Perform computation to test GPU usage
    print("\n=== Testing GPU Computation ===")
    print("Creating large tensor and performing computation...")
    print("Check Task Manager GPU utilization now!")
    
    # Create large tensors on GPU
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    
    # Perform intensive computation for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        z = torch.matmul(x, y)
        z = torch.relu(z)
        z = torch.sum(z)
        print(f"Computation result: {z.item():.2f}")
        time.sleep(0.5)
    
    print("GPU test completed!")
    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")

if __name__ == "__main__":
    test_gpu_usage()
