#!/usr/bin/env python3
"""
Check GPU information and PyTorch CUDA setup
"""

import torch
import subprocess
import sys

def check_gpu_info():
    """Check available GPUs and PyTorch CUDA configuration"""
    print("🔍 GPU Information Check")
    print("=" * 50)
    
    # Check if CUDA is available
    print(f"🚀 CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA support detected")
        return
    
    # Check CUDA version
    print(f"📊 CUDA Version: {torch.version.cuda}")
    print(f"📊 PyTorch Version: {torch.__version__}")
    
    # Check number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"🎯 Number of GPUs: {gpu_count}")
    
    # List all GPUs
    print("\n📋 GPU Details:")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f}GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Multiprocessors: {props.multi_processor_count}")
        
        # Check if this is the current device
        if i == torch.cuda.current_device():
            print(f"    ⭐ CURRENT DEFAULT DEVICE")
        print()
    
    # Check current device
    current_device = torch.cuda.current_device()
    current_name = torch.cuda.get_device_name(current_device)
    print(f"🎯 Current Default Device: GPU {current_device} ({current_name})")
    
    # Check memory usage
    print(f"\n💾 Memory Usage (GPU {current_device}):")
    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    
    print(f"  Allocated: {memory_allocated:.2f}GB")
    print(f"  Reserved: {memory_reserved:.2f}GB") 
    print(f"  Total: {memory_total:.1f}GB")
    print(f"  Free: {memory_total - memory_reserved:.2f}GB")
    
    # Test tensor creation on each GPU
    print(f"\n🧪 GPU Performance Test:")
    for i in range(gpu_count):
        try:
            device = torch.device(f'cuda:{i}')
            # Create a test tensor
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            print(f"  GPU {i}: {elapsed_time:.2f}ms (matrix multiply 1000x1000)")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  GPU {i}: Error - {e}")
    
    # Check which GPU PyTorch will use by default
    print(f"\n🎯 PyTorch Default Behavior:")
    print(f"  torch.cuda.current_device() = {torch.cuda.current_device()}")
    print(f"  torch.device('cuda') points to GPU {torch.cuda.current_device()}")
    
    # Check if we can manually select GPUs
    print(f"\n🔧 Manual GPU Selection Test:")
    for i in range(gpu_count):
        try:
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.tensor([1.0], device=device)
            print(f"  ✅ Can use GPU {i}: {torch.cuda.get_device_name(i)}")
            del test_tensor
        except Exception as e:
            print(f"  ❌ Cannot use GPU {i}: {e}")
    
    # Try nvidia-smi if available
    print(f"\n🖥️ NVIDIA-SMI Output:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    idx, name, mem_total, mem_used, util = parts
                    print(f"  GPU {idx}: {name}")
                    print(f"    Memory: {mem_used}MB / {mem_total}MB")
                    print(f"    Utilization: {util}%")
        else:
            print("  nvidia-smi not available or failed")
    except Exception as e:
        print(f"  nvidia-smi error: {e}")

if __name__ == "__main__":
    check_gpu_info()
