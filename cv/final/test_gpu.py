#!/usr/bin/env python3
"""
Test GPU 1 (NVIDIA RTX 4050) Only
This script will specifically target GPU 1 and monitor Task Manager
"""

import torch
import time
import subprocess
import os
import threading
from datetime import datetime

def get_task_manager_gpu_usage():
    """Get GPU usage that matches Task Manager display"""
    try:
        # PowerShell command to get GPU usage by adapter
        ps_cmd = '''
        Get-Counter "\\GPU Adapter Memory(*)\\Dedicated Usage" -ErrorAction SilentlyContinue | 
        ForEach-Object { 
            $_.CounterSamples | ForEach-Object { 
                "$($_.InstanceName),$($_.CookedValue)" 
            } 
        }
        '''
        
        result = subprocess.run([
            'powershell', '-Command', ps_cmd
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("GPU Memory Usage (Task Manager equivalent):")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        
        # Also get GPU utilization
        ps_cmd2 = '''
        Get-Counter "\\GPU Engine(*)\\Utilization Percentage" -ErrorAction SilentlyContinue | 
        ForEach-Object { 
            $_.CounterSamples | 
            Where-Object { $_.CookedValue -gt 1 } |
            ForEach-Object { 
                "$($_.InstanceName),$($_.CookedValue)" 
            } 
        }
        '''
        
        result2 = subprocess.run([
            'powershell', '-Command', ps_cmd2
        ], capture_output=True, text=True, timeout=5)
        
        if result2.returncode == 0:
            print("GPU Utilization (Task Manager equivalent):")
            lines = result2.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and 'NVIDIA' in line:
                    print(f"  {line}")
        
    except Exception as e:
        print(f"Error getting Task Manager GPU usage: {e}")

def force_gpu_1_usage():
    """Force usage of GPU 1 specifically"""
    print("🔧 Forcing GPU 1 (NVIDIA RTX 4050) usage...")
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Force GPU 1 if available, otherwise use GPU 0
    if gpu_count > 1:
        target_gpu = 1
        print(f"🎯 Targeting GPU 1: {torch.cuda.get_device_name(1)}")
    else:
        target_gpu = 0
        print(f"🎯 Only 1 GPU available, using GPU 0: {torch.cuda.get_device_name(0)}")
    
    # Set environment to force GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
    
    # Now reinitialize CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # This will be GPU 1 due to CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(0)
        print(f"✅ Using device: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("❌ CUDA not available after setting environment")
        return None

def run_gpu_1_stress_test(device, duration=30):
    """Run stress test specifically on GPU 1"""
    print(f"\n🚀 Running GPU 1 stress test for {duration} seconds...")
    print("💡 WATCH TASK MANAGER - GPU 1 should show 90-100% utilization")
    print("💡 GPU 0 should show minimal utilization")
    
    # Create large tensors on GPU 1
    tensor_size = 4000
    tensors = []
    
    # Create multiple tensors
    for i in range(4):
        tensor = torch.randn(tensor_size, tensor_size, device=device, dtype=torch.float32)
        tensors.append(tensor)
    
    print(f"✅ Created {len(tensors)} tensors of size {tensor_size}x{tensor_size} on {device}")
    
    # Show memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"📊 GPU Memory: {memory_allocated:.2f}GB / {memory_total:.1f}GB")
    
    # Run intensive operations
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        # Rotate through tensor operations
        for i in range(len(tensors)):
            for j in range(len(tensors)):
                if i != j:
                    # Matrix multiplication
                    result = torch.matmul(tensors[i], tensors[j])
                    
                    # Additional operations
                    result = torch.relu(result)
                    result = torch.sigmoid(result)
                    result = torch.tanh(result)
                    
                    # Replace one tensor with result
                    tensors[i] = result
        
        # Force synchronization
        torch.cuda.synchronize(device)
        
        iteration += 1
        
        if iteration % 5 == 0:
            elapsed = time.time() - start_time
            memory_current = torch.cuda.memory_allocated(device) / 1024**3
            print(f"⚡ Iteration {iteration} | Time: {elapsed:.1f}s | Memory: {memory_current:.2f}GB")
    
    print(f"✅ Completed {iteration} iterations of intensive GPU operations")
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")

def monitor_task_manager(duration=30):
    """Monitor Task Manager equivalent GPU usage"""
    print(f"\n📊 Monitoring Task Manager GPU usage for {duration} seconds...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        print(f"\n⏰ Time: {time.time() - start_time:.1f}s")
        get_task_manager_gpu_usage()
        time.sleep(2)  # Check every 2 seconds
    
    print("🛑 Task Manager monitoring completed")

def main():
    """Main function to test GPU 1 specifically"""
    print("🎯 GPU 1 (NVIDIA RTX 4050) SPECIFIC TEST")
    print("=" * 60)
    
    # Step 1: Force GPU 1 usage
    device = force_gpu_1_usage()
    if device is None:
        print("❌ Failed to initialize GPU 1")
        return
    
    # Step 2: Start monitoring in background
    monitor_thread = threading.Thread(target=monitor_task_manager, args=(35,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Step 3: Run stress test
    run_gpu_1_stress_test(device, duration=30)
    
    # Step 4: Wait for monitoring to complete
    monitor_thread.join()
    
    print("\n🏁 TEST COMPLETED")
    print("=" * 60)
    print("ANALYSIS:")
    print("✅ If GPU 1 showed high utilization: GPU 1 is working correctly")
    print("❌ If GPU 1 showed low utilization: GPU 1 is not being used")
    print("⚠️  If GPU 0 showed high utilization: System is still using GPU 0")
    print("=" * 60)

if __name__ == "__main__":
    main()