#!/usr/bin/env python3
"""
Real-time GPU monitoring for warehouse tracking system
Run this alongside the main system to see GPU usage
"""

import time
import subprocess
import sys
import os

def get_gpu_usage():
    """Get current GPU usage statistics"""
    try:
        # Run nvidia-smi to get GPU stats
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            line = result.stdout.strip()
            gpu_util, mem_used, mem_total, temp = line.split(', ')
            return {
                'gpu_utilization': int(gpu_util),
                'memory_used': int(mem_used),
                'memory_total': int(mem_total),
                'temperature': int(temp)
            }
    except Exception as e:
        return None

def monitor_gpu():
    """Monitor GPU usage in real-time"""
    print("🚀 REAL-TIME GPU MONITORING FOR WAREHOUSE TRACKING")
    print("=" * 60)
    print("Target GPU Utilization: 80-90%")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        while True:
            stats = get_gpu_usage()
            
            if stats:
                gpu_util = stats['gpu_utilization']
                mem_used = stats['memory_used']
                mem_total = stats['memory_total']
                temp = stats['temperature']
                mem_percent = (mem_used / mem_total) * 100
                
                # Color coding for utilization
                if gpu_util >= 80:
                    util_status = "🟢 OPTIMAL"
                elif gpu_util >= 60:
                    util_status = "🟡 MODERATE"
                elif gpu_util >= 20:
                    util_status = "🟠 LOW"
                else:
                    util_status = "🔴 IDLE"
                
                # Memory status
                if mem_percent >= 90:
                    mem_status = "🔴 HIGH"
                elif mem_percent >= 70:
                    mem_status = "🟡 MODERATE"
                else:
                    mem_status = "🟢 NORMAL"
                
                print(f"\r🚀 GPU: {gpu_util:2d}% {util_status} | "
                      f"Memory: {mem_used:4d}MB/{mem_total:4d}MB ({mem_percent:4.1f}%) {mem_status} | "
                      f"Temp: {temp:2d}°C", end="", flush=True)
                
                # Recommendations
                if gpu_util < 50:
                    print(f"\n💡 GPU utilization low ({gpu_util}%) - Check if detection is running")
                elif gpu_util > 95:
                    print(f"\n⚠️ GPU utilization very high ({gpu_util}%) - May cause slowdowns")
                
            else:
                print("\r❌ Unable to read GPU stats", end="", flush=True)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 GPU monitoring stopped")

def check_pytorch_gpu():
    """Check if PyTorch can see the GPU"""
    print("\n🔍 PYTORCH GPU CHECK:")
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Test allocation
            test_tensor = torch.randn(1000, 1000).cuda()
            print("   ✅ GPU allocation test successful")
            del test_tensor
            torch.cuda.empty_cache()
        else:
            print("   ❌ CUDA not available to PyTorch")
            
    except ImportError:
        print("   ❌ PyTorch not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Main monitoring function"""
    print("🚀 GPU MONITORING TOOL FOR WAREHOUSE TRACKING")
    print("=" * 50)
    
    # Check PyTorch GPU access
    check_pytorch_gpu()
    
    # Check nvidia-smi availability
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        print("✅ nvidia-smi available")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers not installed")
        return
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Start monitoring
    monitor_gpu()

if __name__ == "__main__":
    main()
