import torch
import subprocess
import time
import os

def diagnose_gpu_routing():
    print("🔍 GPU Routing Diagnostic")
    print("=" * 50)
    
    # Force NVIDIA GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda:0')
    print(f"PyTorch using: {torch.cuda.get_device_name(0)}")
    
    # Get baseline readings
    print("\n📊 Baseline GPU Status:")
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    baseline_util, baseline_power, baseline_temp, baseline_mem = result.stdout.strip().split(', ')
    print(f"NVIDIA GPU: {baseline_util}% util, {baseline_power}W, {baseline_temp}°C, {baseline_mem}MB")
    
    print("\n🚀 Starting MAXIMUM GPU load...")
    print("👀 Watch BOTH GPU 0 and GPU 1 in Task Manager")
    print("📊 Also watch the numbers below:")
    
    # Create maximum load
    try:
        # Use as much GPU memory as possible
        x = torch.randn(7000, 7000, device=device)
        y = torch.randn(7000, 7000, device=device)
        
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Run for 30 seconds with monitoring
        for i in range(30):
            # Intensive operations
            z = torch.mm(x, y)
            z = torch.relu(z)
            z = torch.sin(z)
            z = torch.mm(z, x)
            
            # Check GPU status every 5 seconds
            if i % 5 == 0:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                util, power, temp = result.stdout.strip().split(', ')
                print(f"Second {i}: NVIDIA GPU: {util}% util, {power}W, {temp}°C")
            
            time.sleep(1)
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("✅ GPU memory fully utilized!")
        else:
            print(f"❌ Error: {e}")
    
    finally:
        torch.cuda.empty_cache()
        print("\n✅ Test completed")

if __name__ == "__main__":
    diagnose_gpu_routing()