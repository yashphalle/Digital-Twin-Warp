#!/usr/bin/env python3
"""
GPU Maximization Launcher for Grounding DINO
Easily run different GPU utilization strategies
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check for nvidia-ml-py
    try:
        import pynvml
        print("✅ pynvml (NVIDIA monitoring) available")
        nvidia_monitoring = True
    except ImportError:
        print("⚠️  pynvml not available - installing...")
        nvidia_monitoring = False
        
        # Try to install
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-ml-py"])
            print("✅ pynvml installed successfully")
            nvidia_monitoring = True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pynvml - GPU monitoring will be limited")
            nvidia_monitoring = False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            gpu_available = True
        else:
            print("❌ No CUDA GPU detected")
            gpu_available = False
    except ImportError:
        print("❌ PyTorch not available")
        gpu_available = False
    
    return nvidia_monitoring, gpu_available

def show_menu():
    """Show available GPU maximization options"""
    print("\n🚀 GROUNDING DINO GPU MAXIMIZATION LAUNCHER")
    print("=" * 60)
    print("Choose your GPU utilization strategy:")
    print()
    print("1. 🔥 Maximum Parallel Processing")
    print("   - 6 parallel Grounding DINO workers")
    print("   - Smart queue management")
    print("   - Target: 70-80% GPU utilization")
    print()
    print("2. 🚀 EXTREME Batch Processing")
    print("   - Frame batching with 4 detector instances")
    print("   - 8-frame batch processing")
    print("   - Target: 85-95% GPU utilization")
    print()
    print("3. ⚡ Quick Performance Test")
    print("   - Compare both methods quickly")
    print("   - 30-second test runs")
    print()
    print("0. ❌ Exit")
    print("=" * 60)

def run_max_parallel():
    """Run maximum parallel processing version"""
    print("🔥 Starting Maximum Parallel Grounding DINO...")
    print("Target: 70-80% GPU utilization with 6 parallel workers")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, "max_grounding_dino_gpu.py"], 
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except FileNotFoundError:
        print("❌ max_grounding_dino_gpu.py not found")

def run_extreme_batch():
    """Run extreme batch processing version"""
    print("🚀 Starting EXTREME Batch Grounding DINO...")
    print("Target: 85-95% GPU utilization with frame batching")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, "extreme_grounding_dino_gpu.py"], 
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except FileNotFoundError:
        print("❌ extreme_grounding_dino_gpu.py not found")

def run_quick_test():
    """Run quick performance comparison"""
    print("⚡ Quick Performance Test")
    print("Running both methods for 30 seconds each...")
    
    methods = [
        ("Maximum Parallel", "max_grounding_dino_gpu.py"),
        ("EXTREME Batch", "extreme_grounding_dino_gpu.py")
    ]
    
    for method_name, script_name in methods:
        print(f"\n🧪 Testing {method_name}...")
        print("Running for 30 seconds...")
        
        try:
            # Run for 30 seconds
            process = subprocess.Popen([sys.executable, script_name], 
                                     cwd=os.path.dirname(os.path.abspath(__file__)))
            
            import time
            time.sleep(30)
            process.terminate()
            process.wait()
            
            print(f"✅ {method_name} test completed")
            
        except FileNotFoundError:
            print(f"❌ {script_name} not found")
        except Exception as e:
            print(f"❌ Error testing {method_name}: {e}")
    
    print("\n🏁 Quick test completed!")

def main():
    """Main launcher"""
    print("🚀 Grounding DINO GPU Maximization Launcher")
    
    # Check dependencies
    nvidia_monitoring, gpu_available = check_dependencies()
    
    if not gpu_available:
        print("❌ No GPU detected. These scripts require CUDA GPU.")
        return
    
    print("\n💡 Tips for maximum GPU utilization:")
    if nvidia_monitoring:
        print("   - Real-time GPU monitoring available")
    else:
        print("   - Limited GPU monitoring (install nvidia-ml-py for full stats)")
    print("   - Close other GPU applications")
    print("   - Monitor GPU temperature")
    print("   - Use Task Manager to verify GPU usage")
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect option (0-3): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                run_max_parallel()
            elif choice == "2":
                run_extreme_batch()
            elif choice == "3":
                run_quick_test()
            else:
                print("❌ Invalid choice. Please select 0-3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 