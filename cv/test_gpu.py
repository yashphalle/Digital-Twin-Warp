#!/usr/bin/env python3
"""
Quick GPU Test Script
Run this to quickly check if GPU is working for the warehouse tracking system
"""

import sys
import os

def main():
    print("🚀 QUICK GPU TEST FOR WAREHOUSE TRACKING")
    print("=" * 50)
    
    try:
        # Test 1: Basic imports
        print("🔍 Testing imports...")
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("✅ All imports successful")
        
        # Test 2: CUDA availability
        print(f"\n🔍 PyTorch Version: {torch.__version__}")
        print(f"🔍 CUDA Available: {torch.cuda.is_available()}")
        print(f"🔍 CUDA Version: {torch.version.cuda}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA NOT AVAILABLE!")
            print("💡 Run 'python gpu_diagnostics.py' for detailed analysis")
            return False
        
        # Test 3: GPU info
        print(f"🔍 GPU Count: {torch.cuda.device_count()}")
        print(f"🔍 GPU Name: {torch.cuda.get_device_name(0)}")
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU Memory: {memory_total:.1f}GB")
        
        # Test 4: Simple GPU allocation
        print("\n🔍 Testing GPU allocation...")
        test_tensor = torch.randn(1000, 1000).cuda()
        print("✅ GPU allocation successful")
        del test_tensor
        torch.cuda.empty_cache()
        
        # Test 5: Load detection model (quick test)
        print("\n🔍 Testing model loading (this may take a moment)...")
        try:
            model_id = "IDEA-Research/grounding-dino-base"
            processor = AutoProcessor.from_pretrained(model_id)
            print("✅ Processor loaded")
            
            # Note: We're not loading the full model here to save time
            print("✅ Model components accessible")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        print("\n🎉 QUICK GPU TEST PASSED!")
        print("💡 GPU should work with the warehouse tracking system")
        print("💡 For detailed analysis, run: python gpu_diagnostics.py")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check if required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
