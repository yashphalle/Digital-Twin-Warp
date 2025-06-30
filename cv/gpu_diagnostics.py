#!/usr/bin/env python3
"""
GPU Diagnostics Script for Warehouse Tracking System
Comprehensive GPU debugging and performance analysis
"""

import torch
import sys
import os
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    logger.info("🔍 CHECKING NVIDIA DRIVER...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ NVIDIA Driver installed and working")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    logger.info(f"🔍 {line.strip()}")
            return True
        else:
            logger.error("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        logger.error("❌ nvidia-smi not found - NVIDIA drivers not installed")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ nvidia-smi timeout")
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    logger.info("\n🔍 CHECKING CUDA INSTALLATION...")
    
    # Check PyTorch CUDA
    logger.info(f"🔍 PyTorch Version: {torch.__version__}")
    logger.info(f"🔍 PyTorch CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"🔍 PyTorch CUDA Version: {torch.version.cuda}")
    logger.info(f"🔍 PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
    
    if torch.version.cuda is None:
        logger.error("❌ CRITICAL: PyTorch compiled WITHOUT CUDA!")
        logger.error("💡 SOLUTION: Reinstall PyTorch with CUDA:")
        logger.error("   pip uninstall torch torchvision torchaudio")
        logger.error("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    return torch.cuda.is_available()

def check_gpu_devices():
    """Check available GPU devices"""
    logger.info("\n🔍 CHECKING GPU DEVICES...")
    
    if not torch.cuda.is_available():
        logger.error("❌ No CUDA devices available")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"🔍 CUDA Device Count: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"🔍 Device {i}: {props.name}")
        logger.info(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        logger.info(f"   Compute Capability: {props.major}.{props.minor}")
        logger.info(f"   Multi-processors: {props.multi_processor_count}")
    
    return True

def test_gpu_allocation():
    """Test GPU memory allocation"""
    logger.info("\n🔍 TESTING GPU ALLOCATION...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available for allocation test")
        return False
    
    try:
        # Test small allocation
        logger.info("🔍 Testing small allocation (100MB)...")
        test_tensor = torch.randn(100, 1000, 1000).cuda()
        logger.info("✅ Small allocation successful")
        del test_tensor
        torch.cuda.empty_cache()
        
        # Test larger allocation (1GB)
        logger.info("🔍 Testing larger allocation (1GB)...")
        test_tensor = torch.randn(1000, 1000, 1000).cuda()
        logger.info("✅ Large allocation successful")
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU allocation failed: {e}")
        return False

def test_grounding_dino_model():
    """Test Grounding DINO model loading"""
    logger.info("\n🔍 TESTING GROUNDING DINO MODEL...")
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        model_id = "IDEA-Research/grounding-dino-base"
        logger.info(f"🔍 Loading model: {model_id}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("✅ Processor loaded")
        
        # Load model
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        logger.info("✅ Model loaded")
        
        if torch.cuda.is_available():
            logger.info("🔍 Moving model to GPU...")
            model = model.to('cuda')
            
            # Verify device placement
            model_device = next(model.parameters()).device
            logger.info(f"🔍 Model device: {model_device}")
            
            if str(model_device) == 'cuda:0':
                logger.info("✅ Model successfully on GPU")
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"🔍 GPU Memory used by model: {memory_used:.2f}GB")
                
                return True
            else:
                logger.error(f"❌ Model not on GPU: {model_device}")
                return False
        else:
            logger.warning("⚠️ CUDA not available, model on CPU")
            return False
            
    except Exception as e:
        logger.error(f"❌ Grounding DINO test failed: {e}")
        return False

def analyze_memory_requirements():
    """Analyze memory requirements for multi-camera setup"""
    logger.info("\n🔍 ANALYZING MEMORY REQUIREMENTS...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available for memory analysis")
        return
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"🔍 Total GPU Memory: {total_memory:.2f}GB")
    
    # Estimate Grounding DINO memory usage
    estimated_model_size = 6.0  # GB (approximate for grounding-dino-base)
    estimated_per_frame = 0.5   # GB per frame processing
    
    logger.info(f"🔍 Estimated model size: {estimated_model_size:.1f}GB")
    logger.info(f"🔍 Estimated per-frame processing: {estimated_per_frame:.1f}GB")
    
    # Calculate camera capacity
    available_for_frames = total_memory - estimated_model_size - 1.0  # 1GB buffer
    max_cameras = int(available_for_frames / estimated_per_frame)
    
    logger.info(f"🔍 Available memory for frames: {available_for_frames:.1f}GB")
    logger.info(f"🔍 Estimated max cameras (parallel): {max_cameras}")
    
    if max_cameras < 11:
        logger.warning(f"⚠️ GPU may not handle all 11 cameras simultaneously")
        logger.warning(f"💡 Consider sequential processing or frame skipping")
    else:
        logger.info(f"✅ GPU should handle all 11 cameras")

def main():
    """Run comprehensive GPU diagnostics"""
    logger.info("🚀 STARTING GPU DIAGNOSTICS FOR WAREHOUSE TRACKING SYSTEM")
    logger.info("=" * 70)
    logger.info(f"🔍 Timestamp: {datetime.now()}")
    logger.info(f"🔍 Python Version: {sys.version}")
    logger.info(f"🔍 Platform: {sys.platform}")
    
    results = {}
    
    # Run all diagnostic tests
    results['nvidia_driver'] = check_nvidia_driver()
    results['cuda_installation'] = check_cuda_installation()
    results['gpu_devices'] = check_gpu_devices()
    results['gpu_allocation'] = test_gpu_allocation()
    results['grounding_dino'] = test_grounding_dino_model()
    
    # Memory analysis
    analyze_memory_requirements()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 DIAGNOSTIC SUMMARY:")
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {test.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - GPU should work correctly!")
    else:
        logger.info("\n⚠️ SOME TESTS FAILED - GPU issues detected")
        logger.info("💡 Check the error messages above for solutions")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
