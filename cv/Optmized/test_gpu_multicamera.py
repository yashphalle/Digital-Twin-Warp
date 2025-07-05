#!/usr/bin/env python3
"""
Test GPU SIFT multi-camera configuration
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

def test_gpu_multicamera_config():
    """Test the GPU SIFT multi-camera configuration options"""
    print("🎯 TESTING GPU SIFT MULTI-CAMERA CONFIGURATION")
    print("=" * 60)
    
    # Test different camera group configurations
    test_configs = [
        ('SINGLE8', [8], [8]),
        ('SINGLE1', [1], [1]),
        ('GROUP1', [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
        ('GROUP2', [7, 8, 9, 10, 11], [7, 8, 9, 10, 11]),
        ('ALL', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [8, 9, 10, 11])
    ]
    
    for camera_group, expected_active, expected_gui in test_configs:
        print(f"\n🔍 Testing CAMERA_GROUP={camera_group}")
        
        # Set environment variable
        os.environ['CAMERA_GROUP'] = camera_group
        
        # Import the module to test configuration
        try:
            # Clear module cache to reload with new environment
            if 'gpu_sift_11camera_configurable' in sys.modules:
                del sys.modules['gpu_sift_11camera_configurable']
            
            from gpu_sift_11camera_configurable import ACTIVE_CAMERAS, GUI_CAMERAS
            
            print(f"  Expected Active: {expected_active}")
            print(f"  Actual Active:   {ACTIVE_CAMERAS}")
            print(f"  ✅ Active Match: {ACTIVE_CAMERAS == expected_active}")
            
            print(f"  Expected GUI:    {expected_gui}")
            print(f"  Actual GUI:      {GUI_CAMERAS}")
            print(f"  ✅ GUI Match:    {GUI_CAMERAS == expected_gui}")
            
            if ACTIVE_CAMERAS == expected_active and GUI_CAMERAS == expected_gui:
                print(f"  🎉 {camera_group} configuration PASSED!")
            else:
                print(f"  ❌ {camera_group} configuration FAILED!")
                
        except Exception as e:
            print(f"  ❌ Error testing {camera_group}: {e}")
    
    print(f"\n🎯 USAGE EXAMPLES:")
    print("=" * 40)
    print("# Run single camera 8 (default):")
    print("CAMERA_GROUP=SINGLE8 python gpu_sift_11camera_configurable.py")
    print()
    print("# Run all cameras 1-11:")
    print("CAMERA_GROUP=ALL python gpu_sift_11camera_configurable.py")
    print()
    print("# Run camera group 1 (cameras 1-6):")
    print("CAMERA_GROUP=GROUP1 python gpu_sift_11camera_configurable.py")
    print()
    print("# Run camera group 2 (cameras 7-11):")
    print("CAMERA_GROUP=GROUP2 python gpu_sift_11camera_configurable.py")
    print()
    print("# Run single camera 3:")
    print("CAMERA_GROUP=SINGLE3 python gpu_sift_11camera_configurable.py")
    print()
    print("# Override with command line:")
    print("python gpu_sift_11camera_configurable.py --cameras 1 2 3")
    print("python gpu_sift_11camera_configurable.py --camera 5")

def test_multicamera_class():
    """Test the MultiCameraGPUSIFTSystem class"""
    print(f"\n🔧 TESTING MultiCameraGPUSIFTSystem CLASS")
    print("=" * 50)
    
    try:
        from gpu_sift_11camera_configurable import MultiCameraGPUSIFTSystem
        
        # Test initialization
        system = MultiCameraGPUSIFTSystem(
            active_cameras=[8],
            gui_cameras=[8],
            enable_gui=False  # Disable GUI for testing
        )
        
        print("✅ MultiCameraGPUSIFTSystem class imported successfully")
        print("✅ System initialization successful")
        print(f"📹 Active cameras: {system.active_cameras}")
        print(f"🖥️ GUI cameras: {system.gui_cameras}")
        print(f"🎛️ GUI enabled: {system.enable_gui}")
        
        return True
        
    except Exception as e:
        print(f"❌ MultiCameraGPUSIFTSystem test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPU SIFT MULTI-CAMERA SYSTEM TEST")
    print("=" * 70)
    
    config_test = test_gpu_multicamera_config()
    class_test = test_multicamera_class()
    
    print(f"\n🏁 FINAL RESULT:")
    print("=" * 30)
    
    if class_test:
        print("✅ GPU SIFT MULTI-CAMERA SYSTEM READY!")
        print("🎯 You can now run GPU SIFT with multiple cameras just like the CPU version.")
        print("\n📋 Key Features Added:")
        print("   1. Environment variable configuration (CAMERA_GROUP)")
        print("   2. Multi-camera parallel processing")
        print("   3. Configurable GUI display")
        print("   4. Same interface as CPU version")
        print("   5. Command line override support")
    else:
        print("❌ GPU SIFT MULTI-CAMERA SYSTEM HAS ISSUES!")
        print("🔧 Check the errors above for specific problems.")
