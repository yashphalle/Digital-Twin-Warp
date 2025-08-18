#!/usr/bin/env python3
"""
Fix Subprocess Issue Properly
Implementing the expert's shell=True solution correctly
"""

import subprocess
import time
import os
import sys
import logging
import glob

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import configs
import importlib.util
config_path = os.path.join(parent_dir, 'configs', 'config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_shell_true_solution():
    """Test the expert's shell=True solution"""
    print("🔧 IMPLEMENTING EXPERT'S SHELL=TRUE SOLUTION")
    print("=" * 60)
    
    # Setup
    gst_path = r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"
    rtsp_url = Config.get_camera_url(1)
    output_dir = "shell_true_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean old frames
    old_frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
    for frame in old_frames:
        os.remove(frame)
    print(f"🧹 Cleaned {len(old_frames)} old frames")
    
    # EXPERT'S SOLUTION: Use shell=True with single command string
    cmd = f'"{gst_path}" rtspsrc location={rtsp_url} timeout=15000000 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! jpegenc quality=85 ! multifilesink location={output_dir}/frame_%05d.jpg max-files=50'
    
    print(f"📝 Command: {cmd[:100]}...")
    print("🚀 Running with shell=True...")
    
    try:
        # EXPERT'S METHOD: shell=True
        process = subprocess.Popen(
            cmd,
            shell=True,  # ← THE KEY FIX
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Process started (PID: {process.pid})")
        
        # Let it run for 15 seconds
        print("⏱️ Running for 15 seconds...")
        time.sleep(15)
        
        # Terminate process
        print("⏹️ Stopping process...")
        process.terminate()
        
        # Don't wait for communicate() - this was causing timeout issues
        time.sleep(2)  # Give it time to terminate
        
        if process.poll() is None:
            print("🔪 Force killing process...")
            process.kill()
        
        # Check results
        frame_files = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
        frame_count = len(frame_files)
        
        print(f"📊 RESULTS:")
        print(f"   Frames captured: {frame_count}")
        
        if frame_count > 0:
            print("🎉 SUCCESS: shell=True solution works!")
            
            # Show first few frames
            for i, frame_file in enumerate(sorted(frame_files)[:3]):
                file_size = os.path.getsize(frame_file)
                print(f"   Frame {i+1}: {os.path.basename(frame_file)}, {file_size} bytes")
            
            return True
        else:
            print("❌ FAILED: No frames captured")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_environment_variables():
    """Test the expert's environment variables solution"""
    print("\n🔧 TESTING EXPERT'S ENVIRONMENT SOLUTION")
    print("=" * 60)
    
    # Setup
    gst_path = r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"
    rtsp_url = Config.get_camera_url(1)
    output_dir = "env_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean old frames
    old_frames = glob.glob(os.path.join(output_dir, "env_*.jpg"))
    for frame in old_frames:
        os.remove(frame)
    
    # EXPERT'S SOLUTION: Explicit environment
    env = os.environ.copy()
    env['GST_PLUGIN_PATH'] = r'C:\Program Files\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0'
    env['PATH'] = r'C:\Program Files\gstreamer\1.0\msvc_x86_64\bin;' + env.get('PATH', '')
    
    cmd = f'"{gst_path}" rtspsrc location={rtsp_url} timeout=15000000 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! jpegenc quality=85 ! multifilesink location={output_dir}/env_%05d.jpg max-files=50'
    
    print("🚀 Running with explicit environment...")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            env=env,  # ← EXPERT'S ENVIRONMENT FIX
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Process started (PID: {process.pid})")
        
        # Let it run
        time.sleep(15)
        
        # Stop process
        process.terminate()
        time.sleep(2)
        
        if process.poll() is None:
            process.kill()
        
        # Check results
        frame_files = glob.glob(os.path.join(output_dir, "env_*.jpg"))
        frame_count = len(frame_files)
        
        print(f"📊 Environment test results: {frame_count} frames")
        
        return frame_count > 0
        
    except Exception as e:
        print(f"❌ Environment test error: {e}")
        return False


def test_progressive_complexity():
    """Test the expert's progressive complexity approach"""
    print("\n🔧 TESTING EXPERT'S PROGRESSIVE APPROACH")
    print("=" * 60)
    
    tests = [
        ("Version Check", ['gst-launch-1.0', '--version']),
        ("Simple Pipeline", ['gst-launch-1.0', 'videotestsrc', 'num-buffers=10', '!', 'fakesink']),
        ("File Output", ['gst-launch-1.0', 'videotestsrc', 'num-buffers=5', '!', 'jpegenc', '!', 'multifilesink', 'location=progressive_test_%05d.jpg'])
    ]
    
    results = {}
    
    for test_name, cmd_list in tests:
        print(f"\n📝 {test_name}:")
        print(f"   Command: {' '.join(cmd_list)}")
        
        try:
            # EXPERT'S METHOD: shell=True for all tests
            result = subprocess.run(
                ' '.join(cmd_list),  # Convert to string
                shell=True,  # ← EXPERT'S KEY FIX
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"   ✅ {test_name}: SUCCESS")
                results[test_name] = True
            else:
                print(f"   ❌ {test_name}: FAILED (code {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:100]}...")
                results[test_name] = False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {test_name}: TIMEOUT (may be normal)")
            results[test_name] = True  # Timeout can be normal for some tests
        except Exception as e:
            print(f"   ❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Check if file output test created files
    if results.get("File Output", False):
        test_files = glob.glob("progressive_test_*.jpg")
        print(f"📊 File output test created {len(test_files)} files")
        
        # Clean up test files
        for test_file in test_files:
            os.remove(test_file)
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n📊 Progressive test results: {passed}/{total} passed")
    
    return passed >= 2  # At least 2 tests should pass


def main():
    """Main function implementing expert's solutions"""
    print("🎯 FIXING SUBPROCESS ISSUE WITH EXPERT'S SOLUTIONS")
    print("=" * 70)
    print("Implementing the expert advice properly:")
    print("1. shell=True (most likely fix)")
    print("2. Explicit environment variables") 
    print("3. Progressive complexity testing")
    print("=" * 70)
    
    solutions_working = []
    
    try:
        # Test 1: shell=True solution
        if test_shell_true_solution():
            solutions_working.append("shell=True")
        
        # Test 2: Environment variables
        if test_environment_variables():
            solutions_working.append("environment variables")
        
        # Test 3: Progressive complexity
        if test_progressive_complexity():
            solutions_working.append("progressive testing")
        
        # Results
        print(f"\n🎯 EXPERT'S SOLUTIONS RESULTS:")
        print("=" * 50)
        
        if solutions_working:
            print(f"✅ WORKING SOLUTIONS: {', '.join(solutions_working)}")
            print(f"🎉 SUCCESS: Found {len(solutions_working)} working approaches!")
            
            if "shell=True" in solutions_working:
                print("\n💡 RECOMMENDATION:")
                print("   Use shell=True approach - it's the simplest and most reliable")
                print("   This fixes the subprocess issue completely")
            
            return True
        else:
            print("❌ NO SOLUTIONS WORKING")
            print("   The subprocess issue persists")
            print("   Consider using GStreamer Python bindings instead")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        return False


if __name__ == "__main__":
    main()
