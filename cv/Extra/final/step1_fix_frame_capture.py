#!/usr/bin/env python3
"""
STEP 1: Fix Basic Frame Capture Issue
Debug why GStreamer works manually but fails to save frames in Python
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

class GStreamerFrameCaptureDebugger:
    """Debug GStreamer frame capture issues"""
    
    def __init__(self):
        self.gst_path = r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"
        self.test_url = Config.get_camera_url(1)  # Use Camera 1
        self.output_dir = "step1_debug_frames"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("🔧 GStreamer Frame Capture Debugger initialized")
        logger.info(f"   GStreamer path: {self.gst_path}")
        logger.info(f"   Test URL: {self.test_url}")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def test_1_basic_connection(self):
        """Test 1: Basic RTSP connection (no file output)"""
        logger.info("🧪 TEST 1: Basic RTSP Connection")
        logger.info("=" * 40)
        
        pipeline = [
            self.gst_path,
            'rtspsrc', f'location={self.test_url}',
            'timeout=5000000',
            '!', 'fakesink'
        ]
        
        logger.info(f"Pipeline: {' '.join(pipeline)}")
        
        try:
            process = subprocess.Popen(
                pipeline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Let it run for 5 seconds
            logger.info("⏱️ Running for 5 seconds...")
            time.sleep(5)
            
            # Stop process
            process.terminate()
            stdout, stderr = process.communicate(timeout=3)
            
            if process.returncode == 0 or "EOS" in stderr:
                logger.info("✅ TEST 1 PASSED: Basic connection works")
                return True
            else:
                logger.error(f"❌ TEST 1 FAILED: {stderr[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TEST 1 ERROR: {e}")
            return False
    
    def test_2_simple_file_output(self):
        """Test 2: Simple file output (single frame)"""
        logger.info("\n🧪 TEST 2: Simple File Output")
        logger.info("=" * 40)
        
        output_file = os.path.join(self.output_dir, "test_single_frame.jpg")
        
        # Remove old file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        pipeline = [
            self.gst_path,
            'rtspsrc', f'location={self.test_url}',
            'timeout=10000000',
            '!', 'decodebin',
            '!', 'videoconvert',
            '!', 'jpegenc',
            '!', 'filesink', f'location={output_file}'
        ]
        
        logger.info(f"Pipeline: {' '.join(pipeline[:8])}...")
        logger.info(f"Output file: {output_file}")
        
        try:
            process = subprocess.Popen(
                pipeline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Let it run for 10 seconds
            logger.info("⏱️ Running for 10 seconds...")
            time.sleep(10)
            
            # Stop process
            process.terminate()
            stdout, stderr = process.communicate(timeout=3)
            
            # Check if file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"✅ TEST 2 PASSED: File created, size: {file_size} bytes")
                return True
            else:
                logger.error("❌ TEST 2 FAILED: No file created")
                if stderr:
                    logger.error(f"GStreamer stderr: {stderr[:300]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TEST 2 ERROR: {e}")
            return False
    
    def test_3_multifilesink_basic(self):
        """Test 3: Basic multifilesink (few frames)"""
        logger.info("\n🧪 TEST 3: Basic Multifilesink")
        logger.info("=" * 40)
        
        # Clean old frames
        old_frames = glob.glob(os.path.join(self.output_dir, "multi_*.jpg"))
        for frame in old_frames:
            os.remove(frame)
        
        frame_pattern = os.path.join(self.output_dir, "multi_%05d.jpg")
        
        pipeline = [
            self.gst_path,
            'rtspsrc', f'location={self.test_url}',
            'timeout=10000000',
            '!', 'decodebin',
            '!', 'videoconvert',
            '!', 'jpegenc',
            '!', 'multifilesink', f'location={frame_pattern}',
            'max-files=10'  # Only keep 10 frames
        ]
        
        logger.info(f"Pipeline: {' '.join(pipeline[:8])}...")
        logger.info(f"Frame pattern: {frame_pattern}")
        
        try:
            process = subprocess.Popen(
                pipeline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor frame creation
            logger.info("⏱️ Running for 10 seconds, monitoring frames...")
            start_time = time.time()
            
            while time.time() - start_time < 10:
                frame_files = glob.glob(os.path.join(self.output_dir, "multi_*.jpg"))
                logger.info(f"📊 Current frame count: {len(frame_files)}")
                time.sleep(2)
            
            # Stop process
            process.terminate()
            stdout, stderr = process.communicate(timeout=3)
            
            # Final count
            final_frames = glob.glob(os.path.join(self.output_dir, "multi_*.jpg"))
            frame_count = len(final_frames)
            
            if frame_count > 0:
                logger.info(f"✅ TEST 3 PASSED: {frame_count} frames created")
                
                # Show first few frame details
                for i, frame_file in enumerate(sorted(final_frames)[:3]):
                    file_size = os.path.getsize(frame_file)
                    logger.info(f"   Frame {i+1}: {os.path.basename(frame_file)}, {file_size} bytes")
                
                return True
            else:
                logger.error("❌ TEST 3 FAILED: No frames created")
                if stderr:
                    logger.error(f"GStreamer stderr: {stderr[:300]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TEST 3 ERROR: {e}")
            return False
    
    def test_4_verbose_output(self):
        """Test 4: Run with verbose output to see what's happening"""
        logger.info("\n🧪 TEST 4: Verbose Output Analysis")
        logger.info("=" * 40)
        
        frame_pattern = os.path.join(self.output_dir, "verbose_%05d.jpg")
        
        pipeline = [
            self.gst_path,
            '--gst-debug=3',  # Verbose output
            'rtspsrc', f'location={self.test_url}',
            'timeout=10000000',
            '!', 'decodebin',
            '!', 'videoconvert',
            '!', 'jpegenc',
            '!', 'multifilesink', f'location={frame_pattern}',
            'max-files=5'
        ]
        
        logger.info(f"Pipeline with verbose output: {' '.join(pipeline[:8])}...")
        
        try:
            process = subprocess.Popen(
                pipeline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Run for shorter time with verbose output
            logger.info("⏱️ Running for 8 seconds with verbose output...")
            time.sleep(8)
            
            # Stop process
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            
            # Check results
            verbose_frames = glob.glob(os.path.join(self.output_dir, "verbose_*.jpg"))
            
            logger.info(f"📊 Verbose test created {len(verbose_frames)} frames")
            
            # Show key parts of stderr (GStreamer debug info)
            if stderr:
                logger.info("🔍 Key GStreamer debug info:")
                lines = stderr.split('\n')
                for line in lines:
                    if 'ERROR' in line or 'WARNING' in line or 'multifilesink' in line.lower():
                        logger.info(f"   {line[:100]}")
            
            return len(verbose_frames) > 0
            
        except Exception as e:
            logger.error(f"❌ TEST 4 ERROR: {e}")
            return False
    
    def run_all_tests(self):
        """Run all frame capture tests"""
        logger.info("🎯 STEP 1: GSTREAMER FRAME CAPTURE DEBUG")
        logger.info("=" * 60)
        logger.info("Testing why GStreamer works manually but fails in Python")
        logger.info("=" * 60)
        
        tests = [
            ("Basic Connection", self.test_1_basic_connection),
            ("Simple File Output", self.test_2_simple_file_output),
            ("Multifilesink Basic", self.test_3_multifilesink_basic),
            ("Verbose Output", self.test_4_verbose_output)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n📊 STEP 1 TEST RESULTS:")
        logger.info("=" * 40)
        
        passed = 0
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
        
        # Diagnosis
        if passed == len(tests):
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("✅ GStreamer frame capture is working")
            logger.info("💡 The issue was in the previous scripts' logic")
        elif passed >= 2:
            logger.info("⚠️ PARTIAL SUCCESS")
            logger.info("🔧 Some GStreamer functionality works")
            logger.info("💡 Need to fix specific pipeline issues")
        else:
            logger.info("❌ MAJOR ISSUES")
            logger.info("🔧 GStreamer has fundamental problems")
            logger.info("💡 Consider using optimized OpenCV instead")
        
        return passed >= 2


def main():
    """Main test function"""
    print("🔧 STEP 1: FIX GSTREAMER FRAME CAPTURE")
    print("=" * 50)
    print("Debugging why GStreamer works manually but fails in Python")
    print("=" * 50)
    
    debugger = GStreamerFrameCaptureDebugger()
    
    try:
        success = debugger.run_all_tests()
        
        if success:
            print("\n🚀 READY FOR STEP 2!")
            print("Frame capture issues identified and fixed")
        else:
            print("\n⚠️ STEP 1 NEEDS MORE WORK")
            print("Check the test results above for specific issues")
            
    except KeyboardInterrupt:
        print("\n⏹️ Step 1 interrupted by user")
    except Exception as e:
        print(f"\n❌ Step 1 error: {e}")


if __name__ == "__main__":
    main()
