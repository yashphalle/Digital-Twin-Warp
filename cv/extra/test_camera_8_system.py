"""
Test Script for Camera 8 Multi-Camera System
Tests Camera 8 functionality within the 11-camera architecture
"""

import cv2
import time
import logging
from datetime import datetime
from typing import Optional

# Import system components
from config import Config
from multi_camera_tracking_system import MultiCameraTrackingSystem
from rtsp_camera_manager import MultiCameraRTSPManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera8TestSuite:
    """Test suite for Camera 8 functionality"""
    
    def __init__(self):
        logger.info("🧪 Initializing Camera 8 Test Suite")
        self.test_results = {}
        
    def test_configuration(self):
        """Test Camera 8 configuration"""
        logger.info("🔧 Testing Camera 8 Configuration...")
        
        tests = []
        
        # Test Camera 8 is in active cameras
        if 8 in Config.ACTIVE_CAMERAS:
            tests.append(("Camera 8 in active list", True))
        else:
            tests.append(("Camera 8 in active list", False))
        
        # Test Camera 8 has RTSP URL
        if 8 in Config.RTSP_CAMERA_URLS:
            tests.append(("Camera 8 RTSP URL configured", True))
            logger.info(f"   📹 RTSP URL: {Config.RTSP_CAMERA_URLS[8]}")
        else:
            tests.append(("Camera 8 RTSP URL configured", False))
        
        # Test Camera 8 has name
        if 8 in Config.CAMERA_NAMES:
            tests.append(("Camera 8 name configured", True))
            logger.info(f"   🏷️  Name: {Config.CAMERA_NAMES[8]}")
        else:
            tests.append(("Camera 8 name configured", False))
        
        # Test Camera 8 has coverage zone
        if 8 in Config.CAMERA_COVERAGE_ZONES:
            tests.append(("Camera 8 coverage zone configured", True))
            zone = Config.CAMERA_COVERAGE_ZONES[8]
            logger.info(f"   📍 Coverage: {zone['x_start']}-{zone['x_end']}ft x {zone['y_start']}-{zone['y_end']}ft")
        else:
            tests.append(("Camera 8 coverage zone configured", False))
        
        self.test_results["configuration"] = tests
        
        passed = all(result for _, result in tests)
        logger.info(f"✅ Configuration tests: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    def test_camera_manager(self):
        """Test Camera 8 in camera manager"""
        logger.info("📹 Testing Camera Manager...")
        
        tests = []
        
        try:
            # Initialize camera manager
            manager = MultiCameraRTSPManager()
            
            # Test Camera 8 pipeline exists
            if 8 in manager.camera_pipelines:
                tests.append(("Camera 8 pipeline created", True))
            else:
                tests.append(("Camera 8 pipeline created", False))
            
            # Test Camera 8 is in active cameras
            if 8 in manager.active_cameras:
                tests.append(("Camera 8 in active cameras", True))
            else:
                tests.append(("Camera 8 in active cameras", False))
            
            # Test Camera 8 fisheye corrector
            if 8 in manager.fisheye_correctors:
                tests.append(("Camera 8 fisheye corrector created", True))
            else:
                tests.append(("Camera 8 fisheye corrector created", False))
            
            # Test get camera stats
            stats = manager.get_camera_stats(8)
            if stats and stats.get('camera_id') == 8:
                tests.append(("Camera 8 stats available", True))
                logger.info(f"   📊 Status: {stats.get('status', 'unknown')}")
            else:
                tests.append(("Camera 8 stats available", False))
            
            manager.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Camera manager test failed: {e}")
            tests.append(("Camera manager initialization", False))
        
        self.test_results["camera_manager"] = tests
        
        passed = all(result for _, result in tests)
        logger.info(f"✅ Camera Manager tests: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    def test_rtsp_connection(self):
        """Test RTSP connection to Camera 8"""
        logger.info("🔗 Testing Camera 8 RTSP Connection...")
        
        tests = []
        
        try:
            rtsp_url = Config.RTSP_CAMERA_URLS[8]
            logger.info(f"   🔗 Connecting to: {rtsp_url}")
            
            # Test basic OpenCV connection
            cap = cv2.VideoCapture(rtsp_url)
            
            if cap.isOpened():
                tests.append(("RTSP stream opens", True))
                
                # Test frame read
                ret, frame = cap.read()
                if ret and frame is not None:
                    tests.append(("Frame read successful", True))
                    logger.info(f"   📏 Frame size: {frame.shape}")
                    
                    # Test frame is not black
                    frame_mean = frame.mean()
                    if frame_mean > 10:
                        tests.append(("Frame has content", True))
                        logger.info(f"   🌟 Frame brightness: {frame_mean:.1f}")
                    else:
                        tests.append(("Frame has content", False))
                        logger.warning(f"   ⚠️  Frame may be black: brightness {frame_mean:.1f}")
                else:
                    tests.append(("Frame read successful", False))
                    tests.append(("Frame has content", False))
            else:
                tests.append(("RTSP stream opens", False))
                tests.append(("Frame read successful", False))
                tests.append(("Frame has content", False))
            
            cap.release()
            
        except Exception as e:
            logger.error(f"❌ RTSP connection test failed: {e}")
            tests.append(("RTSP stream opens", False))
            tests.append(("Frame read successful", False))
            tests.append(("Frame has content", False))
        
        self.test_results["rtsp_connection"] = tests
        
        passed = all(result for _, result in tests)
        logger.info(f"✅ RTSP Connection tests: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    def test_tracking_integration(self):
        """Test Camera 8 integration with tracking system"""
        logger.info("🎯 Testing Tracking System Integration...")
        
        tests = []
        
        try:
            # Initialize tracking system
            tracking_system = MultiCameraTrackingSystem()
            
            # Test Camera 8 in processing cameras
            if 8 in tracking_system.processing_active_cameras:
                tests.append(("Camera 8 in processing list", True))
            else:
                tests.append(("Camera 8 in processing list", False))
            
            # Test system status
            status = tracking_system.get_system_status()
            if status and isinstance(status, dict):
                tests.append(("System status available", True))
                logger.info(f"   📊 Processing cameras: {status.get('processing_cameras', [])}")
            else:
                tests.append(("System status available", False))
            
            # Test enable/disable functionality
            try:
                # Test enable camera (should already be enabled)
                result = tracking_system.enable_additional_camera(8)
                tests.append(("Enable camera function", result))
                
                # Test disable camera
                result = tracking_system.disable_camera(8)
                tests.append(("Disable camera function", result))
                
                # Re-enable for next tests
                tracking_system.enable_additional_camera(8)
                
            except Exception as e:
                logger.warning(f"⚠️  Enable/disable test failed: {e}")
                tests.append(("Enable camera function", False))
                tests.append(("Disable camera function", False))
            
            tracking_system._cleanup()
            
        except Exception as e:
            logger.error(f"❌ Tracking integration test failed: {e}")
            tests.append(("Camera 8 in processing list", False))
            tests.append(("System status available", False))
            tests.append(("Enable camera function", False))
            tests.append(("Disable camera function", False))
        
        self.test_results["tracking_integration"] = tests
        
        passed = all(result for _, result in tests)
        logger.info(f"✅ Tracking Integration tests: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    def test_frame_processing(self):
        """Test Camera 8 frame processing pipeline"""
        logger.info("🖼️  Testing Frame Processing Pipeline...")
        
        tests = []
        
        try:
            # Initialize camera manager
            manager = MultiCameraRTSPManager()
            manager.start_active_cameras()
            
            # Wait for connection
            time.sleep(3)
            
            # Test frame retrieval
            frame_data = manager.get_frame(8)
            
            if frame_data:
                frame, timestamp = frame_data
                tests.append(("Frame retrieval", True))
                
                # Test frame properties
                if frame is not None and frame.size > 0:
                    tests.append(("Frame validity", True))
                    logger.info(f"   📏 Processed frame size: {frame.shape}")
                    
                    # Test frame is processed size (not original 4K)
                    if frame.shape[1] <= Config.RTSP_PROCESSING_WIDTH:
                        tests.append(("Frame scaling", True))
                        logger.info(f"   📐 Frame properly scaled from 4K to {frame.shape[1]}x{frame.shape[0]}")
                    else:
                        tests.append(("Frame scaling", False))
                    
                    # Test timestamp
                    if timestamp and isinstance(timestamp, datetime):
                        tests.append(("Frame timestamp", True))
                    else:
                        tests.append(("Frame timestamp", False))
                else:
                    tests.append(("Frame validity", False))
                    tests.append(("Frame scaling", False))
                    tests.append(("Frame timestamp", False))
            else:
                tests.append(("Frame retrieval", False))
                tests.append(("Frame validity", False))
                tests.append(("Frame scaling", False))
                tests.append(("Frame timestamp", False))
            
            manager.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Frame processing test failed: {e}")
            tests.append(("Frame retrieval", False))
            tests.append(("Frame validity", False))
            tests.append(("Frame scaling", False))
            tests.append(("Frame timestamp", False))
        
        self.test_results["frame_processing"] = tests
        
        passed = all(result for _, result in tests)
        logger.info(f"✅ Frame Processing tests: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    def run_all_tests(self):
        """Run all Camera 8 tests"""
        logger.info("🚀 Running Camera 8 Complete Test Suite")
        logger.info("=" * 60)
        
        test_methods = [
            ("Configuration", self.test_configuration),
            ("Camera Manager", self.test_camera_manager),
            ("RTSP Connection", self.test_rtsp_connection),
            ("Tracking Integration", self.test_tracking_integration),
            ("Frame Processing", self.test_frame_processing)
        ]
        
        results = {}
        
        for test_name, test_method in test_methods:
            logger.info(f"\n🧪 Running {test_name} Tests...")
            try:
                results[test_name] = test_method()
            except Exception as e:
                logger.error(f"❌ {test_name} test suite failed: {e}")
                results[test_name] = False
        
        # Print summary
        self.print_test_summary(results)
        
        return results
    
    def print_test_summary(self, results):
        """Print comprehensive test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 CAMERA 8 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_suite, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test_suite}: {status}")
            
            # Count individual tests
            if test_suite in self.test_results:
                suite_tests = self.test_results[test_suite]
                suite_passed = sum(1 for _, result in suite_tests if result)
                suite_total = len(suite_tests)
                total_tests += suite_total
                passed_tests += suite_passed
                
                logger.info(f"   📊 {suite_passed}/{suite_total} individual tests passed")
        
        overall_success = all(results.values())
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("-" * 60)
        logger.info(f"🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        logger.info(f"📈 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info("=" * 60)
        
        return overall_success

def run_quick_test():
    """Run a quick Camera 8 connectivity test"""
    logger.info("⚡ Quick Camera 8 Test")
    logger.info("-" * 30)
    
    try:
        # Test RTSP URL
        rtsp_url = Config.RTSP_CAMERA_URLS[8]
        logger.info(f"🔗 Testing: {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"✅ Camera 8 connected: {frame.shape}")
                logger.info(f"🌟 Frame brightness: {frame.mean():.1f}")
            else:
                logger.error("❌ Failed to read frame")
        else:
            logger.error("❌ Failed to connect to RTSP stream")
        cap.release()
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")

def main():
    """Main test execution"""
    logger.info("🏁 Camera 8 Multi-Camera System Test")
    logger.info("=" * 50)
    
    print("\nChoose test mode:")
    print("1. Quick connectivity test")
    print("2. Full test suite")
    print("3. Configuration check only")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_quick_test()
    elif choice == "2":
        test_suite = Camera8TestSuite()
        test_suite.run_all_tests()
    elif choice == "3":
        test_suite = Camera8TestSuite()
        test_suite.test_configuration()
    else:
        logger.info("ℹ️  Running quick test by default...")
        run_quick_test()

if __name__ == "__main__":
    main() 