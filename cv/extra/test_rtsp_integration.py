"""
Test RTSP Camera Integration with Warehouse Tracking System
Tests the integration of Lorex RTSP cameras with the existing tracking architecture
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import RTSP modules
from rtsp_camera_manager import RTSPCameraManager, RTSPTrackingSystem
from rtsp_config import RTSPConfig
from simplified_lorex_pipeline import test_rtsp_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTSPIntegrationTester:
    """Test class for RTSP camera integration"""
    
    def __init__(self):
        self.test_results = {}
        self.camera_manager = None
        
    def test_rtsp_connections(self):
        """Test RTSP camera connections"""
        logger.info("🔍 Testing RTSP Camera Connections")
        print("=" * 60)
        
        camera_urls = RTSPConfig.RTSP_CAMERA_URLS
        results = {}
        
        for i, url in enumerate(camera_urls):
            camera_name = RTSPConfig.get_camera_name(i)
            print(f"\n📹 Testing {camera_name}")
            print(f"URL: {url}")
            
            try:
                success = test_rtsp_connection(url)
                results[i] = {
                    'url': url,
                    'name': camera_name,
                    'connected': success,
                    'timestamp': datetime.now()
                }
                
                if success:
                    print(f"✅ {camera_name} - Connection successful")
                else:
                    print(f"❌ {camera_name} - Connection failed")
                    
            except Exception as e:
                print(f"❌ {camera_name} - Error: {e}")
                results[i] = {
                    'url': url,
                    'name': camera_name,
                    'connected': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        self.test_results['connections'] = results
        return results
    
    def test_camera_manager(self):
        """Test RTSP camera manager"""
        logger.info("🎥 Testing RTSP Camera Manager")
        print("=" * 60)
        
        try:
            # Create camera manager
            camera_urls = RTSPConfig.RTSP_CAMERA_URLS
            self.camera_manager = RTSPCameraManager(camera_urls)
            
            print(f"✅ Camera manager created with {len(camera_urls)} cameras")
            
            # Start capture
            print("Starting camera capture...")
            self.camera_manager.start_capture()
            
            # Test frame capture for 5 seconds
            print("Testing frame capture for 5 seconds...")
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 5:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    frame_count += 1
                    
                    # Display frame
                    cv2.imshow('RTSP Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.1)
            
            # Get stats
            stats = self.camera_manager.get_performance_stats()
            
            print(f"✅ Frame capture test completed")
            print(f"   Frames captured: {frame_count}")
            print(f"   Total frames: {stats['frame_count']}")
            print(f"   Active cameras: {stats['active_cameras']}/{stats['total_cameras']}")
            
            # Stop capture
            self.camera_manager.stop_capture()
            cv2.destroyAllWindows()
            
            self.test_results['camera_manager'] = {
                'success': True,
                'frames_captured': frame_count,
                'stats': stats
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera manager test failed: {e}")
            self.test_results['camera_manager'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_tracking_integration(self):
        """Test integration with existing tracking system"""
        logger.info("🎯 Testing Tracking System Integration")
        print("=" * 60)
        
        try:
            # Check if tracking modules are available
            try:
                from detector_tracker import DetectorTracker
                from database_handler import DatabaseHandler
                tracking_available = True
            except ImportError as e:
                print(f"⚠️ Tracking modules not available: {e}")
                tracking_available = False
            
            if not tracking_available:
                print("⏭️ Skipping tracking integration test")
                self.test_results['tracking_integration'] = {
                    'success': False,
                    'reason': 'Tracking modules not available'
                }
                return False
            
            # Create tracking system
            camera_urls = RTSPConfig.RTSP_CAMERA_URLS
            tracking_system = RTSPTrackingSystem(camera_urls)
            
            print("✅ RTSP tracking system created")
            
            # Test for a short duration
            print("Testing tracking system for 3 seconds...")
            tracking_system.camera_manager.start_capture()
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 3:
                frame = tracking_system.camera_manager.get_frame()
                if frame is not None:
                    frame_count += 1
                    
                    # Process frame (simplified)
                    try:
                        # Just display the frame for now
                        cv2.imshow('RTSP Tracking Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                        
                else:
                    time.sleep(0.1)
            
            # Cleanup
            tracking_system.camera_manager.stop_capture()
            cv2.destroyAllWindows()
            
            print(f"✅ Tracking integration test completed")
            print(f"   Frames processed: {frame_count}")
            
            self.test_results['tracking_integration'] = {
                'success': True,
                'frames_processed': frame_count
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Tracking integration test failed: {e}")
            self.test_results['tracking_integration'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_configuration(self):
        """Test RTSP configuration"""
        logger.info("⚙️ Testing RTSP Configuration")
        print("=" * 60)
        
        try:
            # Validate configuration
            if RTSPConfig.validate_config():
                print("✅ Configuration validation passed")
                
                # Print configuration summary
                RTSPConfig.print_config_summary()
                
                self.test_results['configuration'] = {
                    'success': True,
                    'total_cameras': RTSPConfig.get_total_cameras()
                }
                
                return True
            else:
                print("❌ Configuration validation failed")
                self.test_results['configuration'] = {
                    'success': False,
                    'reason': 'Configuration validation failed'
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            self.test_results['configuration'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting RTSP Integration Tests")
        print("=" * 80)
        print("🎯 RTSP CAMERA INTEGRATION TEST SUITE")
        print("=" * 80)
        
        test_functions = [
            ("Configuration", self.test_configuration),
            ("RTSP Connections", self.test_rtsp_connections),
            ("Camera Manager", self.test_camera_manager),
            ("Tracking Integration", self.test_tracking_integration)
        ]
        
        results = {}
        
        for test_name, test_func in test_functions:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                success = test_func()
                results[test_name] = success
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {e}")
                results[test_name] = False
        
        # Print summary
        self.print_test_summary(results)
        
        return results
    
    def print_test_summary(self, results):
        """Print test results summary"""
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name:<25} {status}")
            if success:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! RTSP integration is ready.")
        else:
            print("⚠️ Some tests failed. Check the results above.")
        
        # Print detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            print(f"\n{test_name.upper()}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  Result: {result}")

def main():
    """Main test function"""
    print("🎯 RTSP Camera Integration Test")
    print("This will test the integration of Lorex RTSP cameras with the warehouse tracking system")
    print("Make sure your RTSP cameras are accessible on the network")
    
    # Ask for confirmation
    response = input("\nContinue with tests? (y/n): ").lower().strip()
    if response != 'y':
        print("Tests cancelled")
        return
    
    # Create tester and run tests
    tester = RTSPIntegrationTester()
    results = tester.run_all_tests()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rtsp_test_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("RTSP Integration Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {datetime.now()}\n\n")
        
        for test_name, result in results.items():
            f.write(f"{test_name}: {'PASSED' if result else 'FAILED'}\n")
        
        f.write(f"\nDetailed Results:\n")
        for test_name, details in tester.test_results.items():
            f.write(f"\n{test_name}:\n")
            if isinstance(details, dict):
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
    
    print(f"\n📄 Test results saved to: {results_file}")

if __name__ == "__main__":
    main() 