"""
Simple test for single RTSP camera (Camera 8)
Tests the basic functionality with one camera to debug issues
"""

import cv2
import sys
import logging
from rtsp_camera_manager import RTSPTrackingSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_camera():
    """Test single RTSP camera functionality"""
    print("🎯 Testing Single RTSP Camera (Camera 8)")
    print("=" * 50)
    
    # Single camera URL
    camera_urls = [
        "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"  # Cam 8 Back
    ]
    
    try:
        # Create tracking system
        print("Creating RTSP tracking system...")
        tracking_system = RTSPTrackingSystem(camera_urls)
        
        print("✅ RTSP tracking system created successfully")
        print("Press 'q' to quit")
        
        # Start the tracking system
        tracking_system.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Test failed: {e}")
    finally:
        print("🧹 Cleaning up...")
        try:
            if 'tracking_system' in locals():
                tracking_system._cleanup()
        except:
            pass
        print("✅ Test completed")

if __name__ == "__main__":
    test_single_camera() 