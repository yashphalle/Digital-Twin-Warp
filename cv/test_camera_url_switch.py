#!/usr/bin/env python3
"""
Test Camera URL Switching
Tests the local/remote RTSP URL switching functionality
"""

import sys
import os

# Add cv directory to path
sys.path.append(os.path.dirname(__file__))

from configs.config import Config

def test_camera_url_switching():
    """Test switching between local and remote camera URLs"""
    
    print("🎥 CAMERA URL SWITCHING TEST")
    print("=" * 50)
    
    # Test Camera 7 (your current camera)
    camera_id = 7
    
    print(f"\n📹 Testing Camera {camera_id}:")
    
    # Show initial configuration
    print(f"\n🔧 Initial Configuration:")
    print(f"USE_LOCAL_CAMERAS: {Config.USE_LOCAL_CAMERAS}")
    info = Config.get_camera_info(camera_id)
    print(f"Current URL: {info['current_url']}")
    print(f"Using Local: {info['using_local']}")
    
    # Test switching to local
    print(f"\n🏠 Switching to LOCAL cameras...")
    Config.switch_to_local_cameras()
    info = Config.get_camera_info(camera_id)
    print(f"Local URL: {info['local_url']}")
    print(f"Current URL: {info['current_url']}")
    
    # Test switching to remote
    print(f"\n🌐 Switching to REMOTE cameras...")
    Config.switch_to_remote_cameras()
    info = Config.get_camera_info(camera_id)
    print(f"Remote URL: {info['remote_url']}")
    print(f"Current URL: {info['current_url']}")
    
    # Switch back to local (default)
    print(f"\n🏠 Switching back to LOCAL cameras...")
    Config.switch_to_local_cameras()
    
    print(f"\n✅ Camera URL switching test completed!")
    
    # Show all camera URLs
    print(f"\n📋 ALL CAMERA URLs (Current Configuration):")
    print(f"Mode: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    print("-" * 50)
    
    for cam_id in range(1, 12):
        url = Config.get_camera_url(cam_id)
        if url:
            network = "LOCAL" if "192.168" in url else "REMOTE"
            print(f"Camera {cam_id:2d}: {network} - {url}")
        else:
            print(f"Camera {cam_id:2d}: No URL configured")

if __name__ == "__main__":
    test_camera_url_switching()
