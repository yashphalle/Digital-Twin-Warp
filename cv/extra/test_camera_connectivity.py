#!/usr/bin/env python3
"""
Test Camera Connectivity
Test RTSP connections to all cameras based on device list
"""

import cv2
import time
import threading
from config import Config

def test_camera_connection(camera_id, rtsp_url, timeout=10):
    """Test connection to a single camera"""
    print(f"🔍 Testing Camera {camera_id}: {rtsp_url}")
    
    try:
        # Try to connect
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set timeout
        start_time = time.time()
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ Camera {camera_id}: Connected successfully - {width}x{height}")
            result = {"camera_id": camera_id, "status": "connected", "resolution": f"{width}x{height}"}
        else:
            print(f"❌ Camera {camera_id}: Failed to read frame")
            result = {"camera_id": camera_id, "status": "no_frame", "resolution": None}
        
        cap.release()
        return result
        
    except Exception as e:
        print(f"❌ Camera {camera_id}: Connection failed - {e}")
        return {"camera_id": camera_id, "status": "failed", "error": str(e)}

def test_all_cameras():
    """Test all camera connections"""
    print("🚀 TESTING ALL CAMERA CONNECTIONS")
    print("=" * 60)
    print("Based on device list IP addresses")
    print("=" * 60)
    
    # Test active cameras first
    active_cameras = Config.ACTIVE_CAMERAS
    results = []
    
    print(f"🎯 Testing Active Cameras: {active_cameras}")
    print("-" * 40)
    
    for camera_id in active_cameras:
        if camera_id in Config.RTSP_CAMERA_URLS:
            rtsp_url = Config.RTSP_CAMERA_URLS[camera_id]
            result = test_camera_connection(camera_id, rtsp_url)
            results.append(result)
        else:
            print(f"❌ Camera {camera_id}: No RTSP URL configured")
            results.append({"camera_id": camera_id, "status": "no_config"})
    
    print("\n" + "=" * 60)
    print("📊 CONNECTIVITY SUMMARY")
    print("=" * 60)
    
    connected = [r for r in results if r["status"] == "connected"]
    failed = [r for r in results if r["status"] != "connected"]
    
    print(f"✅ Connected cameras: {len(connected)}")
    for r in connected:
        print(f"   Camera {r['camera_id']}: {r['resolution']}")
    
    print(f"\n❌ Failed cameras: {len(failed)}")
    for r in failed:
        print(f"   Camera {r['camera_id']}: {r['status']}")
        if "error" in r:
            print(f"      Error: {r['error']}")
    
    print(f"\n📋 Expected vs Actual:")
    print(f"   Expected active: {len(active_cameras)}")
    print(f"   Actually connected: {len(connected)}")
    
    if len(connected) == len(active_cameras):
        print("🎉 All cameras connected successfully!")
    else:
        print("⚠️  Some cameras failed to connect")
    
    return results

def test_specific_ips():
    """Test specific IP addresses from device list"""
    print("\n" + "=" * 60)
    print("🔍 TESTING SPECIFIC IPs FROM DEVICE LIST")
    print("=" * 60)
    
    # Based on your device list
    device_list_ips = {
        7: "192.168.0.78",
        8: "192.168.0.79", 
        9: "192.168.0.80",
        10: "192.168.0.82",  # This was the problem!
        11: "192.168.0.64"   # This was also wrong!
    }
    
    for camera_id, ip in device_list_ips.items():
        rtsp_url = f"rtsp://admin:wearewarp!@{ip}:554/Streaming/channels/1"
        print(f"\n🔍 Testing Camera {camera_id} at {ip}:")
        print(f"   URL: {rtsp_url}")
        
        result = test_camera_connection(camera_id, rtsp_url, timeout=5)
        
        # Compare with config
        config_url = Config.RTSP_CAMERA_URLS.get(camera_id, "Not configured")
        if rtsp_url == config_url:
            print(f"   ✅ Config matches device list")
        else:
            print(f"   ⚠️  Config mismatch:")
            print(f"      Config: {config_url}")
            print(f"      Device: {rtsp_url}")

def main():
    """Main test function"""
    print("🚀 CAMERA CONNECTIVITY TEST")
    print("=" * 60)
    print("Testing camera connections based on device management list")
    print("=" * 60)
    
    # Test all cameras
    results = test_all_cameras()
    
    # Test specific IPs
    test_specific_ips()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    print("1. If Camera 10 now connects, restart the multi-camera system")
    print("2. If still failing, check network connectivity to 192.168.0.82")
    print("3. Verify RTSP credentials and port 554 access")
    print("4. Check if camera is powered on and network cable connected")

if __name__ == "__main__":
    main()
