#!/usr/bin/env python3
"""
Debug Black Screen Issue
Test each component separately to identify the root cause
"""

import cv2
import numpy as np
import time
from lense_correct2 import OptimizedFisheyeCorrector

def test_rtsp_raw():
    """Test 1: Raw RTSP stream without any processing"""
    print("🔍 TEST 1: Raw RTSP Stream")
    print("=" * 40)
    
    rtsp_url = "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Failed to connect to RTSP")
        return False
    
    print("✅ RTSP connected")
    cv2.namedWindow("Raw RTSP Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Raw RTSP Feed", 800, 450)
    
    frame_count = 0
    black_count = 0
    
    for i in range(100):  # Test 100 frames
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            
            # Check if frame is black
            mean_brightness = np.mean(frame)
            if mean_brightness < 10:  # Very dark frame
                black_count += 1
                print(f"⚠️ Frame {frame_count}: BLACK (brightness: {mean_brightness:.1f})")
            
            # Show frame
            display_frame = cv2.resize(frame, (800, 450))
            cv2.putText(display_frame, f"Frame: {frame_count} | Brightness: {mean_brightness:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Raw RTSP Feed", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"❌ Failed to read frame {i}")
    
    cap.release()
    cv2.destroyWindow("Raw RTSP Feed")
    
    print(f"📊 Results: {frame_count} frames, {black_count} black frames ({black_count/frame_count*100:.1f}%)")
    return black_count == 0

def test_fisheye_correction():
    """Test 2: RTSP + Fisheye correction only"""
    print("\n🔍 TEST 2: RTSP + Fisheye Correction")
    print("=" * 40)
    
    rtsp_url = "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"
    cap = cv2.VideoCapture(rtsp_url)
    corrector = OptimizedFisheyeCorrector(lens_mm=2.8)
    
    if not cap.isOpened():
        print("❌ Failed to connect to RTSP")
        return False
    
    print("✅ RTSP connected, testing fisheye correction...")
    cv2.namedWindow("Fisheye Corrected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fisheye Corrected", 800, 450)
    
    frame_count = 0
    black_count = 0
    correction_errors = 0
    
    for i in range(100):  # Test 100 frames
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            
            try:
                # Apply fisheye correction
                corrected = corrector.correct(frame)
                
                # Check if corrected frame is black
                mean_brightness = np.mean(corrected)
                if mean_brightness < 10:  # Very dark frame
                    black_count += 1
                    print(f"⚠️ Frame {frame_count}: BLACK after correction (brightness: {mean_brightness:.1f})")
                
                # Show frame
                display_frame = cv2.resize(corrected, (800, 450))
                cv2.putText(display_frame, f"Frame: {frame_count} | Brightness: {mean_brightness:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "FISHEYE CORRECTED", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Fisheye Corrected", display_frame)
                
            except Exception as e:
                correction_errors += 1
                print(f"❌ Correction error on frame {frame_count}: {e}")
                # Show original frame instead
                display_frame = cv2.resize(frame, (800, 450))
                cv2.putText(display_frame, f"CORRECTION ERROR: {str(e)[:30]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Fisheye Corrected", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"❌ Failed to read frame {i}")
    
    cap.release()
    cv2.destroyWindow("Fisheye Corrected")
    
    print(f"📊 Results: {frame_count} frames, {black_count} black, {correction_errors} errors")
    return black_count == 0 and correction_errors == 0

def test_frame_processing_pipeline():
    """Test 3: Full frame processing pipeline"""
    print("\n🔍 TEST 3: Frame Processing Pipeline")
    print("=" * 40)
    
    # Import the RTSP camera manager
    try:
        from rtsp_camera_manager import RTSPCameraManager
    except ImportError:
        print("❌ Could not import RTSPCameraManager")
        return False
    
    rtsp_urls = ["rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1"]
    
    try:
        camera_manager = RTSPCameraManager(rtsp_urls)
        camera_manager.start_capture()
        
        print("✅ Camera manager started")
        cv2.namedWindow("Pipeline Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pipeline Test", 800, 450)
        
        frame_count = 0
        black_count = 0
        none_count = 0
        
        for i in range(100):  # Test 100 frames
            frame = camera_manager.get_frame()
            
            if frame is not None:
                frame_count += 1
                
                # Check if frame is black
                if frame.size > 0:
                    mean_brightness = np.mean(frame)
                    if mean_brightness < 10:  # Very dark frame
                        black_count += 1
                        print(f"⚠️ Frame {frame_count}: BLACK from pipeline (brightness: {mean_brightness:.1f})")
                    
                    # Show frame
                    display_frame = cv2.resize(frame, (800, 450))
                    cv2.putText(display_frame, f"Pipeline Frame: {frame_count} | Brightness: {mean_brightness:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Pipeline Test", display_frame)
                else:
                    print(f"⚠️ Frame {frame_count}: Empty frame from pipeline")
                    
            else:
                none_count += 1
                if none_count % 10 == 0:
                    print(f"⚠️ {none_count} None frames from pipeline")
                
                # Show placeholder
                placeholder = np.zeros((450, 800, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Waiting for frame... (None count: {none_count})", 
                           (50, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Pipeline Test", placeholder)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.05)  # 20 FPS
        
        camera_manager.stop_capture()
        cv2.destroyWindow("Pipeline Test")
        
        print(f"📊 Results: {frame_count} frames, {black_count} black, {none_count} None")
        return black_count == 0
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests to identify black screen cause"""
    print("🚀 BLACK SCREEN DEBUG TESTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: Raw RTSP
    results.append(("Raw RTSP", test_rtsp_raw()))
    
    # Test 2: Fisheye correction
    results.append(("Fisheye Correction", test_fisheye_correction()))
    
    # Test 3: Full pipeline
    results.append(("Full Pipeline", test_frame_processing_pipeline()))
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    # Diagnosis
    print("\n🔍 DIAGNOSIS:")
    if not results[0][1]:  # Raw RTSP failed
        print("❌ Problem: RTSP stream itself has black frames")
        print("🔧 Solution: Check camera settings or network")
    elif not results[1][1]:  # Fisheye correction failed
        print("❌ Problem: Fisheye correction is causing black frames")
        print("🔧 Solution: Check calibration file or correction parameters")
    elif not results[2][1]:  # Pipeline failed
        print("❌ Problem: Frame processing pipeline has issues")
        print("🔧 Solution: Check camera manager or frame combination logic")
    else:
        print("✅ All tests passed - black screen might be intermittent or CV algorithm related")

if __name__ == "__main__":
    main() 