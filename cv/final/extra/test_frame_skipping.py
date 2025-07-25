#!/usr/bin/env python3
"""
Frame Skipping Test Script
Uses the same modules as main.py to visualize frame skipping behavior
Shows exactly which frames are processed vs skipped
"""

import cv2
import time
import sys
import os
import threading
from typing import Dict, List

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

sys.path.append(root_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

try:
    from configs.config import Config
    from modules.camera_manager import CameraManager
    CAMERA_URLS = Config.RTSP_CAMERA_URLS
    print("✅ Using camera URLs from config")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    # Fallback camera URLs
    CAMERA_URLS = {
        1: "rtsp://admin:wearewarp!@104.181.138.5:5561/Streaming/channels/1"
    }
    print("⚠️  Using fallback camera URLs")

    # Try to import CameraManager directly
    try:
        from modules.camera_manager import CameraManager
        print("✅ CameraManager imported successfully")
    except ImportError as e2:
        print(f"❌ Failed to import CameraManager: {e2}")
        print("Creating simple camera manager fallback...")

        class CameraManager:
            def __init__(self, camera_id, rtsp_url, camera_name):
                self.camera_id = camera_id
                self.rtsp_url = rtsp_url
                self.camera_name = camera_name
                self.cap = None
                self.connected = False

            def connect(self):
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if self.cap.isOpened():
                    # Set same settings as original
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    self.connected = True
                    return True
                return False

            def read_frame(self):
                if self.cap and self.connected:
                    return self.cap.read()
                return False, None

            def disconnect(self):
                if self.cap:
                    self.cap.release()
                self.connected = False

class FrameSkippingTester:
    """Test frame skipping behavior using same modules as main.py"""
    
    def __init__(self, camera_id: int, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = f"Camera {camera_id}"
        
        # Frame skipping settings (same as main.py)
        self.FRAME_SKIP = 20  # Process every 20th frame
        
        # Initialize camera manager (same as main.py)
        self.camera_manager = CameraManager(camera_id, rtsp_url, self.camera_name)
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.failed_reads = 0
        
        # Timing
        self.start_time = None
        self.last_processed_time = None
        self.processing_intervals = []
        
        # Display windows
        self.show_skipped = True  # Show skipped frames in separate window
        self.show_processed = True  # Show processed frames
        
        print(f"🎯 Frame Skipping Tester initialized for {self.camera_name}")
        print(f"   FRAME_SKIP = {self.FRAME_SKIP} (process every {self.FRAME_SKIP}th frame)")
    
    def connect_camera(self) -> bool:
        """Connect to camera using same method as main.py"""
        print(f"🔌 Connecting to {self.camera_name}...")
        
        if self.camera_manager.connect():
            print(f"✅ Connected to {self.camera_name}")
            return True
        else:
            print(f"❌ Failed to connect to {self.camera_name}")
            return False
    
    def test_frame_skipping(self, test_duration: int = 60):
        """
        Test frame skipping behavior - same logic as main.py
        Shows both skipped and processed frames
        """
        if not self.connect_camera():
            return
        
        print(f"\n🎬 Starting frame skipping test for {test_duration} seconds...")
        print("Controls:")
        print("  's' - Toggle showing skipped frames")
        print("  'p' - Toggle showing processed frames") 
        print("  'q' - Quit test")
        
        self.start_time = time.time()
        frame_count = 0
        processed_frame_count = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Stop after test duration
                if elapsed >= test_duration:
                    break
                
                frame_count += 1
                self.total_frames = frame_count
                
                # EXACT SAME LOGIC AS main.py
                if frame_count % self.FRAME_SKIP != 0:
                    # Skip this frame - just read and discard (SAME AS main.py)
                    ret, frame = self.camera_manager.read_frame()
                    if not ret:
                        print(f"⚠️  Failed to read frame {frame_count}")
                        self.failed_reads += 1
                        continue
                    
                    self.skipped_frames += 1
                    
                    # Show skipped frame if enabled
                    if self.show_skipped and frame is not None:
                        # Add frame info overlay
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"SKIPPED Frame {frame_count}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Time: {elapsed:.1f}s", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Resize for display
                        h, w = display_frame.shape[:2]
                        if w > 800:
                            scale = 800 / w
                            new_w, new_h = int(w * scale), int(h * scale)
                            display_frame = cv2.resize(display_frame, (new_w, new_h))
                        
                        cv2.imshow(f"SKIPPED - {self.camera_name}", display_frame)
                    
                    continue
                
                # Read frame normally (SAME AS main.py)
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    print(f"⚠️  Failed to read processed frame {frame_count}")
                    self.failed_reads += 1
                    continue
                
                processed_frame_count += 1
                self.processed_frames = processed_frame_count
                
                # Track processing intervals
                if self.last_processed_time is not None:
                    interval = current_time - self.last_processed_time
                    self.processing_intervals.append(interval)
                self.last_processed_time = current_time
                
                # Show processed frame if enabled
                if self.show_processed and frame is not None:
                    # Add frame info overlay
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"PROCESSED Frame {frame_count}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Processed #{processed_frame_count}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Time: {elapsed:.1f}s", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show timing info
                    if len(self.processing_intervals) > 0:
                        avg_interval = sum(self.processing_intervals) / len(self.processing_intervals)
                        cv2.putText(display_frame, f"Avg Interval: {avg_interval:.2f}s", 
                                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Resize for display
                    h, w = display_frame.shape[:2]
                    if w > 800:
                        scale = 800 / w
                        new_w, new_h = int(w * scale), int(h * scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))
                    
                    cv2.imshow(f"PROCESSED - {self.camera_name}", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Toggle skipped frames
                    self.show_skipped = not self.show_skipped
                    print(f"📺 Skipped frames display: {'ON' if self.show_skipped else 'OFF'}")
                    if not self.show_skipped:
                        cv2.destroyWindow(f"SKIPPED - {self.camera_name}")
                elif key == ord('p'):  # Toggle processed frames
                    self.show_processed = not self.show_processed
                    print(f"📺 Processed frames display: {'ON' if self.show_processed else 'OFF'}")
                    if not self.show_processed:
                        cv2.destroyWindow(f"PROCESSED - {self.camera_name}")
                
                # Log progress every 10 processed frames
                if processed_frame_count % 10 == 0:
                    fps = processed_frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Progress: {processed_frame_count} processed, {self.skipped_frames} skipped, {fps:.2f} processed FPS")
        
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        
        finally:
            self.camera_manager.disconnect()
            cv2.destroyAllWindows()
        
        self.print_results()
    
    def print_results(self):
        """Print detailed test results"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n📈 FRAME SKIPPING TEST RESULTS")
        print("=" * 50)
        print(f"Camera: {self.camera_name}")
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Frame Skip Setting: {self.FRAME_SKIP}")
        print()
        print(f"Total Frames Read: {self.total_frames}")
        print(f"Processed Frames: {self.processed_frames}")
        print(f"Skipped Frames: {self.skipped_frames}")
        print(f"Failed Reads: {self.failed_reads}")
        print()
        
        if total_time > 0:
            total_fps = self.total_frames / total_time
            processed_fps = self.processed_frames / total_time
            print(f"Total Read FPS: {total_fps:.2f}")
            print(f"Processed FPS: {processed_fps:.2f}")
            print(f"Expected Processed FPS: {total_fps / self.FRAME_SKIP:.2f}")
        
        if len(self.processing_intervals) > 1:
            avg_interval = sum(self.processing_intervals) / len(self.processing_intervals)
            min_interval = min(self.processing_intervals)
            max_interval = max(self.processing_intervals)
            print()
            print(f"Processing Intervals:")
            print(f"  Average: {avg_interval:.2f}s")
            print(f"  Range: {min_interval:.2f}s - {max_interval:.2f}s")
            print(f"  Expected: {self.FRAME_SKIP / 20:.2f}s (for 20 FPS camera)")

def main():
    """Main test function"""
    print("🎯 Frame Skipping Test Script")
    print("=" * 50)
    
    # Test configuration
    test_duration = 60  # seconds
    camera_id = 1  # Test Camera 1
    
    if camera_id not in CAMERA_URLS:
        print(f"❌ Camera {camera_id} not found in configuration")
        return
    
    rtsp_url = CAMERA_URLS[camera_id]
    
    # Create tester
    tester = FrameSkippingTester(camera_id, rtsp_url)
    
    # Run test
    tester.test_frame_skipping(test_duration)

if __name__ == "__main__":
    main()
