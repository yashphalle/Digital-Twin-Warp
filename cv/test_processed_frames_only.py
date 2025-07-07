#!/usr/bin/env python3
"""
Show Only Processed Frames Test
Shows exactly which frames are being passed to CV algorithm
"""

import cv2
import time
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your actual modules
try:
    from configs.config import Config
    from final.modules.camera_manager import CPUCameraManager
    from final.fisheye_corrector import OptimizedFisheyeCorrector
    CAMERA_URLS = Config.RTSP_CAMERA_URLS
    print("✅ Using actual CameraManager, FisheyeCorrector and config")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    # Fallback
    CAMERA_URLS = {
        1: "rtsp://admin:wearewarp!@104.181.138.5:5564/Streaming/channels/1",
        2: "rtsp://admin:wearewarp!@104.181.138.5:5562/Streaming/channels/1"
    }
    CPUCameraManager = None
    OptimizedFisheyeCorrector = None
    print("⚠️  Using fallback camera URLs")

class ProcessedFramesViewer:
    """Show only the frames that would be processed by CV algorithm"""

    def __init__(self, camera_id: int, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = f"Camera {camera_id}"

        # Frame skipping settings (same as main.py)
        self.FRAME_SKIP = 20  # Process every 20th frame

        # Use actual CPUCameraManager if available
        if CPUCameraManager:
            self.camera_manager = CPUCameraManager(camera_id, rtsp_url, self.camera_name)
            self.use_real_manager = True
            print(f"🎯 Using REAL CPUCameraManager for {self.camera_name}")
        else:
            # Fallback to simple camera connection
            self.cap = None
            self.use_real_manager = False
            print(f"🎯 Using fallback camera connection for {self.camera_name}")

        # Use actual FisheyeCorrector if available
        if OptimizedFisheyeCorrector:
            self.fisheye_corrector = OptimizedFisheyeCorrector(lens_mm=2.8)
            self.use_real_fisheye = True
            print(f"🔧 Using REAL OptimizedFisheyeCorrector (2.8mm lens)")
        else:
            self.fisheye_corrector = None
            self.use_real_fisheye = False
            print(f"🔧 Fisheye correction disabled (fallback mode)")

        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.start_time = None
        self.fisheye_correction_times = []

        print(f"   Will show every {self.FRAME_SKIP}th frame (frames passed to CV)")
        print(f"   Fisheye correction: {'ENABLED' if self.use_real_fisheye else 'DISABLED'}")

    def connect_camera(self) -> bool:
        """Connect to camera using real or fallback manager"""
        print(f"🔌 Connecting to {self.camera_name}...")

        if self.use_real_manager:
            # Use your actual CPUCameraManager
            if self.camera_manager.connect_camera():
                print(f"✅ Connected using REAL CPUCameraManager")
                print(f"   Timeout settings: {self.camera_manager.read_timeout_ms}ms read timeout")
                return True
            else:
                print(f"❌ Failed to connect using CPUCameraManager")
                return False
        else:
            # Fallback method
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                print(f"✅ Connected using fallback method")
                return True
            else:
                print(f"❌ Failed to connect using fallback method")
                return False
    
    def show_processed_frames(self):
        """
        Show only processed frames - exact same logic as main.py
        These are the frames that get passed to CV algorithm
        """
        if not self.connect_camera():
            return
        
        print(f"\n🎬 Showing PROCESSED frames only...")
        print("These are the exact frames passed to CV algorithm")
        print("Controls:")
        print("  'q' - Quit")
        print("  'f' - Show frame info")
        
        self.start_time = time.time()
        frame_count = 0
        processed_frame_count = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                frame_count += 1
                self.total_frames = frame_count
                
                # EXACT SAME LOGIC AS main.py - Frame skipping
                if frame_count % self.FRAME_SKIP != 0:
                    # Skip this frame - just read and discard (SAME AS main.py)
                    if self.use_real_manager:
                        ret, frame = self.camera_manager.read_frame()
                    else:
                        ret, frame = self.cap.read()
                    if not ret:
                        continue
                    continue

                # Read frame for processing (SAME AS main.py)
                if self.use_real_manager:
                    ret, frame = self.camera_manager.read_frame()
                else:
                    ret, frame = self.cap.read()
                if not ret:
                    continue
                
                processed_frame_count += 1
                self.processed_frames = processed_frame_count
                
                # This is the frame that gets passed to CV algorithm
                if frame is not None:
                    # Apply fisheye correction (SAME AS REAL SYSTEM)
                    corrected_frame = frame.copy()
                    fisheye_time = 0

                    if self.use_real_fisheye:
                        try:
                            fisheye_start = time.time()
                            corrected_frame = self.fisheye_corrector.correct(corrected_frame)
                            fisheye_time = time.time() - fisheye_start
                            self.fisheye_correction_times.append(fisheye_time)

                            # Keep only recent times
                            if len(self.fisheye_correction_times) > 50:
                                self.fisheye_correction_times = self.fisheye_correction_times[-50:]

                        except Exception as e:
                            print(f"⚠️  Fisheye correction failed: {e}")
                            corrected_frame = frame.copy()

                    # Add frame info overlay
                    display_frame = corrected_frame.copy()

                    # Frame info
                    cv2.putText(display_frame, f"PROCESSED Frame #{frame_count}",
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(display_frame, f"CV Frame #{processed_frame_count}",
                              (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Time: {elapsed:.1f}s",
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Processing rate
                    if elapsed > 0:
                        fps = processed_frame_count / elapsed
                        cv2.putText(display_frame, f"Processing Rate: {fps:.2f} FPS",
                                  (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Fisheye correction timing
                    if self.use_real_fisheye and len(self.fisheye_correction_times) > 0:
                        avg_fisheye_time = sum(self.fisheye_correction_times) / len(self.fisheye_correction_times)
                        cv2.putText(display_frame, f"Fisheye: {fisheye_time*1000:.1f}ms (avg: {avg_fisheye_time*1000:.1f}ms)",
                                  (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Frame size info
                    h, w = corrected_frame.shape[:2]
                    orig_h, orig_w = frame.shape[:2]
                    cv2.putText(display_frame, f"Original: {orig_w}x{orig_h}",
                              (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Corrected: {w}x{h}",
                              (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Resize for display if too large
                    if w > 1200:
                        scale = 1200 / w
                        new_w, new_h = int(w * scale), int(h * scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))
                    
                    # Show the processed frame
                    cv2.imshow(f"PROCESSED FRAMES - {self.camera_name}", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('f'):  # Frame info
                    print(f"📊 Frame {frame_count}: Processed #{processed_frame_count}, Time: {elapsed:.1f}s")
                
                # Console progress every 5 processed frames
                if processed_frame_count % 5 == 0:
                    fps = processed_frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Processed {processed_frame_count} frames, {fps:.2f} FPS")
        
        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user")
        
        finally:
            if self.use_real_manager:
                self.camera_manager.cleanup_camera()
                print("🔌 Disconnected using REAL CPUCameraManager")
            elif self.cap:
                self.cap.release()
                print("🔌 Disconnected using fallback method")
            cv2.destroyAllWindows()
        
        self.print_results()
    
    def print_results(self):
        """Print results"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n📈 PROCESSED FRAMES RESULTS")
        print("=" * 40)
        print(f"Camera: {self.camera_name}")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Total Frames: {self.total_frames}")
        print(f"Processed Frames: {self.processed_frames}")
        
        if total_time > 0:
            fps = self.processed_frames / total_time
            print(f"Processing Rate: {fps:.2f} FPS")
            print(f"Expected Rate: ~1.0 FPS (20 FPS ÷ 20)")

        # Fisheye correction statistics
        if self.use_real_fisheye and len(self.fisheye_correction_times) > 0:
            avg_fisheye = sum(self.fisheye_correction_times) / len(self.fisheye_correction_times)
            min_fisheye = min(self.fisheye_correction_times)
            max_fisheye = max(self.fisheye_correction_times)
            print(f"\nFisheye Correction Timing:")
            print(f"  Average: {avg_fisheye*1000:.1f}ms")
            print(f"  Range: {min_fisheye*1000:.1f}ms - {max_fisheye*1000:.1f}ms")
            print(f"  Total corrections: {len(self.fisheye_correction_times)}")
        elif self.use_real_fisheye:
            print(f"\nFisheye Correction: ENABLED but no timing data")
        else:
            print(f"\nFisheye Correction: DISABLED")

def main():
    """Main function"""
    print("🎯 Processed Frames Viewer")
    print("Shows only frames passed to CV algorithm")
    print("=" * 50)
    
    # Test Camera 1
    camera_id = 4
    rtsp_url = CAMERA_URLS[camera_id]
    
    # Create viewer
    viewer = ProcessedFramesViewer(camera_id, rtsp_url)
    
    # Show processed frames
    viewer.show_processed_frames()

if __name__ == "__main__":
    main()
