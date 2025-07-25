#!/usr/bin/env python3
"""
Test Display Bottleneck
Tests if cv2.imshow() and display operations are causing stuck frames
"""

import cv2
import time
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your actual modules (SAME AS main.py)
try:
    from configs.config import Config
    from final.modules.camera_manager import CPUCameraManager
    from final.fisheye_corrector import OptimizedFisheyeCorrector
    from final.modules.gui_display import CPUDisplayManager
    CAMERA_URLS = Config.RTSP_CAMERA_URLS
    print("✅ Using actual modules (including GUI display)")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    CAMERA_URLS = {
        4: "rtsp://admin:wearewarp!@104.181.138.5:5564/Streaming/channels/1"
    }
    CPUCameraManager = None
    OptimizedFisheyeCorrector = None
    CPUDisplayManager = None

class DisplayBottleneckTester:
    """Test if display operations cause frame processing delays"""
    
    def __init__(self, camera_id: int, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = f"Camera {camera_id}"
        self.FRAME_SKIP = 20
        
        # Initialize components
        if CPUCameraManager:
            self.camera_manager = CPUCameraManager(camera_id, rtsp_url, self.camera_name)
            self.use_real_manager = True
        else:
            self.cap = None
            self.use_real_manager = False
            
        if OptimizedFisheyeCorrector:
            self.fisheye_corrector = OptimizedFisheyeCorrector(lens_mm=2.8)
            self.use_real_fisheye = True
        else:
            self.fisheye_corrector = None
            self.use_real_fisheye = False

        # Initialize GUI Display Manager (SAME AS main.py)
        if CPUDisplayManager:
            self.display_manager = CPUDisplayManager(
                camera_name=self.camera_name,
                camera_id=self.camera_id
            )
            self.use_real_display = True
        else:
            self.display_manager = None
            self.use_real_display = False

        # Statistics
        self.stats = {
            'frame_read_times': [],
            'fisheye_times': [],
            'display_times': [],
            'gui_render_times': [],
            'waitkey_times': [],
            'total_processing_times': []
        }

        print(f"🎯 Display Bottleneck Tester for {self.camera_name}")
        print(f"   Camera Manager: {'REAL' if self.use_real_manager else 'FALLBACK'}")
        print(f"   Fisheye Correction: {'REAL' if self.use_real_fisheye else 'DISABLED'}")
        print(f"   GUI Display Manager: {'REAL' if self.use_real_display else 'DISABLED'}")
    
    def connect_camera(self) -> bool:
        """Connect to camera"""
        print(f"🔌 Connecting to {self.camera_name}...")
        
        if self.use_real_manager:
            if self.camera_manager.connect_camera():
                print(f"✅ Connected using REAL CPUCameraManager")
                return True
            else:
                print(f"❌ Failed to connect using CPUCameraManager")
                return False
        else:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"✅ Connected using fallback method")
                return True
            else:
                print(f"❌ Failed to connect using fallback method")
                return False
    
    def test_display_bottleneck(self, test_duration: int = 60):
        """Test different display scenarios to identify bottlenecks"""
        if not self.connect_camera():
            return
        
        print(f"\n🎬 Testing display bottleneck scenarios...")
        print("Scenarios:")
        print("  1. No display (processing only)")
        print("  2. Simple cv2.imshow (basic display)")
        print("  3. REAL GUI Display Manager (same as main.py)")
        print("  4. Full main.py pipeline (GUI + waitKey)")
        print(f"\n🎯 STARTING IN MODE 3: REAL GUI Display Manager")
        print("This tests the exact same display pipeline as main.py")
        print("Press 'q' to quit, '1-4' to switch scenarios")
        
        start_time = time.time()
        frame_count = 0
        processed_count = 0
        display_mode = 3  # Start with REAL GUI Display Manager (Mode 3)
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed >= test_duration:
                    break
                
                frame_count += 1
                
                # Frame skipping (same as real system)
                if frame_count % self.FRAME_SKIP != 0:
                    # Skip frame
                    if self.use_real_manager:
                        ret, frame = self.camera_manager.read_frame()
                    else:
                        ret, frame = self.cap.read()
                    continue
                
                # Process frame (same as real system)
                processing_start = time.time()
                
                # 1. Read frame
                read_start = time.time()
                if self.use_real_manager:
                    ret, frame = self.camera_manager.read_frame()
                else:
                    ret, frame = self.cap.read()
                read_time = time.time() - read_start
                
                if not ret or frame is None:
                    continue
                
                processed_count += 1
                
                # 2. Apply fisheye correction
                fisheye_start = time.time()
                corrected_frame = frame.copy()
                if self.use_real_fisheye:
                    try:
                        corrected_frame = self.fisheye_corrector.correct(corrected_frame)
                    except Exception as e:
                        print(f"⚠️  Fisheye error: {e}")
                fisheye_time = time.time() - fisheye_start

                # 2.5. Resize frame for display (SAME AS REAL SYSTEM)
                display_frame = corrected_frame.copy()
                h, w = display_frame.shape[:2]
                if w > 1600:  # Resize large frames for display performance
                    scale = 1600 / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_frame = cv2.resize(display_frame, (new_w, new_h))
                    print(f"📏 Resized frame: {w}x{h} → {new_w}x{new_h} (scale: {scale:.2f})")
                
                # 3. Display operations (TESTING DIFFERENT SCENARIOS)
                display_start = time.time()
                gui_render_time = 0

                if display_mode == 1:
                    # Scenario 1: No display (baseline)
                    pass

                elif display_mode == 2:
                    # Scenario 2: Simple cv2.imshow (basic display)
                    simple_frame = display_frame.copy()
                    cv2.putText(simple_frame, f"Mode 2: Simple Display (Frame {processed_count})",
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Test - {self.camera_name}", simple_frame)

                elif display_mode == 3:
                    # Scenario 3: REAL GUI Display Manager (SAME AS main.py)
                    if self.use_real_display:
                        gui_start = time.time()
                        # Create complete mock tracker object for display manager
                        mock_frame_processor = type('MockFrameProcessor', (), {
                            'final_tracked_detections': [],
                            'get_detection_counts': lambda self: {'total': 0, 'new': 0, 'tracked': 0},
                            'raw_detections': [],
                            'filtered_detections': [],
                            'grid_filtered_detections': []
                        })()

                        mock_tracker = type('MockTracker', (), {
                            'final_tracked_detections': [],  # Empty detections for testing
                            'frame_processor': mock_frame_processor,
                            'camera_id': self.camera_id,
                            'camera_name': self.camera_name
                        })()

                        # Use REAL GUI display manager (SAME AS main.py)
                        rendered_frame = self.display_manager.render_frame(display_frame, mock_tracker)
                        gui_render_time = time.time() - gui_start
                        cv2.imshow(f"CPU Tracking - {self.camera_name}", rendered_frame)
                    else:
                        # Fallback
                        fallback_frame = display_frame.copy()
                        cv2.putText(fallback_frame, f"Mode 3: GUI Disabled",
                                  (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(f"Test - {self.camera_name}", fallback_frame)

                elif display_mode == 4:
                    # Scenario 4: Full main.py pipeline (GUI + waitKey)
                    if self.use_real_display:
                        gui_start = time.time()
                        mock_frame_processor = type('MockFrameProcessor', (), {
                            'final_tracked_detections': [],
                            'get_detection_counts': lambda self: {'total': 0, 'new': 0, 'tracked': 0},
                            'raw_detections': [],
                            'filtered_detections': [],
                            'grid_filtered_detections': []
                        })()

                        mock_tracker = type('MockTracker', (), {
                            'final_tracked_detections': [],
                            'frame_processor': mock_frame_processor,
                            'camera_id': self.camera_id,
                            'camera_name': self.camera_name
                        })()
                        rendered_frame = self.display_manager.render_frame(display_frame, mock_tracker)
                        gui_render_time = time.time() - gui_start
                        cv2.imshow(f"CPU Tracking - {self.camera_name}", rendered_frame)
                    else:
                        fallback_frame = display_frame.copy()
                        cv2.imshow(f"Test - {self.camera_name}", fallback_frame)

                display_time = time.time() - display_start
                
                # 4. Handle keyboard input (TESTING DIFFERENT WAITKEY VALUES)
                waitkey_start = time.time()
                
                if display_mode == 1:
                    # No waitKey for mode 1
                    key = -1
                elif display_mode == 2:
                    # No waitKey for mode 2
                    key = -1
                elif display_mode == 3:
                    # Standard waitKey(1)
                    key = cv2.waitKey(1) & 0xFF
                elif display_mode == 4:
                    # Longer waitKey(30)
                    key = cv2.waitKey(30) & 0xFF
                
                waitkey_time = time.time() - waitkey_start
                total_processing_time = time.time() - processing_start
                
                # Store timing statistics
                self.stats['frame_read_times'].append(read_time)
                self.stats['fisheye_times'].append(fisheye_time)
                self.stats['display_times'].append(display_time)
                self.stats['gui_render_times'].append(gui_render_time)
                self.stats['waitkey_times'].append(waitkey_time)
                self.stats['total_processing_times'].append(total_processing_time)
                
                # Handle mode switching
                if key == ord('q') or key == 27:
                    break
                elif key == ord('1'):
                    display_mode = 1
                    cv2.destroyAllWindows()
                    print(f"📺 Switched to Mode 1: No Display")
                elif key == ord('2'):
                    display_mode = 2
                    print(f"📺 Switched to Mode 2: Display Only")
                elif key == ord('3'):
                    display_mode = 3
                    print(f"📺 Switched to Mode 3: Full Display")
                elif key == ord('4'):
                    display_mode = 4
                    print(f"📺 Switched to Mode 4: waitKey(30)")
                
                # Progress logging every 10 frames
                if processed_count % 10 == 0:
                    fps = processed_count / elapsed if elapsed > 0 else 0
                    avg_total = sum(self.stats['total_processing_times'][-10:]) / min(10, len(self.stats['total_processing_times']))
                    print(f"📊 Mode {display_mode}: {processed_count} frames, {fps:.2f} FPS, {avg_total*1000:.1f}ms avg processing")
        
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted")
        
        finally:
            if self.use_real_manager:
                self.camera_manager.cleanup_camera()
            elif self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        
        self.print_bottleneck_analysis()
    
    def print_bottleneck_analysis(self):
        """Analyze and print bottleneck results"""
        print(f"\n📈 DISPLAY BOTTLENECK ANALYSIS")
        print("=" * 50)
        
        if not self.stats['total_processing_times']:
            print("❌ No timing data collected")
            return
        
        # Calculate averages
        avg_read = sum(self.stats['frame_read_times']) / len(self.stats['frame_read_times']) * 1000
        avg_fisheye = sum(self.stats['fisheye_times']) / len(self.stats['fisheye_times']) * 1000
        avg_display = sum(self.stats['display_times']) / len(self.stats['display_times']) * 1000
        avg_gui_render = sum(self.stats['gui_render_times']) / len(self.stats['gui_render_times']) * 1000
        avg_waitkey = sum(self.stats['waitkey_times']) / len(self.stats['waitkey_times']) * 1000
        avg_total = sum(self.stats['total_processing_times']) / len(self.stats['total_processing_times']) * 1000

        print(f"Average Timing Breakdown:")
        print(f"  Frame Read: {avg_read:.1f}ms")
        print(f"  Fisheye Correction: {avg_fisheye:.1f}ms")
        print(f"  Display Operations: {avg_display:.1f}ms")
        print(f"  GUI Render (main.py style): {avg_gui_render:.1f}ms")
        print(f"  waitKey Operations: {avg_waitkey:.1f}ms")
        print(f"  Total Processing: {avg_total:.1f}ms")

        # Identify bottlenecks
        print(f"\nBottleneck Analysis:")
        if avg_gui_render > 50:
            print(f"🚨 GUI RENDER is the bottleneck ({avg_gui_render:.1f}ms)")
        elif avg_display > avg_read and avg_display > avg_fisheye:
            print(f"🚨 DISPLAY is the bottleneck ({avg_display:.1f}ms)")
        elif avg_waitkey > 5:
            print(f"🚨 WAITKEY is causing delays ({avg_waitkey:.1f}ms)")
        elif avg_fisheye > avg_read and avg_fisheye > avg_display:
            print(f"⚠️  Fisheye correction is slowest ({avg_fisheye:.1f}ms)")
        else:
            print(f"✅ No obvious display bottleneck detected")

def main():
    """Main test function"""
    print("🎯 Display Bottleneck Test")
    print("=" * 50)
    
    camera_id = 4
    rtsp_url = CAMERA_URLS[camera_id]
    
    tester = DisplayBottleneckTester(camera_id, rtsp_url)
    tester.test_display_bottleneck(120)  # 2 minute test

if __name__ == "__main__":
    main()
