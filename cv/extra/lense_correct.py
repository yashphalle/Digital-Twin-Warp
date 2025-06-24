import cv2
import time
from datetime import datetime
import threading
import queue
import numpy as np

class FisheyeCorrectedLorexPipeline:
    def __init__(self, rtsp_url, buffer_size=10, lens_mm=2.8):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.frame_count = 0
        self.lens_mm = lens_mm
        
        # Fisheye correction parameters
        self.fisheye_correction_enabled = False
        self.setup_fisheye_params()
        
        # Pre-compute undistortion maps for efficiency
        self.map1 = None
        self.map2 = None
        self.maps_computed = False
        
    def setup_fisheye_params(self):
        """Initialize fisheye correction parameters based on your Lorex camera specs"""
        # Camera sensor specifications for 1/2.8" sensor
        self.sensor_width_mm = 6.2
        self.sensor_height_mm = 4.65
        
        # We'll update these once we know the actual resolution
        self.K = None
        self.D = None
        
        # Default distortion coefficients for each lens
        self.default_distortions = {
            2.8: np.array([0.15, -0.05, 0.01, 0.0], dtype=np.float32),
            4.0: np.array([0.08, -0.02, 0.0, 0.0], dtype=np.float32),
            6.0: np.array([0.03, -0.01, 0.0, 0.0], dtype=np.float32)
        }
        
        # Load saved parameters if available
        try:
            params = np.load(f'camera_params_{self.lens_mm}mm.npz')
            self.K = params['K']
            self.D = params['D']
            print(f"Loaded saved fisheye parameters for {self.lens_mm}mm lens")
        except:
            print(f"No saved parameters found for {self.lens_mm}mm lens, using defaults")
    
    def compute_camera_matrix(self, width, height):
        """Compute camera matrix based on resolution"""
        if self.K is None:
            # Calculate focal length in pixels
            fx = self.lens_mm * width / self.sensor_width_mm
            fy = self.lens_mm * height / self.sensor_height_mm
            
            # Principal point (optical center)
            cx = width / 2
            cy = height / 2
            
            self.K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
            
            self.D = self.default_distortions.get(self.lens_mm, 
                                                 np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32))
    
    def compute_undistortion_maps(self, width, height, balance=0.5):
        """Pre-compute undistortion maps for better performance"""
        if self.K is None:
            self.compute_camera_matrix(width, height)
        
        # Generate new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (width, height), np.eye(3), balance=balance
        )
        
        # Create undistortion maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
        )
        self.maps_computed = True
    
    def apply_fisheye_correction(self, frame):
        """Apply fisheye correction to frame"""
        if not self.fisheye_correction_enabled:
            return frame
        
        if not self.maps_computed:
            height, width = frame.shape[:2]
            self.compute_undistortion_maps(width, height)
        
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
    
    def connect_camera(self):
        """Connect to the RTSP stream"""
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Configure capture properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to RTSP stream")
        
        # Get stream properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Stream connected: {width}x{height} @ {fps}fps")
        print(f"Using {self.lens_mm}mm lens configuration")
        
        # Initialize camera matrix if not loaded from file
        if self.K is None:
            self.compute_camera_matrix(width, height)
        
        return True
    
    def frame_capture_thread(self):
        """Separate thread for frame capture to prevent blocking"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply fisheye correction if enabled
                corrected_frame = self.apply_fisheye_correction(frame)
                
                timestamp = datetime.now()
                if not self.frame_queue.full():
                    self.frame_queue.put((corrected_frame, timestamp))
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((corrected_frame, timestamp))
                    except queue.Empty:
                        pass
            else:
                print("Failed to capture frame, attempting reconnection...")
                self.reconnect()
                time.sleep(1)
    
    def reconnect(self):
        """Reconnect to RTSP stream"""
        if self.cap:
            self.cap.release()
        time.sleep(2)
        try:
            self.connect_camera()
        except Exception as e:
            print(f"Reconnection failed: {e}")
    
    def add_overlay(self, frame, timestamp):
        """Add timestamp and info overlay to the frame"""
        # Add timestamp
        cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add fisheye correction status
        status = "ON" if self.fisheye_correction_enabled else "OFF"
        color = (0, 255, 0) if self.fisheye_correction_enabled else (0, 0, 255)
        cv2.putText(frame, f"Fisheye Correction: {status}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add lens info
        cv2.putText(frame, f"Lens: {self.lens_mm}mm",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_grid(self, frame):
        """Draw grid overlay for checking distortion"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Grid color
        grid_color = (0, 255, 0)
        
        # Vertical lines
        for i in range(10):
            x = int(width * i / 9)
            cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
        
        # Horizontal lines
        for i in range(8):
            y = int(height * i / 7)
            cv2.line(overlay, (0, y), (width, y), grid_color, 1)
        
        # Center crosshair
        cv2.line(overlay, (width//2 - 20, height//2), 
                 (width//2 + 20, height//2), (0, 0, 255), 2)
        cv2.line(overlay, (width//2, height//2 - 20), 
                 (width//2, height//2 + 20), (0, 0, 255), 2)
        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def open_tuning_window(self):
        """Open a separate window for fine-tuning fisheye parameters"""
        cv2.namedWindow('Fisheye Tuning', cv2.WINDOW_NORMAL)
        
        # Create trackbars
        cv2.createTrackbar('K1 (*100)', 'Fisheye Tuning', 
                          int(self.D[0]*100 + 50), 100, self.update_params)
        cv2.createTrackbar('K2 (*100)', 'Fisheye Tuning', 
                          int(self.D[1]*100 + 50), 100, self.update_params)
        cv2.createTrackbar('K3 (*100)', 'Fisheye Tuning', 
                          int(self.D[2]*100 + 50), 100, self.update_params)
        cv2.createTrackbar('K4 (*100)', 'Fisheye Tuning', 
                          int(self.D[3]*100 + 50), 100, self.update_params)
        cv2.createTrackbar('Balance', 'Fisheye Tuning', 50, 100, self.update_params)
    
    def update_params(self, val):
        """Update fisheye parameters from trackbars"""
        self.D[0] = (cv2.getTrackbarPos('K1 (*100)', 'Fisheye Tuning') - 50) / 100.0
        self.D[1] = (cv2.getTrackbarPos('K2 (*100)', 'Fisheye Tuning') - 50) / 100.0
        self.D[2] = (cv2.getTrackbarPos('K3 (*100)', 'Fisheye Tuning') - 50) / 100.0
        self.D[3] = (cv2.getTrackbarPos('K4 (*100)', 'Fisheye Tuning') - 50) / 100.0
        balance = cv2.getTrackbarPos('Balance', 'Fisheye Tuning') / 100.0
        
        # Recompute maps with new parameters
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.compute_undistortion_maps(width, height, balance)
    
    def run(self, show_display=True):
        """Main execution loop"""
        try:
            # Connect to camera
            self.connect_camera()
            self.running = True
            
            # Start capture thread
            capture_thread = threading.Thread(target=self.frame_capture_thread)
            capture_thread.daemon = True
            capture_thread.start()
            
            print("\nLorex Camera Feed Started with Fisheye Correction!")
            print("\nControls:")
            print("  'q' - Quit")
            print("  's' - Take screenshot")
            print("  'f' - Toggle fullscreen")
            print("  'c' - Toggle fisheye correction")
            print("  'g' - Toggle grid overlay")
            print("  't' - Open tuning window")
            print("  'p' - Save current parameters")
            
            # Create main window
            cv2.namedWindow('Lorex Camera Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lorex Camera Feed', 1280, 720)
            cv2.moveWindow('Lorex Camera Feed', 100, 100)
            
            fullscreen = False
            show_grid = False
            tuning_mode = False
            
            while self.running:
                try:
                    # Get frame from queue
                    frame, timestamp = self.frame_queue.get(timeout=1)
                    
                    # Add overlay
                    display_frame = self.add_overlay(frame.copy(), timestamp)
                    
                    # Add grid if enabled
                    if show_grid:
                        display_frame = self.draw_grid(display_frame)
                    
                    if show_display:
                        # Display the frame
                        cv2.imshow('Lorex Camera Feed', display_frame)
                        
                        # Show tuning window if enabled
                        if tuning_mode and hasattr(self, 'last_frame'):
                            tuning_frame = self.apply_fisheye_correction(self.last_frame)
                            tuning_frame = self.draw_grid(tuning_frame)
                            cv2.imshow('Fisheye Tuning', tuning_frame)
                        
                        # Store last frame for tuning
                        self.last_frame = frame.copy()
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            screenshot_name = f"screenshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(screenshot_name, frame)
                            print(f"Screenshot saved: {screenshot_name}")
                        elif key == ord('f'):
                            if fullscreen:
                                cv2.setWindowProperty('Lorex Camera Feed', 
                                                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                                fullscreen = False
                            else:
                                cv2.setWindowProperty('Lorex Camera Feed', 
                                                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                fullscreen = True
                        elif key == ord('c'):
                            self.fisheye_correction_enabled = not self.fisheye_correction_enabled
                            status = "enabled" if self.fisheye_correction_enabled else "disabled"
                            print(f"Fisheye correction {status}")
                        elif key == ord('g'):
                            show_grid = not show_grid
                            print(f"Grid overlay {'enabled' if show_grid else 'disabled'}")
                        elif key == ord('t'):
                            if not tuning_mode:
                                self.open_tuning_window()
                                tuning_mode = True
                                print("Tuning mode enabled")
                            else:
                                cv2.destroyWindow('Fisheye Tuning')
                                tuning_mode = False
                                print("Tuning mode disabled")
                        elif key == ord('p'):
                            np.savez(f'camera_params_{self.lens_mm}mm.npz', K=self.K, D=self.D)
                            print(f"Parameters saved to camera_params_{self.lens_mm}mm.npz")
                            print(f"D coefficients: {self.D}")
                    
                    self.frame_count += 1
                    
                except queue.Empty:
                    print("No frames received, checking connection...")
                    continue
                except KeyboardInterrupt:
                    break
        
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Your RTSP URL
    rtsp_url = "rtsp://admin:wearewarp!@192.168.0.78:554/Streaming/channels/1"
    
    # Specify which lens you're using (2.8, 4.0, or 6.0 mm)
    lens_mm = 2.8  # Change this based on your actual lens
    
    # Create and run pipeline with fisheye correction
    pipeline = FisheyeCorrectedLorexPipeline(rtsp_url, lens_mm=lens_mm)
    
    try:
        pipeline.run(show_display=True)
    except KeyboardInterrupt:
        print("Pipeline stopped by user")
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()