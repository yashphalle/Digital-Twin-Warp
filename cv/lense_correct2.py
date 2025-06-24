import cv2
import numpy as np
import os

class OptimizedFisheyeCorrector:
    def __init__(self, lens_mm=2.8):
        self.lens_mm = lens_mm
        self.K = None
        self.D = None
        self.map1 = None
        self.map2 = None
        
        # Load saved parameters
        self.load_parameters()
    
    def load_parameters(self):
        """Load saved fisheye parameters"""
        param_file = f'camera_params_{self.lens_mm}mm.npz'
        
        if os.path.exists(param_file):
            params = np.load(param_file)
            self.K = params['K']
            self.D = params['D']
            print(f"Loaded fisheye parameters for {self.lens_mm}mm lens")
            return True
        else:
            print(f"No saved parameters found for {self.lens_mm}mm lens!")
            print(f"Please run the tuning pipeline first to generate {param_file}")
            return False
    
    def initialize_maps(self, width, height, balance=0.5):
        """Pre-compute undistortion maps once"""
        if self.K is None or self.D is None:
            return False
        
        # Generate new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (width, height), np.eye(3), balance=balance
        )
        
        # Create undistortion maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
        )
        
        print(f"Initialized undistortion maps for {width}x{height}")
        return True
    
    def correct(self, frame):
        """Apply fisheye correction to a frame"""
        if self.map1 is None or self.map2 is None:
            # Initialize maps on first frame
            height, width = frame.shape[:2]
            if not self.initialize_maps(width, height):
                return frame  # Return uncorrected if initialization fails
        
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

class SimpleFisheyeRTSP:
    def __init__(self, rtsp_url, lens_mm=2.8):
        self.rtsp_url = rtsp_url
        self.lens_mm = lens_mm  # Store lens_mm as instance attribute
        self.corrector = OptimizedFisheyeCorrector(lens_mm)
        self.cap = None
        self.display_size = None
        self.window_name = 'Corrected Feed'
    
    def get_display_size(self, frame_width, frame_height):
        """Calculate appropriate display size to fit screen"""
        # Get screen dimensions
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            # Default to common resolution
            screen_width, screen_height = 1920, 1080
        
        # Use 80% of screen size as max window size
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        # Calculate scaling factor
        scale_w = max_width / frame_width
        scale_h = max_height / frame_height
        scale = min(scale_w, scale_h)
        
        # Calculate display dimensions
        display_width = int(frame_width * scale)
        display_height = int(frame_height * scale)
        
        return display_width, display_height
    
    def run(self):
        """Simple RTSP stream with fisheye correction"""
        # Suppress H.264 warnings
        os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
        os.environ["OPENCV_LOG_LEVEL"] = "OFF"
        
        # Configure capture with better error handling
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set capture properties to reduce errors
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print("Failed to connect to RTSP stream")
            return
        
        # Get stream info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Stream: {width}x{height} @ {fps}fps")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Calculate and set appropriate window size
        if self.display_size is None:
            self.display_size = self.get_display_size(width, height)
        
        cv2.resizeWindow(self.window_name, *self.display_size)
        cv2.moveWindow(self.window_name, 50, 50)  # Center on screen
        
        print(f"Display window: {self.display_size[0]}x{self.display_size[1]}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Screenshot")
        print("  'f' - Toggle fullscreen")
        print("  'r' - Reset window size")
        
        fullscreen = False
        frame_errors = 0
        max_errors = 10
        
        while True:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    frame_errors += 1
                    if frame_errors > max_errors:
                        print("Too many frame errors, attempting reconnection...")
                        self.reconnect()
                        frame_errors = 0
                    continue
                
                # Reset error counter on successful frame
                frame_errors = 0
                
                # Apply correction
                corrected = self.corrector.correct(frame)
                
                # Add simple overlay
                cv2.putText(corrected, f"Fisheye Corrected - {self.lens_mm}mm lens", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow(self.window_name, corrected)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'screenshot_{cv2.getTickCount()}.jpg'
                    cv2.imwrite(filename, corrected)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('f'):
                    if fullscreen:
                        cv2.setWindowProperty(self.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.window_name, *self.display_size)
                        fullscreen = False
                    else:
                        cv2.setWindowProperty(self.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_FULLSCREEN)
                        fullscreen = True
                elif key == ord('r'):
                    # Reset window size
                    cv2.setWindowProperty(self.window_name, 
                                        cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, *self.display_size)
                    cv2.moveWindow(self.window_name, 50, 50)
                    fullscreen = False
                    print("Window size reset")
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def reconnect(self):
        """Attempt to reconnect to stream"""
        if self.cap:
            self.cap.release()
        
        import time
        time.sleep(2)
        
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            print("Reconnection successful")
        else:
            print("Reconnection failed")

# Fixed AutoSizedFisheyeRTSP class
class AutoSizedFisheyeRTSP(SimpleFisheyeRTSP):
    def __init__(self, rtsp_url, lens_mm=2.8):
        # Properly call parent constructor
        super().__init__(rtsp_url, lens_mm)
    
    def get_display_size(self, frame_width, frame_height):
        """Calculate display size based on actual monitor"""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
        except:
            screen_w, screen_h = 1920, 1080
        
        # Use 80% of screen size
        max_width = int(screen_w * 0.8)
        max_height = int(screen_h * 0.8)
        
        # For 4K content on smaller screens, scale down more aggressively
        if frame_width > 3000:  # 4K content
            max_width = min(max_width, 1600)
            max_height = min(max_height, 900)
        
        # Calculate scaling
        scale_w = max_width / frame_width
        scale_h = max_height / frame_height
        scale = min(scale_w, scale_h)
        
        display_width = int(frame_width * scale)
        display_height = int(frame_height * scale)
        
        print(f"Monitor: {screen_w}x{screen_h}")
        print(f"Scaling 4K to: {display_width}x{display_height}")
        
        return display_width, display_height

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:wearewarp!@192.168.0.78:554/Streaming/channels/1"
    
    # Use auto-sized version
    pipeline = AutoSizedFisheyeRTSP(rtsp_url, lens_mm=2.8)
    pipeline.run()