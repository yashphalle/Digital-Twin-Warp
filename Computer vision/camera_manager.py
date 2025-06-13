"""
Dual ZED 2i Camera Management and Frame Stitching
Handles camera capture, synchronization, and basic stitching operations

ZED Camera Setup:
- Each ZED camera outputs stereo images (left + right side-by-side)
- We extract only the LEFT view from each camera
- Final stitched image: Left view of Camera 1 + Left view of Camera 2
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from config import Config
import logging

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        """Initialize dual camera manager"""
        logger.info("Initializing Dual Camera Manager...")
        
        # Camera objects
        self.camera1 = None
        self.camera2 = None
        self.cameras_initialized = False
        
        # Frame synchronization
        self.frame1 = None
        self.frame2 = None
        self.last_frame_time = 0
        
        # Stitching parameters
        self.overlap_enabled = Config.OVERLAP_ENABLED
        self.overlap_percentage = Config.OVERLAP_PERCENTAGE
        self.stitch_mode = Config.STITCH_MODE
        
        # Calibration data
        self.homography_matrix = None
        self.calibrated = False
        self.calibration_frames_captured = 0
        
        # Performance tracking
        self.frame_count = 0
        self.capture_times = []
        self.stitch_times = []
        
        # Initialize cameras
        self._initialize_cameras()
        
    def _initialize_cameras(self):
        """Initialize both ZED cameras"""
        try:
            logger.info(f"Connecting to Camera 1 (ID: {Config.CAMERA_1_ID})...")
            self.camera1 = cv2.VideoCapture(Config.CAMERA_1_ID)
            
            if not self.camera1.isOpened():
                raise Exception(f"Failed to open Camera 1 (ID: {Config.CAMERA_1_ID})")
            
            logger.info(f"Connecting to Camera 2 (ID: {Config.CAMERA_2_ID})...")
            self.camera2 = cv2.VideoCapture(Config.CAMERA_2_ID)
            
            if not self.camera2.isOpened():
                self.camera1.release()
                raise Exception(f"Failed to open Camera 2 (ID: {Config.CAMERA_2_ID})")
            
            # Configure camera settings
            self._configure_cameras()
            
            self.cameras_initialized = True
            logger.info("Both cameras initialized successfully")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self._cleanup_cameras()
            raise
    
    def _configure_cameras(self):
        """Configure camera settings for optimal performance"""
        cameras = [self.camera1, self.camera2]
        camera_names = ["Camera 1", "Camera 2"]
        
        for i, (camera, name) in enumerate(zip(cameras, camera_names)):
            logger.info(f"Configuring {name}...")
            
            # Set resolution
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            
            # Set FPS
            camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            # Additional settings for better performance
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            
            # Verify settings
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"{name}: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    def capture_synchronized_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Capture synchronized frames from both cameras (left view only from stereo cameras)
        Returns: (frame1_left, frame2_left, success)
        """
        if not self.cameras_initialized:
            return None, None, False
        
        capture_start = time.time()
        
        try:
            # Capture from both cameras simultaneously
            ret1, frame1_full = self.camera1.read()
            ret2, frame2_full = self.camera2.read()
            
            if not (ret1 and ret2):
                logger.warning("Failed to capture from one or both cameras")
                return None, None, False
            
            # Extract left view from stereo cameras
            # ZED cameras typically provide side-by-side stereo image
            # Left half = left camera, Right half = right camera
            frame1_left = self._extract_left_view(frame1_full)
            frame2_left = self._extract_left_view(frame2_full)
            
            # Store frames for stitching
            self.frame1 = frame1_left.copy()
            self.frame2 = frame2_left.copy()
            self.frame_count += 1
            
            # Track capture performance
            capture_time = time.time() - capture_start
            self.capture_times.append(capture_time)
            if len(self.capture_times) > Config.FPS_CALCULATION_FRAMES:
                self.capture_times.pop(0)
            
            return frame1_left, frame2_left, True
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None, None, False
    
    def _extract_left_view(self, stereo_frame: np.ndarray) -> np.ndarray:
        """
        Extract left view from ZED stereo camera frame
        ZED cameras output side-by-side stereo images
        """
        if stereo_frame is None:
            return None
        
        height, width = stereo_frame.shape[:2]
        
        # For ZED cameras, the frame is typically side-by-side stereo
        # Left half = left camera view, Right half = right camera view
        left_view = stereo_frame[:, :width//2]
        
        logger.debug(f"Extracted left view: {left_view.shape} from stereo frame: {stereo_frame.shape}")
        
        return left_view
    
    def stitch_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """
        Stitch two frames together based on configuration
        
        Current Implementation:
        - "side_by_side": Simple horizontal concatenation (no feature matching)
        - "overlap": Basic overlap with simple blending (no feature matching)  
        - "blend": Placeholder for future feature-based stitching
        
        NOTE: This is basic stitching without feature matching algorithms.
        For production use, consider implementing SIFT/ORB feature matching.
        """
        if frame1 is None or frame2 is None:
            return None
        
        stitch_start = time.time()
        
        try:
            if self.stitch_mode == "side_by_side":
                # Simple horizontal concatenation - NO feature matching
                stitched = self._stitch_side_by_side(frame1, frame2)
            elif self.stitch_mode == "overlap":
                # Basic overlap handling - NO feature matching
                stitched = self._stitch_with_overlap(frame1, frame2)
            elif self.stitch_mode == "blend":
                # Placeholder for advanced feature-based stitching
                stitched = self._stitch_with_blend(frame1, frame2)
            else:
                logger.warning(f"Unknown stitch mode: {self.stitch_mode}, using side_by_side")
                stitched = self._stitch_side_by_side(frame1, frame2)
            
            # Track stitching performance
            stitch_time = time.time() - stitch_start
            self.stitch_times.append(stitch_time)
            if len(self.stitch_times) > Config.FPS_CALCULATION_FRAMES:
                self.stitch_times.pop(0)
            
            return stitched
            
        except Exception as e:
            logger.error(f"Frame stitching error: {e}")
            return None
    
    def _stitch_side_by_side(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Simple side-by-side stitching - NO feature matching
        Just horizontally concatenates the two frames
        """
        return np.hstack([frame1, frame2])
    
    def _stitch_with_overlap(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Basic overlap handling - NO feature matching
        Assumes fixed overlap percentage and simple blending
        """
        h, w = frame1.shape[:2]
        overlap_pixels = int(w * self.overlap_percentage)
        
        if overlap_pixels <= 0:
            return self._stitch_side_by_side(frame1, frame2)
        
        # Extract overlap regions (assumes perfect alignment)
        frame1_overlap = frame1[:, -overlap_pixels:]
        frame2_overlap = frame2[:, :overlap_pixels]
        
        # Simple averaging in overlap zone - NO feature-based alignment
        overlap_blended = cv2.addWeighted(frame1_overlap, 0.5, frame2_overlap, 0.5, 0)
        
        # Combine frames
        frame1_main = frame1[:, :-overlap_pixels]
        frame2_main = frame2[:, overlap_pixels:]
        
        stitched = np.hstack([frame1_main, overlap_blended, frame2_main])
        return stitched
    
    def _stitch_with_blend(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Placeholder for advanced feature-based stitching
        
        TODO: Implement proper feature matching:
        1. Extract SIFT/ORB keypoints from both frames
        2. Match keypoints between frames
        3. Calculate homography matrix
        4. Warp and blend frames based on feature matches
        """
        logger.info("Advanced blending not implemented yet, using overlap method")
        return self._stitch_with_overlap(frame1, frame2)
    
    def auto_calibrate(self, frames_needed: int = None) -> bool:
        """
        Auto-calibrate cameras for better stitching
        """
        if not Config.AUTO_CALIBRATE:
            return True
        
        frames_needed = frames_needed or Config.CALIBRATION_FRAMES
        
        logger.info(f"Starting auto-calibration ({frames_needed} frames)...")
        
        # Placeholder for calibration logic
        # In a full implementation, this would:
        # 1. Capture multiple frame pairs
        # 2. Find matching features between cameras
        # 3. Calculate homography matrix
        # 4. Store calibration parameters
        
        calibration_frames = []
        
        for i in range(frames_needed):
            frame1, frame2, success = self.capture_synchronized_frames()
            if success:
                calibration_frames.append((frame1, frame2))
                self.calibration_frames_captured += 1
            
            if len(calibration_frames) >= frames_needed:
                break
        
        if len(calibration_frames) >= frames_needed // 2:
            # Simplified calibration - just mark as calibrated
            self.calibrated = True
            logger.info("Auto-calibration completed successfully")
            return True
        else:
            logger.warning("Auto-calibration failed - insufficient frames")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get camera performance statistics"""
        stats = {
            'frames_captured': self.frame_count,
            'cameras_initialized': self.cameras_initialized,
            'calibrated': self.calibrated,
            'avg_capture_time': np.mean(self.capture_times) if self.capture_times else 0,
            'avg_stitch_time': np.mean(self.stitch_times) if self.stitch_times else 0,
            'capture_fps': 1.0 / np.mean(self.capture_times) if self.capture_times else 0,
            'stitch_fps': 1.0 / np.mean(self.stitch_times) if self.stitch_times else 0
        }
        return stats
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        if not self.cameras_initialized:
            return {}
        
        info = {}
        cameras = [self.camera1, self.camera2]
        
        for i, camera in enumerate(cameras, 1):
            full_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            full_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate left view dimensions (half width for stereo)
            left_width = full_width // 2
            
            info[f'camera_{i}'] = {
                'full_width': full_width,
                'full_height': full_height,
                'left_view_width': left_width,
                'left_view_height': full_height,
                'fps': camera.get(cv2.CAP_PROP_FPS),
                'buffer_size': int(camera.get(cv2.CAP_PROP_BUFFERSIZE)),
                'stereo_mode': True
            }
        
        return info
    
    def create_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add grid overlay to stitched frame"""
        if not Config.SHOW_GRID_OVERLAY:
            return frame
        
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate grid dimensions
        rows = len(Config.GRID_ROWS)
        cols = len(Config.GRID_COLUMNS)
        
        cell_height = h // rows
        cell_width = w // cols
        
        # Draw grid lines
        color = (100, 100, 100)  # Gray
        thickness = 1
        
        # Vertical lines
        for i in range(1, cols):
            x = i * cell_width
            cv2.line(overlay_frame, (x, 0), (x, h), color, thickness)
        
        # Horizontal lines
        for i in range(1, rows):
            y = i * cell_height
            cv2.line(overlay_frame, (0, y), (w, y), color, thickness)
        
        # Add grid labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (150, 150, 150)
        
        for i, row in enumerate(Config.GRID_ROWS):
            for j, col in enumerate(Config.GRID_COLUMNS):
                label = f"{row}{col}"
                x = j * cell_width + 5
                y = i * cell_height + 20
                cv2.putText(overlay_frame, label, (x, y), font, font_scale, font_color, 1)
        
        return overlay_frame
    
    def _cleanup_cameras(self):
        """Clean up camera resources"""
        if self.camera1:
            self.camera1.release()
            self.camera1 = None
        
        if self.camera2:
            self.camera2.release()
            self.camera2 = None
        
        self.cameras_initialized = False
        logger.info("Camera resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cameras are released"""
        self._cleanup_cameras()
    
    def close(self):
        """Explicitly close cameras"""
        self._cleanup_cameras()

# Test function for the camera manager
def test_camera_manager():
    """Test the camera manager functionality"""
    print("Testing Camera Manager...")
    
    try:
        # Initialize camera manager
        cam_manager = CameraManager()
        
        # Test frame capture
        for i in range(10):
            frame1, frame2, success = cam_manager.capture_synchronized_frames()
            
            if success:
                print(f"Frame {i+1}: Captured successfully")
                
                # Test stitching
                stitched = cam_manager.stitch_frames(frame1, frame2)
                if stitched is not None:
                    print(f"Frame {i+1}: Stitched successfully")
                
                # Add grid overlay
                if stitched is not None:
                    with_grid = cam_manager.create_grid_overlay(stitched)
                    print(f"Frame {i+1}: Grid overlay added")
            else:
                print(f"Frame {i+1}: Capture failed")
            
            time.sleep(0.1)  # Small delay
        
        # Print performance stats
        stats = cam_manager.get_performance_stats()
        print("\nPerformance Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Print camera info
        info = cam_manager.get_camera_info()
        print("\nCamera Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        cam_manager.close()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_camera_manager()