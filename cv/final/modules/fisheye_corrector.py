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
        # Look for parameter file in multiple locations
        param_filename = f'camera_params_{self.lens_mm}mm.npz'
        search_paths = [
            param_filename,  # Current directory
            os.path.join('..', '..', param_filename),  # Project root
            os.path.join('..', param_filename),  # Parent directory
        ]

        for param_file in search_paths:
            if os.path.exists(param_file):
                try:
                    params = np.load(param_file)
                    self.K = params['K']
                    self.D = params['D']
                    print(f"Loaded fisheye parameters for {self.lens_mm}mm lens from {param_file}")
                    return True
                except Exception as e:
                    print(f"Error loading parameters from {param_file}: {e}")
                    continue

        print(f"No saved parameters found for {self.lens_mm}mm lens")
        print(f"Using optimized default parameters for warehouse cameras")
        self._set_optimized_parameters()
        return False
    
    def _set_optimized_parameters(self):
        """Set optimized fisheye parameters for warehouse cameras"""
        # Optimized camera matrix for 4K resolution (3840x2160)
        # Based on typical 2.8mm lens characteristics
        focal_length = 1200  # Adjusted for 2.8mm lens on 4K sensor
        self.K = np.array([[focal_length, 0, 1920],  # cx = width/2
                          [0, focal_length, 1080],   # cy = height/2
                          [0, 0, 1]], dtype=np.float32)

        # Optimized distortion coefficients for 2.8mm fisheye lens
        # Based on warehouse camera testing
        self.D = np.array([0.15, -0.05, 0.01, 0.0], dtype=np.float32)

        print(f"Using optimized parameters:")
        print(f"  Focal length: {focal_length}")
        print(f"  Principal point: (1920, 1080)")
        print(f"  Distortion coefficients: {self.D}")
    
    def initialize_maps(self, width, height):
        """Initialize undistortion maps"""
        if self.K is None or self.D is None:
            print("Warning: Camera parameters not loaded, cannot initialize fisheye correction")
            return False

        try:
            # Adjust camera matrix for actual frame size
            scale_x = width / 3840.0  # Scale from 4K reference
            scale_y = height / 2160.0

            adjusted_K = self.K.copy()
            adjusted_K[0, 0] *= scale_x  # fx
            adjusted_K[1, 1] *= scale_y  # fy
            adjusted_K[0, 2] *= scale_x  # cx
            adjusted_K[1, 2] *= scale_y  # cy

            # Compute new camera matrix with balance for better field of view
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                adjusted_K, self.D, (width, height), np.eye(3), balance=0.2
            )

            # Generate undistortion maps
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                adjusted_K, self.D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
            )

            print(f"Fisheye correction maps initialized for {width}x{height}")
            return True

        except Exception as e:
            print(f"Error initializing fisheye correction maps: {e}")
            return False
    
    def correct(self, frame):
        """Apply fisheye correction to a frame"""
        if frame is None:
            return frame

        if self.map1 is None or self.map2 is None:
            # Initialize maps on first frame
            height, width = frame.shape[:2]
            if not self.initialize_maps(width, height):
                print("Fisheye correction disabled - using original frame")
                return frame  # Return uncorrected if initialization fails

        try:
            # Apply fisheye correction
            corrected_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            return corrected_frame
        except Exception as e:
            print(f"Error applying fisheye correction: {e}")
            return frame  # Return original frame on error
