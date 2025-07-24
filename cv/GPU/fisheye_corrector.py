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
            return True
        else:
            print(f"No saved parameters found for {self.lens_mm}mm lens!")
            return False
    def initialize_maps(self, width, height, balance=0.5):
        """Pre-compute undistortion maps once"""
        if self.K is None or self.D is None:
            return False

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (width, height), np.eye(3), balance=balance
        )

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
        )

        return True
    
    def correct(self, frame):
        """Apply fisheye correction to a frame"""
        if frame is None:
            return frame

        if self.map1 is None or self.map2 is None:
            height, width = frame.shape[:2]
            if not self.initialize_maps(width, height):
                return frame

        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
