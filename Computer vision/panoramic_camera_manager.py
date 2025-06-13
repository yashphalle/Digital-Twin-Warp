"""
Panoramic Camera Manager with Feature-Based Stitching
Implements proper panoramic stitching using SIFT/ORB feature matching
Identifies overlap zones and stitches without repeating content
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional, Dict, Any, List
from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class PanoramicCameraManager:
    def __init__(self):
        """Initialize panoramic camera manager with advanced stitching"""
        logger.info("Initializing Panoramic Camera Manager...")
        
        # Camera objects
        self.camera1 = None
        self.camera2 = None
        self.cameras_initialized = False
        
        # Feature detection and matching
        self.feature_detector = None
        self.feature_matcher = None
        self.init_feature_detection()
        
        # Stitching parameters
        self.homography_matrix = None
        self.calibrated = False
        self.calibration_confidence = 0.0
        
        # Performance tracking
        self.frame_count = 0
        self.stitch_times = []
        self.feature_match_times = []
        
        # Stitching statistics
        self.successful_stitches = 0
        self.failed_stitches = 0
        self.average_matches = 0
        
        # Initialize cameras
        self._initialize_cameras()
        
    def init_feature_detection(self):
        """Initialize feature detection and matching algorithms"""
        try:
            # Try SIFT first (better quality)
            self.feature_detector = cv2.SIFT_create(
                nfeatures=1000,  # More features for stitching
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
            logger.info("Using SIFT for feature detection")
            
        except Exception as e:
            # Fallback to ORB if SIFT not available
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8
            )
            logger.info("SIFT not available, using ORB for feature detection")
        
        # Initialize feature matcher
        if isinstance(self.feature_detector, cv2.SIFT):
            # FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BF matcher for ORB
            self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
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
            
            # Configure cameras
            self._configure_cameras()
            
            self.cameras_initialized = True
            logger.info("Both cameras initialized for panoramic stitching")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self._cleanup_cameras()
            raise
    
    def _configure_cameras(self):
        """Configure cameras for optimal panoramic stitching"""
        cameras = [self.camera1, self.camera2]
        camera_names = ["Camera 1", "Camera 2"]
        
        for camera, name in zip(cameras, camera_names):
            logger.info(f"Configuring {name} for panoramic stitching...")
            
            # Set resolution
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            
            # Set FPS
            camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            # Important settings for stitching
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Consistent exposure
            
            # Verify settings
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"{name}: {actual_width}x{actual_height}")
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """Capture frames from both cameras (left views only)"""
        if not self.cameras_initialized:
            return None, None, False
        
        try:
            # Capture simultaneously for better sync
            ret1, frame1_full = self.camera1.read()
            ret2, frame2_full = self.camera2.read()
            
            if not (ret1 and ret2):
                logger.warning("Failed to capture from one or both cameras")
                return None, None, False
            
            # Extract left views from stereo cameras
            frame1_left = self._extract_left_view(frame1_full)
            frame2_left = self._extract_left_view(frame2_full)
            
            self.frame_count += 1
            return frame1_left, frame2_left, True
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None, None, False
    
    def _extract_left_view(self, stereo_frame: np.ndarray) -> np.ndarray:
        """Extract left view from ZED stereo camera frame"""
        if stereo_frame is None:
            return None
        
        height, width = stereo_frame.shape[:2]
        # Left half = left camera view
        left_view = stereo_frame[:, :width//2]
        return left_view
    
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect features and find matches between two images
        Returns: (keypoints1, keypoints2, good_matches)
        """
        feature_start = time.time()
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # Detect keypoints and descriptors
            kp1, des1 = self.feature_detector.detectAndCompute(gray1, None)
            kp2, des2 = self.feature_detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                logger.warning("No features detected in one or both images")
                return [], [], []
            
            # Match features
            if isinstance(self.feature_detector, cv2.SIFT):
                # FLANN matching for SIFT
                matches = self.feature_matcher.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:  # Stricter ratio for stitching
                            good_matches.append(m)
            else:
                # BF matching for ORB
                matches = self.feature_matcher.match(des1, des2)
                # Sort by distance and take best matches
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = matches[:min(100, len(matches) // 2)]
            
            # Track performance
            feature_time = time.time() - feature_start
            self.feature_match_times.append(feature_time)
            if len(self.feature_match_times) > 30:
                self.feature_match_times.pop(0)
            
            logger.debug(f"Found {len(good_matches)} good matches")
            return kp1, kp2, good_matches
            
        except Exception as e:
            logger.error(f"Feature matching error: {e}")
            return [], [], []
    
    def calculate_homography(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        """Calculate homography matrix from matched features"""
        if len(matches) < 10:  # Need minimum matches for reliable homography
            logger.warning(f"Insufficient matches for homography: {len(matches)}")
            return None
        
        try:
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Calculate homography using RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                ransacReprojThreshold=5.0,
                confidence=0.99,
                maxIters=2000
            )
            
            if homography is not None:
                # Calculate inlier ratio for confidence
                inliers = np.sum(mask)
                confidence = inliers / len(matches)
                logger.debug(f"Homography confidence: {confidence:.2f} ({inliers}/{len(matches)} inliers)")
                
                if confidence > 0.3:  # Minimum confidence threshold
                    return homography
                else:
                    logger.warning(f"Low homography confidence: {confidence:.2f}")
                    return None
            
        except Exception as e:
            logger.error(f"Homography calculation error: {e}")
            return None
        
        return None
    
    def stitch_panoramic(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Create panoramic stitch of two images using feature matching
        Automatically detects overlap and stitches without repeating content
        """
        stitch_start = time.time()
        
        try:
            # Detect and match features
            kp1, kp2, matches = self.detect_and_match_features(img1, img2)
            
            if len(matches) < 10:
                logger.warning("Insufficient matches for stitching")
                self.failed_stitches += 1
                return self._fallback_stitch(img1, img2)
            
            # Calculate homography
            homography = self.calculate_homography(kp1, kp2, matches)
            
            if homography is None:
                logger.warning("Failed to calculate homography")
                self.failed_stitches += 1
                return self._fallback_stitch(img1, img2)
            
            # Update calibration data
            self.homography_matrix = homography
            self.calibrated = True
            self.calibration_confidence = len(matches) / 100.0  # Normalize confidence
            
            # Determine output size
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Get corners of img2 transformed to img1 coordinate system
            corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_img2, homography)
            
            # Find bounding box of the stitched image
            corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
            all_corners = np.concatenate([corners_img1, transformed_corners], axis=0)
            
            x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
            x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
            
            # Create translation matrix to handle negative coordinates
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
            
            # Calculate output size
            output_width = x_max - x_min
            output_height = y_max - y_min
            
            # Warp img2 to align with img1
            warped_img2 = cv2.warpPerspective(
                img2, 
                translation @ homography, 
                (output_width, output_height)
            )
            
            # Create output image and place img1
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Place img1 in the result
            img1_x_offset = -x_min
            img1_y_offset = -y_min
            result[img1_y_offset:img1_y_offset + h1, img1_x_offset:img1_x_offset + w1] = img1
            
            # Blend the images
            result = self._blend_images(result, warped_img2, img1_x_offset, img1_y_offset, w1, h1)
            
            # Track performance
            stitch_time = time.time() - stitch_start
            self.stitch_times.append(stitch_time)
            if len(self.stitch_times) > 30:
                self.stitch_times.pop(0)
            
            self.successful_stitches += 1
            self.average_matches = (self.average_matches + len(matches)) / 2
            
            logger.debug(f"Panoramic stitch successful: {len(matches)} matches, {stitch_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Panoramic stitching error: {e}")
            self.failed_stitches += 1
            return self._fallback_stitch(img1, img2)
    
    def _blend_images(self, base_img: np.ndarray, overlay_img: np.ndarray, 
                     img1_x_offset: int, img1_y_offset: int, w1: int, h1: int) -> np.ndarray:
        """Advanced blending to avoid double-exposure in overlap regions"""
        result = base_img.copy()
        
        # Create masks
        mask_base = np.zeros(base_img.shape[:2], dtype=np.uint8)
        mask_overlay = np.zeros(base_img.shape[:2], dtype=np.uint8)
        
        # Mark regions
        mask_base[img1_y_offset:img1_y_offset + h1, img1_x_offset:img1_x_offset + w1] = 255
        mask_overlay[overlay_img[:, :, 0] > 0] = 255
        
        # Find overlap region
        overlap_mask = cv2.bitwise_and(mask_base, mask_overlay)
        
        # Blend in overlap region using distance transform for smooth transition
        if np.any(overlap_mask):
            # Distance transforms for smooth blending
            dist_base = cv2.distanceTransform(mask_base, cv2.DIST_L2, 5)
            dist_overlay = cv2.distanceTransform(mask_overlay, cv2.DIST_L2, 5)
            
            # Normalize distances in overlap region
            total_dist = dist_base + dist_overlay
            total_dist[total_dist == 0] = 1  # Avoid division by zero
            
            alpha = dist_base / total_dist
            alpha = np.expand_dims(alpha, axis=2)
            
            # Apply blending only in overlap region
            overlap_indices = np.where(overlap_mask)
            for i, j in zip(overlap_indices[0], overlap_indices[1]):
                result[i, j] = (alpha[i, j] * base_img[i, j] + 
                               (1 - alpha[i, j]) * overlay_img[i, j]).astype(np.uint8)
        
        # Add non-overlapping regions from overlay
        non_overlap_overlay = cv2.bitwise_and(mask_overlay, cv2.bitwise_not(mask_base))
        result[non_overlap_overlay > 0] = overlay_img[non_overlap_overlay > 0]
        
        return result
    
    def _fallback_stitch(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Fallback to simple side-by-side stitching if feature matching fails"""
        logger.info("Using fallback side-by-side stitching")
        return np.hstack([img1, img2])
    
    def get_stitching_stats(self) -> Dict[str, Any]:
        """Get comprehensive stitching statistics"""
        total_attempts = self.successful_stitches + self.failed_stitches
        success_rate = (self.successful_stitches / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            'total_frames_processed': self.frame_count,
            'successful_stitches': self.successful_stitches,
            'failed_stitches': self.failed_stitches,
            'success_rate_percent': success_rate,
            'average_matches': int(self.average_matches),
            'calibrated': self.calibrated,
            'calibration_confidence': self.calibration_confidence,
            'avg_stitch_time': np.mean(self.stitch_times) if self.stitch_times else 0,
            'avg_feature_time': np.mean(self.feature_match_times) if self.feature_match_times else 0,
            'stitch_fps': 1.0 / np.mean(self.stitch_times) if self.stitch_times else 0
        }
    
    def _cleanup_cameras(self):
        """Clean up camera resources"""
        if self.camera1:
            self.camera1.release()
            self.camera1 = None
        
        if self.camera2:
            self.camera2.release()
            self.camera2 = None
        
        self.cameras_initialized = False
        logger.info("Panoramic camera resources cleaned up")
    
    def close(self):
        """Close cameras and cleanup"""
        self._cleanup_cameras()
    
    def __del__(self):
        """Destructor"""
        self.close()

# Test function for panoramic stitching
def test_panoramic_stitching():
    """Test panoramic camera manager"""
    print("Testing Panoramic Camera Manager...")
    
    try:
        # Initialize panoramic camera manager
        pano_manager = PanoramicCameraManager()
        
        print("Testing panoramic stitching for 10 frames...")
        
        for i in range(10):
            # Capture frames
            frame1, frame2, success = pano_manager.capture_frames()
            
            if success:
                print(f"Frame {i+1}: Captured successfully")
                
                # Perform panoramic stitching
                stitched = pano_manager.stitch_panoramic(frame1, frame2)
                
                if stitched is not None:
                    print(f"Frame {i+1}: Stitched successfully - Size: {stitched.shape}")
                    
                    # Optional: Save result for inspection
                    # cv2.imwrite(f"panoramic_test_{i+1}.jpg", stitched)
                else:
                    print(f"Frame {i+1}: Stitching failed")
            else:
                print(f"Frame {i+1}: Capture failed")
            
            time.sleep(0.5)  # Brief pause
        
        # Print statistics
        stats = pano_manager.get_stitching_stats()
        print("\nPanoramic Stitching Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        pano_manager.close()
        print("\nPanoramic test completed!")
        
    except Exception as e:
        print(f"Panoramic test failed: {e}")

if __name__ == "__main__":
    test_panoramic_stitching()