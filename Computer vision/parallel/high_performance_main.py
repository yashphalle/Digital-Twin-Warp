"""
Improved Panoramic Warehouse Tracking System
Focus on better image blending with simpler but more reliable stitching
"""

import cv2
import numpy as np
import time
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import modules
from config import Config
from detector_tracker import DetectorTracker
from database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ImprovedPanoramicCameraManager:
    """Improved panoramic camera manager with better blending"""
    
    def __init__(self):
        logger.info("Initializing Improved Panoramic Camera Manager...")
        
        # Camera setup
        self.camera1 = None
        self.camera2 = None
        self.cameras_initialized = False
        
        # Feature detection for panoramic stitching
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher()
        
        # Threading
        self.frame_buffer = queue.Queue(maxsize=3)
        self.capture_thread = None
        self.running = False
        
        # Stitching parameters
        self.overlap_ratio = 0.25  # 25% overlap
        self.blend_width = 50      # Pixels for blending zone
        
        # Calibration
        self.homography_matrix = None
        self.calibrated = False
        self.calibration_confidence = 0.0
        
        # Performance tracking
        self.frame_count = 0
        self.successful_stitches = 0
        self.failed_stitches = 0
        self.capture_times = []
        self.stitch_times = []
        
        # Initialize cameras
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize cameras"""
        try:
            # Camera 1
            self.camera1 = cv2.VideoCapture(Config.CAMERA_1_ID)
            if not self.camera1.isOpened():
                raise Exception(f"Failed to open Camera 1 (ID: {Config.CAMERA_1_ID})")
            
            # Camera 2
            self.camera2 = cv2.VideoCapture(Config.CAMERA_2_ID)
            if not self.camera2.isOpened():
                self.camera1.release()
                raise Exception(f"Failed to open Camera 2 (ID: {Config.CAMERA_2_ID})")
            
            # Configure cameras
            for i, camera in enumerate([self.camera1, self.camera2], 1):
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Consistent exposure
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                logger.info(f"Camera {i} configured")
            
            self.cameras_initialized = True
            logger.info("Cameras initialized for improved panoramic stitching")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _extract_left_view(self, stereo_frame):
        """Extract left view from stereo frame"""
        if stereo_frame is None:
            return None
        height, width = stereo_frame.shape[:2]
        return stereo_frame[:, :width//2]
    
    def find_overlap_region(self, img1, img2):
        """Find the optimal overlap region between two images"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Extract features
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return None, 0
            
            # Match features
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                return None, 0
            
            # Calculate homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if homography is not None:
                inliers = np.sum(mask)
                confidence = inliers / len(good_matches)
                return homography, confidence
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Overlap detection error: {e}")
            return None, 0
    
    def create_improved_panoramic(self, img1, img2):
        """Create panoramic image with improved blending"""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Try to find optimal overlap
            homography, confidence = self.find_overlap_region(img1, img2)
            
            if homography is not None and confidence > 0.3:
                # Use feature-based stitching
                result = self._stitch_with_homography(img1, img2, homography)
                self.calibrated = True
                self.calibration_confidence = confidence
                self.homography_matrix = homography
                return result
            else:
                # Fall back to improved geometric stitching
                logger.debug("Using improved geometric stitching")
                return self._improved_geometric_stitch(img1, img2)
                
        except Exception as e:
            logger.error(f"Panoramic creation error: {e}")
            return self._simple_stitch(img1, img2)
    
    def _stitch_with_homography(self, img1, img2, homography):
        """Stitch using homography with improved blending"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Transform corners to find output size
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, homography)
        
        # Find bounding box
        all_corners = np.concatenate([
            np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
            transformed_corners
        ])
        
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        # Output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        # Warp second image
        warped_img2 = cv2.warpPerspective(
            img2, translation @ homography, (output_width, output_height)
        )
        
        # Create result
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Place first image
        x_offset = -x_min
        y_offset = -y_min
        result[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1
        
        # Improved blending
        return self._advanced_blend(result, warped_img2, x_offset, y_offset, w1, h1)
    
    def _improved_geometric_stitch(self, img1, img2):
        """Improved geometric stitching with better blending"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Assume cameras are aligned horizontally with some overlap
        overlap_pixels = int(min(w1, w2) * self.overlap_ratio)
        
        if overlap_pixels <= 0:
            return self._simple_stitch(img1, img2)
        
        # Extract overlap regions
        img1_overlap = img1[:, -overlap_pixels:]
        img2_overlap = img2[:, :overlap_pixels]
        
        # Calculate similarity to find best alignment
        best_offset = self._find_best_vertical_alignment(img1_overlap, img2_overlap)
        
        # Create panoramic image with alignment
        return self._blend_with_alignment(img1, img2, overlap_pixels, best_offset)
    
    def _find_best_vertical_alignment(self, overlap1, overlap2):
        """Find best vertical alignment between overlap regions"""
        h1, w1 = overlap1.shape[:2]
        h2, w2 = overlap2.shape[:2]
        
        if h1 != h2:
            # Resize to same height for comparison
            overlap2 = cv2.resize(overlap2, (w2, h1))
        
        # Find best match using template matching
        result = cv2.matchTemplate(overlap1, overlap2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        return max_loc[1] if max_val > 0.5 else 0
    
    def _blend_with_alignment(self, img1, img2, overlap_pixels, y_offset):
        """Blend images with vertical alignment"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate output dimensions
        output_height = max(h1, h2 + abs(y_offset))
        output_width = w1 + w2 - overlap_pixels
        
        # Create result image
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Place first image
        result[:h1, :w1] = img1
        
        # Place second image with offset
        img2_y_start = max(0, y_offset)
        img2_y_end = min(output_height, y_offset + h2)
        img2_x_start = w1 - overlap_pixels
        img2_x_end = img2_x_start + w2
        
        # Ensure bounds
        if img2_y_end > img2_y_start and img2_x_end <= output_width:
            img2_height = img2_y_end - img2_y_start
            img2_to_place = img2[:img2_height, :] if y_offset >= 0 else img2[-y_offset:-y_offset + img2_height, :]
            
            # Create blending mask for overlap region
            blend_start_x = w1 - overlap_pixels
            blend_end_x = w1
            
            # Place non-overlapping part of img2
            if img2_x_end > blend_end_x:
                result[img2_y_start:img2_y_end, blend_end_x:img2_x_end] = img2_to_place[:, overlap_pixels:]
            
            # Blend overlap region
            if overlap_pixels > 0:
                self._linear_blend_region(
                    result, img2_to_place,
                    (img2_y_start, img2_y_end, blend_start_x, blend_end_x),
                    overlap_pixels
                )
        
        return result
    
    def _linear_blend_region(self, result, img2, region, overlap_pixels):
        """Apply linear blending in overlap region"""
        y_start, y_end, x_start, x_end = region
        
        for i in range(overlap_pixels):
            x = x_start + i
            if x < result.shape[1] and x < x_end:
                # Linear blend weight (0 = full img1, 1 = full img2)
                alpha = i / (overlap_pixels - 1) if overlap_pixels > 1 else 0.5
                
                # Blend pixels
                if y_end <= result.shape[0]:
                    result[y_start:y_end, x] = (
                        (1 - alpha) * result[y_start:y_end, x] +
                        alpha * img2[:y_end - y_start, i]
                    ).astype(np.uint8)
    
    def _advanced_blend(self, base, overlay, x_offset, y_offset, w1, h1):
        """Advanced blending for homography-based stitching"""
        # Create masks
        mask1 = np.zeros(base.shape[:2], dtype=np.float32)
        mask2 = np.zeros(base.shape[:2], dtype=np.float32)
        
        # Mark valid regions
        mask1[y_offset:y_offset + h1, x_offset:x_offset + w1] = 1.0
        mask2[overlay.sum(axis=2) > 0] = 1.0
        
        # Find overlap
        overlap = (mask1 > 0) & (mask2 > 0)
        
        if np.any(overlap):
            # Create feathering weights
            dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            
            # Normalize
            total_dist = dist1 + dist2
            total_dist[total_dist == 0] = 1
            
            alpha = dist1 / total_dist
            
            # Apply blending
            for c in range(3):
                base[overlap, c] = (
                    alpha[overlap] * base[overlap, c] +
                    (1 - alpha[overlap]) * overlay[overlap, c]
                ).astype(np.uint8)
        
        # Add non-overlapping regions
        non_overlap = (mask2 > 0) & (mask1 == 0)
        base[non_overlap] = overlay[non_overlap]
        
        return base
    
    def _simple_stitch(self, img1, img2):
        """Simple side-by-side stitching as fallback"""
        return np.hstack([img1, img2])
    
    def _capture_thread_func(self):
        """Background thread for camera capture and improved panoramic stitching"""
        logger.info("Improved panoramic capture thread started")
        
        while self.running:
            try:
                capture_start = time.time()
                
                # Capture frames
                ret1, frame1_full = self.camera1.read()
                ret2, frame2_full = self.camera2.read()
                
                if ret1 and ret2:
                    # Extract left views
                    frame1 = self._extract_left_view(frame1_full)
                    frame2 = self._extract_left_view(frame2_full)
                    
                    # Create improved panoramic stitch
                    stitched = self.create_improved_panoramic(frame1, frame2)
                    
                    if stitched is not None:
                        self.successful_stitches += 1
                    else:
                        self.failed_stitches += 1
                        stitched = self._simple_stitch(frame1, frame2)
                    
                    # Add to buffer
                    try:
                        self.frame_buffer.put_nowait({
                            'frame': stitched,
                            'timestamp': time.time(),
                            'frame1': frame1,
                            'frame2': frame2,
                            'is_panoramic': True
                        })
                    except queue.Full:
                        # Remove old frame and add new
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put_nowait({
                                'frame': stitched,
                                'timestamp': time.time(),
                                'frame1': frame1,
                                'frame2': frame2,
                                'is_panoramic': True
                            })
                        except queue.Empty:
                            pass
                    
                    # Track performance
                    capture_time = time.time() - capture_start
                    self.capture_times.append(capture_time)
                    if len(self.capture_times) > 30:
                        self.capture_times.pop(0)
                    
                    self.frame_count += 1
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Improved panoramic capture error: {e}")
                break
        
        logger.info("Improved panoramic capture thread stopped")
    
    def start_capture(self):
        """Start background capture thread"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_thread_func, daemon=True)
        self.capture_thread.start()
        logger.info("Improved panoramic capture started")
    
    def get_frame(self):
        """Get latest panoramic frame"""
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop capture thread"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        total_attempts = self.successful_stitches + self.failed_stitches
        success_rate = (self.successful_stitches / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            'frame_count': self.frame_count,
            'successful_stitches': self.successful_stitches,
            'failed_stitches': self.failed_stitches,
            'success_rate_percent': success_rate,
            'calibrated': self.calibrated,
            'calibration_confidence': self.calibration_confidence,
            'capture_fps': 1.0 / np.mean(self.capture_times) if self.capture_times else 0,
            'buffer_size': self.frame_buffer.qsize()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_capture()
        if self.camera1:
            self.camera1.release()
        if self.camera2:
            self.camera2.release()
        logger.info("Improved panoramic camera manager cleaned up")

class ImprovedPanoramicSystem:
    """System with improved panoramic blending"""
    
    def __init__(self):
        logger.info("Initializing Improved Panoramic System...")
        
        # Components
        self.camera_manager = ImprovedPanoramicCameraManager()
        self.detector_tracker = DetectorTracker(force_gpu=Config.FORCE_GPU)
        
        try:
            self.database_handler = DatabaseHandler()
        except Exception as e:
            logger.warning(f"Database failed: {e}")
            self.database_handler = None
        
        # Processing
        self.processing_executor = ThreadPoolExecutor(max_workers=4)
        
        # State
        self.running = False
        self.frame_count = 0
        self.session_start = datetime.now()
        
        # Performance
        self.total_times = []
        self.detection_times = []
        
        logger.info("Improved panoramic system initialized")
    
    def run(self):
        """Main run loop"""
        logger.info("Starting Improved Panoramic System...")
        print("🎯 Improved Panoramic Warehouse Tracking")
        print("🔥 Better Image Blending | Seamless Panoramic Views")
        print("Press 'q' to quit, 's' for stats")
        
        try:
            # Start capture
            self.camera_manager.start_capture()
            
            # Wait for initialization
            print("Initializing improved panoramic stitching...")
            time.sleep(3)
            
            self.running = True
            
            while self.running:
                # Get panoramic frame
                frame_data = self.camera_manager.get_frame()
                
                if frame_data is not None:
                    # Process frame
                    process_start = time.time()
                    
                    frame = frame_data['frame']
                    
                    # Detection and tracking
                    detection_start = time.time()
                    tracked_objects, detection_stats = self.detector_tracker.process_frame(frame)
                    detection_time = time.time() - detection_start
                    
                    # Draw annotations
                    annotated_frame = self.detector_tracker.draw_tracked_objects(frame, tracked_objects)
                    
                    # Store to database (async)
                    if self.database_handler and tracked_objects:
                        self.processing_executor.submit(self._store_objects_async, tracked_objects)
                    
                    # Performance tracking
                    total_time = time.time() - process_start
                    self.total_times.append(total_time)
                    self.detection_times.append(detection_time)
                    
                    if len(self.total_times) > 30:
                        self.total_times.pop(0)
                        self.detection_times.pop(0)
                    
                    # Create display with stats
                    display_frame = self._create_overlay(annotated_frame, {
                        'tracked_objects': tracked_objects,
                        'detection_stats': detection_stats
                    })
                    
                    # Show frame
                    cv2.imshow('Panoramic Warehouse Tracking', display_frame)
                    
                    self.frame_count += 1
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._print_stats()
                else:
                    time.sleep(0.01)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _store_objects_async(self, tracked_objects):
        """Async database storage"""
        try:
            for obj in tracked_objects:
                db_object = {
                    'persistent_id': obj['id'],
                    'center': obj['center'],
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence'],
                    'age_seconds': obj['age'],
                    'times_seen': obj['times_seen'],
                    'frame_number': self.frame_count,
                    'last_seen': datetime.now(),
                    'processing_mode': 'improved_panoramic'
                }

                if obj.get('status') == 'new':
                    db_object['first_seen'] = datetime.now()

                self.database_handler.upsert_object(db_object)

        except Exception as e:
            logger.error(f"Database error: {e}")
    
    def _create_overlay(self, frame, result_data):
        """Create performance overlay"""
        if not Config.SHOW_INFO_OVERLAY:
            return frame
        
        _, width = frame.shape[:2]
        overlay_height = 160
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        
        # Get stats
        camera_stats = self.camera_manager.get_performance_stats()
        detection_stats = result_data.get('detection_stats', {})
        
        # Calculate performance
        system_fps = 1.0 / np.mean(self.total_times) if self.total_times else 0
        detection_fps = 1.0 / np.mean(self.detection_times) if self.detection_times else 0
        
        # Colors
        color_white = (255, 255, 255)
        color_green = (0, 255, 0)
        color_yellow = (255, 255, 0)
        color_red = (0, 0, 255)
        color_cyan = (255, 255, 0)
        
        # Display
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 20
        
        # Title
        cv2.putText(overlay, "PANORAMIC Warehouse Tracking", 
                   (10, y_pos), font, 0.6, color_cyan, 2)
        y_pos += 25
        
        # Performance
        fps_color = color_green if system_fps > 10 else color_yellow if system_fps > 5 else color_red
        cv2.putText(overlay, f"System FPS: {system_fps:.1f} | Frame: {self.frame_count}", 
                   (10, y_pos), font, 0.5, fps_color, 1)
        y_pos += 20
        
        # Panoramic stats
        success_rate = camera_stats['success_rate_percent']
        stitch_color = color_green if success_rate > 90 else color_yellow if success_rate > 70 else color_red
        
        cv2.putText(overlay, f"Panoramic Success: {success_rate:.1f}% | Capture: {camera_stats['capture_fps']:.1f}fps", 
                   (10, y_pos), font, 0.45, stitch_color, 1)
        y_pos += 20
        
        # Detection
        cv2.putText(overlay, f"Detection: {detection_fps:.1f}fps | Objects: {detection_stats.get('active_objects', 0)}", 
                   (10, y_pos), font, 0.45, color_white, 1)
        y_pos += 20
        
        # Calibration
        calibrated = camera_stats['calibrated']
        calib_color = color_green if calibrated else color_yellow
        cv2.putText(overlay, f"Calibration: {'YES' if calibrated else 'AUTO'} | Confidence: {camera_stats['calibration_confidence']:.2f}", 
                   (10, y_pos), font, 0.45, calib_color, 1)
        y_pos += 20
        
        # Database
        db_status = "Connected" if self.database_handler and self.database_handler.connected else "Disconnected"
        db_color = color_green if self.database_handler and self.database_handler.connected else color_red
        cv2.putText(overlay, f"Database: {db_status} | Mode: Improved Panoramic", 
                   (10, y_pos), font, 0.45, db_color, 1)
        y_pos += 20
        
        # Controls
        cv2.putText(overlay, "Controls: 'q'=Quit | 's'=Statistics", 
                   (10, y_pos), font, 0.4, color_yellow, 1)
        
        # Combine
        combined = np.vstack([overlay, frame])
        return combined
    
    def _print_stats(self):
        """Print system statistics"""
        print("\n" + "="*60)
        print("IMPROVED PANORAMIC SYSTEM STATISTICS")
        print("="*60)
        
        uptime = (datetime.now() - self.session_start).total_seconds()
        system_fps = 1.0 / np.mean(self.total_times) if self.total_times else 0
        
        camera_stats = self.camera_manager.get_performance_stats()
        
        print(f"Uptime: {uptime:.1f}s | Frames: {self.frame_count}")
        print(f"System FPS: {system_fps:.2f}")
        print(f"Panoramic Success Rate: {camera_stats['success_rate_percent']:.1f}%")
        print(f"Successful Stitches: {camera_stats['successful_stitches']}")
        print(f"Failed Stitches: {camera_stats['failed_stitches']}")
        print(f"Calibrated: {camera_stats['calibrated']}")
        print(f"Calibration Confidence: {camera_stats['calibration_confidence']:.2f}")
        
        print("="*60)
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up improved panoramic system...")
        
        self.running = False
        
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        if self.detector_tracker:
            self.detector_tracker.cleanup()
        
        if self.database_handler:
            self.database_handler.close_connection()
        
        self.processing_executor.shutdown(wait=True)
        
        cv2.destroyAllWindows()
        
        self._print_stats()
        
        logger.info("Improved panoramic cleanup complete")

def main():
    """Main entry point"""
    print("🎯 IMPROVED PANORAMIC WAREHOUSE TRACKING")
    print("🔥 Better Image Blending | Seamless Panoramic Views")
    print("="*60)
    
    try:
        system = ImprovedPanoramicSystem()
        system.run()
    except Exception as e:
        logger.error(f"System failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()