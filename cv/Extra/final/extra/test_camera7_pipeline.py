#!/usr/bin/env python3
"""
Camera 7 Complete Pipeline Test
Uses existing camera manager and modules from main_optimized_threading
1) Camera capture
2) Fisheye correction  
3) Grounding DINO detection
4) GUI display (optional)
"""

import sys
import os
import time
import logging
import argparse
import cv2
import numpy as np
from typing import Dict, List

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import existing modules from optimized threading system
from modules.camera_manager import CPUCameraManager
from fisheye_corrector import OptimizedFisheyeCorrector
from modules.detector import CPUSimplePalletDetector
from modules.gui_display import CPUDisplayManager
from configs.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Camera7PipelineTester:
    """Complete pipeline tester for Camera 7 using existing modules"""
    
    def __init__(self, enable_gui: bool = True):
        self.camera_id = 7
        self.enable_gui = enable_gui
        self.running = False

        # Test OpenCV GUI support if GUI is requested
        if self.enable_gui:
            self.enable_gui = self._test_opencv_gui()
            if not self.enable_gui:
                logger.warning("⚠️ OpenCV GUI not working, falling back to headless mode")
        
        # Get Camera 7 configuration
        self.camera_name = f"Camera {self.camera_id}"
        self.rtsp_url = Config.RTSP_CAMERA_URLS.get(self.camera_id, "")
        
        if not self.rtsp_url:
            logger.error(f"❌ No RTSP URL configured for Camera {self.camera_id}")
            raise ValueError(f"Camera {self.camera_id} not configured")
        
        logger.info(f"🎯 Testing Camera 7 Pipeline")
        logger.info(f"📹 Camera: {self.camera_name}")
        logger.info(f"🔗 RTSP URL: {self.rtsp_url}")
        logger.info(f"🖥️ GUI: {'ENABLED' if enable_gui else 'DISABLED'}")
        
        # Initialize components using existing modules
        self._initialize_components()

    def _test_opencv_gui(self) -> bool:
        """Test if OpenCV GUI functions work"""
        try:
            # Create a small test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[:] = (0, 255, 0)  # Green

            # Try to create window and show image
            test_window = "OpenCV_GUI_Test"
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.imshow(test_window, test_img)
            cv2.waitKey(1)  # Process window events
            cv2.destroyWindow(test_window)

            logger.info("✅ OpenCV GUI test passed")
            return True

        except Exception as e:
            logger.error(f"❌ OpenCV GUI test failed: {e}")
            return False
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("🔧 Initializing pipeline components...")
        
        # 1. Camera Manager (same as optimized threading)
        self.camera_manager = CPUCameraManager(
            camera_id=self.camera_id,
            rtsp_url=self.rtsp_url,
            camera_name=self.camera_name
        )
        
        # 2. Fisheye Corrector (same as optimized threading)
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # 3. Grounding DINO Detector (same as optimized threading)
        self.pallet_detector = CPUSimplePalletDetector()
        
        # 4. GUI Display Manager (if enabled)
        if self.enable_gui:
            self.display_manager = CPUDisplayManager(
                camera_name=self.camera_name,
                camera_id=self.camera_id
            )
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'fps': 0.0,
            'avg_detection_time': 0.0,
            'avg_fisheye_time': 0.0
        }
        
        logger.info("✅ All components initialized")
    
    def connect_camera(self) -> bool:
        """Connect to Camera 7"""
        logger.info(f"📡 Connecting to {self.camera_name}...")
        
        if not self.camera_manager.connect_camera():
            logger.error(f"❌ Failed to connect to {self.camera_name}")
            return False
        
        logger.info(f"✅ Connected to {self.camera_name}")
        return True
    
    def process_single_frame(self) -> Dict:
        """Process a single frame through the complete pipeline"""
        # Step 1: Capture frame
        frame_start = time.time()
        ret, raw_frame = self.camera_manager.read_frame()
        
        if not ret or raw_frame is None:
            return {'success': False, 'error': 'Failed to capture frame'}
        
        capture_time = time.time() - frame_start
        
        # Step 2: Fisheye correction
        fisheye_start = time.time()
        corrected_frame = self.fisheye_corrector.correct(raw_frame)
        fisheye_time = time.time() - fisheye_start
        
        # Step 3: Grounding DINO detection
        detection_start = time.time()
        detections = self.pallet_detector.detect_pallets(corrected_frame)
        detection_time = time.time() - detection_start
        
        total_time = time.time() - frame_start
        fps = 1.0 / total_time if total_time > 0 else 0.0
        
        # Update stats
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['fps'] = fps
        self.stats['avg_detection_time'] = detection_time
        self.stats['avg_fisheye_time'] = fisheye_time
        
        return {
            'success': True,
            'raw_frame': raw_frame,
            'corrected_frame': corrected_frame,
            'detections': detections,
            'timing': {
                'capture_time': capture_time,
                'fisheye_time': fisheye_time,
                'detection_time': detection_time,
                'total_time': total_time,
                'fps': fps
            }
        }
    
    def run_pipeline_test(self):
        """Run the complete pipeline test"""
        logger.info("🚀 Starting Camera 7 Pipeline Test")
        logger.info("=" * 60)
        logger.info("Pipeline: Capture → Fisheye → Grounding DINO → Display")
        logger.info("Press 'q' to quit, 'p' to pause/resume")
        logger.info("=" * 60)
        
        # Connect to camera
        if not self.connect_camera():
            return
        
        self.running = True
        frame_count = 0
        last_stats_time = time.time()
        
        try:
            while self.running:
                # Process frame
                result = self.process_single_frame()
                
                if not result['success']:
                    logger.warning(f"⚠️ Frame processing failed: {result.get('error', 'Unknown error')}")
                    continue
                
                frame_count += 1
                
                # Display frame using EXACT same GUI thread approach as OptimizedPipelineSystem
                if self.enable_gui:
                    try:
                        # Use EXACT same mock tracker structure as OptimizedPipelineSystem (lines 309-314)
                        mock_tracker = type('MockTracker', (), {
                            'frame_processor': type('MockFrameProcessor', (), {
                                'final_tracked_detections': result['detections'],
                                'get_detection_counts': lambda self: {
                                    'raw_detections': len(result['detections']),
                                    'area_filtered_detections': len(result['detections']),
                                    'grid_filtered_detections': len(result['detections']),
                                    'size_filtered_detections': len(result['detections']),
                                    'final_tracked_detections': len(result['detections']),
                                    'new_objects': 0,
                                    'existing_objects': len(result['detections'])
                                },
                                'raw_detections': result['detections'],
                                'area_filtered_detections': result['detections'],
                                'grid_filtered_detections': result['detections'],
                                'size_filtered_detections': result['detections']
                            })(),
                            'camera_id': self.camera_id,
                            'camera_name': f"Camera {self.camera_id}"
                        })()

                        # EXACT same display method as OptimizedPipelineSystem (lines 316-317)
                        display_frame = self.display_manager.render_frame(result['corrected_frame'], mock_tracker)

                        # EXACT same GUI window code as OptimizedPipelineSystem (lines 319-321)
                        window_name = f"CPU Tracking - Camera {self.camera_id}"
                        cv2.imshow(window_name, display_frame)

                        # EXACT same keyboard handling as OptimizedPipelineSystem (lines 323-332)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("[GUI] 'q' key pressed - shutting down system")
                            break
                        elif key == ord('n'):
                            logger.info("[GUI] 'n' key pressed - next detection prompt")
                        elif key == ord('p'):
                            logger.info("[GUI] 'p' key pressed - previous detection prompt")

                    except Exception as e:
                        # EXACT same error handling as OptimizedPipelineSystem (lines 334-335)
                        logger.error(f"[GUI] Display error for Camera {self.camera_id}: {e}")
                        # Fall back to headless mode
                        logger.info("🔄 Falling back to headless mode")
                        self.enable_gui = False
                
                # Print periodic stats and real-time FPS
                current_time = time.time()
                if current_time - last_stats_time >= 5.0:  # Every 5 seconds
                    self._print_stats(frame_count, current_time - last_stats_time)
                    last_stats_time = current_time

                # Print real-time FPS every 30 frames
                if frame_count % 30 == 0:
                    timing = result['timing']  # Get timing from result
                    logger.info(f"🚀 Frame {frame_count}: FPS = {timing['fps']:.2f} | "
                               f"Detections = {len(result['detections'])} | "
                               f"Grounding DINO = {timing['detection_time']*1000:.1f}ms")
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def _print_stats(self, frame_count: int, elapsed_time: float):
        """Print performance statistics"""
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        logger.info("📊 PERFORMANCE STATS")
        logger.info(f"   Frames Processed: {self.stats['frames_processed']}")
        logger.info(f"   Average FPS: {avg_fps:.2f}")
        logger.info(f"   Total Detections: {self.stats['total_detections']}")
        logger.info(f"   Avg Detection Time: {self.stats['avg_detection_time']*1000:.1f}ms")
        logger.info(f"   Avg Fisheye Time: {self.stats['avg_fisheye_time']*1000:.1f}ms")
    
    def cleanup(self):
        """Cleanup resources (same method as main_optimized_threading)"""
        logger.info("🧹 Cleaning up resources...")

        self.running = False

        if hasattr(self, 'camera_manager'):
            self.camera_manager.cleanup_camera()

        # Close GUI windows using same method as main_optimized_threading
        if self.enable_gui:
            try:
                cv2.destroyAllWindows()
                logger.info("✅ GUI windows closed")
            except Exception as e:
                logger.error(f"⚠️ Error closing GUI windows: {e}")

        logger.info("✅ Cleanup complete")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Camera 7 Complete Pipeline Test')
    parser.add_argument('--no-gui', action='store_true', 
                       help='Disable GUI display (headless mode)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline tester
    enable_gui = not args.no_gui
    
    try:
        tester = Camera7PipelineTester(enable_gui=enable_gui)
        tester.run_pipeline_test()
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
