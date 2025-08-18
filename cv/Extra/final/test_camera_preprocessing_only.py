#!/usr/bin/env python3
"""
Focused Camera Preprocessing Test
Tests ONLY camera acquisition + skipping + fisheye correction
Uses existing OptimizedCameraThreadManager directly
Measures FPS at each stage and validates frame quality and timing
"""

import cv2
import time
import threading
import logging
import sys
import os
from typing import List, Dict
import numpy as np

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')  # cv/
grandparent_dir = os.path.join(current_dir, '..', '..')  # root
sys.path.insert(0, current_dir)  # cv/final/
sys.path.insert(0, parent_dir)   # cv/
sys.path.insert(0, grandparent_dir)  # root

# Import existing optimized components
from warehouse_threading.optimized_camera_threads import OptimizedCameraThreadManager
from warehouse_threading.optimized_queue_manager import OptimizedQueueManager
from warehouse_threading.queue_manager import FrameData

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import new optimized RTSP manager
try:
    from modules.optimized_rtsp_manager import MultiCameraRTSPManager, RTSPConfig
    OPTIMIZED_RTSP_AVAILABLE = True
    logger.info("✅ Optimized RTSP Manager available")
except ImportError as e:
    OPTIMIZED_RTSP_AVAILABLE = False
    logger.warning(f"⚠️ Optimized RTSP Manager not available: {e}")

# Import configs using correct paths
import importlib.util
config_path = os.path.join(parent_dir, 'configs', 'config.py')  # cv/configs/config.py
warehouse_config_path = os.path.join(parent_dir, 'configs', 'warehouse_config.py')  # cv/configs/warehouse_config.py

spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config

spec2 = importlib.util.spec_from_file_location("warehouse_config", warehouse_config_path)
warehouse_config_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(warehouse_config_module)
get_warehouse_config = warehouse_config_module.get_warehouse_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraPreprocessingTester:
    """
    Focused test for camera preprocessing pipeline
    Tests ONLY: Camera acquisition → Frame skipping → Fisheye correction
    NO detection, processing, or database operations
    """
    
    def __init__(self, active_cameras: List[int] = [7], test_duration: int = 60, enable_gui: bool = True):
        self.active_cameras = active_cameras
        self.test_duration = test_duration
        self.enable_gui = enable_gui
        self.running = False
        
        # Performance tracking
        self.stats = {
            'test_start_time': None,
            'frames_received': 0,
            'frames_processed': 0,
            'camera_stats': {},
            'fps_measurements': {
                'camera_acquisition': [],
                'fisheye_correction': [],
                'overall_preprocessing': []
            },
            'frame_quality_checks': {
                'valid_frames': 0,
                'invalid_frames': 0,
                'fisheye_corrected': 0,
                'size_validation': 0
            }
        }
        
        # Initialize camera stats for each camera
        for camera_id in self.active_cameras:
            self.stats['camera_stats'][camera_id] = {
                'frames_read': 0,
                'frames_skipped': 0,
                'frames_processed': 0,
                'connection_time': 0,
                'avg_read_time': 0,
                'avg_correction_time': 0,
                'read_times': [],
                'correction_times': []
            }
        
        # Initialize components
        self.queue_manager = OptimizedQueueManager(max_cameras=len(active_cameras))
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)

        # Initialize GUI display managers if enabled
        self.display_managers = {}
        if self.enable_gui:
            logger.info(f"[GUI] Initializing display managers for cameras: {self.active_cameras}")
            try:
                from modules.gui_display import CPUDisplayManager
                for cam_id in self.active_cameras:
                    camera_name = f"Camera {cam_id}"
                    self.display_managers[cam_id] = CPUDisplayManager(camera_name, cam_id)
                    logger.info(f"[GUI] Display manager initialized for {camera_name}")
            except Exception as e:
                logger.error(f"[GUI] Failed to initialize display managers: {e}")
                self.enable_gui = False

        # Frame consumer thread
        self.consumer_thread = None
        
        logger.info(f"🎯 Camera Preprocessing Tester initialized")
        logger.info(f"   Active cameras: {active_cameras}")
        logger.info(f"   Test duration: {test_duration} seconds")
        logger.info(f"   GUI display: {'ENABLED' if self.enable_gui else 'DISABLED'}")
        logger.info(f"   Testing: Camera acquisition + Frame skipping + Fisheye correction ONLY")
    
    def start_test(self):
        """Start the preprocessing test"""
        if self.running:
            logger.warning("Test already running")
            return
        
        self.running = True
        self.stats['test_start_time'] = time.time()
        
        logger.info("🚀 Starting Camera Preprocessing Test...")
        logger.info("=" * 60)
        
        try:
            # Start camera threads (handles acquisition + skipping + fisheye)
            logger.info("📹 Starting optimized camera threads...")
            self.camera_manager.start_camera_threads()
            
            # Start frame consumer (measures output quality and timing)
            logger.info("📊 Starting frame consumer for measurements...")
            self.consumer_thread = threading.Thread(
                target=self._frame_consumer,
                name="FrameConsumer",
                daemon=True
            )
            self.consumer_thread.start()
            
            # Start performance monitor
            logger.info("📈 Starting performance monitor...")
            monitor_thread = threading.Thread(
                target=self._performance_monitor,
                name="PerformanceMonitor", 
                daemon=True
            )
            monitor_thread.start()
            
            logger.info("✅ All components started successfully!")
            logger.info("📊 Monitoring preprocessing pipeline performance...")
            
            # Run test for specified duration
            time.sleep(self.test_duration)
            
        except KeyboardInterrupt:
            logger.info("⏹️  Test interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
        finally:
            self.stop_test()
    
    def _frame_consumer(self):
        """
        Consume preprocessed frames and measure quality/timing
        This validates the output of camera acquisition + skipping + fisheye correction
        """
        logger.info("📊 Frame consumer started - measuring preprocessing output...")
        
        while self.running:
            try:
                # Get preprocessed frame from camera threads
                frame_data = self.queue_manager.get_frame('camera_to_detection', timeout=1.0)
                if frame_data is None:
                    continue
                
                self.stats['frames_received'] += 1
                camera_id = frame_data.camera_id
                
                # Measure frame quality and timing
                self._validate_frame_quality(frame_data)
                self._measure_preprocessing_performance(frame_data)
                
                # Update camera-specific stats
                if camera_id in self.stats['camera_stats']:
                    self.stats['camera_stats'][camera_id]['frames_processed'] += 1

                # Display GUI if enabled
                if self.enable_gui and camera_id in self.display_managers:
                    self._display_fisheye_frame(frame_data)

                # Log progress every 50 frames
                if self.stats['frames_received'] % 50 == 0:
                    elapsed = time.time() - self.stats['test_start_time']
                    fps = self.stats['frames_received'] / elapsed if elapsed > 0 else 0
                    logger.info(f"📊 Processed {self.stats['frames_received']} frames, Current FPS: {fps:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Frame consumer error: {e}")
                time.sleep(0.1)
        
        logger.info("📊 Frame consumer stopped")
    
    def _validate_frame_quality(self, frame_data: FrameData):
        """Validate frame quality and properties"""
        try:
            frame = frame_data.frame
            metadata = frame_data.metadata
            
            # Basic frame validation
            if frame is not None and frame.size > 0:
                self.stats['frame_quality_checks']['valid_frames'] += 1
                
                # Check if fisheye corrected
                if metadata.get('corrected', False):
                    self.stats['frame_quality_checks']['fisheye_corrected'] += 1
                
                # Check frame size (should be resized if > 1600 width)
                height, width = frame.shape[:2]
                if width <= 1600:
                    self.stats['frame_quality_checks']['size_validation'] += 1
                    
            else:
                self.stats['frame_quality_checks']['invalid_frames'] += 1
                logger.warning(f"❌ Invalid frame from Camera {frame_data.camera_id}")
                
        except Exception as e:
            logger.error(f"❌ Frame validation error: {e}")
            self.stats['frame_quality_checks']['invalid_frames'] += 1
    
    def _measure_preprocessing_performance(self, frame_data: FrameData):
        """Measure preprocessing performance metrics"""
        try:
            metadata = frame_data.metadata
            camera_id = frame_data.camera_id
            
            # Extract timing information from metadata
            if 'cpu_savings' in metadata:
                # This comes from OptimizedCameraThreadManager
                pass
            
            # Calculate FPS for this camera
            current_time = time.time()
            if camera_id in self.stats['camera_stats']:
                camera_stats = self.stats['camera_stats'][camera_id]
                
                # Simple FPS calculation based on frame intervals
                if hasattr(self, '_last_frame_time'):
                    if camera_id in self._last_frame_time:
                        interval = current_time - self._last_frame_time[camera_id]
                        if interval > 0:
                            fps = 1.0 / interval
                            self.stats['fps_measurements']['overall_preprocessing'].append(fps)
                else:
                    self._last_frame_time = {}
                
                self._last_frame_time[camera_id] = current_time
                
        except Exception as e:
            logger.error(f"❌ Performance measurement error: {e}")

    def _display_fisheye_frame(self, frame_data: FrameData):
        """Display fisheye corrected frame in GUI window"""
        try:
            camera_id = frame_data.camera_id
            frame = frame_data.frame

            if camera_id not in self.display_managers:
                return

            display_manager = self.display_managers[camera_id]

            # Create a simple mock tracker for display compatibility
            mock_tracker = type('MockTracker', (), {
                'frame_processor': type('MockFrameProcessor', (), {
                    'final_tracked_detections': [],  # No detections, just show fisheye corrected frame
                    'get_detection_counts': lambda self: {'total': 0, 'new': 0, 'tracked': 0},
                    'raw_detections': [],
                    'filtered_detections': [],
                    'grid_filtered_detections': []
                })(),
                'camera_id': camera_id,
                'camera_name': f"Camera {camera_id}"
            })()

            # Render frame (will show fisheye corrected frame with info overlay)
            display_frame = display_manager.render_frame(frame, mock_tracker)

            # Add preprocessing info overlay
            self._add_preprocessing_overlay(display_frame, frame_data)

            # Show GUI window
            window_name = f"Fisheye Corrected - Camera {camera_id}"
            cv2.imshow(window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("[GUI] 'q' key pressed - shutting down test")
                self.stop_test()
            elif key == ord('s'):
                # Save current frame
                filename = f"camera_{camera_id}_fisheye_corrected_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"[GUI] Frame saved as {filename}")

        except Exception as e:
            logger.error(f"❌ GUI display error for Camera {camera_id}: {e}")

    def _add_preprocessing_overlay(self, frame: np.ndarray, frame_data: FrameData):
        """Add preprocessing information overlay to frame"""
        try:
            metadata = frame_data.metadata
            camera_id = frame_data.camera_id

            # Add preprocessing status text
            y_offset = 50
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 255)  # Cyan
            thickness = 2

            # Frame info
            cv2.putText(frame, f"Frame #{frame_data.frame_number}",
                       (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25

            # Fisheye correction status
            corrected = metadata.get('corrected', False)
            status = "CORRECTED" if corrected else "NOT CORRECTED"
            cv2.putText(frame, f"Fisheye: {status}",
                       (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25

            # Frame skipping info
            if 'cpu_savings' in metadata:
                cv2.putText(frame, f"CPU Savings: {metadata['cpu_savings']}",
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += 25

            # Original size info
            if 'original_size' in metadata:
                orig_w, orig_h = metadata['original_size']
                curr_h, curr_w = frame.shape[:2]
                cv2.putText(frame, f"Size: {orig_w}x{orig_h} -> {curr_w}x{curr_h}",
                           (10, y_offset), font, font_scale, color, thickness)

        except Exception as e:
            logger.error(f"❌ Overlay error: {e}")

    def _performance_monitor(self):
        """Monitor and log performance statistics during test"""
        logger.info("📈 Performance monitor started")

        while self.running:
            try:
                time.sleep(10)  # Report every 10 seconds

                elapsed = time.time() - self.stats['test_start_time']
                overall_fps = self.stats['frames_received'] / elapsed if elapsed > 0 else 0

                # Get camera optimization stats
                camera_stats = self.camera_manager.get_optimization_stats()
                queue_stats = self.queue_manager.get_optimization_stats()

                logger.info("📊 PREPROCESSING PERFORMANCE REPORT:")
                logger.info(f"   ⏱️  Elapsed: {elapsed:.1f}s")
                logger.info(f"   📹 Frames received: {self.stats['frames_received']}")
                logger.info(f"   🎯 Overall FPS: {overall_fps:.2f}")
                logger.info(f"   ✅ Valid frames: {self.stats['frame_quality_checks']['valid_frames']}")
                logger.info(f"   🔧 Fisheye corrected: {self.stats['frame_quality_checks']['fisheye_corrected']}")
                logger.info(f"   📏 Size validated: {self.stats['frame_quality_checks']['size_validation']}")

                # Camera-specific stats
                for camera_id, stats in self.stats['camera_stats'].items():
                    logger.info(f"   📹 Camera {camera_id}: {stats['frames_processed']} frames processed")

                # Optimization stats
                logger.info(f"   🚀 Expected CPU savings: {camera_stats.get('expected_cpu_savings', 'N/A')}")
                logger.info(f"   ⚡ GPU feed efficiency: {queue_stats.get('optimization', {}).get('gpu_feed_efficiency', 'N/A')}")

            except Exception as e:
                logger.error(f"❌ Performance monitor error: {e}")
                time.sleep(5)

    def stop_test(self):
        """Stop the preprocessing test and generate final report"""
        if not self.running:
            return

        logger.info("⏹️  Stopping Camera Preprocessing Test...")
        self.running = False

        # Close GUI windows if enabled
        if self.enable_gui:
            try:
                cv2.destroyAllWindows()
                logger.info("[GUI] All GUI windows closed")
            except Exception as e:
                logger.error(f"[GUI] Error closing windows: {e}")

        # Stop camera threads
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop_camera_threads()

        # Wait for consumer thread
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2.0)

        # Generate final report
        self._generate_final_report()

        logger.info("✅ Camera Preprocessing Test completed")

    def _generate_final_report(self):
        """Generate comprehensive final test report"""
        if not self.stats['test_start_time']:
            return

        total_time = time.time() - self.stats['test_start_time']
        overall_fps = self.stats['frames_received'] / total_time if total_time > 0 else 0

        logger.info("=" * 60)
        logger.info("📊 FINAL CAMERA PREPROCESSING TEST REPORT")
        logger.info("=" * 60)

        # Overall performance
        logger.info("🎯 OVERALL PERFORMANCE:")
        logger.info(f"   Test duration: {total_time:.1f} seconds")
        logger.info(f"   Total frames received: {self.stats['frames_received']}")
        logger.info(f"   Average FPS: {overall_fps:.2f}")
        logger.info(f"   Target FPS: ~1.0 (with FRAME_SKIP=5)")

        # Frame quality validation
        quality = self.stats['frame_quality_checks']
        total_frames = quality['valid_frames'] + quality['invalid_frames']
        if total_frames > 0:
            valid_percent = (quality['valid_frames'] / total_frames) * 100
            fisheye_percent = (quality['fisheye_corrected'] / total_frames) * 100 if total_frames > 0 else 0
            size_percent = (quality['size_validation'] / total_frames) * 100 if total_frames > 0 else 0

            logger.info("✅ FRAME QUALITY VALIDATION:")
            logger.info(f"   Valid frames: {quality['valid_frames']}/{total_frames} ({valid_percent:.1f}%)")
            logger.info(f"   Fisheye corrected: {quality['fisheye_corrected']}/{total_frames} ({fisheye_percent:.1f}%)")
            logger.info(f"   Size validated: {quality['size_validation']}/{total_frames} ({size_percent:.1f}%)")

        # Camera-specific performance
        logger.info("📹 CAMERA-SPECIFIC PERFORMANCE:")
        for camera_id, stats in self.stats['camera_stats'].items():
            camera_fps = stats['frames_processed'] / total_time if total_time > 0 else 0
            logger.info(f"   Camera {camera_id}: {stats['frames_processed']} frames, {camera_fps:.2f} FPS")

        # FPS measurements analysis
        if self.stats['fps_measurements']['overall_preprocessing']:
            fps_measurements = self.stats['fps_measurements']['overall_preprocessing']
            avg_fps = sum(fps_measurements) / len(fps_measurements)
            min_fps = min(fps_measurements)
            max_fps = max(fps_measurements)

            logger.info("📈 FPS ANALYSIS:")
            logger.info(f"   Average FPS: {avg_fps:.2f}")
            logger.info(f"   Min FPS: {min_fps:.2f}")
            logger.info(f"   Max FPS: {max_fps:.2f}")
            logger.info(f"   FPS samples: {len(fps_measurements)}")

        # Test results summary
        logger.info("🎯 TEST RESULTS SUMMARY:")

        # Check if FPS is within expected range (0.8 - 1.2 FPS with FRAME_SKIP=5)
        if 0.8 <= overall_fps <= 1.2:
            logger.info("   ✅ FPS: PASS - Within expected range (0.8-1.2 FPS)")
        else:
            logger.info(f"   ❌ FPS: FAIL - {overall_fps:.2f} FPS outside expected range (0.8-1.2 FPS)")

        # Check frame quality
        if total_frames > 0 and (quality['valid_frames'] / total_frames) >= 0.95:
            logger.info("   ✅ Frame Quality: PASS - >95% valid frames")
        else:
            logger.info("   ❌ Frame Quality: FAIL - <95% valid frames")

        # Check fisheye correction
        if total_frames > 0 and (quality['fisheye_corrected'] / total_frames) >= 0.95:
            logger.info("   ✅ Fisheye Correction: PASS - >95% frames corrected")
        else:
            logger.info("   ❌ Fisheye Correction: FAIL - <95% frames corrected")

        logger.info("=" * 60)


def main():
    """Main test function"""
    print("🎯 CAMERA PREPROCESSING TEST")
    print("=" * 50)
    print("Tests ONLY: Camera acquisition + Frame skipping + Fisheye correction")
    print("Uses existing OptimizedCameraThreadManager directly")
    print("NO detection, processing, or database operations")
    print("=" * 50)

    # Test configuration
    test_cameras = [7]  # Test with cameras 1 and 2
    test_duration = 60     # Run for 60 seconds
    enable_gui = True    # GUI display disabled by default

    # Allow command line arguments
    if len(sys.argv) > 1:
        try:
            test_cameras = [int(x) for x in sys.argv[1].split(',')]
            print(f"📹 Using cameras from command line: {test_cameras}")
        except:
            print(f"⚠️  Invalid camera argument, using default: {test_cameras}")

    if len(sys.argv) > 2:
        try:
            test_duration = int(sys.argv[2])
            print(f"⏱️  Using test duration from command line: {test_duration}s")
        except:
            print(f"⚠️  Invalid duration argument, using default: {test_duration}s")

    if len(sys.argv) > 3:
        try:
            enable_gui = sys.argv[3].lower() in ['true', '1', 'yes', 'gui']
            print(f"🖥️  Using GUI setting from command line: {enable_gui}")
        except:
            print(f"⚠️  Invalid GUI argument, using default: {enable_gui}")

    print(f"📹 Testing cameras: {test_cameras}")
    print(f"⏱️  Test duration: {test_duration} seconds")
    print(f"🖥️  GUI display: {'ENABLED' if enable_gui else 'DISABLED'}")
    print(f"🎯 Expected FPS: ~1.0 (with FRAME_SKIP=5)")
    if enable_gui:
        print("🎮 GUI Controls:")
        print("   'q' - Quit test")
        print("   's' - Save current frame")
    print("=" * 50)

    # Create and run test
    tester = CameraPreprocessingTester(
        active_cameras=test_cameras,
        test_duration=test_duration,
        enable_gui=enable_gui
    )

    try:
        tester.start_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.stop_test()


if __name__ == "__main__":
    main()
