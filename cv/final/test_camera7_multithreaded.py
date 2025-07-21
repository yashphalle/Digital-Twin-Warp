#!/usr/bin/env python3
"""
Camera 7 Multi-Threaded Pipeline Test
Uses 2 detection workers to increase total processing FPS
Extracted threading components from OptimizedPipelineSystem
"""

import sys
import os
import time
import logging
import argparse
import threading
import queue
from typing import Dict, List, Optional
import cv2
import numpy as np

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import existing modules
from modules.camera_manager import CPUCameraManager
from fisheye_corrector import OptimizedFisheyeCorrector
from modules.detector import CPUSimplePalletDetector
from modules.gui_display import CPUDisplayManager
from configs.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Camera7MultiThreadedTester:
    """Multi-threaded Camera 7 pipeline tester with 2 detection workers"""
    
    def __init__(self, enable_gui: bool = True, num_detection_workers: int = 2):
        self.camera_id = 7
        self.enable_gui = enable_gui
        self.num_detection_workers = num_detection_workers
        self.running = False
        
        # Get Camera 7 configuration
        self.camera_name = f"Camera {self.camera_id}"
        self.rtsp_url = Config.RTSP_CAMERA_URLS.get(self.camera_id, "")
        
        if not self.rtsp_url:
            logger.error(f"❌ No RTSP URL configured for Camera {self.camera_id}")
            raise ValueError(f"Camera {self.camera_id} not configured")
        
        logger.info(f"🎯 Multi-Threaded Camera 7 Pipeline Test")
        logger.info(f"📹 Camera: {self.camera_name}")
        logger.info(f"🔗 RTSP URL: {self.rtsp_url}")
        logger.info(f"🧵 Detection Workers: {self.num_detection_workers}")
        logger.info(f"🖥️ GUI: {'ENABLED' if enable_gui else 'DISABLED'}")
        
        # Threading components (extracted from OptimizedPipelineSystem)
        self.frame_queue = queue.Queue(maxsize=10)  # Camera → Detection
        self.result_queue = queue.Queue(maxsize=20)  # Detection → Display
        self.detection_workers = []
        self.threads = []
        
        # Performance tracking
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'total_detections': 0,
            'capture_fps': 0.0,
            'processing_fps': 0.0,
            'avg_detection_time': 0.0,
            'worker_stats': {}
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("🔧 Initializing multi-threaded components...")
        
        # 1. Camera Manager
        self.camera_manager = CPUCameraManager(
            camera_id=self.camera_id,
            rtsp_url=self.rtsp_url,
            camera_name=self.camera_name
        )
        
        # 2. Fisheye Corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # 3. Detection Workers (2 workers sharing GPU)
        for i in range(self.num_detection_workers):
            detector = CPUSimplePalletDetector()
            self.detection_workers.append(detector)
            self.stats['worker_stats'][i] = {
                'frames_processed': 0,
                'total_time': 0.0,
                'avg_fps': 0.0
            }
        
        # 4. GUI Display Manager (if enabled)
        if self.enable_gui:
            try:
                self.display_manager = CPUDisplayManager(
                    camera_name=self.camera_name,
                    camera_id=self.camera_id
                )
                logger.info("✅ GUI Display Manager initialized")
            except Exception as e:
                logger.error(f"❌ GUI initialization failed: {e}")
                self.enable_gui = False
        
        logger.info("✅ All multi-threaded components initialized")
    
    def _camera_capture_thread(self):
        """Camera capture and fisheye correction thread (extracted from OptimizedPipelineSystem)"""
        logger.info("🎥 Starting camera capture thread...")
        
        if not self.camera_manager.connect_camera():
            logger.error(f"❌ Failed to connect to {self.camera_name}")
            return
        
        frame_count = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                ret, raw_frame = self.camera_manager.read_frame()
                if not ret or raw_frame is None:
                    continue
                
                # Fisheye correction
                corrected_frame = self.fisheye_corrector.correct(raw_frame)
                
                # Create frame data package
                frame_data = {
                    'frame_id': frame_count,
                    'timestamp': time.time(),
                    'raw_frame': raw_frame,
                    'corrected_frame': corrected_frame
                }
                
                # Add to queue (non-blocking, replace old frames if full)
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.stats['frames_captured'] += 1
                except queue.Full:
                    # Remove old frame and add new one (same as OptimizedPipelineSystem)
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                frame_count += 1
                
                # Calculate capture FPS
                current_time = time.time()
                if current_time - last_fps_time >= 5.0:
                    self.stats['capture_fps'] = frame_count / (current_time - last_fps_time)
                    logger.info(f"📹 Capture FPS: {self.stats['capture_fps']:.2f}")
                    frame_count = 0
                    last_fps_time = current_time
                
            except Exception as e:
                logger.error(f"❌ Camera capture error: {e}")
                time.sleep(0.1)
        
        self.camera_manager.cleanup_camera()
        logger.info("🎥 Camera capture thread stopped")
    
    def _detection_worker_thread(self, worker_id: int):
        """Detection worker thread (extracted from OptimizedPipelineSystem)"""
        logger.info(f"🧠 Starting detection worker {worker_id}...")
        
        detector = self.detection_workers[worker_id]
        worker_stats = self.stats['worker_stats'][worker_id]
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Process detection
                start_time = time.time()
                detections = detector.detect_pallets(frame_data['corrected_frame'])
                detection_time = time.time() - start_time
                
                # Create result package
                result_data = {
                    'frame_id': frame_data['frame_id'],
                    'timestamp': frame_data['timestamp'],
                    'corrected_frame': frame_data['corrected_frame'],
                    'detections': detections,
                    'detection_time': detection_time,
                    'worker_id': worker_id
                }
                
                # Add to result queue
                try:
                    self.result_queue.put_nowait(result_data)
                except queue.Full:
                    # Remove old result and add new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_data)
                    except queue.Empty:
                        pass
                
                # Update worker stats
                worker_stats['frames_processed'] += 1
                worker_stats['total_time'] += detection_time
                worker_stats['avg_fps'] = 1.0 / (worker_stats['total_time'] / worker_stats['frames_processed'])
                
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += len(detections)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Detection worker {worker_id} error: {e}")
                time.sleep(0.1)
        
        logger.info(f"🧠 Detection worker {worker_id} stopped")
    
    def _display_thread(self):
        """Display thread (extracted from OptimizedPipelineSystem)"""
        logger.info("🖥️ Starting display thread...")
        
        last_fps_time = time.time()
        display_count = 0
        
        while self.running:
            try:
                # Get result from queue (blocking with timeout)
                result_data = self.result_queue.get(timeout=1.0)
                
                if self.enable_gui:
                    # Create mock tracker for display (same as single-threaded version)
                    mock_tracker = type('MockTracker', (), {
                        'frame_processor': type('MockFrameProcessor', (), {
                            'final_tracked_detections': result_data['detections'],
                            'get_detection_counts': lambda self: {
                                'raw_detections': len(result_data['detections']),
                                'area_filtered_detections': len(result_data['detections']),
                                'grid_filtered_detections': len(result_data['detections']),
                                'size_filtered_detections': len(result_data['detections']),
                                'final_tracked_detections': len(result_data['detections']),
                                'new_objects': 0,
                                'existing_objects': len(result_data['detections'])
                            }
                        })(),
                        'camera_id': self.camera_id,
                        'camera_name': f"Camera {self.camera_id}"
                    })()
                    
                    # Display frame
                    display_frame = self.display_manager.render_frame(result_data['corrected_frame'], mock_tracker)
                    
                    # Add multi-threading stats overlay
                    stats_text = [
                        f"🚀 Multi-Threaded FPS Test",
                        f"Worker {result_data['worker_id']}: {1.0/result_data['detection_time']:.2f} FPS",
                        f"Detection: {result_data['detection_time']*1000:.1f}ms",
                        f"Total Workers: {self.num_detection_workers}"
                    ]
                    
                    y_start = 400
                    for i, text in enumerate(stats_text):
                        color = (0, 255, 0) if "FPS" in text else (255, 255, 255)
                        cv2.putText(display_frame, text, (20, y_start + i*20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    
                    # Show frame
                    window_name = f"Multi-Threaded Camera {self.camera_id}"
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("🛑 'q' key pressed - shutting down")
                        self.running = False
                        break
                
                display_count += 1
                
                # Calculate display FPS
                current_time = time.time()
                if current_time - last_fps_time >= 5.0:
                    self.stats['processing_fps'] = display_count / (current_time - last_fps_time)
                    logger.info(f"🖥️ Processing FPS: {self.stats['processing_fps']:.2f}")
                    display_count = 0
                    last_fps_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Display thread error: {e}")
                time.sleep(0.1)
        
        if self.enable_gui:
            cv2.destroyAllWindows()
        logger.info("🖥️ Display thread stopped")
    
    def start_multithreaded_test(self):
        """Start the multi-threaded pipeline test"""
        logger.info("🚀 Starting Multi-Threaded Camera 7 Pipeline Test")
        logger.info("=" * 70)
        logger.info(f"Architecture: Camera → Fisheye → Queue → {self.num_detection_workers} GPU Workers → Display")
        logger.info("Press 'q' to quit")
        logger.info("=" * 70)
        
        self.running = True
        
        # Start all threads (same pattern as OptimizedPipelineSystem)
        try:
            # 1. Camera capture thread
            camera_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            camera_thread.start()
            self.threads.append(camera_thread)
            
            # 2. Detection worker threads
            for i in range(self.num_detection_workers):
                worker_thread = threading.Thread(target=self._detection_worker_thread, args=(i,), daemon=True)
                worker_thread.start()
                self.threads.append(worker_thread)
            
            # 3. Display thread
            display_thread = threading.Thread(target=self._display_thread, daemon=True)
            display_thread.start()
            self.threads.append(display_thread)
            
            logger.info(f"✅ Started {len(self.threads)} threads")
            
            # Monitor performance
            self._monitor_performance()
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        finally:
            self.cleanup()
    
    def _monitor_performance(self):
        """Monitor and report performance statistics"""
        start_time = time.time()
        
        while self.running:
            try:
                time.sleep(10)  # Report every 10 seconds
                
                elapsed = time.time() - start_time
                
                logger.info("📊 MULTI-THREADED PERFORMANCE STATS")
                logger.info(f"   Runtime: {elapsed:.1f}s")
                logger.info(f"   Frames Captured: {self.stats['frames_captured']}")
                logger.info(f"   Frames Processed: {self.stats['frames_processed']}")
                logger.info(f"   Capture FPS: {self.stats['capture_fps']:.2f}")
                logger.info(f"   Processing FPS: {self.stats['processing_fps']:.2f}")
                logger.info(f"   Total Detections: {self.stats['total_detections']}")
                
                # Worker-specific stats
                for worker_id, stats in self.stats['worker_stats'].items():
                    if stats['frames_processed'] > 0:
                        logger.info(f"   Worker {worker_id}: {stats['frames_processed']} frames, {stats['avg_fps']:.2f} FPS")
                
                logger.info("-" * 50)
                
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Cleanup all resources"""
        logger.info("🧹 Cleaning up multi-threaded resources...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.enable_gui:
            cv2.destroyAllWindows()
        
        logger.info("✅ Multi-threaded cleanup complete")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Multi-Threaded Camera 7 Pipeline Test')
    parser.add_argument('--no-gui', action='store_true', 
                       help='Disable GUI display (headless mode)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of detection workers (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run multi-threaded tester
    enable_gui = not args.no_gui
    
    try:
        tester = Camera7MultiThreadedTester(
            enable_gui=enable_gui, 
            num_detection_workers=args.workers
        )
        tester.start_multithreaded_test()
        
    except Exception as e:
        logger.error(f"❌ Multi-threaded test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
