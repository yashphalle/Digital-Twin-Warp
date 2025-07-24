#!/usr/bin/env python3
"""
Multi-Camera Display System - Optimized for Fisheye Processing
Shows feeds from all 11 cameras simultaneously with improved resource management
"""

import cv2
import numpy as np
import threading
import time
import logging
import argparse
from datetime import datetime
import sys
import os
import queue
import gc
from threading import Lock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global fisheye correction pool to avoid thread safety issues
class FisheyeCorrectionPool:
    """Thread-safe fisheye correction with resource pooling"""
    
    def __init__(self, lens_mm, pool_size=4):
        self.lens_mm = lens_mm
        self.pool_size = pool_size
        self.correctors = queue.Queue(maxsize=pool_size)
        self.lock = Lock()
        self.semaphore = BoundedSemaphore(pool_size)
        
        # Pre-create corrector instances
        for _ in range(pool_size):
            corrector = OptimizedFisheyeCorrector(lens_mm)
            self.correctors.put(corrector)
        
        logger.info(f"Created fisheye correction pool with {pool_size} instances")
    
    def correct_frame(self, frame):
        """Apply fisheye correction using pooled corrector"""
        # Limit concurrent corrections to prevent resource exhaustion
        self.semaphore.acquire()
        try:
            corrector = self.correctors.get(timeout=1.0)
            try:
                corrected_frame = corrector.correct(frame)
                return corrected_frame
            finally:
                self.correctors.put(corrector)
        except queue.Empty:
            logger.warning("Fisheye correction pool exhausted, skipping correction")
            return frame
        except Exception as e:
            logger.error(f"Fisheye correction failed: {e}")
            return frame
        finally:
            self.semaphore.release()

# Global correction pool (initialized later)
fisheye_pool = None

class SingleCameraProcessor:
    """Single camera processor for multi-camera system - Optimized"""
    
    def __init__(self, camera_id: int, enable_fisheye: bool = True):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()
        
        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            # Fallback to config.py settings
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        
        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False
        self.processing_thread = None
        
        # Frame processing with rate limiting
        self.frame_count = 0
        self.total_frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.last_fps_report = time.time()
        self.target_fps = 15  # Limit target FPS to reduce load
        self.frame_interval = 1.0 / self.target_fps
        self.last_process_time = 0
        
        # Fisheye correction settings
        self.use_fisheye_correction = enable_fisheye
        
        # Fisheye debug counters
        self.fisheye_processed_frames = 0
        self.fisheye_failed_frames = 0
        self.fisheye_debug_interval = 500  # Less frequent debug output
        
        # Error handling
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        # Failed read tracking to debug hidden frame drops
        self.failed_reads = 0
        self.successful_reads = 0
        
        # Threading optimization
        self.processing_lock = Lock()
        
        logger.info(f"Initialized processor for {self.camera_name}")
        logger.info(f"Fisheye Correction: {'ENABLED' if self.use_fisheye_correction else 'DISABLED'}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"RTSP URL: {self.rtsp_url}")
    
    def connect_camera(self) -> bool:
        """Connect to the camera with optimized settings"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name}...")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Optimized capture settings for stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            
            # Try to limit resolution if possible to reduce load
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                return False
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"{self.camera_name} connected successfully")
            logger.info(f"Stream: {width}x{height} @ {fps:.1f}fps")
            
            self.connected = True
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False

    def disconnect_camera(self):
        """Disconnect from the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        logger.info(f"Disconnected from {self.camera_name}")
    
    def start_processing(self):
        """Start camera processing"""
        if self.running:
            logger.warning(f"{self.camera_name} processing already running")
            return

        if not self.connect_camera():
            logger.error(f"Failed to connect to {self.camera_name}")
            return

        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"Camera{self.camera_id}Processor",
            daemon=True
        )
        self.processing_thread.start()

        logger.info(f"Started processing for {self.camera_name}")
    
    def stop_processing(self):
        """Stop camera processing"""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3)
        
        self.disconnect_camera()
        logger.info(f"Stopped processing for {self.camera_name}")
    
    def _processing_loop(self):
        """Main processing loop for the camera - Optimized"""
        global fisheye_pool
        
        while self.running:
            try:
                current_time = time.time()
                
                # Rate limiting to prevent overwhelming the system
                if current_time - self.last_process_time < self.frame_interval:
                    time.sleep(0.01)  # Short sleep to prevent busy waiting
                    continue
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.failed_reads += 1
                    self.consecutive_failures += 1
                    
                    # Debug: Print failed read info
                    if self.failed_reads % 10 == 1:  # Print every 10th failure to avoid spam
                        print(f"DEBUG [{self.camera_name}]: Failed read #{self.failed_reads} (consecutive: {self.consecutive_failures})")
                    
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error(f"{self.camera_name}: Too many consecutive failures, stopping")
                        break
                    
                    logger.warning(f"{self.camera_name}: Failed to capture frame (failure #{self.consecutive_failures})")
                    if not self._reconnect_camera():
                        break
                    continue
                
                # Successful frame capture
                self.successful_reads += 1
                
                # Reset failure counter on successful capture
                self.consecutive_failures = 0
                self.total_frame_count += 1
                self.last_process_time = current_time
                
                # Process frame with fisheye correction using pool
                if self.use_fisheye_correction and fisheye_pool:
                    try:
                        # Debug: Log fisheye processing attempt (less frequent)
                        if self.total_frame_count % self.fisheye_debug_interval == 0:
                            logger.info(f"FISHEYE DEBUG [{self.camera_name}]: Processing frame {self.total_frame_count}")
                        
                        original_frame = frame.copy()
                        frame = fisheye_pool.correct_frame(frame)
                        
                        if np.array_equal(frame, original_frame):
                            self.fisheye_failed_frames += 1
                        else:
                            self.fisheye_processed_frames += 1
                        
                        # Debug: Log successful processing (less frequent)
                        if self.total_frame_count % self.fisheye_debug_interval == 0:
                            success_rate = (self.fisheye_processed_frames / max(1, self.fisheye_processed_frames + self.fisheye_failed_frames)) * 100
                            logger.info(f"FISHEYE DEBUG [{self.camera_name}]: Success rate: {success_rate:.1f}%")
                        
                    except Exception as e:
                        self.fisheye_failed_frames += 1
                        logger.warning(f"FISHEYE DEBUG [{self.camera_name}]: Correction failed: {e}")
                
                # Longer delay for fisheye processing to reduce system load
                time.sleep(0.01 if self.use_fisheye_correction else 0.005)
                
                # Update FPS
                self._update_fps()
                
                # Periodic garbage collection to prevent memory buildup
                if self.total_frame_count % 1000 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error in {self.camera_name} processing loop: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    break
                time.sleep(1)  # Wait before retrying
        
        self.running = False

    def _update_fps(self):
        """Update FPS counter and print to terminal"""
        self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
            
            # Print FPS to terminal every 3 seconds (less frequent for multiple cameras)
            if (current_time - self.last_fps_report) >= 3.0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                status = "Connected" if self.connected else "Disconnected"
                fisheye_status = "ON" if self.use_fisheye_correction else "OFF"
                
                # Include fisheye stats in terminal output
                if self.use_fisheye_correction:
                    success_rate = (self.fisheye_processed_frames / max(1, self.fisheye_processed_frames + self.fisheye_failed_frames)) * 100
                    total_attempts = self.successful_reads + self.failed_reads
                    read_success_rate = (self.successful_reads / max(1, total_attempts)) * 100
                    print(f"[{timestamp}] {self.camera_name} | FPS: {self.fps:.1f} | Status: {status} | Fisheye: {fisheye_status} ({success_rate:.1f}% success) | Failures: {self.consecutive_failures} | Read Success: {read_success_rate:.1f}% ({self.failed_reads} failed)")
                else:
                    total_attempts = self.successful_reads + self.failed_reads
                    read_success_rate = (self.successful_reads / max(1, total_attempts)) * 100
                    print(f"[{timestamp}] {self.camera_name} | FPS: {self.fps:.1f} | Status: {status} | Fisheye: {fisheye_status} | Failures: {self.consecutive_failures} | Read Success: {read_success_rate:.1f}% ({self.failed_reads} failed)")
                
                self.last_fps_report = current_time

    def _reconnect_camera(self) -> bool:
        """Attempt to reconnect to camera with backoff"""
        logger.info(f"Attempting {self.camera_name} reconnection...")

        self.disconnect_camera()
        
        # Exponential backoff based on failure count
        backoff_time = min(2 ** min(self.consecutive_failures, 5), 30)  # Max 30 seconds
        time.sleep(backoff_time)

        return self.connect_camera()

    def is_running(self) -> bool:
        """Check if processor is running"""
        return self.running


class MultiCameraSystem:
    """Multi-camera system manager - Optimized"""
    
    def __init__(self, camera_ids: list, enable_fisheye: bool = True):
        global fisheye_pool
        
        self.camera_ids = camera_ids
        self.enable_fisheye = enable_fisheye
        self.camera_processors = {}
        self.running = False
        
        # Initialize fisheye correction pool if needed
        if enable_fisheye:
            pool_size = min(len(camera_ids) // 2, 6)  # Limit concurrent corrections
            fisheye_pool = FisheyeCorrectionPool(Config.FISHEYE_LENS_MM, pool_size)
        
        # Initialize camera processors
        for camera_id in camera_ids:
            self.camera_processors[camera_id] = SingleCameraProcessor(
                camera_id=camera_id,
                enable_fisheye=enable_fisheye
            )
        
        logger.info(f"Initialized multi-camera system with {len(camera_ids)} cameras")
        logger.info(f"Camera IDs: {camera_ids}")
        logger.info(f"Fisheye correction: {'ENABLED' if enable_fisheye else 'DISABLED'}")
        if enable_fisheye:
            logger.info(f"Fisheye correction pool size: {pool_size}")
    
    def start_all_cameras(self):
        """Start processing for all cameras with staggered startup"""
        if self.running:
            logger.warning("Multi-camera system already running")
            return
        
        logger.info("Starting all camera processors...")
        self.running = True
        
        # Start each camera processor with longer stagger for fisheye
        stagger_delay = 2.0 if self.enable_fisheye else 0.5
        
        for camera_id, processor in self.camera_processors.items():
            processor.start_processing()
            time.sleep(stagger_delay)  # Longer stagger to reduce startup load
        
        logger.info("All camera processors started")
    
    def stop_all_cameras(self):
        """Stop processing for all cameras"""
        if not self.running:
            return
        
        logger.info("Stopping all camera processors...")
        self.running = False
        
        # Stop each camera processor
        for camera_id, processor in self.camera_processors.items():
            processor.stop_processing()
        
        logger.info("All camera processors stopped")
        
        # Clean up fisheye pool
        global fisheye_pool
        if fisheye_pool:
            fisheye_pool = None
            gc.collect()
    
    def get_system_status(self) -> dict:
        """Get status of all cameras"""
        status = {
            'total_cameras': len(self.camera_processors),
            'running_cameras': 0,
            'connected_cameras': 0,
            'camera_details': {}
        }
        
        for camera_id, processor in self.camera_processors.items():
            if processor.is_running():
                status['running_cameras'] += 1
            if processor.connected:
                status['connected_cameras'] += 1
            
            status['camera_details'][camera_id] = {
                'name': processor.camera_name,
                'running': processor.is_running(),
                'connected': processor.connected,
                'fps': processor.fps,
                'total_frames': processor.total_frame_count,
                'fisheye_processed': processor.fisheye_processed_frames,
                'fisheye_failed': processor.fisheye_failed_frames,
                'consecutive_failures': processor.consecutive_failures,
                'failed_reads': processor.failed_reads,
                'successful_reads': processor.successful_reads
            }
        
        return status
    
    def is_running(self) -> bool:
        """Check if any camera is still running"""
        return any(processor.is_running() for processor in self.camera_processors.values())


def main():
    """Main function to run the multi-camera system"""
    parser = argparse.ArgumentParser(description='Multi-Camera Display System - Optimized')
    parser.add_argument('--cameras', nargs='+', type=int, 
                       default=list(range(1, 12)),  # Cameras 1-11 by default
                       help='Camera IDs to process (default: 1-11)')
    parser.add_argument('--disable-fisheye', action='store_true',
                       help='Disable fisheye correction for all cameras')
    parser.add_argument('--status-interval', type=int, default=45,
                       help='System status report interval in seconds (default: 45)')
    parser.add_argument('--limit-cameras', type=int, default=None,
                       help='Limit number of cameras for testing (default: all)')
    
    args = parser.parse_args()
    
    # Limit cameras if specified
    if args.limit_cameras:
        args.cameras = args.cameras[:args.limit_cameras]
    
    # Determine fisheye setting
    enable_fisheye = not args.disable_fisheye
    
    print("MULTI-CAMERA DISPLAY SYSTEM - OPTIMIZED")
    print("=" * 60)
    print(f"Processing Cameras: {args.cameras}")
    print(f"Total Cameras: {len(args.cameras)}")
    print(f"Mode: HEADLESS (No GUI)")
    print(f"Fisheye Correction: {'ENABLED' if enable_fisheye else 'DISABLED'}")
    if enable_fisheye:
        print(f"Fisheye Lens MM: {Config.FISHEYE_LENS_MM}")
        print(f"Fisheye Pool Size: {min(len(args.cameras) // 2, 6)}")
    print("Press Ctrl+C to quit")
    print("=" * 60)

    # Create multi-camera system
    multi_cam_system = MultiCameraSystem(
        camera_ids=args.cameras,
        enable_fisheye=enable_fisheye
    )

    try:
        # Start all cameras
        multi_cam_system.start_all_cameras()
        
        # Status reporting loop
        last_status_report = time.time()
        
        # Keep main thread alive and provide periodic status updates
        while multi_cam_system.is_running():
            time.sleep(5)  # Check every 5 seconds
            
            # Periodic status report
            current_time = time.time()
            if (current_time - last_status_report) >= args.status_interval:
                status = multi_cam_system.get_system_status()
                
                print("\n" + "=" * 60)
                print(f"SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                print(f"Total Cameras: {status['total_cameras']}")
                print(f"Running: {status['running_cameras']}")
                print(f"Connected: {status['connected_cameras']}")
                
                if enable_fisheye:
                    total_fisheye_processed = sum(details['fisheye_processed'] for details in status['camera_details'].values())
                    total_fisheye_failed = sum(details['fisheye_failed'] for details in status['camera_details'].values())
                    total_failed_reads = sum(details['failed_reads'] for details in status['camera_details'].values())
                    total_successful_reads = sum(details['successful_reads'] for details in status['camera_details'].values())
                    total_read_attempts = total_failed_reads + total_successful_reads
                    fisheye_success_rate = (total_fisheye_processed / max(1, total_fisheye_processed + total_fisheye_failed)) * 100
                    overall_read_success_rate = (total_successful_reads / max(1, total_read_attempts)) * 100
                    print(f"Total Fisheye Processed: {total_fisheye_processed}")
                    print(f"Total Fisheye Failed: {total_fisheye_failed}")
                    print(f"Overall Fisheye Success Rate: {fisheye_success_rate:.1f}%")
                    print(f"Total Failed Reads: {total_failed_reads}")
                    print(f"Overall Read Success Rate: {overall_read_success_rate:.1f}%")
                else:
                    total_failed_reads = sum(details['failed_reads'] for details in status['camera_details'].values())
                    total_successful_reads = sum(details['successful_reads'] for details in status['camera_details'].values())
                    total_read_attempts = total_failed_reads + total_successful_reads
                    overall_read_success_rate = (total_successful_reads / max(1, total_read_attempts)) * 100
                    print(f"Total Failed Reads: {total_failed_reads}")
                    print(f"Overall Read Success Rate: {overall_read_success_rate:.1f}%")
                
                print("\nIndividual Camera Status:")
                for camera_id, details in status['camera_details'].items():
                    status_indicator = "?" if details['connected'] else "?"
                    fisheye_info = f" | Fisheye: {details['fisheye_processed']}/{details['fisheye_failed']}" if enable_fisheye else ""
                    failure_info = f" | Failures: {details['consecutive_failures']}"
                    print(f"  {status_indicator} {details['name']}: {details['fps']:.1f} FPS | Frames: {details['total_frames']}{fisheye_info}{failure_info}")
                
                print("=" * 60)
                last_status_report = current_time

    except KeyboardInterrupt:
        print("\n\nShutting down multi-camera system...")
    finally:
        multi_cam_system.stop_all_cameras()
        print("Multi-camera system stopped.")


if __name__ == "__main__":
    main()