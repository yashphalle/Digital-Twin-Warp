"""
Simplified Lorex Pipeline for RTSP Camera Integration
Handles RTSP camera connections and frame capture for warehouse tracking
"""

import cv2
import numpy as np
import threading
import time
import queue
from datetime import datetime
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleLorexPipeline:
    """Simplified pipeline for Lorex RTSP cameras"""
    
    def __init__(self, rtsp_url: str, buffer_size: int = 5, timeout: int = 10):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.timeout = timeout
        
        # Frame queue for buffering
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Camera state
        self.cap = None
        self.running = False
        self.connected = False
        self.last_frame_time = None
        self.frame_count = 0
        
        # Performance tracking
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Camera properties
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        
    def connect_camera(self) -> bool:
        """Connect to RTSP camera"""
        logger.info(f"Connecting to RTSP camera: {self.rtsp_url}")
        
        try:
            # Create VideoCapture with RTSP URL
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set RTSP-specific options for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
            
            # Test connection
            if not self.cap.isOpened():
                raise Exception("Failed to open RTSP stream")
            
            # Read a test frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Failed to read test frame")
            
            # Get camera properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.connected = True
            self.connection_attempts = 0
            
            logger.info(f"✅ Connected to RTSP camera: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RTSP camera: {e}")
            self.connected = False
            self.connection_attempts += 1
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            return False
    
    def frame_capture_thread(self):
        """Background thread for continuous frame capture"""
        logger.info(f"Starting frame capture thread for {self.rtsp_url}")
        
        frame_errors = 0
        max_frame_errors = 10
        
        while self.running:
            try:
                if not self.connected:
                    # Try to reconnect
                    if self.connection_attempts < self.max_connection_attempts:
                        logger.info(f"Attempting to reconnect to {self.rtsp_url} (attempt {self.connection_attempts + 1})")
                        if self.connect_camera():
                            logger.info("Reconnection successful")
                            frame_errors = 0  # Reset error counter
                        else:
                            logger.warning(f"Reconnection failed, waiting {self.reconnect_delay}s")
                            time.sleep(self.reconnect_delay)
                            continue
                    else:
                        logger.error(f"Max reconnection attempts reached for {self.rtsp_url}")
                        # Reset connection attempts after a longer wait
                        time.sleep(30)
                        self.connection_attempts = 0
                        continue
                
                # Read frame from camera
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None and frame.size > 0:
                        # Update frame info
                        self.last_frame_time = datetime.now()
                        self.frame_count += 1
                        frame_errors = 0  # Reset error counter
                        
                        # Add frame to queue (non-blocking)
                        try:
                            if not self.frame_queue.full():
                                self.frame_queue.put((frame, self.last_frame_time), timeout=0.1)
                            else:
                                # Queue is full, remove oldest frame
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put((frame, self.last_frame_time), timeout=0.1)
                                except queue.Empty:
                                    pass
                        except queue.Full:
                            # Queue is still full, skip this frame
                            pass
                    else:
                        # Frame read failed
                        frame_errors += 1
                        if frame_errors > max_frame_errors:
                            logger.warning(f"Too many frame errors ({frame_errors}) from {self.rtsp_url}, reconnecting...")
                            self.connected = False
                            frame_errors = 0
                            if self.cap:
                                self.cap.release()
                                self.cap = None
                        else:
                            # Brief pause before retry
                            time.sleep(0.1)
                else:
                    # Camera not connected
                    self.connected = False
                    time.sleep(1)
                
                # Control frame rate (only if connected)
                if self.connected:
                    time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in frame capture thread: {e}")
                self.connected = False
                frame_errors += 1
                time.sleep(1)
        
        logger.info(f"Frame capture thread stopped for {self.rtsp_url}")
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, datetime]]:
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        return {
            'connected': self.connected,
            'frame_count': self.frame_count,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps,
            'queue_size': self.frame_queue.qsize(),
            'last_frame_time': self.last_frame_time,
            'connection_attempts': self.connection_attempts
        }
    
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self.connected and self.cap and self.cap.isOpened()
    
    def cleanup(self):
        """Cleanup camera resources"""
        logger.info(f"Cleaning up camera: {self.rtsp_url}")
        
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.connected = False
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

class RTSPCameraPool:
    """Pool of RTSP cameras for multi-camera setups"""
    
    def __init__(self, camera_urls: list, buffer_size: int = 5):
        self.camera_urls = camera_urls
        self.buffer_size = buffer_size
        self.pipelines = []
        self.running = False
        
        # Initialize pipelines
        for url in camera_urls:
            pipeline = SimpleLorexPipeline(url, buffer_size)
            self.pipelines.append(pipeline)
    
    def start_all(self):
        """Start all camera pipelines"""
        logger.info(f"Starting {len(self.pipelines)} RTSP camera pipelines")
        
        self.running = True
        
        for i, pipeline in enumerate(self.pipelines):
            try:
                # Connect camera
                if pipeline.connect_camera():
                    pipeline.running = True
                    
                    # Start capture thread
                    capture_thread = threading.Thread(
                        target=pipeline.frame_capture_thread,
                        name=f"RTSP-Camera-{i+1}"
                    )
                    capture_thread.daemon = True
                    capture_thread.start()
                    
                    logger.info(f"✅ Camera {i+1} started successfully")
                else:
                    logger.error(f"❌ Failed to start camera {i+1}")
                    
            except Exception as e:
                logger.error(f"❌ Error starting camera {i+1}: {e}")
    
    def get_all_frames(self) -> list:
        """Get frames from all cameras"""
        frames = []
        
        for i, pipeline in enumerate(self.pipelines):
            frame_data = pipeline.get_frame()
            if frame_data:
                frame, timestamp = frame_data
                frames.append((frame, timestamp, i+1))
            else:
                # Create placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {i+1} - No Signal", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frames.append((placeholder, datetime.now(), i+1))
        
        return frames
    
    def get_stats(self) -> dict:
        """Get statistics for all cameras"""
        stats = {
            'total_cameras': len(self.pipelines),
            'connected_cameras': 0,
            'total_frames': 0,
            'cameras': []
        }
        
        for i, pipeline in enumerate(self.pipelines):
            camera_stats = pipeline.get_stats()
            camera_stats['camera_id'] = i + 1
            camera_stats['url'] = pipeline.rtsp_url
            
            stats['cameras'].append(camera_stats)
            
            if camera_stats['connected']:
                stats['connected_cameras'] += 1
            
            stats['total_frames'] += camera_stats['frame_count']
        
        return stats
    
    def stop_all(self):
        """Stop all camera pipelines"""
        logger.info("Stopping all RTSP camera pipelines")
        
        self.running = False
        
        for pipeline in self.pipelines:
            pipeline.cleanup()
        
        logger.info("All RTSP camera pipelines stopped")

def test_rtsp_connection(rtsp_url: str) -> bool:
    """Test RTSP connection without starting full pipeline"""
    logger.info(f"Testing RTSP connection: {rtsp_url}")
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to read frame from RTSP stream")
            cap.release()
            return False
        
        # Get camera info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"✅ RTSP connection successful: {width}x{height} @ {fps}fps")
        
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"❌ RTSP connection test failed: {e}")
        return False

def main():
    """Test function for RTSP camera connections"""
    # Test camera URLs
    camera_urls = [
        "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1",  # Cam 8 Back
        "rtsp://admin:wearewarp!@192.168.0.80:554/Streaming/channels/1",  # Cam 9 Back 
        "rtsp://admin:wearewarp!@192.168.0.82:554/Streaming/channels/1"   # Cam 10 Back
    ]
    
    print("🔍 Testing RTSP Camera Connections")
    print("=" * 50)
    
    # Test each camera connection
    for i, url in enumerate(camera_urls):
        print(f"\nTesting Camera {i+1}: {url}")
        if test_rtsp_connection(url):
            print(f"✅ Camera {i+1} connection successful")
        else:
            print(f"❌ Camera {i+1} connection failed")
    
    print("\n🎯 RTSP Connection Test Complete")

if __name__ == "__main__":
    main() 