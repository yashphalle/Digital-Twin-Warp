#!/usr/bin/env python3
"""
Optimized RTSP Manager for Multi-Camera Scenarios
Addresses corruption issues when scaling to 11 cameras
"""

import cv2
import numpy as np
import threading
import time
import logging
import queue
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RTSPConfig:
    """RTSP connection configuration"""
    buffer_size: int = 1
    open_timeout_ms: int = 3000
    read_timeout_ms: int = 3000
    fps_limit: int = 15
    width: int = 1280
    height: int = 720
    reconnect_attempts: int = 3
    reconnect_delay: float = 2.0
    use_tcp: bool = True
    codec_preference: str = "h264"

class OptimizedRTSPCamera:
    """
    Optimized RTSP camera with corruption prevention
    """
    
    def __init__(self, camera_id: int, rtsp_url: str, camera_name: str, config: RTSPConfig = None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.config = config or RTSPConfig()
        
        self.cap = None
        self.connected = False
        self.running = False
        
        # Performance tracking
        self.frames_read = 0
        self.failed_reads = 0
        self.last_successful_read = 0
        self.connection_attempts = 0
        
        # Frame queue for buffering
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        
        logger.info(f"✅ Optimized RTSP Camera initialized: {self.camera_name}")
    
    def _build_optimized_rtsp_url(self) -> str:
        """Build RTSP URL with optimization parameters"""
        base_url = self.rtsp_url
        
        # Add TCP transport for reliability
        if self.config.use_tcp and "?tcp" not in base_url:
            base_url += "?tcp"
        
        return base_url
    
    def connect(self) -> bool:
        """Connect with optimized settings"""
        if self.connected:
            return True
        
        self.connection_attempts += 1
        logger.info(f"🔌 Connecting to {self.camera_name} (attempt {self.connection_attempts})...")
        
        try:
            # Release existing connection
            if self.cap:
                self.cap.release()
            
            # Build optimized URL
            optimized_url = self._build_optimized_rtsp_url()
            
            # Create VideoCapture with FFMPEG backend for better RTSP support
            self.cap = cv2.VideoCapture(optimized_url, cv2.CAP_FFMPEG)
            
            # Apply optimized settings
            self._apply_optimized_settings()
            
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open RTSP stream: {self.camera_name}")
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"❌ Failed to capture test frame: {self.camera_name}")
                self.cap.release()
                self.cap = None
                return False
            
            self.connected = True
            self.last_successful_read = time.time()
            
            logger.info(f"✅ {self.camera_name} connected successfully")
            logger.info(f"   Frame size: {frame.shape}")
            logger.info(f"   Optimized URL: {optimized_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection error for {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.connected = False
            return False
    
    def _apply_optimized_settings(self):
        """Apply optimized capture settings"""
        if not self.cap:
            return
        
        # Buffer settings - minimal to reduce latency and memory
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        
        # Timeout settings - prevent hanging
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.open_timeout_ms)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.read_timeout_ms)
        
        # FPS limiting to reduce load
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_limit)
        
        # Resolution limiting to reduce bandwidth
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        
        # Codec preferences
        if self.config.codec_preference == "h264":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        logger.info(f"🔧 Applied optimized settings to {self.camera_name}")
    
    def start_threaded_capture(self):
        """Start threaded frame capture for better performance"""
        if self.running:
            return
        
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_worker,
            name=f"RTSPCapture-{self.camera_id}",
            daemon=True
        )
        self.capture_thread.start()
        logger.info(f"🧵 Started threaded capture for {self.camera_name}")
    
    def _capture_worker(self):
        """Background frame capture worker"""
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            try:
                if not self.connected:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Success - reset failure counter
                    consecutive_failures = 0
                    self.frames_read += 1
                    self.last_successful_read = time.time()
                    
                    # Add to queue (non-blocking, drop old frames)
                    try:
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass  # Queue full, drop frame
                
                else:
                    # Failure - increment counter
                    consecutive_failures += 1
                    self.failed_reads += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"⚠️ {self.camera_name}: Too many consecutive failures, attempting reconnect...")
                        self.connected = False
                        
                        # Attempt reconnection
                        if self.connect():
                            consecutive_failures = 0
                        else:
                            time.sleep(self.config.reconnect_delay)
                    
                    time.sleep(0.01)  # Brief pause on failure
                
            except Exception as e:
                logger.error(f"❌ Capture worker error for {self.camera_name}: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read latest frame from queue"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop_capture(self):
        """Stop threaded capture"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        logger.info(f"⏹️ Stopped capture for {self.camera_name}")
    
    def disconnect(self):
        """Disconnect from camera"""
        self.stop_capture()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.connected = False
        logger.info(f"🔌 {self.camera_name} disconnected")
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        current_time = time.time()
        time_since_last_read = current_time - self.last_successful_read if self.last_successful_read > 0 else 0
        
        success_rate = 0
        if self.frames_read + self.failed_reads > 0:
            success_rate = self.frames_read / (self.frames_read + self.failed_reads) * 100
        
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'connected': self.connected,
            'running': self.running,
            'frames_read': self.frames_read,
            'failed_reads': self.failed_reads,
            'success_rate': success_rate,
            'time_since_last_read': time_since_last_read,
            'connection_attempts': self.connection_attempts,
            'queue_size': self.frame_queue.qsize()
        }


class MultiCameraRTSPManager:
    """
    Multi-camera RTSP manager with corruption prevention
    """
    
    def __init__(self, camera_configs: Dict[int, dict], rtsp_config: RTSPConfig = None):
        self.camera_configs = camera_configs
        self.rtsp_config = rtsp_config or RTSPConfig()
        self.cameras = {}
        self.active_cameras = []
        
        # Initialize cameras
        for camera_id, config in camera_configs.items():
            camera = OptimizedRTSPCamera(
                camera_id=camera_id,
                rtsp_url=config['rtsp_url'],
                camera_name=config['camera_name'],
                config=self.rtsp_config
            )
            self.cameras[camera_id] = camera
        
        logger.info(f"🎥 Multi-Camera RTSP Manager initialized for {len(camera_configs)} cameras")
    
    def connect_cameras(self, camera_ids: List[int], staggered_delay: float = 0.5) -> Dict[int, bool]:
        """Connect cameras with staggered timing to prevent network overload"""
        results = {}
        
        for i, camera_id in enumerate(camera_ids):
            if camera_id in self.cameras:
                # Staggered connection to prevent network overload
                if i > 0:
                    time.sleep(staggered_delay)
                
                success = self.cameras[camera_id].connect()
                results[camera_id] = success
                
                if success:
                    self.active_cameras.append(camera_id)
                    # Start threaded capture
                    self.cameras[camera_id].start_threaded_capture()
                    logger.info(f"✅ Camera {camera_id} connected and started")
                else:
                    logger.error(f"❌ Camera {camera_id} failed to connect")
            else:
                logger.error(f"❌ Camera {camera_id} not configured")
                results[camera_id] = False
        
        logger.info(f"📊 Connected {len(self.active_cameras)}/{len(camera_ids)} cameras")
        return results
    
    def read_all_frames(self) -> Dict[int, np.ndarray]:
        """Read frames from all active cameras"""
        frames = {}
        
        for camera_id in self.active_cameras:
            camera = self.cameras[camera_id]
            ret, frame = camera.read_frame()
            
            if ret and frame is not None:
                frames[camera_id] = frame
        
        return frames
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        for camera in self.cameras.values():
            camera.disconnect()
        
        self.active_cameras.clear()
        logger.info("🔌 All cameras disconnected")
    
    def get_all_stats(self) -> Dict[int, dict]:
        """Get statistics for all cameras"""
        stats = {}
        for camera_id, camera in self.cameras.items():
            stats[camera_id] = camera.get_stats()
        return stats
    
    def print_status_report(self):
        """Print comprehensive status report"""
        stats = self.get_all_stats()
        
        print("\n📊 MULTI-CAMERA RTSP STATUS REPORT")
        print("=" * 60)
        
        total_cameras = len(stats)
        connected_cameras = sum(1 for s in stats.values() if s['connected'])
        
        print(f"📹 Total cameras: {total_cameras}")
        print(f"✅ Connected: {connected_cameras}")
        print(f"❌ Disconnected: {total_cameras - connected_cameras}")
        print()
        
        for camera_id, stat in stats.items():
            status = "🟢" if stat['connected'] else "🔴"
            print(f"{status} Camera {camera_id}: {stat['camera_name']}")
            print(f"   Frames read: {stat['frames_read']}")
            print(f"   Success rate: {stat['success_rate']:.1f}%")
            print(f"   Queue size: {stat['queue_size']}")
            print()


if __name__ == "__main__":
    # Example usage
    print("🔧 Optimized RTSP Manager for Multi-Camera Scenarios")
    print("Addresses corruption issues when scaling to 11 cameras")
