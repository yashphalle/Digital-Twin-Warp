#!/usr/bin/env python3
"""
GStreamer-based Camera Manager for Multi-Camera RTSP Streaming
Much more robust than OpenCV for handling multiple RTSP streams
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple
import subprocess
import os

logger = logging.getLogger(__name__)

class GStreamerCameraManager:
    """
    GStreamer-based camera manager for robust RTSP streaming
    Handles network issues, codec problems, and multi-camera scenarios better than OpenCV
    """
    
    def __init__(self, camera_id: int, rtsp_url: str, camera_name: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.connected = False
        self.cap = None
        
        # GStreamer pipeline settings
        self.width = 1920
        self.height = 1080
        self.fps = 20
        
        # Connection settings
        self.buffer_size = 1
        self.timeout_ms = 5000
        
        # Performance tracking
        self.frames_read = 0
        self.connection_attempts = 0
        self.last_frame_time = 0
        
        logger.info(f"✅ GStreamer Camera Manager initialized for {self.camera_name}")
    
    def _build_gstreamer_pipeline(self) -> str:
        """
        Build optimized GStreamer pipeline for RTSP streaming
        Much more robust than OpenCV's default RTSP handling
        """
        pipeline = (
            f"rtspsrc location={self.rtsp_url} "
            f"latency=100 "
            f"buffer-mode=1 "
            f"timeout=5000000 "
            f"tcp-timeout=5000000 "
            f"retry=3 "
            f"! rtph264depay "
            f"! h264parse "
            f"! avdec_h264 "
            f"! videoconvert "
            f"! videoscale "
            f"! video/x-raw,width={self.width},height={self.height},format=BGR "
            f"! appsink drop=1 max-buffers=1 sync=0"
        )
        
        logger.info(f"🔧 GStreamer pipeline for {self.camera_name}:")
        logger.info(f"   {pipeline}")
        
        return pipeline
    
    def connect(self) -> bool:
        """Connect to camera using GStreamer pipeline"""
        if self.connected:
            return True
        
        self.connection_attempts += 1
        logger.info(f"🔌 Connecting to {self.camera_name} (attempt {self.connection_attempts})...")
        
        try:
            # Release existing connection
            if self.cap:
                self.cap.release()
            
            # Build GStreamer pipeline
            pipeline = self._build_gstreamer_pipeline()
            
            # Create VideoCapture with GStreamer backend
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open GStreamer pipeline for {self.camera_name}")
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"❌ Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                self.cap = None
                return False
            
            self.connected = True
            self.last_frame_time = time.time()
            logger.info(f"✅ {self.camera_name} connected successfully via GStreamer")
            logger.info(f"   Frame size: {frame.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GStreamer connection error for {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.connected = False
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with robust error handling"""
        if not self.connected or not self.cap:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                self.frames_read += 1
                self.last_frame_time = time.time()
                return True, frame
            else:
                logger.warning(f"⚠️ {self.camera_name}: Failed to read frame")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ {self.camera_name}: Frame read error: {e}")
            return False, None
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.connected = False
        logger.info(f"🔌 {self.camera_name} disconnected")
    
    def get_stats(self) -> dict:
        """Get camera performance statistics"""
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time if self.last_frame_time > 0 else 0
        
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'connected': self.connected,
            'frames_read': self.frames_read,
            'connection_attempts': self.connection_attempts,
            'time_since_last_frame': time_since_last_frame,
            'backend': 'GStreamer'
        }


class GStreamerMultiCameraManager:
    """
    Multi-camera manager using GStreamer for robust RTSP handling
    Handles 11 cameras with better network resilience
    """
    
    def __init__(self, camera_configs: dict):
        self.camera_configs = camera_configs
        self.camera_managers = {}
        self.active_cameras = []
        
        # Initialize camera managers
        for camera_id, config in camera_configs.items():
            manager = GStreamerCameraManager(
                camera_id=camera_id,
                rtsp_url=config['rtsp_url'],
                camera_name=config['camera_name']
            )
            self.camera_managers[camera_id] = manager
        
        logger.info(f"🎥 GStreamer Multi-Camera Manager initialized for {len(camera_configs)} cameras")
    
    def connect_cameras(self, camera_ids: list) -> dict:
        """Connect to specified cameras"""
        results = {}
        
        for camera_id in camera_ids:
            if camera_id in self.camera_managers:
                success = self.camera_managers[camera_id].connect()
                results[camera_id] = success
                
                if success:
                    self.active_cameras.append(camera_id)
                    logger.info(f"✅ Camera {camera_id} connected")
                else:
                    logger.error(f"❌ Camera {camera_id} failed to connect")
            else:
                logger.error(f"❌ Camera {camera_id} not configured")
                results[camera_id] = False
        
        logger.info(f"📊 Connected {len(self.active_cameras)}/{len(camera_ids)} cameras")
        return results
    
    def read_all_frames(self) -> dict:
        """Read frames from all active cameras"""
        frames = {}
        
        for camera_id in self.active_cameras:
            manager = self.camera_managers[camera_id]
            ret, frame = manager.read_frame()
            
            if ret and frame is not None:
                frames[camera_id] = frame
            else:
                # Try to reconnect on failure
                logger.warning(f"⚠️ Camera {camera_id}: Attempting reconnection...")
                if manager.connect():
                    ret, frame = manager.read_frame()
                    if ret and frame is not None:
                        frames[camera_id] = frame
        
        return frames
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        for manager in self.camera_managers.values():
            manager.disconnect()
        
        self.active_cameras.clear()
        logger.info("🔌 All cameras disconnected")
    
    def get_all_stats(self) -> dict:
        """Get statistics for all cameras"""
        stats = {}
        for camera_id, manager in self.camera_managers.items():
            stats[camera_id] = manager.get_stats()
        return stats


def check_gstreamer_support() -> bool:
    """Check if GStreamer is available"""
    try:
        # Test GStreamer availability
        result = subprocess.run(['gst-launch-1.0', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ GStreamer is available")
            return True
        else:
            logger.warning("⚠️ GStreamer not found")
            return False
    except Exception as e:
        logger.warning(f"⚠️ GStreamer check failed: {e}")
        return False


def install_gstreamer_guide():
    """Provide installation guide for GStreamer"""
    guide = """
    🔧 GStreamer Installation Guide:
    
    Windows:
    1. Download GStreamer from: https://gstreamer.freedesktop.org/download/
    2. Install both runtime and development packages
    3. Add to PATH: C:\\gstreamer\\1.0\\msvc_x86_64\\bin
    4. Restart terminal
    
    Linux (Ubuntu/Debian):
    sudo apt-get update
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
    sudo apt-get install gstreamer1.0-plugins-ugly gstreamer1.0-libav
    
    Test installation:
    gst-launch-1.0 --version
    """
    print(guide)


if __name__ == "__main__":
    # Test GStreamer support
    if check_gstreamer_support():
        print("✅ GStreamer is ready for multi-camera RTSP streaming")
    else:
        print("❌ GStreamer not available")
        install_gstreamer_guide()
