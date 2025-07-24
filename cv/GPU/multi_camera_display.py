#!/usr/bin/env python3
"""
Multi-Camera Display Manager
Handles 11-camera grid display with fisheye correction
"""

import cv2
import numpy as np
import threading
import time
import logging
import queue
from typing import List, Dict, Optional
import sys
import os

# Add path for config imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from per_camera_queue_manager import PerCameraQueueManager
from queue_manager import FrameData

logger = logging.getLogger(__name__)

class MultiCameraDisplayManager:
    """
    Manages 11-camera grid display
    Reads from per-camera display queues and creates unified grid view
    """
    
    def __init__(self, active_cameras: List[int], queue_manager: PerCameraQueueManager):
        self.active_cameras = active_cameras
        self.queue_manager = queue_manager
        self.running = False
        
        # Display configuration
        self.window_name = "11-Camera Warehouse View"
        self.grid_cols = 4  # 4 columns
        self.grid_rows = 3  # 3 rows (11 cameras fit in 3x4 grid)
        self.cell_width = 320
        self.cell_height = 180
        
        # Frame storage
        self.latest_frames = {}  # {camera_id: frame}
        self.frame_timestamps = {}  # {camera_id: timestamp}
        
        # Performance tracking
        self.display_fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Add display queues to queue manager
        self._add_display_queues()
        
        logger.info(f"✅ Multi-Camera Display Manager initialized for {len(active_cameras)} cameras")
        logger.info(f"📺 Grid layout: {self.grid_rows}x{self.grid_cols} ({self.cell_width}x{self.cell_height} per cell)")
    
    def _add_display_queues(self):
        """Add display queues to the queue manager"""
        if not hasattr(self.queue_manager, 'camera_display_queues'):
            self.queue_manager.camera_display_queues = {
                camera_id: queue.Queue(maxsize=2)  # Small buffer for real-time display
                for camera_id in self.active_cameras
            }
            logger.info(f"📊 Added {len(self.active_cameras)} display queues")
    
    def put_display_frame(self, camera_id: int, frame_data: FrameData) -> bool:
        """Put frame into camera-specific display queue"""
        if camera_id not in self.queue_manager.camera_display_queues:
            return False
            
        try:
            display_queue = self.queue_manager.camera_display_queues[camera_id]
            
            # Replace old frame if queue full (keep latest for display)
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass
            
            display_queue.put(frame_data, timeout=0.1)
            return True
        except:
            return False
    
    def update_frames(self):
        """Get latest frames from all camera display queues"""
        for camera_id in self.active_cameras:
            try:
                if camera_id in self.queue_manager.camera_display_queues:
                    display_queue = self.queue_manager.camera_display_queues[camera_id]
                    frame_data = display_queue.get_nowait()
                    self.latest_frames[camera_id] = frame_data.frame
                    self.frame_timestamps[camera_id] = frame_data.timestamp
            except queue.Empty:
                pass  # Keep previous frame
    
    def create_grid_display(self) -> np.ndarray:
        """Create 11-camera grid display"""
        # Create empty grid
        grid_height = self.grid_rows * self.cell_height
        grid_width = self.grid_cols * self.cell_width
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Camera positioning (Camera 1 at top-right, Camera 11 at bottom-left as per memory)
        camera_positions = self._get_camera_positions()
        
        for camera_id in self.active_cameras:
            if camera_id in camera_positions and camera_id in self.latest_frames:
                row, col = camera_positions[camera_id]
                frame = self.latest_frames[camera_id]
                
                # Resize frame to cell size
                resized_frame = cv2.resize(frame, (self.cell_width, self.cell_height))
                
                # Calculate position in grid
                y1 = row * self.cell_height
                y2 = y1 + self.cell_height
                x1 = col * self.cell_width
                x2 = x1 + self.cell_width
                
                # Place frame in grid
                grid_image[y1:y2, x1:x2] = resized_frame
                
                # Add camera label
                cv2.putText(grid_image, f"Camera {camera_id}", 
                           (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add timestamp if available
                if camera_id in self.frame_timestamps:
                    timestamp = self.frame_timestamps[camera_id]
                    # Handle both float (time.time()) and datetime objects
                    if isinstance(timestamp, float):
                        import datetime
                        timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                    else:
                        timestamp_str = timestamp.strftime("%H:%M:%S")
                    cv2.putText(grid_image, timestamp_str,
                               (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add overall FPS
        cv2.putText(grid_image, f"Display FPS: {self.display_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return grid_image
    
    def _get_camera_positions(self) -> Dict[int, tuple]:
        """Get camera positions in grid (Camera 1 top-right, Camera 11 bottom-left)"""
        positions = {}
        
        # Simple left-to-right, top-to-bottom arrangement for now
        # TODO: Implement proper warehouse layout based on camera positions
        for i, camera_id in enumerate(sorted(self.active_cameras)):
            row = i // self.grid_cols
            col = i % self.grid_cols
            positions[camera_id] = (row, col)
        
        return positions
    
    def start_display(self):
        """Start the display thread"""
        self.running = True
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.grid_cols * self.cell_width, self.grid_rows * self.cell_height)
        
        logger.info(f"🚀 Starting multi-camera display for {len(self.active_cameras)} cameras")
        
        while self.running:
            try:
                # Update frames from queues
                self.update_frames()
                
                # Create grid display
                grid_image = self.create_grid_display()
                
                # Show display
                cv2.imshow(self.window_name, grid_image)
                
                # Update FPS
                self._update_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Display shutdown requested")
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    self._toggle_fullscreen()
                
            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(0.1)
        
        self.stop_display()
    
    def _update_fps(self):
        """Update display FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.display_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        # Simple fullscreen toggle
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        logger.info("🖥️  Toggled fullscreen mode")
    
    def stop_display(self):
        """Stop the display"""
        self.running = False
        cv2.destroyAllWindows()
        logger.info("🛑 Multi-camera display stopped")
