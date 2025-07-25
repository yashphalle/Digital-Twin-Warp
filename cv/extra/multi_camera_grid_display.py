#!/usr/bin/env python3
"""
Multi-Camera Grid Display System
Shows all camera feeds in a single window with warehouse layout grid
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from configs.config import Config
from configs.warehouse_config import get_warehouse_config, get_camera_zone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraFeed:
    """Camera feed information"""
    camera_id: int
    camera_name: str
    rtsp_url: str
    active: bool
    column: int
    row: int
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    frame: Optional[np.ndarray] = None
    last_update: float = 0
    connected: bool = False

class MultiCameraGridDisplay:
    """Multi-camera grid display system"""
    
    def __init__(self):
        self.warehouse_config = get_warehouse_config()
        self.camera_feeds: Dict[int, CameraFeed] = {}
        self.display_running = False
        self.display_thread = None
        
        # Display settings - ADJUSTED for 3×4 grid
        self.grid_width = 1920   # Total display width
        self.grid_height = 1080  # Total display height
        self.cell_width = 640    # Individual camera cell width (1920/3 = 640)
        self.cell_height = 240   # Individual camera cell height (960/4 = 240, leaving space for headers)
        self.border_size = 2
        
        # Colors
        self.active_border_color = (0, 255, 0)  # Green for active cameras
        self.standby_border_color = (128, 128, 128)  # Gray for standby cameras
        self.disconnected_border_color = (0, 0, 255)  # Red for disconnected
        self.text_color = (255, 255, 255)  # White text
        self.bg_color = (40, 40, 40)  # Dark background
        
        self._initialize_camera_feeds()
    
    def _initialize_camera_feeds(self):
        """Initialize camera feed configurations"""
        logger.info("🏭 Initializing camera feed configurations")
        
        for camera_id, zone in self.warehouse_config.camera_zones.items():
            self.camera_feeds[camera_id] = CameraFeed(
                camera_id=camera_id,
                camera_name=zone.camera_name,
                rtsp_url=zone.rtsp_url,
                active=zone.active,
                column=zone.column,
                row=zone.row,
                x_start=zone.x_start,
                x_end=zone.x_end,
                y_start=zone.y_start,
                y_end=zone.y_end
            )
        
        logger.info(f"📹 Configured {len(self.camera_feeds)} camera feeds")
        active_count = sum(1 for feed in self.camera_feeds.values() if feed.active)
        logger.info(f"🎯 Active cameras: {active_count}")
    
    def _get_grid_position(self, camera_id: int) -> Tuple[int, int]:
        """Get grid position for camera based on warehouse layout - FIXED to match frontend"""
        # Frontend layout mapping (exactly as in frontend):
        # Camera 8-11: Column 0 (left side)
        # Camera 5-7: Column 1 (middle)
        # Camera 1-4: Column 2 (right side)

        if camera_id in [8, 9, 10, 11]:  # Column 3 cameras → Left side (column 0)
            display_col = 0
            display_row = camera_id - 8  # 8→0, 9→1, 10→2, 11→3
        elif camera_id in [5, 6, 7]:  # Column 2 cameras → Middle (column 1)
            display_col = 1
            display_row = camera_id - 5  # 5→0, 6→1, 7→2
        elif camera_id in [1, 2, 3, 4]:  # Column 1 cameras → Right side (column 2)
            display_col = 2
            display_row = camera_id - 1  # 1→0, 2→1, 3→2, 4→3
        else:
            # Fallback for any other cameras
            display_col = 0
            display_row = 0

        return display_col, display_row
    
    def _create_camera_cell(self, camera_id: int) -> np.ndarray:
        """Create display cell for a camera"""
        feed = self.camera_feeds[camera_id]
        
        # Create cell background
        cell = np.full((self.cell_height, self.cell_width, 3), self.bg_color, dtype=np.uint8)
        
        # Add camera frame if available
        if feed.frame is not None and feed.connected:
            # Resize frame to fit cell (adjusted for smaller height)
            frame_height = self.cell_height - 40  # Leave space for text
            resized_frame = cv2.resize(feed.frame, (self.cell_width - 4, frame_height))
            cell[25:25+frame_height, 2:self.cell_width-2] = resized_frame
        else:
            # Show placeholder for disconnected cameras
            placeholder_text = "DISCONNECTED" if not feed.connected else "NO SIGNAL"
            cv2.putText(cell, placeholder_text, (self.cell_width//2 - 80, self.cell_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add border based on camera status
        if feed.active and feed.connected:
            border_color = self.active_border_color
        elif feed.active and not feed.connected:
            border_color = self.disconnected_border_color
        else:
            border_color = self.standby_border_color
        
        cv2.rectangle(cell, (0, 0), (self.cell_width-1, self.cell_height-1), 
                     border_color, self.border_size)
        
        # Add camera information
        # Camera ID and name
        cv2.putText(cell, f"Camera {camera_id}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        
        # Status indicator (adjusted for smaller height)
        status = "🟢 LIVE" if (feed.active and feed.connected) else "⚪ STANDBY" if not feed.active else "🔴 OFFLINE"
        cv2.putText(cell, status, (10, self.cell_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)

        # Coverage area (smaller text)
        coverage = f"{feed.x_start:.0f}-{feed.x_end:.0f}×{feed.y_start:.0f}-{feed.y_end:.0f}ft"
        cv2.putText(cell, coverage, (10, self.cell_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)

        # Column and row info (smaller)
        col_row = f"C{feed.column}R{feed.row}"
        cv2.putText(cell, col_row, (self.cell_width - 60, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
        
        return cell
    
    def _create_grid_display(self) -> np.ndarray:
        """Create the complete grid display"""
        # Create main display canvas
        display = np.full((self.grid_height, self.grid_width, 3), self.bg_color, dtype=np.uint8)
        
        # Add title
        title = f"WARP Warehouse - Multi-Camera Grid View ({len([f for f in self.camera_feeds.values() if f.active and f.connected])} Active)"
        cv2.putText(display, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.text_color, 2)
        
        # Add warehouse layout info
        layout_info = f"Warehouse: {self.warehouse_config.width_ft}ft × {self.warehouse_config.length_ft}ft"
        cv2.putText(display, layout_info, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        
        # Calculate grid layout (3 columns × 4 rows)
        start_y = 80
        cols = 3
        rows = 4
        
        # Place camera cells in grid - FIXED to show all cameras
        for camera_id in range(1, 12):  # Cameras 1-11 in order
            if camera_id in self.camera_feeds:
                feed = self.camera_feeds[camera_id]
                display_col, display_row = self._get_grid_position(camera_id)

                # Calculate position
                x = display_col * self.cell_width
                y = start_y + display_row * self.cell_height

                # Ensure we don't exceed display bounds
                if x + self.cell_width <= self.grid_width and y + self.cell_height <= self.grid_height:
                    cell = self._create_camera_cell(camera_id)
                    display[y:y+self.cell_height, x:x+self.cell_width] = cell
        
        # Add legend
        legend_y = self.grid_height - 60
        cv2.putText(display, "🟢 Active Camera", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.active_border_color, 1)
        cv2.putText(display, "⚪ Standby Camera", (200, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.standby_border_color, 1)
        cv2.putText(display, "🔴 Disconnected", (400, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.disconnected_border_color, 1)
        
        # Add coordinate system info
        coord_info = "Coordinate System: Origin (0,0) at top-right, Camera 8-11 on LEFT side (120-180ft)"
        cv2.putText(display, coord_info, (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
        
        return display
    
    def update_camera_frame(self, camera_id: int, frame: np.ndarray):
        """Update frame for a specific camera"""
        if camera_id in self.camera_feeds:
            self.camera_feeds[camera_id].frame = frame.copy()
            self.camera_feeds[camera_id].last_update = time.time()
            self.camera_feeds[camera_id].connected = True
    
    def set_camera_disconnected(self, camera_id: int):
        """Mark camera as disconnected"""
        if camera_id in self.camera_feeds:
            self.camera_feeds[camera_id].connected = False
            self.camera_feeds[camera_id].frame = None
    
    def _display_loop(self):
        """Main display loop"""
        logger.info("🖥️ Starting grid display loop")
        
        while self.display_running:
            try:
                # Create grid display
                display = self._create_grid_display()
                
                # Show display
                cv2.imshow("WARP Warehouse - Multi-Camera Grid", display)
                
                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    logger.info("👋 Display quit requested")
                    self.stop_display()
                    break
                elif key == ord('f'):  # 'f' for fullscreen toggle
                    cv2.setWindowProperty("WARP Warehouse - Multi-Camera Grid", 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('w'):  # 'w' for windowed mode
                    cv2.setWindowProperty("WARP Warehouse - Multi-Camera Grid", 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"❌ Display loop error: {e}")
                time.sleep(1)
        
        cv2.destroyAllWindows()
        logger.info("🖥️ Grid display stopped")
    
    def start_display(self):
        """Start the grid display"""
        if not self.display_running:
            self.display_running = True
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
            logger.info("🚀 Multi-camera grid display started")
            logger.info("💡 Controls: 'q' or ESC to quit, 'f' for fullscreen, 'w' for windowed")
    
    def stop_display(self):
        """Stop the grid display"""
        self.display_running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        logger.info("🛑 Multi-camera grid display stopped")
    
    def is_running(self) -> bool:
        """Check if display is running"""
        return self.display_running

def main():
    """Test the multi-camera grid display"""
    print("🚀 MULTI-CAMERA GRID DISPLAY TEST")
    print("=" * 60)
    print("This will show a grid layout of all camera feeds")
    print("Controls:")
    print("  'q' or ESC - Quit")
    print("  'f' - Fullscreen")
    print("  'w' - Windowed mode")
    print("=" * 60)
    
    # Create display system
    grid_display = MultiCameraGridDisplay()
    
    # Start display
    grid_display.start_display()
    
    # Simulate some camera feeds (for testing)
    try:
        while grid_display.is_running():
            # In real implementation, this would be fed by the camera manager
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    
    grid_display.stop_display()

if __name__ == "__main__":
    main()
