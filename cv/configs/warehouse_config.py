#!/usr/bin/env python3
"""
Warehouse Configuration System
Manages warehouse-specific settings including dimensions, camera layout, and coordinate system
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraZone:
    """Camera coverage zone definition"""
    camera_id: int
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    column: int
    row: int
    camera_name: str
    rtsp_url: str
    active: bool = False

@dataclass
class WarehouseConfig:
    """Complete warehouse configuration"""
    warehouse_id: str
    name: str
    width_ft: float
    length_ft: float
    origin_position: str  # "top-right", "top-left", "bottom-right", "bottom-left"
    x_axis_direction: str  # "left-to-right" or "right-to-left"
    y_axis_direction: str  # "top-to-bottom" or "bottom-to-top"
    camera_zones: Dict[int, CameraZone]
    active_cameras: List[int]
    
    def get_global_coordinates(self, camera_id: int, local_x: float, local_y: float) -> Tuple[float, float]:
        """Convert local camera coordinates to global warehouse coordinates"""
        if camera_id not in self.camera_zones:
            raise ValueError(f"Camera {camera_id} not found in warehouse configuration")
        
        zone = self.camera_zones[camera_id]
        
        # For direct global mapping, local coordinates should already be global
        # This method is for future use when we need coordinate transformation
        return local_x, local_y
    
    def get_camera_coverage_area(self, camera_id: int) -> Dict[str, float]:
        """Get camera coverage area in global coordinates"""
        if camera_id not in self.camera_zones:
            raise ValueError(f"Camera {camera_id} not found in warehouse configuration")
        
        zone = self.camera_zones[camera_id]
        return {
            'x_start': zone.x_start,
            'x_end': zone.x_end,
            'y_start': zone.y_start,
            'y_end': zone.y_end,
            'width': zone.x_end - zone.x_start,
            'height': zone.y_end - zone.y_start,
            'center_x': (zone.x_start + zone.x_end) / 2,
            'center_y': (zone.y_start + zone.y_end) / 2
        }

class WarehouseConfigManager:
    """Manages warehouse configurations"""
    
    def __init__(self, config_dir: str = "configs/warehouse_configs"):
        self.config_dir = config_dir
        self.current_config: Optional[WarehouseConfig] = None
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            logger.info(f"Created warehouse config directory: {self.config_dir}")
    
    def create_default_warp_warehouse(self) -> WarehouseConfig:
        """Create default configuration for WARP warehouse"""
        
        # Define camera zones for WARP warehouse
        camera_zones = {}
        
        # Column 3 (LEFT side in display, 120-180ft X-axis) - Cameras 8,9,10,11
        camera_zones[8] = CameraZone(
            camera_id=8, x_start=120, x_end=180, y_start=0, y_end=25,
            column=3, row=1, camera_name="Camera 8 - Column 3 Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1",
            active=True
        )
        camera_zones[9] = CameraZone(
            camera_id=9, x_start=120, x_end=180, y_start=25, y_end=50,
            column=3, row=2, camera_name="Camera 9 - Column 3 Mid-Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.80:554/Streaming/channels/1",
            active=True
        )
        camera_zones[10] = CameraZone(
            camera_id=10, x_start=120, x_end=180, y_start=50, y_end=75,
            column=3, row=3, camera_name="Camera 10 - Column 3 Mid-Bottom",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.81:554/Streaming/channels/1",
            active=True
        )
        camera_zones[11] = CameraZone(
            camera_id=11, x_start=120, x_end=180, y_start=75, y_end=90,
            column=3, row=4, camera_name="Camera 11 - Column 3 Bottom",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.82:554/Streaming/channels/1",
            active=True
        )
        
        # Column 2 (MIDDLE, 60-120ft X-axis) - Cameras 5,6,7
        camera_zones[5] = CameraZone(
            camera_id=5, x_start=60, x_end=120, y_start=0, y_end=22.5,
            column=2, row=1, camera_name="Camera 5 - Column 2 Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.75:554/Streaming/channels/1",
            active=False
        )
        camera_zones[6] = CameraZone(
            camera_id=6, x_start=60, x_end=120, y_start=22.5, y_end=45,
            column=2, row=2, camera_name="Camera 6 - Column 2 Mid-Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.76:554/Streaming/channels/1",
            active=False
        )
        camera_zones[7] = CameraZone(
            camera_id=7, x_start=60, x_end=120, y_start=45, y_end=67.5,
            column=2, row=3, camera_name="Camera 7 - Column 2 Mid-Bottom",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.77:554/Streaming/channels/1",
            active=False
        )
        
        # Column 1 (RIGHT side in display, 0-60ft X-axis) - Cameras 1,2,3,4
        camera_zones[1] = CameraZone(
            camera_id=1, x_start=0, x_end=60, y_start=0, y_end=22.5,
            column=1, row=1, camera_name="Camera 1 - Column 1 Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.71:554/Streaming/channels/1",
            active=False
        )
        camera_zones[2] = CameraZone(
            camera_id=2, x_start=0, x_end=60, y_start=22.5, y_end=45,
            column=1, row=2, camera_name="Camera 2 - Column 1 Mid-Top",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.72:554/Streaming/channels/1",
            active=False
        )
        camera_zones[3] = CameraZone(
            camera_id=3, x_start=0, x_end=60, y_start=45, y_end=67.5,
            column=1, row=3, camera_name="Camera 3 - Column 1 Mid-Bottom",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.73:554/Streaming/channels/1",
            active=False
        )
        camera_zones[4] = CameraZone(
            camera_id=4, x_start=0, x_end=60, y_start=67.5, y_end=90,
            column=1, row=4, camera_name="Camera 4 - Column 1 Bottom",
            rtsp_url="rtsp://admin:wearewarp!@192.168.0.74:554/Streaming/channels/1",
            active=False
        )
        
        # Create warehouse config
        config = WarehouseConfig(
            warehouse_id="warp_main",
            name="WARP Main Warehouse",
            width_ft=180.0,
            length_ft=90.0,
            origin_position="top-right",
            x_axis_direction="right-to-left",
            y_axis_direction="top-to-bottom",
            camera_zones=camera_zones,
            active_cameras=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # ALL CAMERAS ENABLED
        )
        
        return config
    
    def save_config(self, config: WarehouseConfig, filename: Optional[str] = None):
        """Save warehouse configuration to file"""
        if filename is None:
            filename = f"{config.warehouse_id}.json"
        
        filepath = os.path.join(self.config_dir, filename)
        
        # Convert to serializable format
        config_dict = {
            'warehouse_id': config.warehouse_id,
            'name': config.name,
            'width_ft': config.width_ft,
            'length_ft': config.length_ft,
            'origin_position': config.origin_position,
            'x_axis_direction': config.x_axis_direction,
            'y_axis_direction': config.y_axis_direction,
            'active_cameras': config.active_cameras,
            'camera_zones': {}
        }
        
        for camera_id, zone in config.camera_zones.items():
            config_dict['camera_zones'][str(camera_id)] = {
                'camera_id': zone.camera_id,
                'x_start': zone.x_start,
                'x_end': zone.x_end,
                'y_start': zone.y_start,
                'y_end': zone.y_end,
                'column': zone.column,
                'row': zone.row,
                'camera_name': zone.camera_name,
                'rtsp_url': zone.rtsp_url,
                'active': zone.active
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Warehouse configuration saved to: {filepath}")
    
    def load_config(self, filename: str) -> WarehouseConfig:
        """Load warehouse configuration from file"""
        filepath = os.path.join(self.config_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Warehouse config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert camera zones
        camera_zones = {}
        for camera_id_str, zone_dict in config_dict['camera_zones'].items():
            camera_id = int(camera_id_str)
            camera_zones[camera_id] = CameraZone(**zone_dict)
        
        config = WarehouseConfig(
            warehouse_id=config_dict['warehouse_id'],
            name=config_dict['name'],
            width_ft=config_dict['width_ft'],
            length_ft=config_dict['length_ft'],
            origin_position=config_dict['origin_position'],
            x_axis_direction=config_dict['x_axis_direction'],
            y_axis_direction=config_dict['y_axis_direction'],
            camera_zones=camera_zones,
            active_cameras=config_dict['active_cameras']
        )
        
        self.current_config = config
        logger.info(f"Warehouse configuration loaded: {config.name}")
        return config
    
    def get_current_config(self) -> WarehouseConfig:
        """Get current warehouse configuration"""
        if self.current_config is None:
            # Try to load default config, or create it
            try:
                return self.load_config("warp_main.json")
            except FileNotFoundError:
                logger.info("Creating default WARP warehouse configuration")
                config = self.create_default_warp_warehouse()
                self.save_config(config)
                self.current_config = config
                return config
        
        return self.current_config

# Global instance
warehouse_config_manager = WarehouseConfigManager()

def get_warehouse_config() -> WarehouseConfig:
    """Get current warehouse configuration"""
    return warehouse_config_manager.get_current_config()

def get_camera_zone(camera_id: int) -> CameraZone:
    """Get camera zone configuration"""
    config = get_warehouse_config()
    if camera_id not in config.camera_zones:
        raise ValueError(f"Camera {camera_id} not found in warehouse configuration")
    return config.camera_zones[camera_id]

def get_active_cameras() -> List[int]:
    """Get list of active cameras"""
    config = get_warehouse_config()
    return config.active_cameras
