#!/usr/bin/env python3
"""
Warehouse Setup Tool
Interactive tool for configuring different warehouses
"""

import json
import os
from typing import Dict, List, Tuple
from warehouse_config import WarehouseConfigManager, WarehouseConfig, CameraZone

def create_warehouse_config_interactive():
    """Interactive warehouse configuration creation"""
    print("🏭 WAREHOUSE CONFIGURATION SETUP")
    print("=" * 60)
    
    # Basic warehouse info
    warehouse_id = input("Enter warehouse ID (e.g., 'warp_main', 'facility_2'): ").strip()
    warehouse_name = input("Enter warehouse name: ").strip()
    
    # Dimensions
    print("\n📏 Warehouse Dimensions:")
    width_ft = float(input("Enter warehouse width (feet): "))
    length_ft = float(input("Enter warehouse length (feet): "))
    
    # Coordinate system
    print("\n🎯 Coordinate System:")
    print("1. Top-right origin (X: right→left, Y: top→bottom)")
    print("2. Top-left origin (X: left→right, Y: top→bottom)")
    print("3. Bottom-right origin (X: right→left, Y: bottom→top)")
    print("4. Bottom-left origin (X: left→right, Y: bottom→top)")
    
    coord_choice = input("Choose coordinate system (1-4): ").strip()
    
    coord_systems = {
        "1": ("top-right", "right-to-left", "top-to-bottom"),
        "2": ("top-left", "left-to-right", "top-to-bottom"),
        "3": ("bottom-right", "right-to-left", "bottom-to-top"),
        "4": ("bottom-left", "left-to-right", "bottom-to-top")
    }
    
    origin_position, x_axis_direction, y_axis_direction = coord_systems.get(coord_choice, coord_systems["1"])
    
    # Camera layout
    print(f"\n📹 Camera Layout for {width_ft}ft × {length_ft}ft warehouse:")
    num_cameras = int(input("Enter number of cameras: "))
    
    print("\nCamera layout options:")
    print("1. Grid layout (automatic)")
    print("2. Manual camera positioning")
    
    layout_choice = input("Choose layout method (1-2): ").strip()
    
    camera_zones = {}
    
    if layout_choice == "1":
        # Automatic grid layout
        print("\n🔢 Automatic Grid Layout:")
        columns = int(input("Enter number of columns: "))
        rows = int(input("Enter number of rows: "))
        
        if columns * rows < num_cameras:
            print(f"Warning: {columns}×{rows} grid only supports {columns*rows} cameras")
            rows = (num_cameras + columns - 1) // columns
            print(f"Adjusted to {columns}×{rows} grid")
        
        # Calculate camera zones
        col_width = width_ft / columns
        row_height = length_ft / rows
        
        camera_id = 1
        for col in range(columns):
            for row in range(rows):
                if camera_id > num_cameras:
                    break
                
                # Calculate zone boundaries
                if origin_position.startswith("top"):
                    if x_axis_direction == "left-to-right":
                        x_start = col * col_width
                        x_end = (col + 1) * col_width
                    else:  # right-to-left
                        x_start = (columns - col - 1) * col_width
                        x_end = (columns - col) * col_width
                    
                    y_start = row * row_height
                    y_end = (row + 1) * row_height
                else:  # bottom origin
                    if x_axis_direction == "left-to-right":
                        x_start = col * col_width
                        x_end = (col + 1) * col_width
                    else:  # right-to-left
                        x_start = (columns - col - 1) * col_width
                        x_end = (columns - col) * col_width
                    
                    y_start = (rows - row - 1) * row_height
                    y_end = (rows - row) * row_height
                
                # Get camera details
                print(f"\nCamera {camera_id} ({x_start:.1f}-{x_end:.1f}ft × {y_start:.1f}-{y_end:.1f}ft):")
                camera_name = input(f"  Camera name [Camera {camera_id}]: ").strip() or f"Camera {camera_id}"
                rtsp_ip = input(f"  RTSP IP address: ").strip()
                rtsp_url = f"rtsp://admin:wearewarp!@{rtsp_ip}:554/Streaming/channels/1"
                
                active = input(f"  Active? (y/n) [n]: ").strip().lower() == 'y'
                
                camera_zones[camera_id] = CameraZone(
                    camera_id=camera_id,
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                    column=col + 1,
                    row=row + 1,
                    camera_name=camera_name,
                    rtsp_url=rtsp_url,
                    active=active
                )
                
                camera_id += 1
    
    else:
        # Manual positioning
        print("\n✋ Manual Camera Positioning:")
        for camera_id in range(1, num_cameras + 1):
            print(f"\nCamera {camera_id}:")
            camera_name = input(f"  Camera name: ").strip()
            x_start = float(input(f"  X start (0-{width_ft}ft): "))
            x_end = float(input(f"  X end (0-{width_ft}ft): "))
            y_start = float(input(f"  Y start (0-{length_ft}ft): "))
            y_end = float(input(f"  Y end (0-{length_ft}ft): "))
            rtsp_ip = input(f"  RTSP IP address: ").strip()
            rtsp_url = f"rtsp://admin:wearewarp!@{rtsp_ip}:554/Streaming/channels/1"
            active = input(f"  Active? (y/n) [n]: ").strip().lower() == 'y'
            
            # Determine column and row (approximate)
            col = int(x_start // (width_ft / 3)) + 1  # Assume 3 columns
            row = int(y_start // (length_ft / 4)) + 1  # Assume 4 rows
            
            camera_zones[camera_id] = CameraZone(
                camera_id=camera_id,
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
                column=col,
                row=row,
                camera_name=camera_name,
                rtsp_url=rtsp_url,
                active=active
            )
    
    # Active cameras
    active_cameras = [cam_id for cam_id, zone in camera_zones.items() if zone.active]
    
    # Create warehouse config
    config = WarehouseConfig(
        warehouse_id=warehouse_id,
        name=warehouse_name,
        width_ft=width_ft,
        length_ft=length_ft,
        origin_position=origin_position,
        x_axis_direction=x_axis_direction,
        y_axis_direction=y_axis_direction,
        camera_zones=camera_zones,
        active_cameras=active_cameras
    )
    
    # Save configuration
    manager = WarehouseConfigManager()
    manager.save_config(config)
    
    print(f"\n✅ Warehouse configuration saved!")
    print(f"   Warehouse: {warehouse_name}")
    print(f"   Dimensions: {width_ft}ft × {length_ft}ft")
    print(f"   Cameras: {len(camera_zones)} total, {len(active_cameras)} active")
    print(f"   Active cameras: {active_cameras}")
    
    return config

def list_warehouse_configs():
    """List available warehouse configurations"""
    print("📋 AVAILABLE WAREHOUSE CONFIGURATIONS")
    print("=" * 60)
    
    config_dir = "warehouse_configs"
    if not os.path.exists(config_dir):
        print("No warehouse configurations found.")
        return
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
    
    if not config_files:
        print("No warehouse configurations found.")
        return
    
    for i, filename in enumerate(config_files, 1):
        try:
            with open(os.path.join(config_dir, filename), 'r') as f:
                config_data = json.load(f)
            
            print(f"{i}. {config_data['name']} ({config_data['warehouse_id']})")
            print(f"   File: {filename}")
            print(f"   Dimensions: {config_data['width_ft']}ft × {config_data['length_ft']}ft")
            print(f"   Cameras: {len(config_data['camera_zones'])}")
            print(f"   Active: {config_data['active_cameras']}")
            print()
        except Exception as e:
            print(f"Error reading {filename}: {e}")

def main():
    """Main warehouse setup tool"""
    print("🏭 WAREHOUSE SETUP TOOL")
    print("=" * 60)
    print("Configure warehouses for different facilities")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Create new warehouse configuration")
        print("2. List existing configurations")
        print("3. Load existing configuration")
        print("4. Exit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            create_warehouse_config_interactive()
        elif choice == "2":
            list_warehouse_configs()
        elif choice == "3":
            list_warehouse_configs()
            filename = input("\nEnter config filename: ").strip()
            try:
                manager = WarehouseConfigManager()
                config = manager.load_config(filename)
                print(f"✅ Loaded configuration: {config.name}")
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
