#!/usr/bin/env python3
"""
Comprehensive Coordinate System and Grid Confirmation
Tests and validates the warehouse coordinate system implementation
"""

import cv2
import numpy as np
import json
from detector_tracker import CoordinateMapper, DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector

def test_coordinate_mapping():
    """Test coordinate mapping with known test points"""
    print("\n🎯 COORDINATE MAPPING TEST")
    print("=" * 50)
    
    mapper = CoordinateMapper(camera_id=8)
    mapper.load_calibration()
    
    # Test points based on our calibration
    test_points = [
        (705, 691, "Object 1", (163, 7)),     # Our main test object
        (1761, 1108, "Object 2", (150.6, 13.6)),  # Second object
        (112, 254, "Top-left corner", None),   # Calibration corners
        (3766, 281, "Top-right corner", None),
        (3754, 2030, "Bottom-right corner", None),
        (147, 1949, "Bottom-left corner", None)
    ]
    
    print("Testing pixel → local → global coordinate transformation:")
    print()
    
    for px, py, name, expected in test_points:
        # Get coordinates
        global_x, global_y = mapper.pixel_to_real(px, py)
        
        # Calculate local coordinates manually
        pixel_point = np.array([[[px, py]]], dtype=np.float32)
        real_point = cv2.perspectiveTransform(pixel_point, mapper.homography_matrix)
        local_x = float(real_point[0][0][0])
        local_y = float(real_point[0][0][1])
        
        print(f"{name}:")
        print(f"  Pixel: ({px}, {py})")
        print(f"  Local: ({local_x:.1f}, {local_y:.1f})")
        print(f"  Global: ({global_x:.1f}, {global_y:.1f})")
        
        if expected:
            exp_x, exp_y = expected
            diff_x = abs(global_x - exp_x)
            diff_y = abs(global_y - exp_y)
            status = "✅ PASS" if diff_x <= 1 and diff_y <= 1 else "❌ FAIL"
            print(f"  Expected: ({exp_x}, {exp_y}) → {status}")
        print()

def confirm_coordinate_system():
    """Confirm the coordinate system design"""
    print("\n🗺️  COORDINATE SYSTEM CONFIRMATION")
    print("=" * 50)
    
    print("Warehouse Layout:")
    print("  Total Size: 180ft × 90ft")
    print("  Origin (0,0): TOP-RIGHT corner")
    print("  X-axis: RIGHT → LEFT (0ft to 180ft)")
    print("  Y-axis: TOP → BOTTOM (0ft to 90ft)")
    print()
    
    print("Column Layout:")
    print("  Column 1 (Rightmost): X = 0-60ft from right edge")
    print("  Column 2 (Middle):    X = 60-120ft from right edge")
    print("  Column 3 (Leftmost):  X = 120-180ft from right edge")
    print()
    
    print("Camera 8 Configuration:")
    zone = Config.CAMERA_COVERAGE_ZONES[8]
    print(f"  Zone: {zone['x_start']}-{zone['x_end']}ft (X), {zone['y_start']}-{zone['y_end']}ft (Y)")
    print(f"  Center: ({zone['center_x']:.1f}ft, {zone['center_y']:.1f}ft)")
    print(f"  Column: {zone['column']}")
    print()
    
    print("Expected Object Coordinates:")
    print("  Object 1: (163, 7) = 163ft from right, 7ft from top")
    print("  Object 2: (150.6, 13.6) = 150.6ft from right, 13.6ft from top")
    print()

def test_fisheye_correction():
    """Test fisheye correction functionality"""
    print("\n🔍 FISHEYE CORRECTION TEST")
    print("=" * 50)
    
    corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Connect to camera and test one frame
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    
    if cap.isOpened():
        ret, raw_frame = cap.read()
        if ret:
            corrected_frame = corrector.correct(raw_frame)
            print(f"✅ Frame correction successful")
            print(f"  Raw frame size: {raw_frame.shape}")
            print(f"  Corrected frame size: {corrected_frame.shape}")
            
            # Show difference
            raw_center = raw_frame[raw_frame.shape[0]//2, raw_frame.shape[1]//2]
            corrected_center = corrected_frame[corrected_frame.shape[0]//2, corrected_frame.shape[1]//2]
            print(f"  Center pixel change: {raw_center} → {corrected_center}")
        else:
            print("❌ Failed to read test frame")
        cap.release()
    else:
        print("❌ Failed to connect to Camera 8")

def test_calibration_files():
    """Test all calibration files"""
    print("\n📁 CALIBRATION FILES TEST")
    print("=" * 50)
    
    # Test Camera 8 specific file
    try:
        with open("warehouse_calibration_camera_8.json", 'r') as f:
            cal_data = json.load(f)
        
        print("Camera 8 Calibration:")
        print(f"  File: warehouse_calibration_camera_8.json")
        print(f"  Image corners: {len(cal_data['image_corners'])} points")
        print(f"  Real corners: {cal_data['real_world_corners']}")
        print(f"  Area: {cal_data['warehouse_area']['width_ft']}ft × {cal_data['warehouse_area']['height_ft']}ft")
        print(f"  Fisheye corrected: {cal_data.get('fisheye_corrected', 'Not specified')}")
        print("  ✅ Valid")
    except Exception as e:
        print(f"  ❌ Error loading Camera 8 calibration: {e}")
    
    print()

def test_grid_system():
    """Test the coordinate grid system"""
    print("\n🔲 GRID SYSTEM TEST")
    print("=" * 50)
    
    # Test grid calculations
    mapper = CoordinateMapper(camera_id=8)
    mapper.load_calibration()
    
    print("Grid Configuration:")
    max_dim = max(mapper.floor_width_ft, mapper.floor_length_ft)
    if max_dim <= 10.0:
        grid_spacing = 2.0
    elif max_dim <= 30.0:
        grid_spacing = 5.0
    else:
        grid_spacing = 10.0
    
    print(f"  Camera area: {mapper.floor_width_ft}ft × {mapper.floor_length_ft}ft")
    print(f"  Grid spacing: {grid_spacing}ft")
    
    # Calculate number of grid lines
    v_lines = int(mapper.floor_width_ft / grid_spacing) - 1
    h_lines = int(mapper.floor_length_ft / grid_spacing) - 1
    print(f"  Vertical lines: {v_lines}")
    print(f"  Horizontal lines: {h_lines}")
    print("  ✅ Grid system configured")

def run_live_confirmation():
    """Run live confirmation with fisheye correction"""
    print("\n🎥 LIVE SYSTEM CONFIRMATION")
    print("=" * 50)
    print("Press 'q' to quit, 's' to save a test frame")
    
    # Initialize everything
    tracker = DetectorTracker()
    tracker.set_camera_id(8)
    corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    
    # Connect to camera
    camera_url = Config.RTSP_CAMERA_URLS[8]
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print("❌ Failed to connect to Camera 8")
        return
    
    frame_count = 0
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply fisheye correction
        corrected_frame = corrector.correct(raw_frame)
        
        # Process for detection
        tracked_objects, _ = tracker.process_frame(corrected_frame)
        
        # Draw everything
        annotated_frame = tracker.draw_tracked_objects(corrected_frame, tracked_objects)
        annotated_frame = tracker.draw_calibrated_zone_overlay(annotated_frame)
        
        # Add comprehensive status
        info_y = 30
        cv2.putText(annotated_frame, f"COORDINATE SYSTEM CONFIRMATION - Frame {frame_count}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(annotated_frame, "FISHEYE CORRECTED + NEW COORDINATE SYSTEM", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(annotated_frame, f"Objects: {len(tracked_objects)} | Origin: Top-Right (0,0)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display coordinates for each object
        if tracked_objects:
            for i, obj in enumerate(tracked_objects):
                real_center = obj.get('real_center')
                if real_center and real_center[0] is not None:
                    x, y = real_center
                    print(f"📍 Object {i+1}: ({x:.1f}ft from right, {y:.1f}ft from top)")
        
        # Resize and display
        height, width = annotated_frame.shape[:2]
        display_width = 1280
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        
        cv2.imshow('Coordinate System Confirmation', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'coordinate_test_frame_{frame_count}.jpg', annotated_frame)
            print(f"✅ Saved test frame: coordinate_test_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()

def main():
    print("🔧 COMPREHENSIVE COORDINATE SYSTEM CONFIRMATION")
    print("=" * 60)
    
    # Run all tests
    confirm_coordinate_system()
    test_calibration_files()
    test_coordinate_mapping()
    test_grid_system()
    test_fisheye_correction()
    
    print("\n" + "=" * 60)
    print("All tests completed! Starting live confirmation...")
    print("=" * 60)
    
    # Run live test
    run_live_confirmation()
    
    print("\n✅ COORDINATE SYSTEM CONFIRMATION COMPLETE")

if __name__ == "__main__":
    main() 