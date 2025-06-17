"""
Complete System Setup Tool
Handles both camera detection/selection and coordinate calibration
"""

import os
import sys
import json
from camera_detector import DynamicCameraDetector
from calibration_tool import WarehouseCalibrationTool

def print_banner():
    """Print system banner"""
    print("🎯 WAREHOUSE TRACKING SYSTEM SETUP")
    print("=" * 60)
    print("Complete setup for camera detection and coordinate calibration")
    print("=" * 60)

def setup_cameras():
    """Setup camera detection and selection"""
    print("\n📷 CAMERA SETUP")
    print("-" * 30)
    
    detector = DynamicCameraDetector()
    
    # Check if camera config already exists
    if os.path.exists("camera_config.json"):
        print("📁 Existing camera configuration found")
        
        choice = input("Use existing configuration? (y/n): ").lower().strip()
        if choice == 'y':
            if detector.load_camera_config():
                capabilities = detector.get_camera_capabilities()
                print(f"✅ Loaded configuration with {capabilities['total_cameras']} cameras")
                return True
    
    # Detect and select cameras
    print("🔍 Detecting available cameras...")
    cameras = detector.detect_cameras()
    
    if not cameras:
        print("❌ No cameras found!")
        return False
    
    print(f"✅ Found {len(cameras)} cameras")
    
    # Show selection interface
    selected = detector.display_camera_selection()
    
    if selected:
        detector.save_camera_config()
        capabilities = detector.get_camera_capabilities()
        
        print("\n📊 CAMERA SYSTEM SUMMARY:")
        print(f"  • Selected cameras: {selected}")
        print(f"  • Total cameras: {capabilities['total_cameras']}")
        print(f"  • Stereo cameras: {capabilities['stereo_cameras']}")
        print(f"  • Max resolution: {capabilities['max_resolution']}")
        print(f"  • Panoramic capable: {'Yes' if capabilities['panoramic_capable'] else 'No'}")
        
        return True
    else:
        print("❌ No cameras selected")
        return False

def setup_calibration():
    """Setup coordinate calibration"""
    print("\n📐 COORDINATE CALIBRATION SETUP")
    print("-" * 40)
    
    # Check if calibration already exists
    if os.path.exists("warehouse_calibration.json"):
        print("📁 Existing calibration found")
        
        try:
            with open("warehouse_calibration.json", 'r') as f:
                cal_data = json.load(f)
            
            dims = cal_data.get("warehouse_dimensions", {})
            width = dims.get("width_meters", 0)
            length = dims.get("length_meters", 0)
            
            print(f"📏 Current warehouse: {width:.2f}m x {length:.2f}m")
            
            choice = input("Use existing calibration? (y/n): ").lower().strip()
            if choice == 'y':
                print("✅ Using existing calibration")
                return True
        except Exception as e:
            print(f"⚠️ Error reading existing calibration: {e}")
    
    # Run calibration tool
    print("🎯 Starting interactive calibration...")
    print("Instructions:")
    print("1. Position camera to see the warehouse floor area")
    print("2. Click 4 corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("3. Enter physical warehouse dimensions")
    
    calibrator = WarehouseCalibrationTool()
    success = calibrator.run_calibration()
    
    if success:
        print("✅ Calibration completed successfully!")
        return True
    else:
        print("❌ Calibration failed or cancelled")
        return False

def verify_setup():
    """Verify complete setup"""
    print("\n🔍 SETUP VERIFICATION")
    print("-" * 25)
    
    issues = []
    
    # Check camera config
    if os.path.exists("camera_config.json"):
        try:
            with open("camera_config.json", 'r') as f:
                cam_config = json.load(f)
            
            cameras = cam_config.get("selected_cameras", [])
            if cameras:
                print(f"✅ Camera configuration: {len(cameras)} cameras selected")
            else:
                issues.append("No cameras selected in configuration")
        except Exception as e:
            issues.append(f"Camera config error: {e}")
    else:
        issues.append("Camera configuration file missing")
    
    # Check calibration
    if os.path.exists("warehouse_calibration.json"):
        try:
            with open("warehouse_calibration.json", 'r') as f:
                cal_config = json.load(f)
            
            dims = cal_config.get("warehouse_dimensions", {})
            corners = cal_config.get("image_corners", [])
            
            if len(corners) == 4 and dims.get("width_meters", 0) > 0:
                width = dims["width_meters"]
                length = dims["length_meters"]
                print(f"✅ Coordinate calibration: {width:.2f}m x {length:.2f}m")
            else:
                issues.append("Invalid calibration data")
        except Exception as e:
            issues.append(f"Calibration config error: {e}")
    else:
        issues.append("Coordinate calibration file missing")
    
    # Check dependencies
    try:
        import cv2
        import torch
        import pymongo
        print("✅ Required dependencies available")
    except ImportError as e:
        issues.append(f"Missing dependency: {e}")
    
    if issues:
        print("\n⚠️ SETUP ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("\n🎉 SETUP COMPLETE - SYSTEM READY!")
        return True

def show_next_steps():
    """Show next steps after setup"""
    print("\n🚀 NEXT STEPS")
    print("-" * 15)
    print("1. Run the tracking system:")
    print("   python high_performance_main.py")
    print()
    print("2. Or run individual components:")
    print("   python camera_detector.py     # Camera setup only")
    print("   python calibration_tool.py    # Calibration only")
    print()
    print("3. Configuration files created:")
    print("   • camera_config.json          # Camera selection")
    print("   • warehouse_calibration.json  # Coordinate mapping")
    print()
    print("4. Features available:")
    print("   • Real-time box detection and tracking")
    print("   • Persistent object IDs with SIFT matching")
    print("   • Real-world coordinate mapping")
    print("   • MongoDB database storage")
    print("   • Panoramic camera stitching")

def main():
    """Main setup function"""
    print_banner()
    
    # Setup cameras
    camera_success = setup_cameras()
    
    if not camera_success:
        print("\n❌ Camera setup failed - cannot continue")
        return
    
    # Setup calibration
    calibration_success = setup_calibration()
    
    if not calibration_success:
        print("\n⚠️ Calibration setup failed")
        choice = input("Continue without calibration? (y/n): ").lower().strip()
        if choice != 'y':
            print("❌ Setup cancelled")
            return
    
    # Verify setup
    if verify_setup():
        show_next_steps()
    else:
        print("\n❌ Setup verification failed")
        print("Please resolve the issues and run setup again")

if __name__ == "__main__":
    main()
