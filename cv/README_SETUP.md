# Warehouse Tracking System Setup Guide

## 🎯 Overview

This enhanced warehouse tracking system now includes:
1. **Dynamic Camera Detection** - Automatically detect and select cameras
2. **Interactive Calibration** - Point-and-click coordinate mapping setup
3. **Physical Coordinate Mapping** - Real-world positioning in meters
4. **Panoramic Stitching** - Multi-camera support with automatic blending

---

## 🚀 Quick Setup (Recommended)

### One-Command Setup
```bash
cd integration
python setup_system.py
```

This will guide you through:
1. Camera detection and selection
2. Coordinate calibration with physical dimensions
3. System verification

---

## 📷 Camera Setup

### Automatic Detection
```bash
python camera_detector.py
```

**Features:**
- Scans for all available cameras (0-9)
- Tests each camera for functionality
- Detects stereo cameras (like ZED)
- Interactive preview and selection
- Saves configuration to `camera_config.json`

**Controls:**
- `1-9`: Select/deselect cameras
- `a`: Auto-select best cameras
- `s`: Save selection
- `q`: Quit

### Manual Configuration
Edit `camera_config.json`:
```json
{
  "selected_cameras": [1, 2],
  "panoramic_config": {
    "stitch_mode": "panoramic",
    "overlap_percentage": 0.2
  }
}
```

---

## 📐 Coordinate Calibration

### Interactive Calibration
```bash
python calibration_tool.py
```

**Process:**
1. **Capture Image**: Press SPACE to capture calibration frame
2. **Select Corners**: Click 4 corners in order:
   - Top-Left → Top-Right → Bottom-Right → Bottom-Left
3. **Enter Dimensions**: Input warehouse size in various formats:
   - `10.5` (meters, default)
   - `10.5m` (meters)
   - `34.5ft` (feet)
   - `34'6"` (feet and inches)

**Controls:**
- `Left Click`: Select corner points
- `s`: Save calibration
- `r`: Reset points
- `q`: Quit

### Dimension Input Examples
```
Width: 10.5      # 10.5 meters
Width: 34.5ft    # 34.5 feet
Width: 34'6"     # 34 feet 6 inches
```

### Output Files
- `warehouse_calibration.json`: Complete calibration data
- Includes pixel corners and real-world mapping

---

## 🎛️ Configuration Files

### Camera Configuration (`camera_config.json`)
```json
{
  "selected_cameras": [1, 2],
  "camera_details": {
    "1": {
      "name": "Stereo Camera 1",
      "resolution": "1344x376",
      "is_stereo": true
    }
  },
  "panoramic_config": {
    "stitch_mode": "panoramic",
    "overlap_percentage": 0.2
  }
}
```

### Coordinate Calibration (`warehouse_calibration.json`)
```json
{
  "warehouse_dimensions": {
    "width_meters": 10.0,
    "length_meters": 8.0
  },
  "image_corners": [
    [150, 100], [1200, 120], 
    [1180, 600], [170, 580]
  ],
  "real_world_corners": [
    [0, 0], [10.0, 0], 
    [10.0, 8.0], [0, 8.0]
  ]
}
```

---

## 🔧 Advanced Setup

### Individual Component Setup

#### 1. Camera Detection Only
```bash
python camera_detector.py
```

#### 2. Calibration Only
```bash
python calibration_tool.py
```

#### 3. Manual Configuration
Edit config files directly and run:
```bash
python high_performance_main.py
```

### Multiple Camera Scenarios

#### Single Camera
- System automatically detects single camera
- Disables panoramic stitching
- Uses single feed for tracking

#### Dual Cameras
- Panoramic stitching enabled
- Automatic overlap detection
- Improved field of view

#### Stereo Cameras (ZED)
- Automatically detected as stereo
- Uses left view only
- Maintains stereo capabilities

---

## 📊 System Capabilities

### Detection & Tracking
- ✅ Grounding DINO object detection
- ✅ SIFT-based persistent tracking
- ✅ Real-world coordinate mapping
- ✅ MongoDB database storage

### Camera Support
- ✅ Single camera operation
- ✅ Multi-camera panoramic stitching
- ✅ Stereo camera support (ZED)
- ✅ Dynamic camera detection
- ✅ Hot-swappable camera configuration

### Coordinate Mapping
- ✅ Interactive calibration
- ✅ Multiple unit support (meters, feet, inches)
- ✅ Real-time coordinate conversion
- ✅ Pixel-to-world mapping

---

## 🎮 Running the System

### Complete System
```bash
python high_performance_main.py
```

### Runtime Controls
- `q`: Quit
- `s`: Show statistics
- `c`: Clear old database entries
- `d`: Show database statistics

### Display Information
- **Object Labels**: `ID:5 Duration:23s S:0.85`
- **Real Coordinates**: `Real: 3.45m, 2.10m`
- **Color Coding**:
  - 🟡 Yellow: New objects
  - 🔵 Cyan: Tracking objects
  - 🟢 Green: Established objects

---

## 🔍 Troubleshooting

### Camera Issues
```bash
# Test camera detection
python camera_detector.py

# Check camera IDs manually
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### Calibration Issues
- Ensure good lighting for corner detection
- Select corners in correct order
- Verify physical dimensions are accurate
- Check that all 4 corners are visible

### Database Issues
```bash
# Test MongoDB connection
python -c "from database_handler import DatabaseHandler; db = DatabaseHandler(); print(db.health_check())"
```

---

## 📁 File Structure

```
integration/
├── setup_system.py           # Complete setup tool
├── camera_detector.py        # Camera detection & selection
├── calibration_tool.py       # Interactive calibration
├── high_performance_main.py  # Main tracking system
├── detector_tracker.py       # Detection & tracking logic
├── database_handler.py       # MongoDB operations
├── config.py                 # System configuration
├── camera_config.json        # Camera selection (generated)
└── warehouse_calibration.json # Coordinate mapping (generated)
```

---

## 🎉 Success Indicators

After successful setup, you should see:
- ✅ Camera configuration saved
- ✅ Coordinate calibration completed
- ✅ Real-time object tracking with persistent IDs
- ✅ Physical coordinates displayed (meters)
- ✅ Database storage working
- ✅ Panoramic stitching (if multiple cameras)

The system is now ready for production warehouse monitoring! 🚀
