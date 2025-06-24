# Digital Twin Warehouse System Setup Guide

## System Overview
The Digital Twin Warehouse system uses RTSP cameras to track objects in a 180ft x 90ft warehouse environment. This system currently processes **Camera 8** for object detection and tracking.

## Prerequisites

### Hardware Requirements
- RTSP-capable cameras (Lorex cameras configured)
- Network connectivity to cameras (IPs 192.168.0.71-82)
- Computer with sufficient processing power for real-time video processing

### Software Requirements
- Python 3.8+ with OpenCV
- Node.js 16+ for frontend
- MongoDB for data storage
- Required Python packages (see cv/ folder requirements)

## Quick Start

### 1. Test Camera 8 Connection
```bash
cd cv
python test_camera_8_system.py
```
Choose option "1" for quick connectivity test.

### 2. Start the Backend Server
```bash
cd backend
# Activate virtual environment if using one
# source digitaltwin_venv/bin/activate  # Linux/Mac
# digitaltwin_venv\Scripts\activate     # Windows

python live_server.py
```
Server will start on `http://localhost:8000`

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend will start on `http://localhost:5173`

### 4. Start Camera Processing
```bash
cd cv
python multi_camera_tracking_system.py
```

## Detailed Setup Instructions

### Backend Setup

1. **Install Python Dependencies**
   ```bash
   cd backend
   pip install flask flask-cors pymongo opencv-python numpy
   ```

2. **MongoDB Setup**
   - Install MongoDB locally or use MongoDB Atlas
   - Default connection: `mongodb://localhost:27017/`
   - Database: `warehouse_tracking`
   - Collection: `detected_objects`

3. **Start Backend Server**
   ```bash
   cd backend
   python live_server.py
   ```

### Frontend Setup

1. **Install Node.js Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Development Mode**
   ```bash
   npm run dev
   ```

3. **Production Build**
   ```bash
   npm run build
   npm run preview
   ```

### Computer Vision Setup

1. **Install CV Dependencies**
   ```bash
   cd cv
   pip install opencv-python numpy pymongo datetime
   ```

2. **Camera Configuration**
   - Camera 8 is active by default
   - RTSP URL: `rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1`
   - Coverage area: 60-96ft (X) × 25-55ft (Y)

3. **Test Camera 8**
   ```bash
   python test_camera_8_system.py
   ```

4. **Start Object Tracking**
   ```bash
   python multi_camera_tracking_system.py
   ```

## System Components

### 1. Camera Configuration (`cv/config.py`)
- **Active Camera**: Camera 8 only
- **RTSP Settings**: All 11 cameras configured, only Camera 8 processing
- **Warehouse**: 180ft × 90ft mapped area
- **Processing**: 4K→1080p scaling with fisheye correction

### 2. Camera Manager (`cv/rtsp_camera_manager.py`)
- Manages all 11 camera pipelines
- Only starts active cameras (Camera 8)
- Handles fisheye correction and frame scaling
- Provides individual camera control

### 3. Tracking System (`cv/multi_camera_tracking_system.py`)
- Main processing loop for active cameras
- Object detection using Grounding DINO
- SIFT-based object tracking
- Real-world coordinate mapping
- Database storage with camera attribution

### 4. Backend API (`backend/live_server.py`)
- **Camera Streams**: `/api/cameras/8/stream`
- **Camera Info**: `/api/cameras/8/info`
- **Camera Control**: `/api/cameras/8/enable`, `/api/cameras/8/disable`
- **System Status**: `/api/system/multi-camera/status`
- **Coverage Zones**: `/api/cameras/coverage-zones`

### 5. Frontend (`frontend/src/components/MultiCameraWarehouseView.tsx`)
- 180ft × 90ft warehouse visualization
- Interactive camera zone display
- Real-time object tracking overlay
- Camera status indicators

## API Endpoints

### Camera Endpoints
- `GET /api/cameras/8/stream` - Camera 8 video stream
- `GET /api/cameras/8/info` - Camera 8 information
- `POST /api/cameras/8/enable` - Enable Camera 8
- `POST /api/cameras/8/disable` - Disable Camera 8

### System Endpoints
- `GET /api/system/multi-camera/status` - System status
- `GET /api/cameras/coverage-zones` - All camera coverage areas
- `GET /api/tracking/objects` - Current tracked objects

## Camera Layout

```
Warehouse Layout (180ft × 90ft):
Row 1 (Front):   [C1] [C2] [C3] [C4]        (4 cameras)
Row 2 (Middle):  [C5] [C6] [C7]             (3 cameras)
Row 3 (Back):    [C8*][C9] [C10] [C11]      (4 cameras)

* = Currently Active (Camera 8 - Back Left)
```

### Camera 8 Specifications
- **Location**: Back Left (Row 3, Position 1)
- **IP**: 192.168.0.79
- **RTSP URL**: `rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1`
- **Coverage**: 0-45ft (X) × 50-80ft (Y)
- **Lens**: 2.8mm fisheye (corrected)
- **Resolution**: 4K capture → 1080p processing

## Troubleshooting

### Camera Connection Issues
1. **Test RTSP URL directly**:
   ```bash
   python test_camera_8_system.py
   ```

2. **Check network connectivity**:
   ```bash
   ping 192.168.0.79
   ```

3. **Verify camera credentials**:
   - Username: `admin`
   - Password: `wearewarp!`

### Backend Issues
1. **MongoDB Connection**:
   - Ensure MongoDB is running
   - Check connection string in database_handler.py

2. **Port Conflicts**:
   - Backend uses port 8000
   - Change in live_server.py if needed

### Frontend Issues
1. **API Connection**:
   - Ensure backend is running on port 8000
   - Check CORS settings

2. **Build Issues**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

## Performance Optimization

### Processing Settings
- **Frame Rate**: 20 FPS target
- **Resolution**: 1080p processing (scaled from 4K)
- **Detection**: Optimized for warehouse objects
- **Tracking**: SIFT-based for accuracy

### System Resources
- **CPU**: Multi-threaded processing
- **Memory**: ~2GB for single camera processing
- **Network**: Stable connection to camera required

## Scaling to Multiple Cameras

To enable additional cameras:

1. **Update Configuration**:
   ```python
   # In cv/config.py
   ACTIVE_CAMERAS = [6, 7, 8, 9]  # Add more cameras
   ```

2. **Update Calibration**:
   - Add calibration data for each camera
   - Update warehouse_calibration.json

3. **Test Each Camera**:
   ```bash
   python test_camera_8_system.py  # Modify for each camera
   ```

## System Status Monitoring

### Health Checks
- Camera connectivity status
- Frame processing rate
- Object detection performance
- Database connection status

### Logs
- Backend: Flask server logs
- CV System: Multi-camera tracking logs
- Frontend: Browser developer console

## Development Notes

### File Structure
```
Digital-Twin-Warp/
├── backend/          # Flask API server
├── cv/              # Computer vision processing
│   ├── config.py    # System configuration
│   ├── multi_camera_tracking_system.py  # Main processor
│   ├── rtsp_camera_manager.py          # Camera management
│   └── test_camera_8_system.py         # Testing suite
└── frontend/        # React web interface
```

### Key Features
- **Scalable Architecture**: Ready for all 11 cameras
- **Real-time Processing**: Live object detection and tracking
- **Web Interface**: Modern React-based dashboard
- **API-driven**: RESTful API for all operations
- **Database Storage**: Persistent object tracking history

## Next Steps

1. **Test Current Setup**: Verify Camera 8 functionality
2. **Calibrate Camera 8**: Fine-tune coordinate mapping
3. **Add More Cameras**: Scale to additional cameras as needed
4. **Optimize Performance**: Tune for specific warehouse requirements
5. **Deploy Production**: Set up production environment

For technical support or questions, refer to the individual component documentation in each folder. 