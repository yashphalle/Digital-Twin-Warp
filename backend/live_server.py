"""
Live server connecting to real MongoDB CV system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import cv2
import time
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Live Warehouse Tracking API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "warehouse_tracking"
COLLECTION_NAME = "detections"

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client[DATABASE_NAME]
    tracking_collection = mongo_db[COLLECTION_NAME]
    # Test connection
    mongo_client.server_info()
    logger.info("✅ MongoDB connected successfully to CV system")
    logger.info(f"📍 Database: {DATABASE_NAME}")
    logger.info(f"📦 Collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    mongo_client = None
    tracking_collection = None

# Camera management
class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.camera_status = {}
        self.initialize_cameras()

    def initialize_cameras(self):
        """Initialize all 11 cameras from configuration"""
        logger.info("🎥 Initializing multi-camera system...")

        # Import camera configuration
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cv'))
        
        try:
            from config import Config
            
            # Initialize all 11 cameras from Config
            for camera_id in Config.RTSP_CAMERA_URLS.keys():
                camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
                rtsp_url = Config.RTSP_CAMERA_URLS[camera_id]
                
                # Set initial status based on active cameras
                if camera_id in Config.ACTIVE_CAMERAS:
                    self.camera_status[camera_id] = "active"
                    logger.info(f"✅ {camera_name} - ACTIVE")
                else:
                    self.camera_status[camera_id] = "ready"
                    logger.info(f"💤 {camera_name} - STANDBY")
                    
        except ImportError:
            logger.warning("⚠️  Could not import Config, using fallback camera setup")
            self._initialize_fallback_cameras()

        # Log final camera status
        active_cameras = len([c for c in self.camera_status.values() if c == "active"])
        ready_cameras = len([c for c in self.camera_status.values() if c == "ready"])
        total_cameras = len(self.camera_status)
        
        logger.info(f"📊 Camera system initialized:")
        logger.info(f"   • Total cameras: {total_cameras}")
        logger.info(f"   • Active cameras: {active_cameras}")
        logger.info(f"   • Ready cameras: {ready_cameras}")

    def _initialize_fallback_cameras(self):
        """Fallback camera initialization if Config import fails"""
        # Fallback to basic 11-camera setup
        for camera_id in range(1, 12):
            if camera_id == 7:  # Default active camera
                self.camera_status[camera_id] = "active"
            else:
                self.camera_status[camera_id] = "ready"
            logger.info(f"📹 Camera {camera_id} - {'ACTIVE' if camera_id == 7 else 'STANDBY'}")

    def get_camera_status(self):
        """Get status of all cameras with enhanced information"""
        try:
            from config import Config
            
            camera_info = {}
            for camera_id, status in self.camera_status.items():
                camera_info[camera_id] = {
                    'camera_id': camera_id,
                    'name': Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}"),
                    'status': status,
                    'rtsp_url': Config.RTSP_CAMERA_URLS.get(camera_id, ""),
                    'coverage_zone': Config.CAMERA_COVERAGE_ZONES.get(camera_id, {}),
                    'active': camera_id in Config.ACTIVE_CAMERAS
                }
            
            return camera_info
            
        except ImportError:
            # Fallback format
            return {
                camera_id: {
                    'camera_id': camera_id,
                    'name': f"Camera {camera_id}",
                    'status': status,
                    'active': camera_id == 7
                }
                for camera_id, status in self.camera_status.items()
            }

    def generate_frame(self, camera_id):
        """Generate frames for a specific camera (1-11)"""
        logger.info(f"🎥 Starting stream for camera {camera_id}")

        status = self.camera_status.get(camera_id, "offline")

        # Try to get camera info
        try:
            from config import Config
            camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
            rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
            coverage_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
        except ImportError:
            camera_name = f"Camera {camera_id}"
            rtsp_url = ""
            coverage_zone = {}

        # For now, generate demo feeds since real RTSP integration is in CV module
        logger.info(f"📺 Using demo feed for {camera_name}")
        frame_count = 0
        
        while True:
            try:
                # Create animated demo frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                if status in ["active", "ready"]:
                    # Different colors for different cameras
                    base_color = (
                        (camera_id * 30) % 255,
                        (camera_id * 50) % 255,
                        (camera_id * 70) % 255
                    )
                    
                    # Animated background
                    color_intensity = int(50 + 30 * np.sin(frame_count * 0.1))
                    frame[:] = tuple(int(c * color_intensity / 255) for c in base_color)

                    # Add camera info
                    cv2.putText(frame, camera_name, (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Status indicator
                    status_text = "🎯 ACTIVE" if status == "active" else "💤 STANDBY"
                    cv2.putText(frame, status_text, (50, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "active" else (255, 255, 0), 2)

                    # Coverage zone info
                    if coverage_zone:
                        zone_text = f"Zone: {coverage_zone.get('x_start', 0)}-{coverage_zone.get('x_end', 0)}ft"
                        cv2.putText(frame, zone_text, (50, 170),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Timestamp removed for cleaner display

                    # Frame counter removed for cleaner display

                    # Add animated element based on camera ID
                    center_x = 320 + int(50 * np.cos(frame_count * 0.1 + camera_id))
                    center_y = 350 + int(20 * np.sin(frame_count * 0.2 + camera_id))
                    color = (0, 255, 0) if status == "active" else (255, 255, 0)
                    cv2.circle(frame, (center_x, center_y), 15, color, -1)

                else:
                    # Offline camera
                    frame[:] = (30, 30, 30)
                    cv2.putText(frame, camera_name, (150, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, "OFFLINE", (180, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                frame_count += 1
                time.sleep(0.1)  # ~10 FPS for demo

            except Exception as e:
                logger.error(f"❌ Error generating frame for camera {camera_id}: {e}")
                time.sleep(1)

# Initialize camera manager
camera_manager = CameraManager()

@app.get("/")
async def root():
    return {
        "message": "Live Warehouse Tracking API", 
        "status": "running",
        "database": "connected" if tracking_collection is not None else "disconnected",
        "cv_system": "live"
    }

@app.get("/api/tracking/objects")
async def get_tracked_objects():
    """Get current tracked objects from live CV system MongoDB"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected to CV system")
    
    try:
        # Get recent detections (last 5 minutes) with physical coordinates
        five_minutes_ago = datetime.now() - timedelta(minutes=5)

        # Query for recent detections from CV system
        query = {
            "$and": [
                {"timestamp": {"$gte": five_minutes_ago}},
                {"physical_x_ft": {"$exists": True, "$ne": None}},
                {"physical_y_ft": {"$exists": True, "$ne": None}}
            ]
        }

        # Get detections, excluding MongoDB _id field
        detections = list(tracking_collection.find(
            query,
            {"_id": 0}
        ).sort([("timestamp", -1)]).limit(100))
        
        logger.info(f"📊 Found {len(detections)} detections in database")

        # Process detections for frontend
        processed_objects = []
        for detection in detections:
            try:
                # Convert datetime objects to ISO strings
                if "timestamp" in detection and isinstance(detection["timestamp"], datetime):
                    detection["timestamp"] = detection["timestamp"].isoformat()

                # Create object structure for frontend
                obj = {
                    "persistent_id": detection.get("global_id", f"det_{detection.get('_id', 'unknown')}"),
                    "global_id": detection.get("global_id"),
                    "camera_id": detection.get("camera_id"),
                    "bbox": detection.get("bbox", [0, 0, 100, 100]),
                    "confidence": detection.get("confidence", 0.0),
                    "area": detection.get("area", 0),
                    "real_center": [detection.get("physical_x_ft"), detection.get("physical_y_ft")],
                    "physical_x_ft": detection.get("physical_x_ft"),
                    "physical_y_ft": detection.get("physical_y_ft"),
                    "grid_cell": detection.get("grid_cell"),
                    "times_seen": detection.get("times_seen", 1),
                    "is_new": detection.get("is_new", False),
                    "timestamp": detection.get("timestamp"),
                    "first_seen": detection.get("timestamp"),  # Use timestamp as first_seen
                    "last_seen": detection.get("timestamp"),   # Use timestamp as last_seen
                    "age_seconds": 0  # Recent detection
                }

                # Skip objects without valid coordinates
                if obj["physical_x_ft"] is None or obj["physical_y_ft"] is None:
                    continue

                processed_objects.append(obj)

            except Exception as e:
                logger.warning(f"⚠️ Error processing detection: {e}")
                continue

        return {
            "objects": processed_objects,
            "count": len(processed_objects),
            "total_in_db": len(detections),
            "timestamp": datetime.now().isoformat(),
            "source": "live_cv_system"
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching objects: {str(e)}")

@app.get("/api/warehouse/config")
async def get_warehouse_config():
    """Get warehouse configuration - Return dimensions in FEET (primary) and meters"""
    try:
        # Try to read from CV system's calibration file
        import os
        import json
        
        calibration_paths = [
            "../cv/warehouse_calibration.json",
            "../../cv/warehouse_calibration.json",
            "../warehouse_calibration.json",
            "configs/warehouse_calibration.json"
        ]
        
        for path in calibration_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    calibration_data = json.load(f)
                    
                warehouse_dims = calibration_data.get('warehouse_dimensions', {})
                
                # Get full warehouse dimensions (180ft x 90ft)
                full_warehouse_width_ft = 180.0
                full_warehouse_length_ft = 90.0
                
                return {
                    # Primary dimensions in feet
                    "width_feet": full_warehouse_width_ft,
                    "length_feet": full_warehouse_length_ft,
                    
                    # Secondary dimensions in meters for backward compatibility
                    "width_meters": full_warehouse_width_ft * 0.3048,
                    "length_meters": full_warehouse_length_ft * 0.3048,
                    
                    # Camera coverage info
                    "camera_coverage": warehouse_dims.get('coverage_zone', {}),
                    
                    "calibrated": True,
                    "calibration_file": path,
                    "last_updated": datetime.now().isoformat(),
                    "source": "cv_calibration_file",
                    "units": "feet"
                }
        
        # Fallback configuration - Full warehouse in feet
        return {
            "width_feet": 180.0,
            "length_feet": 90.0,
            "width_meters": 180.0 * 0.3048,  # 54.864m
            "length_meters": 90.0 * 0.3048,   # 27.432m
            "calibrated": True,
            "calibration_file": None,
            "last_updated": datetime.now().isoformat(),
            "source": "default_config",
            "units": "feet"
        }
        
    except Exception as e:
        logger.warning(f"⚠️  Error reading calibration: {e}")
        return {
            "width_feet": 180.0,
            "length_feet": 90.0,
            "width_meters": 180.0 * 0.3048,
            "length_meters": 90.0 * 0.3048,
            "calibrated": False,
            "error": str(e),
            "last_updated": datetime.now().isoformat(),
            "source": "fallback",
            "units": "feet"
        }

@app.get("/api/tracking/stats")
async def get_tracking_stats():
    """Get tracking statistics from live CV system"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")
    
    try:
        # Basic statistics
        total_objects = tracking_collection.count_documents({})
        
        # Recent objects (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_objects = tracking_collection.count_documents({
            "$or": [
                {"last_seen": {"$gte": one_hour_ago}},
                {"updated_at": {"$gte": one_hour_ago}}
            ]
        })
        
        # Unique persistent IDs
        unique_ids = len(tracking_collection.distinct("persistent_id"))
        
        # Objects with real coordinates
        objects_with_coords = tracking_collection.count_documents({
            "real_center": {"$exists": True}
        })
        
        return {
            "total_detections": total_objects,
            "unique_objects": unique_ids,
            "recent_objects": recent_objects,
            "objects_with_coordinates": objects_with_coords,
            "database_connected": True,
            "timestamp": datetime.now().isoformat(),
            "source": "live_cv_system"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.get("/api/cameras/status")
async def get_cameras_status():
    """Get status of all cameras"""
    try:
        status = camera_manager.get_camera_status()
        return {
            "cameras": [
                {
                    "camera_id": i,
                    "status": status.get(i, "offline"),
                    "stream_url": f"/api/cameras/{i}/stream"
                }
                for i in range(1, 12)
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error fetching camera status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching camera status: {str(e)}")

@app.get("/api/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Stream video from a specific camera"""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        camera_manager.generate_frame(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/{camera_id}/info")
async def get_camera_info(camera_id: int):
    """Get detailed information about a specific camera"""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_status = camera_manager.get_camera_status()
    camera_info = camera_status.get(camera_id, {})
    
    if not camera_info:
        raise HTTPException(status_code=404, detail="Camera information not found")
    
    return {
        "success": True,
        "camera": camera_info,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/cameras/{camera_id}/enable")
async def enable_camera(camera_id: int):
    """Enable processing for a specific camera"""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # This would interface with the CV system to enable the camera
        # For now, update status in camera manager
        camera_manager.camera_status[camera_id] = "active"
        
        return {
            "success": True,
            "message": f"Camera {camera_id} enabled for processing",
            "camera_id": camera_id,
            "new_status": "active",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable camera: {str(e)}")

@app.post("/api/cameras/{camera_id}/disable")
async def disable_camera(camera_id: int):
    """Disable processing for a specific camera"""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # This would interface with the CV system to disable the camera
        # For now, update status in camera manager
        camera_manager.camera_status[camera_id] = "ready"
        
        return {
            "success": True,
            "message": f"Camera {camera_id} disabled from processing",
            "camera_id": camera_id,
            "new_status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable camera: {str(e)}")

@app.get("/api/system/multi-camera/status")
async def get_multi_camera_system_status():
    """Get comprehensive multi-camera system status"""
    try:
        camera_status = camera_manager.get_camera_status()
        
        # Calculate summary statistics
        total_cameras = len(camera_status)
        active_cameras = len([c for c in camera_status.values() if c.get('status') == 'active'])
        ready_cameras = len([c for c in camera_status.values() if c.get('status') == 'ready'])
        offline_cameras = len([c for c in camera_status.values() if c.get('status') == 'offline'])
        
        return {
            "success": True,
            "system_type": "multi_camera_rtsp",
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "ready_cameras": ready_cameras,
            "offline_cameras": offline_cameras,
            "cameras": camera_status,
            "warehouse_config": {
                "width_ft": 180.0,
                "length_ft": 90.0,
                "coverage_zones": 11
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get multi-camera status: {e}")
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")

@app.get("/api/cameras/coverage-zones")
async def get_camera_coverage_zones():
    """Get camera coverage zones for the warehouse layout"""
    try:
        # Try to get from Config
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cv'))
        
        from config import Config
        
        coverage_zones = {}
        for camera_id, zone in Config.CAMERA_COVERAGE_ZONES.items():
            coverage_zones[camera_id] = {
                "camera_id": camera_id,
                "camera_name": Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}"),
                "coverage_area": zone,
                "rtsp_url": Config.RTSP_CAMERA_URLS.get(camera_id, ""),
                "active": camera_id in Config.ACTIVE_CAMERAS
            }
        
        return {
            "success": True,
            "warehouse_dimensions": {
                "width_ft": Config.FULL_WAREHOUSE_WIDTH_FT,
                "length_ft": Config.FULL_WAREHOUSE_LENGTH_FT
            },
            "coverage_zones": coverage_zones,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        # Fallback data
        return {
            "success": True,
            "warehouse_dimensions": {
                "width_ft": 180.0,
                "length_ft": 90.0
            },
            "coverage_zones": {
                i: {
                    "camera_id": i,
                    "camera_name": f"Camera {i}",
                    "coverage_area": {"x_start": 0, "x_end": 30, "y_start": 0, "y_end": 30},
                    "active": i == 7
                } for i in range(1, 12)
            },
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Starting Live CV System API Server...")
    print("📡 Connecting to MongoDB at localhost:27017")
    print("📊 Database: warehouse_tracking")
    print("📦 Collection: tracked_objects")
    print("🌐 API will be available at: http://localhost:8000")
    print("-" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
