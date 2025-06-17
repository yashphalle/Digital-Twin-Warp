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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "warehouse_tracking"
COLLECTION_NAME = "tracked_objects"

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
        """Initialize available cameras - try real cameras first, fallback to demo"""
        logger.info("🎥 Initializing camera system...")

        # Try to access real cameras with different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for camera_id in range(4):
            camera_found = False

            for backend in backends:
                try:
                    logger.info(f"🔍 Trying camera {camera_id} with backend {backend}")
                    cap = cv2.VideoCapture(camera_id, backend)

                    if cap.isOpened():
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce conflicts
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer

                        # Test if camera actually works
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.cameras[camera_id + 1] = cap  # Use 1-based indexing
                            self.camera_status[camera_id + 1] = "active"
                            logger.info(f"✅ Real Camera {camera_id + 1} initialized successfully with backend {backend}")
                            camera_found = True
                            break
                        else:
                            cap.release()

                except Exception as e:
                    logger.warning(f"⚠️  Camera {camera_id} backend {backend} failed: {e}")
                    continue

            if not camera_found:
                self.camera_status[camera_id + 1] = "offline"
                logger.info(f"❌ Camera {camera_id + 1} not available")

        # Log final camera status
        active_cameras = len([c for c in self.camera_status.values() if c == "active"])
        logger.info(f"📊 Camera initialization complete: {active_cameras}/4 cameras active")

        # If no real cameras, set demo status
        if active_cameras == 0:
            logger.info("📹 No real cameras available, using demo feeds")
            for camera_id in range(1, 5):
                self.camera_status[camera_id] = "demo" if camera_id <= 2 else "offline"

    def get_camera_status(self):
        """Get status of all cameras"""
        return self.camera_status

    def generate_frame(self, camera_id):
        """Generate frames for a specific camera"""
        logger.info(f"🎥 Starting stream for camera {camera_id}")

        # Check if we have a real camera
        if camera_id in self.cameras and self.camera_status[camera_id] == "active":
            logger.info(f"📹 Using REAL camera {camera_id}")
            cap = self.cameras[camera_id]
            frame_count = 0

            while True:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning(f"⚠️  Camera {camera_id} read failed, switching to demo")
                        break

                    # Resize frame
                    frame = cv2.resize(frame, (640, 480))

                    # Add overlay information
                    cv2.putText(frame, f"Camera {camera_id} - LIVE", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 460),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.putText(frame, f"Frame: {frame_count}", (450, 460),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # Add border
                    cv2.rectangle(frame, (2, 2), (638, 478), (0, 255, 0), 2)

                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                    frame_count += 1
                    time.sleep(0.067)  # ~15 FPS

                except Exception as e:
                    logger.error(f"❌ Real camera {camera_id} error: {e}")
                    break

        # Fallback to demo/animated feed
        logger.info(f"📺 Using demo feed for camera {camera_id}")
        frame_count = 0
        while True:
            try:
                # Create animated demo frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                status = self.camera_status.get(camera_id, "offline")

                if status in ["active", "demo"]:
                    # Animated background
                    color_intensity = int(50 + 30 * np.sin(frame_count * 0.1))
                    frame[:] = (color_intensity, color_intensity//2, color_intensity//3)

                    # Add camera info
                    cv2.putText(frame, f"Camera {camera_id}", (180, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                    cv2.putText(frame, "DEMO FEED", (200, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Add animated timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (220, 250),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (220, 300),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Add animated circle
                    center_x = 320 + int(50 * np.cos(frame_count * 0.2))
                    center_y = 350 + int(20 * np.sin(frame_count * 0.3))
                    cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)

                else:
                    # Offline camera
                    frame[:] = (30, 30, 30)
                    cv2.putText(frame, f"Camera {camera_id}", (180, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.putText(frame, "OFFLINE", (220, 250),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add border
                cv2.rectangle(frame, (5, 5), (635, 475), (100, 100, 100), 2)

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                frame_count += 1
                time.sleep(0.1)  # 10 FPS for demo

            except Exception as e:
                logger.error(f"❌ Error generating demo frame for camera {camera_id}: {e}")
                time.sleep(1)

# Initialize camera manager
camera_manager = CameraManager()

@app.get("/")
async def root():
    return {
        "message": "Live Warehouse Tracking API", 
        "status": "running",
        "database": "connected" if tracking_collection else "disconnected",
        "cv_system": "live"
    }

@app.get("/api/tracking/objects")
async def get_tracked_objects():
    """Get current tracked objects from live CV system MongoDB"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected to CV system")
    
    try:
        # Get recent objects (last 5 minutes) with real_center coordinates
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        
        # Query for recent objects with real_center data
        query = {
            "$or": [
                {"last_seen": {"$gte": five_minutes_ago}},
                {"updated_at": {"$gte": five_minutes_ago}},
                {"_id": {"$exists": True}}  # Fallback to get any objects
            ]
        }
        
        # Get objects, excluding MongoDB _id field
        objects = list(tracking_collection.find(
            query,
            {"_id": 0}
        ).sort([("last_seen", -1), ("updated_at", -1)]).limit(50))
        
        logger.info(f"📊 Found {len(objects)} objects in database")
        
        # Process objects for frontend
        processed_objects = []
        for obj in objects:
            try:
                # Convert datetime objects to ISO strings
                if "first_seen" in obj and isinstance(obj["first_seen"], datetime):
                    obj["first_seen"] = obj["first_seen"].isoformat()
                if "last_seen" in obj and isinstance(obj["last_seen"], datetime):
                    obj["last_seen"] = obj["last_seen"].isoformat()
                if "created_at" in obj and isinstance(obj["created_at"], datetime):
                    obj["created_at"] = obj["created_at"].isoformat()
                if "updated_at" in obj and isinstance(obj["updated_at"], datetime):
                    obj["updated_at"] = obj["updated_at"].isoformat()
                
                # Ensure required fields exist
                if "persistent_id" not in obj:
                    continue
                    
                # Calculate age_seconds if not present
                if "age_seconds" not in obj:
                    if "first_seen" in obj:
                        try:
                            if isinstance(obj["first_seen"], str):
                                first_seen = datetime.fromisoformat(obj["first_seen"].replace('Z', '+00:00'))
                            else:
                                first_seen = obj["first_seen"]
                            obj["age_seconds"] = (datetime.now() - first_seen).total_seconds()
                        except:
                            obj["age_seconds"] = 0
                    else:
                        obj["age_seconds"] = 0
                
                # Ensure confidence exists
                if "confidence" not in obj:
                    obj["confidence"] = 0.8  # Default confidence
                
                # Ensure times_seen exists
                if "times_seen" not in obj:
                    obj["times_seen"] = 1
                
                # Only include objects with real_center coordinates
                if "real_center" in obj and obj["real_center"]:
                    processed_objects.append(obj)
                    
            except Exception as e:
                logger.warning(f"⚠️  Error processing object: {e}")
                continue
        
        logger.info(f"✅ Returning {len(processed_objects)} processed objects")
        
        return {
            "objects": processed_objects,
            "count": len(processed_objects),
            "total_in_db": len(objects),
            "timestamp": datetime.now().isoformat(),
            "source": "live_cv_system"
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching objects: {str(e)}")

@app.get("/api/warehouse/config")
async def get_warehouse_config():
    """Get warehouse configuration"""
    try:
        # Try to read from CV system's calibration file
        import os
        import json
        
        calibration_paths = [
            "../cv/warehouse_calibration.json",
            "../../cv/warehouse_calibration.json",
            "../warehouse_calibration.json",
            "warehouse_calibration.json"
        ]
        
        for path in calibration_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    calibration_data = json.load(f)
                    
                warehouse_dims = calibration_data.get('warehouse_dimensions', {})
                return {
                    "width_meters": warehouse_dims.get('width_meters', 10.0),
                    "length_meters": warehouse_dims.get('length_meters', 8.0),
                    "calibrated": True,
                    "calibration_file": path,
                    "last_updated": datetime.now().isoformat(),
                    "source": "cv_calibration_file"
                }
        
        # Fallback configuration
        return {
            "width_meters": 10.0,
            "length_meters": 8.0,
            "calibrated": True,
            "calibration_file": None,
            "last_updated": datetime.now().isoformat(),
            "source": "default_config"
        }
        
    except Exception as e:
        logger.warning(f"⚠️  Error reading calibration: {e}")
        return {
            "width_meters": 10.0,
            "length_meters": 8.0,
            "calibrated": False,
            "error": str(e),
            "last_updated": datetime.now().isoformat(),
            "source": "fallback"
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
                for i in range(1, 5)
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error fetching camera status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching camera status: {str(e)}")

@app.get("/api/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Stream video from a specific camera"""
    if camera_id < 1 or camera_id > 4:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        return StreamingResponse(
            camera_manager.generate_frame(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"❌ Error streaming camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming camera {camera_id}: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Live CV System API Server...")
    print("📡 Connecting to MongoDB at localhost:27017")
    print("📊 Database: warehouse_tracking")
    print("📦 Collection: tracked_objects")
    print("🌐 API will be available at: http://localhost:8000")
    print("-" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
