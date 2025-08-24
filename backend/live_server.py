"""
Live server connecting to real MongoDB CV system
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import cv2
import time
import numpy as np
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
class WarpIdLinkRequest(BaseModel):
    warp_id: str

class PalletWarpLinkRequest(BaseModel):
    pallet_id: int  # This will be the global_id
    warp_id: str

class WarpIdResponse(BaseModel):
    success: bool
    message: str
    persistent_id: Optional[int] = None
    global_id: Optional[int] = None
    warp_id: Optional[str] = None

app = FastAPI(title="Live Warehouse Tracking API")

# Add CORS middleware
import os
# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://localhost:5174').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve saved crops as static files under /images (relative repo path)
try:
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(repo_root, 'images')
    os.makedirs(images_dir, exist_ok=True)
    app.mount('/images', StaticFiles(directory=images_dir), name='images')
except Exception as e:
    logger.warning(f"Static mount for /images failed: {e}")

# Normalize GOOGLE_APPLICATION_CREDENTIALS to absolute path if provided relative in .env
try:
    import os as _os
    cred = _os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if cred and not _os.path.isabs(cred):
        # repo_root already computed above
        abs_cred = _os.path.abspath(_os.path.join(repo_root, cred))
        if _os.path.exists(abs_cred):
            _os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_cred
            logger.info(f"Using GOOGLE_APPLICATION_CREDENTIALS at: {abs_cred}")
        else:
            logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS not found at {abs_cred}")
except Exception as _e:
    logger.warning(f"Failed to normalize GOOGLE_APPLICATION_CREDENTIALS: {_e}")


# Load .env for RTSP URLs if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Prefer RTSP URLs from .env: RTSP_URL_1..RTSP_URL_11
def _load_rtsp_env_urls():
    urls = {}
    for i in range(1, 12):
        u = os.getenv(f"RTSP_URL_{i}")
        if u:
            urls[i] = u
    return urls

RTSP_ENV_URLS = _load_rtsp_env_urls()

def get_rtsp_url(camera_id: int) -> str:
    if camera_id in RTSP_ENV_URLS:
        return RTSP_ENV_URLS[camera_id]
    try:
        # Use remote streams by default as requested
        from cv.config.config import Config
        url = ""
        if hasattr(Config, 'REMOTE_RTSP_CAMERA_URLS'):
            url = Config.REMOTE_RTSP_CAMERA_URLS.get(camera_id, "")
        if not url and hasattr(Config, 'RTSP_CAMERA_URLS'):
            url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        return url or ""
    except Exception:
        return ""

def rtsp_mjpeg_generator(camera_id: int):
    """Stream actual RTSP as MJPEG if URL available; fallback to demo generator."""
    url = get_rtsp_url(camera_id)
    if not url:
        # Fallback to demo frames
        yield from camera_manager.generate_frame(camera_id)
        return
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.warning(f"RTSP open failed for camera {camera_id}, falling back to demo stream")
        yield from camera_manager.generate_frame(camera_id)
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning(f"RTSP read failed for camera {camera_id}, stopping stream")
                break
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"RTSP stream error (camera {camera_id}): {e}")
    finally:
        cap.release()

# Import Config for database settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv.config.config import Config


# MongoDB connection using Config
MONGO_URI = Config.MONGO_URI  # Will use local or online based on Config.USE_LOCAL_DATABASE
DATABASE_NAME = Config.DATABASE_NAME
COLLECTION_NAME = Config.COLLECTION_NAME

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
            from cv.config.config import Config

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
            from cv.config.config import Config

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
            from cv.config.config import Config
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

    def get_single_frame(self, camera_id):
        """Generate a single JPEG frame for a specific camera (demo feed)."""
        status = self.camera_status.get(camera_id, "offline")
        try:
            from cv.config.config import Config
            camera_name = Config.CAMERA_NAMES.get(camera_id, f"Camera {camera_id}")
            coverage_zone = Config.CAMERA_COVERAGE_ZONES.get(camera_id, {})
        except ImportError:
            camera_name = f"Camera {camera_id}"
            coverage_zone = {}

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if status in ["active", "ready"]:
            base_color = (
                (camera_id * 30) % 255,
                (camera_id * 50) % 255,
                (camera_id * 70) % 255,
            )
            color_intensity = int(50 + 30 * np.sin(time.time()))
            frame[:] = tuple(int(c * color_intensity / 255) for c in base_color)
            cv2.putText(frame, camera_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            status_text = "🎯 ACTIVE" if status == "active" else "💤 STANDBY"
            cv2.putText(frame, status_text, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "active" else (255, 255, 0), 2)
            if coverage_zone:
                zone_text = f"Zone: {coverage_zone.get('x_start', 0)}-{coverage_zone.get('x_end', 0)}ft"
                cv2.putText(frame, zone_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            frame[:] = (30, 30, 30)
            cv2.putText(frame, camera_name, (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        return buffer.tobytes()
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
    """Get ALL tracked objects in warehouse (complete warehouse state)"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected to CV system")

    try:
        # Get active objects with valid physical coordinates
        # Active = status in ['new', 'tracked']
        query = {
            "physical_x_ft": {"$exists": True, "$ne": None},
            "physical_y_ft": {"$exists": True, "$ne": None},
            "status": {"$in": ["new", "tracked"]}
        }

        # Get ALL detections with coordinates, excluding MongoDB _id field
        detections = list(tracking_collection.find(
            query,
            {"_id": 0}
        ).sort([("last_seen", -1), ("timestamp", -1), ("first_seen", -1)]))

        logger.info(f"📊 Found {len(detections)} detections in database")

        # Process detections for frontend
        processed_objects = []
        for detection in detections:
            try:
                # Convert datetime objects to ISO strings (handle multiple timestamp fields)
                for timestamp_field in ["timestamp", "last_seen", "first_seen"]:
                    if timestamp_field in detection and isinstance(detection[timestamp_field], datetime):
                        detection[timestamp_field] = detection[timestamp_field].isoformat()

                # Create object structure for frontend
                obj = {
                    "persistent_id": detection.get("global_id", f"det_{detection.get('_id', 'unknown')}"),
                    "global_id": detection.get("global_id"),
                    "warp_id": detection.get("warp_id"),  # NEW: Include Warp ID from QR code
                    "camera_id": detection.get("camera_id"),
                    "bbox": detection.get("bbox", [0, 0, 100, 100]),
                    "corners": detection.get("corners", []),  # 4-point pixel coordinates
                    "physical_corners": detection.get("physical_corners", []),  # 4-point physical coordinates
                    "shape_type": detection.get("shape_type", "rectangle"),
                    "confidence": detection.get("confidence", 0.0),
                    "area": detection.get("area", 0),
                    "real_center": [detection.get("physical_x_ft"), detection.get("physical_y_ft")],
                    "physical_x_ft": detection.get("physical_x_ft"),
                    "physical_y_ft": detection.get("physical_y_ft"),
                    "grid_cell": detection.get("grid_cell"),
                    "times_seen": detection.get("times_seen", 1),
                    "is_new": detection.get("is_new", False),
                    "timestamp": detection.get("timestamp"),
                    "first_seen": detection.get("first_seen"),  # Use actual first_seen from DB
                    "last_seen": detection.get("last_seen"),    # Use actual last_seen from DB
                    "age_seconds": 0,  # Recent detection
                    # Color information from CV system
                    "color_rgb": detection.get("color_rgb"),
                    "color_hsv": detection.get("color_hsv"),
                    "color_hex": detection.get("color_hex"),
                    "color_name": detection.get("color_name"),
                    "color_confidence": detection.get("color_confidence"),
                    "extraction_method": detection.get("extraction_method"),
                    # Warp ID metadata
                    "warp_id_linked_at": detection.get("warp_id_linked_at"),
                    # Current lifecycle status for frontend filtering
                    "status": detection.get("status", "tracked"),
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
            "source": "warehouse_complete_state"
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

@app.put("/api/tracking/objects/{persistent_id}/warp-id")
async def link_warp_id(persistent_id: int, request: WarpIdLinkRequest) -> WarpIdResponse:
    """Link a Warp ID from QR code to existing pallet by persistent_id"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        warp_id = request.warp_id.strip()

        # Validate warp_id format (basic validation)
        if not warp_id or len(warp_id) < 3:
            return WarpIdResponse(
                success=False,
                message="Invalid Warp ID format. Must be at least 3 characters."
            )

        # Check if warp_id is already used by another object
        existing_object = tracking_collection.find_one({"warp_id": warp_id})
        if existing_object and existing_object.get("persistent_id") != persistent_id:
            return WarpIdResponse(
                success=False,
                message=f"Warp ID '{warp_id}' is already linked to another object (ID: {existing_object.get('persistent_id')})"
            )

        # Find the object by persistent_id
        target_object = tracking_collection.find_one({"persistent_id": persistent_id})
        if not target_object:
            return WarpIdResponse(
                success=False,
                message=f"No object found with persistent_id {persistent_id}"
            )

        # Update the object with warp_id
        result = tracking_collection.update_many(
            {"persistent_id": persistent_id},
            {
                "$set": {
                    "warp_id": warp_id,
                    "warp_id_linked_at": datetime.now()
                }
            }
        )

        if result.modified_count > 0:
            logger.info(f"✅ Linked Warp ID '{warp_id}' to persistent_id {persistent_id}")
            return WarpIdResponse(
                success=True,
                message=f"Successfully linked Warp ID '{warp_id}' to object {persistent_id}",
                persistent_id=persistent_id,
                global_id=target_object.get("global_id"),
                warp_id=warp_id
            )
        else:
            return WarpIdResponse(
                success=False,
                message=f"Failed to update object {persistent_id}"
            )

    except Exception as e:
        logger.error(f"❌ Error linking Warp ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error linking Warp ID: {str(e)}")

@app.get("/api/tracking/objects/by-warp-id/{warp_id}")
async def get_object_by_warp_id(warp_id: str):
    """Find object by Warp ID"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        # Find object with the specified warp_id
        object_data = tracking_collection.find_one(
            {"warp_id": warp_id},
            {"_id": 0},  # Exclude MongoDB _id field
            sort=[("last_seen", -1)]  # Get most recent if multiple
        )

        if not object_data:
            raise HTTPException(status_code=404, detail=f"No object found with Warp ID '{warp_id}'")

        logger.info(f"🔍 Found object with Warp ID '{warp_id}': persistent_id {object_data.get('persistent_id')}")

        return {
            "success": True,
            "object": object_data,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error finding object by Warp ID '{warp_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding object: {str(e)}")

@app.get("/api/tracking/warp-ids")
async def get_all_warp_ids():
    """Get all objects with Warp IDs for inventory management"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        # Find all objects that have warp_id set
        objects_with_warp_ids = list(tracking_collection.find(
            {"warp_id": {"$exists": True, "$ne": None}},
            {
                "_id": 0,
                "persistent_id": 1,
                "global_id": 1,
                "warp_id": 1,
                "real_center": 1,
                "physical_x_ft": 1,
                "physical_y_ft": 1,
                "camera_id": 1,
                "first_seen": 1,
                "last_seen": 1,
                "warp_id_linked_at": 1
            }
        ).sort([("warp_id_linked_at", -1)]))

        logger.info(f"📊 Found {len(objects_with_warp_ids)} objects with Warp IDs")

        return {
            "success": True,
            "count": len(objects_with_warp_ids),
            "objects": objects_with_warp_ids,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error fetching objects with Warp IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching Warp IDs: {str(e)}")

@app.post("/api/robot/simulate-assignment")
async def simulate_robot_assignment():
    """🤖 TEST API: Simulate robot assignment of Warp ID to a random pallet"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        # Find objects without Warp IDs
        unlinked_objects = list(tracking_collection.find(
            {
                "physical_x_ft": {"$exists": True, "$ne": None},
                "physical_y_ft": {"$exists": True, "$ne": None},
                "$or": [
                    {"warp_id": {"$exists": False}},
                    {"warp_id": None}
                ]
            },
            {
                "_id": 0,
                "persistent_id": 1,
                "global_id": 1,
                "physical_x_ft": 1,
                "physical_y_ft": 1,
                "camera_id": 1
            }
        ).limit(10))

        if not unlinked_objects:
            return {
                "success": False,
                "message": "No unlinked objects found to assign Warp ID",
                "timestamp": datetime.now().isoformat()
            }

        # Select first unlinked object for simulation
        target_object = unlinked_objects[0]
        persistent_id = target_object["persistent_id"]

        # Generate simulated Warp ID
        import time
        simulated_warp_id = f"WARP-ROBOT-{int(time.time())}"

        # Simulate robot workflow
        robot_steps = [
            f"🔍 Robot scans warehouse area at ({target_object['physical_x_ft']:.1f}ft, {target_object['physical_y_ft']:.1f}ft)",
            f"📦 Robot identifies pallet with Object ID: {persistent_id}",
            f"📱 Robot scans QR code and reads: {simulated_warp_id}",
            f"🔗 Robot calls API to link Warp ID..."
        ]

        # Actually link the Warp ID
        result = tracking_collection.update_many(
            {"persistent_id": persistent_id},
            {
                "$set": {
                    "warp_id": simulated_warp_id,
                    "warp_id_linked_at": datetime.now()
                }
            }
        )

        if result.modified_count > 0:
            robot_steps.append(f"✅ Successfully linked {simulated_warp_id} to Object {persistent_id}")

            logger.info(f"🤖 Robot simulation: Linked {simulated_warp_id} to object {persistent_id}")

            return {
                "success": True,
                "message": "Robot assignment simulation completed successfully",
                "robot_steps": robot_steps,
                "assignment": {
                    "persistent_id": persistent_id,
                    "global_id": target_object.get("global_id"),
                    "warp_id": simulated_warp_id,
                    "position": [target_object["physical_x_ft"], target_object["physical_y_ft"]],
                    "camera_id": target_object.get("camera_id")
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            robot_steps.append(f"❌ Failed to link Warp ID")
            return {
                "success": False,
                "message": "Failed to update object with Warp ID",
                "robot_steps": robot_steps,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"❌ Error in robot simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Robot simulation error: {str(e)}")

@app.post("/api/link-warp-id")
async def link_pallet_warp_id(request: PalletWarpLinkRequest) -> WarpIdResponse:
    """🔗 Simple POST API: Link Warp ID to Pallet ID"""
    if tracking_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        pallet_id = request.pallet_id
        warp_id = request.warp_id.strip()

        # Basic validation
        if not warp_id or len(warp_id) < 3:
            return WarpIdResponse(
                success=False,
                message="Invalid Warp ID format. Must be at least 3 characters."
            )

        # Check if warp_id is already used
        existing_object = tracking_collection.find_one({"warp_id": warp_id})
        if existing_object and existing_object.get("global_id") != pallet_id:
            return WarpIdResponse(
                success=False,
                message=f"Warp ID '{warp_id}' is already linked to another pallet (Global ID: {existing_object.get('global_id')})"
            )

        # Check if pallet exists (using global_id)
        target_object = tracking_collection.find_one({"global_id": pallet_id})
        if not target_object:
            return WarpIdResponse(
                success=False,
                message=f"Pallet with Global ID {pallet_id} not found in system"
            )

        # Link the Warp ID (update all documents with this global_id)
        result = tracking_collection.update_many(
            {"global_id": pallet_id},
            {
                "$set": {
                    "warp_id": warp_id,
                    "warp_id_linked_at": datetime.now()
                }
            }
        )

        if result.modified_count > 0:
            logger.info(f"✅ Linked Warp ID '{warp_id}' to Global ID {pallet_id}")
            return WarpIdResponse(
                success=True,
                message=f"Successfully linked Warp ID '{warp_id}' to Global ID {pallet_id}",
                persistent_id=target_object.get("persistent_id"),
                global_id=pallet_id,
                warp_id=warp_id
            )
        else:
            return WarpIdResponse(
                success=False,
                message=f"Failed to update Global ID {pallet_id}"
            )

    except Exception as e:
        logger.error(f"❌ Error linking Warp ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error linking Warp ID: {str(e)}")

@app.get("/api/cameras/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: int):
    """Return a single JPEG frame from actual RTSP if provided in .env, else from Config, else demo."""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")
    url = get_rtsp_url(camera_id)
    img_bytes = None
    try:
        if url:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    # try a second frame to warm up
                    ok2, frame2 = cap.read()
                    frame = frame2 if ok2 else frame
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        img_bytes = buffer.tobytes()
                cap.release()
        if img_bytes is None:
            img_bytes = camera_manager.get_single_frame(camera_id)
        if img_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot error: {str(e)}")

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
        logger.error(f"❌ Error fetching cameras status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching camera status: {str(e)}")

from typing import List as _List

@app.get("/api/crops/{persistent_id}")
async def list_crops(persistent_id: int, camera_id: int | None = None, date: str | None = None, limit: int = 20, offset: int = 0):
    """List saved crop image URLs for a persistent_id. Optional camera_id and date (YYYY-MM-DD).
    If USE_GCS_CROPS=true, list from GCS bucket; else, list from local images/ directory.
    """
    try:
        import os
        # Load .env to allow configuring GCS via .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        use_gcs = os.getenv('USE_GCS_CROPS', 'false').lower() == 'true'
        results = []
        # Determine dates to scan
        dates_to_check: _List[str] = []
        if date:
            dates_to_check = [date]
        else:
            from datetime import datetime
            dates_to_check = [datetime.now().strftime('%Y-%m-%d')]

        if use_gcs:
            from google.cloud import storage
            bucket_name = os.getenv('GCS_BUCKET')
            prefix_base = os.getenv('GCS_PREFIX', 'images')
            if not bucket_name:
                raise HTTPException(status_code=500, detail="GCS_BUCKET not set")
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            for d in dates_to_check:
                # Cameras: specific or all under date
                cam_dirs = [str(camera_id)] if camera_id is not None else None
                # Build prefix(s): images/{date}/{cam}/{pid}/
                if cam_dirs is None:
                    prefix = f"{prefix_base}/{d}/"
                    # List cams under date (1 level); then list per pid folder
                    # Simpler: just list with startswith of pid dir
                    iterator = client.list_blobs(bucket, prefix=prefix)
                    for blob in iterator:
                        # Expect names like images/{d}/{cam}/{pid}/{fname}
                        parts = blob.name.split('/')
                        if len(parts) < 5:
                            continue
                        _, dd, cam_s, pid_s, fname = parts[0], parts[1], parts[2], parts[3], parts[-1]
                        if dd != d or pid_s != str(persistent_id):
                            continue
                        if not fname.lower().endswith('.jpg') or not fname.startswith(f"{persistent_id}_"):
                            continue
                        cam_int = int(cam_s) if cam_s.isdigit() else None
                        mtime_ms = int(blob.updated.timestamp() * 1000) if getattr(blob, 'updated', None) else 0
                        # Build URL (signed or public)
                        if os.getenv('GCS_SIGNED_URLS', 'true').lower() == 'true':
                            from datetime import timedelta
                            ttl = int(os.getenv('GCS_SIGNED_URL_TTL', '600'))
                            url = blob.generate_signed_url(expiration=timedelta(seconds=ttl), version='v4', method='GET')
                        else:
                            url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
                        results.append({
                            'camera_id': cam_int,
                            'date': d,
                            'filename': fname,
                            'url': url,
                            'mtime_ms': mtime_ms,
                        })
                else:
                    for cam_s in cam_dirs:
                        prefix = f"{prefix_base}/{d}/{cam_s}/{persistent_id}/"
                        for blob in client.list_blobs(bucket, prefix=prefix):
                            fname = blob.name.split('/')[-1]
                            if not fname.lower().endswith('.jpg') or not fname.startswith(f"{persistent_id}_"):
                                continue
                            mtime_ms = int(blob.updated.timestamp() * 1000) if getattr(blob, 'updated', None) else 0
                            if os.getenv('GCS_SIGNED_URLS', 'true').lower() == 'true':
                                from datetime import timedelta
                                ttl = int(os.getenv('GCS_SIGNED_URL_TTL', '600'))
                                url = blob.generate_signed_url(expiration=timedelta(seconds=ttl), version='v4', method='GET')
                            else:
                                url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
                            results.append({
                                'camera_id': int(cam_s),
                                'date': d,
                                'filename': fname,
                                'url': url,
                                'mtime_ms': mtime_ms,
                            })
        else:
            base = images_dir  # from static mount
            for d in dates_to_check:
                day_dir = os.path.join(base, d)
                if not os.path.isdir(day_dir):
                    continue
                cams = [str(camera_id)] if camera_id is not None else [name for name in os.listdir(day_dir) if os.path.isdir(os.path.join(day_dir, name))]
                for cam in cams:
                    pid_dir = os.path.join(day_dir, cam, str(persistent_id))
                    if not os.path.isdir(pid_dir):
                        continue
                    for fname in os.listdir(pid_dir):
                        if not fname.lower().endswith('.jpg'):
                            continue
                        if not fname.startswith(f"{persistent_id}_"):
                            continue
                        fpath = os.path.join(pid_dir, fname)
                        try:
                            mtime_ms = int(os.path.getmtime(fpath) * 1000)
                        except Exception:
                            mtime_ms = 0
                        results.append({
                            'camera_id': int(cam),
                            'date': d,
                            'filename': fname,
                            'url': f"/images/{d}/{cam}/{persistent_id}/{fname}",
                            'mtime_ms': mtime_ms,
                        })
        # Sort newest first
        results.sort(key=lambda x: x['mtime_ms'], reverse=True)
        sliced = results[offset: offset + limit]
        return {'persistent_id': persistent_id, 'count': len(results), 'items': sliced}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing crops: {e}")

@app.get("/api/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Stream video from a specific camera"""
    if camera_id < 1 or camera_id > 11:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        rtsp_mjpeg_generator(camera_id),
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

        from cv.config.config import Config

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
    print("📦 Collection: detections")
    print("🌐 API will be available at: http://localhost:8000")
    print("-" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
