from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import asyncio
import json
from typing import Dict, Optional
import threading
import time

app = FastAPI(title="Warehouse Digital Twin API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple config without pydantic-settings
DATABASE_URL = "sqlite:///./warehouse.db"
CAMERA_ZONES = {
    1: {"rows": ["A", "B", "C", "D", "E", "F", "G"], "cols": list(range(1, 11))},
    2: {"rows": ["A", "B", "C", "D", "E", "F", "G"], "cols": list(range(11, 21))},
    3: {"rows": ["H", "I", "J", "K", "L", "M", "N"], "cols": list(range(1, 11))},
    4: {"rows": ["H", "I", "J", "K", "L", "M", "N"], "cols": list(range(11, 21))}
}

# Camera management
camera_instances: Dict[int, Optional[cv2.VideoCapture]] = {1: None, 2: None, 3: None, 4: None}
camera_status: Dict[int, str] = {1: "disconnected", 2: "disconnected", 3: "disconnected", 4: "disconnected"}

def get_camera_source(camera_id: int) -> int:
    """Get camera source for different camera IDs"""
    # For development, use different camera indices or same camera
    # In production, this would map to actual camera IPs/serials
    camera_sources = {
        1: 0,  # Default camera
        2: 0,  # Same camera for demo (in production: different IP)
        3: 0,  # Same camera for demo (in production: different IP)
        4: 0   # Same camera for demo (in production: different IP)
    }
    return camera_sources.get(camera_id, 0)

def generate_frames(camera_id: int):
    """Generate video frames for streaming"""
    camera = camera_instances[camera_id]
    if camera is None or not camera.isOpened():
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Add camera ID overlay
        cv2.putText(frame, f"Camera {camera_id} - Zone {camera_id}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
async def root():
    return {
        "message": "Warehouse Digital Twin Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/camera/{camera_id}/status")
async def get_camera_status(camera_id: int):
    """Get camera connection status"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {
        "camera_id": camera_id,
        "status": camera_status[camera_id],
        "zone": camera_id
    }

@app.post("/api/camera/{camera_id}/connect")
async def connect_camera(camera_id: int):
    """Connect to a camera"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        # Disconnect if already connected
        if camera_instances[camera_id] is not None:
            camera_instances[camera_id].release()

        # Connect to camera
        camera_source = get_camera_source(camera_id)
        camera = cv2.VideoCapture(camera_source)

        if camera.isOpened():
            camera_instances[camera_id] = camera
            camera_status[camera_id] = "connected"
            return {
                "camera_id": camera_id,
                "status": "connected",
                "message": f"Camera {camera_id} connected successfully"
            }
        else:
            camera_status[camera_id] = "error"
            raise HTTPException(status_code=500, detail="Failed to connect to camera")

    except Exception as e:
        camera_status[camera_id] = "error"
        raise HTTPException(status_code=500, detail=f"Camera connection error: {str(e)}")

@app.post("/api/camera/{camera_id}/disconnect")
async def disconnect_camera(camera_id: int):
    """Disconnect from a camera"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")

    if camera_instances[camera_id] is not None:
        camera_instances[camera_id].release()
        camera_instances[camera_id] = None

    camera_status[camera_id] = "disconnected"
    return {
        "camera_id": camera_id,
        "status": "disconnected",
        "message": f"Camera {camera_id} disconnected"
    }

@app.get("/api/camera/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Get live camera stream"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")

    if camera_status[camera_id] != "connected":
        raise HTTPException(status_code=503, detail="Camera not connected")

    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/status")
async def get_all_cameras_status():
    """Get status of all cameras"""
    return {
        "cameras": [
            {
                "camera_id": i,
                "status": camera_status[i],
                "zone": i
            }
            for i in [1, 2, 3, 4]
        ]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup cameras on shutdown"""
    for camera_id in camera_instances:
        if camera_instances[camera_id] is not None:
            camera_instances[camera_id].release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/config")
async def get_config():
    return {
        "database": DATABASE_URL,
        "zones": CAMERA_ZONES
    }

@app.get("/pallets")
async def get_pallets():
    return {
        "pallets": [],
        "message": "Ready to add database integration"
    }

@app.post("/pallets")
async def create_pallet(pallet_data: dict):
    return {
        "message": "Pallet received",
        "data": pallet_data,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
