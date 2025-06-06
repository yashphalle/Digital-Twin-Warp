from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from datetime import datetime

router = APIRouter()

# In-memory storage for camera states
camera_states: Dict[int, Dict] = {}

@router.get("/")
async def get_cameras():
    """Get all available cameras and their status"""
    return {
        "cameras": [
            {
                "id": 1,
                "name": "Camera 1",
                "zone": 1,
                "status": camera_states.get(1, {}).get("status", "inactive"),
                "device_id": camera_states.get(1, {}).get("device_id"),
                "last_active": camera_states.get(1, {}).get("last_active")
            },
            {
                "id": 2,
                "name": "Camera 2", 
                "zone": 2,
                "status": camera_states.get(2, {}).get("status", "inactive"),
                "device_id": camera_states.get(2, {}).get("device_id"),
                "last_active": camera_states.get(2, {}).get("last_active")
            },
            {
                "id": 3,
                "name": "Camera 3",
                "zone": 3,
                "status": camera_states.get(3, {}).get("status", "inactive"),
                "device_id": camera_states.get(3, {}).get("device_id"),
                "last_active": camera_states.get(3, {}).get("last_active")
            },
            {
                "id": 4,
                "name": "Camera 4",
                "zone": 4,
                "status": camera_states.get(4, {}).get("status", "inactive"),
                "device_id": camera_states.get(4, {}).get("device_id"),
                "last_active": camera_states.get(4, {}).get("last_active")
            }
        ]
    }

@router.get("/{camera_id}")
async def get_camera(camera_id: int):
    """Get specific camera information"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return {
        "id": camera_id,
        "name": f"Camera {camera_id}",
        "zone": camera_id,
        "status": camera_states.get(camera_id, {}).get("status", "inactive"),
        "device_id": camera_states.get(camera_id, {}).get("device_id"),
        "last_active": camera_states.get(camera_id, {}).get("last_active")
    }

@router.post("/{camera_id}/start")
async def start_camera(camera_id: int, device_id: Optional[str] = None):
    """Start a camera feed"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # Update camera state
        camera_states[camera_id] = {
            "status": "active",
            "device_id": device_id,
            "last_active": datetime.now().isoformat()
        }
        
        return {
            "message": f"Camera {camera_id} started successfully",
            "camera_id": camera_id,
            "status": "active",
            "device_id": device_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")

@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """Stop a camera feed"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # Update camera state
        if camera_id in camera_states:
            camera_states[camera_id]["status"] = "inactive"
        
        return {
            "message": f"Camera {camera_id} stopped successfully",
            "camera_id": camera_id,
            "status": "inactive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {str(e)}")

@router.get("/{camera_id}/devices")
async def get_camera_devices(camera_id: int):
    """Get available camera devices for a specific camera"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # This would typically enumerate actual camera devices
    # For now, return mock data
    return {
        "camera_id": camera_id,
        "devices": [
            {
                "device_id": "default",
                "name": "Default Camera",
                "type": "built-in"
            },
            {
                "device_id": "external_1",
                "name": "External Camera 1",
                "type": "usb"
            }
        ]
    }

@router.post("/{camera_id}/configure")
async def configure_camera(camera_id: int, config: Dict):
    """Configure camera settings"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Update camera configuration
    if camera_id not in camera_states:
        camera_states[camera_id] = {}
    
    camera_states[camera_id].update({
        "config": config,
        "last_configured": datetime.now().isoformat()
    })
    
    return {
        "message": f"Camera {camera_id} configured successfully",
        "camera_id": camera_id,
        "config": config
    }

@router.get("/{camera_id}/status")
async def get_camera_status(camera_id: int):
    """Get real-time camera status"""
    if camera_id not in [1, 2, 3, 4]:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    state = camera_states.get(camera_id, {})
    
    return {
        "camera_id": camera_id,
        "status": state.get("status", "inactive"),
        "device_id": state.get("device_id"),
        "last_active": state.get("last_active"),
        "config": state.get("config", {}),
        "timestamp": datetime.now().isoformat()
    }
