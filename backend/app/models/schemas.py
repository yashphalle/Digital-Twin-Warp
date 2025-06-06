from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

class PalletStatus(str, Enum):
    ACTIVE = "active"
    REMOVED = "removed"
    MATCHED = "matched"

class PalletBase(BaseModel):
    temp_id: str
    grid_position: str
    zone: int
    x_coord: float
    y_coord: float
    confidence: float
    dimensions: Optional[Dict] = None

class PalletCreate(PalletBase):
    pass

class PalletUpdate(BaseModel):
    real_id: Optional[str] = None
    status: Optional[PalletStatus] = None
    grid_position: Optional[str] = None
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None

class PalletResponse(PalletBase):
    id: int
    real_id: Optional[str]
    status: str
    detected_at: datetime
    matched_at: Optional[datetime]
    last_seen: datetime
    
    class Config:
        from_attributes = True

class RobotStatus(BaseModel):
    connected: bool
    battery_level: Optional[float]
    current_position: Optional[Dict[str, float]]
    current_task: Optional[str]
    last_patrol: Optional[datetime]

class WebSocketMessage(BaseModel):
    type: str
    data: Dict