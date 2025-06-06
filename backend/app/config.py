from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080"
    ]
    
    # Database
    DATABASE_URL: str = "postgresql://warehouse:password@localhost:5432/warehouse_db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Warehouse Layout
    WAREHOUSE_GRID: Dict = {
        "rows": 8,
        "cols": 8,
        "cell_size": 1.5,
        "origin_x": 0.0,
        "origin_y": 0.0
    }
    
    CAMERA_ZONES: Dict = {
        1: {"rows": ["A", "B", "C", "D"], "cols": [1, 2, 3, 4]},
        2: {"rows": ["A", "B", "C", "D"], "cols": [5, 6, 7, 8]},
        3: {"rows": ["E", "F", "G", "H"], "cols": [1, 2, 3, 4]},
        4: {"rows": ["E", "F", "G", "H"], "cols": [5, 6, 7, 8]}
    }
    
    # Cameras
    OAK_CAMERAS: Dict = {
        1: {"ip": "192.168.1.101", "type": "OAK-D-LR", "zone": 1},
        2: {"ip": "192.168.1.102", "type": "OAK-D-LR", "zone": 2}
    }
    
    ZED_CAMERAS: Dict = {
        3: {"serial": "SN12345", "type": "ZED-2i", "zone": 3},
        4: {"serial": "SN12346", "type": "ZED-2i", "zone": 4}
    }
    
    # Robot
    ROBOT_IP: str = "192.168.1.200"
    ROSBRIDGE_PORT: int = 9090
    
    # CV Settings
    YOLO_MODEL_PATH: str = "./models/pallet_detection_v8.pt"
    CONFIDENCE_THRESHOLD: float = 0.7
    TEMP_ID_PREFIX: str = "TEMP"
    
    # Operations
    PATROL_INTERVAL: int = 300
    QR_SCAN_TIMEOUT: int = 10
    ID_MATCH_DISTANCE_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"

settings = Settings()