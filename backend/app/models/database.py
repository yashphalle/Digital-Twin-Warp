from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class Pallet(Base):
    __tablename__ = "pallets"
    
    id = Column(Integer, primary_key=True, index=True)
    temp_id = Column(String, unique=True, index=True)
    real_id = Column(String, unique=True, index=True, nullable=True)
    
    grid_position = Column(String, index=True)
    zone = Column(Integer, index=True)
    x_coord = Column(Float)
    y_coord = Column(Float)
    
    detected_at = Column(DateTime, server_default=func.now())
    matched_at = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, server_default=func.now())
    
    status = Column(String, default="active")
    confidence = Column(Float, default=0.0)
    dimensions = Column(JSON)

class RobotPatrol(Base):
    __tablename__ = "robot_patrols"
    
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, server_default=func.now())
    end_time = Column(DateTime, nullable=True)
    zones_covered = Column(JSON)
    qr_scans_attempted = Column(Integer, default=0)
    qr_scans_successful = Column(Integer, default=0)
    temp_ids_matched = Column(Integer, default=0)
    status = Column(String, default="active")

class QRScanResult(Base):
    __tablename__ = "qr_scan_results"
    
    id = Column(Integer, primary_key=True, index=True)
    qr_code = Column(String, index=True)
    x_coord = Column(Float)
    y_coord = Column(Float)
    timestamp = Column(DateTime, server_default=func.now())
    matched_temp_id = Column(String, nullable=True)
    match_distance = Column(Float, nullable=True)
    match_confidence = Column(Float, nullable=True)