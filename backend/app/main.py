from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import camera router
from api.v1.endpoints.cameras import router as camera_router

app = FastAPI(title="Warehouse Digital Twin API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include camera router
app.include_router(camera_router, prefix="/api/v1/cameras", tags=["cameras"])

# Simple config without pydantic-settings
DATABASE_URL = "sqlite:///./warehouse.db"
CAMERA_ZONES = {
    1: {"rows": ["A", "B", "C", "D"], "cols": [1, 2, 3, 4]},
    2: {"rows": ["A", "B", "C", "D"], "cols": [5, 6, 7, 8]},
    3: {"rows": ["E", "F", "G", "H"], "cols": [1, 2, 3, 4]},
    4: {"rows": ["E", "F", "G", "H"], "cols": [5, 6, 7, 8]}
}

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
        app,
        host="0.0.0.0",
        port=8001,
        reload=False
    )