from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.services.pallet_service import PalletService
from app.models.schemas import PalletCreate, PalletUpdate, PalletResponse

router = APIRouter()

@router.post("/", response_model=PalletResponse)
async def create_pallet(
    pallet: PalletCreate,
    db: Session = Depends(get_db)
):
    service = PalletService(db)
    return service.create_pallet(pallet)

@router.get("/", response_model=List[PalletResponse])
async def get_all_pallets(db: Session = Depends(get_db)):
    service = PalletService(db)
    return service.get_all_pallets()

@router.get("/zone/{zone_id}", response_model=List[PalletResponse])
async def get_pallets_by_zone(
    zone_id: int,
    db: Session = Depends(get_db)
):
    service = PalletService(db)
    return service.get_pallets_by_zone(zone_id)

@router.get("/{temp_id}", response_model=PalletResponse)
async def get_pallet(
    temp_id: str,
    db: Session = Depends(get_db)
):
    service = PalletService(db)
    pallet = service.get_pallet_by_temp_id(temp_id)
    if not pallet:
        raise HTTPException(status_code=404, detail="Pallet not found")
    return pallet

@router.put("/{temp_id}", response_model=PalletResponse)
async def update_pallet(
    temp_id: str,
    update_data: PalletUpdate,
    db: Session = Depends(get_db)
):
    service = PalletService(db)
    pallet = service.update_pallet(temp_id, update_data)
    if not pallet:
        raise HTTPException(status_code=404, detail="Pallet not found")
    return pallet

@router.post("/{temp_id}/match")
async def match_pallet_id(
    temp_id: str,
    real_id: str,
    db: Session = Depends(get_db)
):
    service = PalletService(db)
    success = service.match_temp_to_real_id(temp_id, real_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pallet not found")
    return {"message": "Pallet matched successfully"}