from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.database import Pallet
from app.models.schemas import PalletCreate, PalletUpdate
from typing import List, Optional
import math

class PalletService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_pallet(self, pallet_data: PalletCreate) -> Pallet:
        db_pallet = Pallet(**pallet_data.dict())
        self.db.add(db_pallet)
        self.db.commit()
        self.db.refresh(db_pallet)
        return db_pallet
    
    def get_pallet_by_temp_id(self, temp_id: str) -> Optional[Pallet]:
        return self.db.query(Pallet).filter(Pallet.temp_id == temp_id).first()
    
    def get_all_pallets(self) -> List[Pallet]:
        return self.db.query(Pallet).all()
    
    def get_pallets_by_zone(self, zone: int) -> List[Pallet]:
        return self.db.query(Pallet).filter(Pallet.zone == zone).all()
    
    def update_pallet(self, temp_id: str, update_data: PalletUpdate) -> Optional[Pallet]:
        pallet = self.get_pallet_by_temp_id(temp_id)
        if pallet:
            for key, value in update_data.dict(exclude_unset=True).items():
                setattr(pallet, key, value)
            self.db.commit()
            self.db.refresh(pallet)
        return pallet
    
    def find_nearest_pallet(self, x: float, y: float, max_distance: float = 0.5) -> Optional[Pallet]:
        pallets = self.db.query(Pallet).filter(Pallet.status == "active").all()
        
        nearest_pallet = None
        min_distance = float('inf')
        
        for pallet in pallets:
            distance = math.sqrt((pallet.x_coord - x)**2 + (pallet.y_coord - y)**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_pallet = pallet
        
        return nearest_pallet
    
    def match_temp_to_real_id(self, temp_id: str, real_id: str) -> bool:
        pallet = self.get_pallet_by_temp_id(temp_id)
        if pallet:
            pallet.real_id = real_id
            pallet.status = "matched"
            self.db.commit()
            return True
        return False