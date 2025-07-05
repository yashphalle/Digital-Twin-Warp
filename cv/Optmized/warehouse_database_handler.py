#!/usr/bin/env python3
"""
Warehouse Database Handler
Extracted from working GPU script for MongoDB integration
Handles detection saving, batching, and database operations
"""

import logging
from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class WarehouseDatabaseHandler:
    """MongoDB database handler for warehouse detections (extracted from GPU script)"""
    
    def __init__(self, 
                 mongodb_url: str = "mongodb://localhost:27017/",
                 database_name: str = "warehouse_tracking",
                 collection_name: str = "detections",
                 batch_save_size: int = 10,
                 enable_mongodb: bool = True):
        
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.collection_name = collection_name
        self.batch_save_size = batch_save_size
        self.enable_mongodb = enable_mongodb
        
        # MongoDB connection objects
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_collection = None
        
        # Batching for performance
        self.detection_batch = []
        self.total_saved_detections = 0
        
        # Initialize connection if enabled
        if self.enable_mongodb:
            self.initialize_mongodb()
    
    def initialize_mongodb(self):
        """Initialize MongoDB connection (same as GPU script)"""
        try:
            self.mongodb_client = MongoClient(self.mongodb_url)
            self.mongodb_db = self.mongodb_client[self.database_name]
            self.mongodb_collection = self.mongodb_db[self.collection_name]

            # Test connection
            self.mongodb_client.admin.command('ping')
            logger.info(f"✅ MongoDB connected: {self.database_name}.{self.collection_name}")

            # Create indexes for better performance (no unique constraint to avoid insert failures)
            self.mongodb_collection.create_index([("global_id", 1), ("camera_id", 1)])
            self.mongodb_collection.create_index([("camera_id", 1), ("last_seen", -1)])
            self.mongodb_collection.create_index([("first_seen", -1)])
            # NEW: Index for warp_id lookups
            self.mongodb_collection.create_index([("warp_id", 1)], sparse=True)  # sparse=True for nullable field

            logger.info("✅ MongoDB indexes created for optimal upsert performance")

            # Test database write capability
            test_doc = {"test": "connection", "timestamp": datetime.utcnow()}
            test_result = self.mongodb_collection.insert_one(test_doc)
            if test_result.inserted_id:
                logger.info("✅ Database write test successful")
                # Clean up test document
                self.mongodb_collection.delete_one({"_id": test_result.inserted_id})
            else:
                logger.error("❌ Database write test failed")

        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.warning("⚠️ Continuing without MongoDB integration")
            self.mongodb_client = None
    
    def save_detection_to_db(self, camera_id: int, detection: Dict):
        """Save detection using proper upsert logic - no duplicates"""
        if not self.enable_mongodb or not self.mongodb_client:
            logger.debug(f"❌ MongoDB not enabled or not connected")
            return

        global_id = detection.get('global_id')
        tracking_status = detection.get('tracking_status', 'unknown')

        logger.debug(f"🔍 Attempting to save: Camera {camera_id}, Global ID {global_id}, Status: {tracking_status}")
        logger.info(f"🎨 Color data in detection: RGB={detection.get('rgb')}, HSV={detection.get('hsv')}, Hex={detection.get('hex')}")

        if global_id is None or global_id == -1:
            logger.debug(f"❌ Skipping invalid global ID: {global_id}")
            return  # Skip invalid global IDs

        try:
            current_time = datetime.utcnow()

            if tracking_status == 'new':
                # Insert new object document
                detection_doc = {
                    "camera_id": camera_id,
                    "global_id": global_id,
                    "warp_id": detection.get('warp_id'),  # NEW: Warp ID from QR code (None by default)
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "times_seen": 1,
                    "bbox": detection.get('bbox'),
                    "corners": detection.get('corners'),  # 4-point pixel coordinates
                    "physical_corners": detection.get('physical_corners'),  # 4-point physical coordinates
                    "shape_type": detection.get('shape_type', 'rectangle'),
                    "real_center": detection.get('real_center'),  # Physical center coordinates
                    "confidence": detection.get('confidence'),
                    "area": detection.get('area'),
                    "center": detection.get('center'),
                    "physical_x_ft": detection.get('physical_x_ft'),
                    "physical_y_ft": detection.get('physical_y_ft'),
                    "coordinate_status": detection.get('coordinate_status'),
                    "tracking_status": tracking_status,
                    "similarity_score": detection.get('similarity_score', 1.0),
                    # Color information
                    "color_rgb": detection.get('rgb'),
                    "color_hsv": detection.get('hsv'),
                    "color_hex": detection.get('hex'),
                    "color_name": detection.get('color_name'),
                    "color_confidence": detection.get('color_confidence'),
                    "extraction_method": detection.get('extraction_method')
                }

                result = self.mongodb_collection.insert_one(detection_doc)
                if result.inserted_id:
                    self.total_saved_detections += 1
                    logger.info(f"💾 NEW object saved to DB: Global ID {global_id}, Camera {camera_id}")
                else:
                    logger.error(f"❌ Failed to insert NEW object: Global ID {global_id}, Camera {camera_id}")

            elif tracking_status == 'existing':
                # Update existing object document
                update_doc = {
                    "$set": {
                        "last_seen": current_time,
                        "bbox": detection.get('bbox'),
                        "corners": detection.get('corners'),  # 4-point pixel coordinates
                        "physical_corners": detection.get('physical_corners'),  # 4-point physical coordinates
                        "shape_type": detection.get('shape_type', 'rectangle'),
                        "real_center": detection.get('real_center'),  # Physical center coordinates
                        "confidence": detection.get('confidence'),
                        "area": detection.get('area'),
                        "center": detection.get('center'),
                        "physical_x_ft": detection.get('physical_x_ft'),
                        "physical_y_ft": detection.get('physical_y_ft'),
                        "coordinate_status": detection.get('coordinate_status'),
                        "similarity_score": detection.get('similarity_score', 0.0),
                        # Update color information
                        "color_rgb": detection.get('rgb'),
                        "color_hsv": detection.get('hsv'),
                        "color_hex": detection.get('hex'),
                        "color_name": detection.get('color_name'),
                        "color_confidence": detection.get('color_confidence'),
                        "extraction_method": detection.get('extraction_method')
                    },
                    "$inc": {
                        "times_seen": 1
                    }
                }

                result = self.mongodb_collection.update_one(
                    {"global_id": global_id, "camera_id": camera_id},
                    update_doc
                )

                if result.modified_count > 0:
                    logger.debug(f"🔄 UPDATED object in DB: Global ID {global_id}")
                elif result.matched_count == 0:
                    # Object not found, insert as new (fallback)
                    logger.warning(f"⚠️ Object {global_id} not found in DB, inserting as new")
                    detection['tracking_status'] = 'new'
                    self.save_detection_to_db(camera_id, detection)

        except Exception as e:
            logger.error(f"❌ Failed to save detection to MongoDB: {e}")
            # Fallback to batch system for failed individual saves
            self._add_to_batch_fallback(camera_id, detection)

    def _add_to_batch_fallback(self, camera_id: int, detection: Dict):
        """Fallback batch system for failed individual saves"""
        detection_doc = {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow(),
            "global_id": detection.get('global_id'),
            "bbox": detection.get('bbox'),
            "confidence": detection.get('confidence'),
            "area": detection.get('area'),
            "center": detection.get('center'),
            "physical_x_ft": detection.get('physical_x_ft'),
            "physical_y_ft": detection.get('physical_y_ft'),
            "coordinate_status": detection.get('coordinate_status'),
            "tracking_status": detection.get('tracking_status'),
            "similarity_score": detection.get('similarity_score')
        }

        self.detection_batch.append(detection_doc)

        if len(self.detection_batch) >= self.batch_save_size:
            self.flush_detection_batch()

    def flush_detection_batch(self):
        """Save all batched detections to MongoDB (fallback batch system)"""
        if not self.detection_batch or not self.mongodb_client:
            return

        try:
            # Use insert_many for fallback batch (these are usually failed individual saves)
            result = self.mongodb_collection.insert_many(self.detection_batch)
            self.total_saved_detections += len(result.inserted_ids)

            if len(self.detection_batch) >= 5:  # Only log for larger batches
                logger.info(f"💾 Batch saved {len(self.detection_batch)} detections to MongoDB (Total: {self.total_saved_detections})")

            self.detection_batch.clear()

        except Exception as e:
            logger.error(f"❌ Failed to save batch detections to MongoDB: {e}")
            self.detection_batch.clear()  # Clear to prevent memory buildup
    
    def save_detections_list(self, camera_id: int, detections: List[Dict]):
        """Save multiple detections at once"""
        if not detections:
            return
            
        for detection in detections:
            self.save_detection_to_db(camera_id, detection)
    
    def get_detection_count(self) -> int:
        """Get total number of detections saved"""
        return self.total_saved_detections
    
    def get_batch_size(self) -> int:
        """Get current batch size"""
        return len(self.detection_batch)
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self.mongodb_client is not None

    def update_warp_id(self, global_id: int, warp_id: str) -> bool:
        """Update warp_id for an existing object by global_id"""
        if not self.mongodb_client:
            logger.error("❌ MongoDB not connected")
            return False

        try:
            # Update all documents with this global_id (across all cameras)
            result = self.mongodb_collection.update_many(
                {"global_id": global_id},
                {
                    "$set": {
                        "warp_id": warp_id,
                        "warp_id_linked_at": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"✅ Updated warp_id '{warp_id}' for global_id {global_id} ({result.modified_count} documents)")
                return True
            else:
                logger.warning(f"⚠️ No documents found with global_id {global_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error updating warp_id for global_id {global_id}: {e}")
            return False

    def find_object_by_warp_id(self, warp_id: str) -> Dict:
        """Find object by warp_id"""
        if not self.mongodb_client:
            logger.error("❌ MongoDB not connected")
            return None

        try:
            result = self.mongodb_collection.find_one(
                {"warp_id": warp_id},
                sort=[("last_seen", -1)]  # Get most recent if multiple
            )

            if result:
                logger.debug(f"🔍 Found object with warp_id '{warp_id}': global_id {result.get('global_id')}")
            else:
                logger.debug(f"🔍 No object found with warp_id '{warp_id}'")

            return result

        except Exception as e:
            logger.error(f"❌ Error finding object by warp_id '{warp_id}': {e}")
            return None

    def cleanup(self):
        """Cleanup database connection and save remaining detections (same as GPU script)"""
        if self.enable_mongodb and self.mongodb_client:
            # Save any remaining detections
            self.flush_detection_batch()
            
            # Close connection
            self.mongodb_client.close()
            logger.info(f"💾 MongoDB: Total {self.total_saved_detections} detections saved")
            logger.info("✅ Database handler cleanup complete")

# Default configuration (same as GPU script)
DEFAULT_CONFIG = {
    'mongodb_url': "mongodb://localhost:27017/",
    'database_name': "warehouse_tracking", 
    'collection_name': "detections",
    'batch_save_size': 10,
    'enable_mongodb': True
}

def create_database_handler(**kwargs) -> WarehouseDatabaseHandler:
    """Factory function to create database handler with default config"""
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return WarehouseDatabaseHandler(**config)

# Example usage:
if __name__ == "__main__":
    # Test the database handler
    db_handler = create_database_handler()
    
    # Test detection
    test_detection = {
        'global_id': 1001,
        'bbox': [100, 100, 200, 200],
        'confidence': 0.85,
        'area': 10000,
        'center': [150, 150],
        'physical_x_ft': 45.2,
        'physical_y_ft': 12.8,
        'coordinate_status': 'SUCCESS',
        'tracking_status': 'existing',
        'similarity_score': 0.75
    }
    
    print("Testing database handler...")
    db_handler.save_detection_to_db(camera_id=8, detection=test_detection)
    print(f"Batch size: {db_handler.get_batch_size()}")
    print(f"Total saved: {db_handler.get_detection_count()}")
    
    # Cleanup
    db_handler.cleanup()
    print("Test complete!")
