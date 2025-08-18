# cv/GPU/pipelines/simple_database_writer.py

"""
Simple synchronous database writer - no threads, no async
Just basic MongoDB writes to isolate the issue
"""

import time
import logging
from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from GPU.configs.config import Config

logger = logging.getLogger(__name__)

class SimpleDatabaseWriter:
    """
    Simplified database writer - no threads, just direct writes
    """
    
    def __init__(self):
        """Initialize simple database connection"""
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.buffer = []
        self.stats = {
            'total_written': 0,
            'total_errors': 0,
            'connection_status': 'disconnected'
        }
        
    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            logger.info(f"Connecting to MongoDB at {Config.MONGO_URI}...")
            self.mongo_client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            self.db = self.mongo_client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            
            self.stats['connection_status'] = 'connected'
            logger.info(f"✅ Connected to MongoDB: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.stats['connection_status'] = f'failed: {e}'
            return False
    
    def add_detection(self, detection: Dict, camera_id: int):
        """Add detection to buffer"""
        try:
            # Create document
            doc = {
                'camera_id': camera_id,
                'global_id': detection.get('track_id'),
                'timestamp': datetime.utcnow(),
                'bbox': detection.get('bbox', []),
                'confidence': float(detection.get('confidence', 0.0)),
                'class': detection.get('class', 'unknown'),
                'track_age': detection.get('track_age', 0)
            }
            
            self.buffer.append(doc)
            
            # Write immediately if buffer gets large
            if len(self.buffer) >= 100:
                self.flush()
                
        except Exception as e:
            logger.error(f"Error adding detection: {e}")
            self.stats['total_errors'] += 1
    
    def flush(self):
        """Write buffer to database"""
        if not self.buffer or not self.collection:
            return
        
        try:
            # Simple insert
            result = self.collection.insert_many(self.buffer, ordered=False)
            self.stats['total_written'] += len(result.inserted_ids)
            logger.info(f"✅ Wrote {len(self.buffer)} detections to database")
            self.buffer.clear()
            
        except Exception as e:
            logger.error(f"❌ Database write failed: {e}")
            self.stats['total_errors'] += len(self.buffer)
            self.buffer.clear()
    
    def close(self):
        """Close database connection"""
        # Final flush
        if self.buffer:
            self.flush()
            
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Database connection closed")
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer)
        }