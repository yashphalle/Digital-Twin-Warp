"""
MongoDB Database Handler
Manages all database operations for tracked objects
Adapted from the working database integration
"""

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from config import Config
import logging

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DatabaseHandler:
    """MongoDB database handler for warehouse tracking system"""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        self.connection_string = connection_string or Config.MONGO_URI
        self.database_name = Config.DATABASE_NAME
        self.collection_name = Config.COLLECTION_NAME
        
        # Connection objects
        self.client = None
        self.database = None
        self.collection = None
        self.connected = False
        
        # Statistics
        self.operations_count = {
            'inserts': 0,
            'updates': 0,
            'queries': 0,
            'deletes': 0
        }
        
        # Initialize connection
        self._connect()
        
    def _connect(self):
        """Establish database connection"""
        try:
            logger.info(f"Connecting to MongoDB: {self.connection_string}")
            
            # Create client with timeout
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=Config.CONNECTION_TIMEOUT
            )
            
            # Test connection
            self.client.server_info()
            logger.info("MongoDB connection successful")
            
            # Get database and collection
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            # Create indexes if configured
            if Config.AUTO_CREATE_INDEXES:
                self._create_indexes()
            
            self.connected = True
            logger.info(f"Database '{self.database_name}' and collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.connected = False
            raise
    
    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Index on persistent_id for fast lookups
            self.collection.create_index("persistent_id", unique=True)
            
            # Index on timestamps for time-based queries
            self.collection.create_index("last_seen")
            self.collection.create_index("first_seen")
            
            # Index on frame_number for frame-based queries
            self.collection.create_index("frame_number")
            
            # Compound index for common queries
            self.collection.create_index([
                ("persistent_id", ASCENDING),
                ("last_seen", DESCENDING)
            ])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_single_object(self, object_data: Dict[str, Any]) -> Optional[str]:
        """Store a single tracked object"""
        if not self.connected:
            logger.error("Database not connected")
            return None
        
        try:
            # Add metadata
            object_data['created_at'] = datetime.now()
            object_data['updated_at'] = datetime.now()
            
            # Insert document
            result = self.collection.insert_one(object_data)
            self.operations_count['inserts'] += 1
            
            logger.debug(f"Stored object with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except pymongo.errors.DuplicateKeyError:
            logger.warning(f"Object with persistent_id {object_data.get('persistent_id')} already exists")
            return None
        except Exception as e:
            logger.error(f"Error storing object: {e}")
            return None
    
    def store_multiple_objects(self, objects_data: List[Dict[str, Any]]) -> List[str]:
        """Store multiple tracked objects efficiently"""
        if not self.connected or not objects_data:
            return []
        
        try:
            # Add metadata to all objects
            timestamp = datetime.now()
            for obj in objects_data:
                obj['created_at'] = timestamp
                obj['updated_at'] = timestamp
            
            # Bulk insert
            result = self.collection.insert_many(objects_data, ordered=False)
            self.operations_count['inserts'] += len(result.inserted_ids)
            
            logger.debug(f"Stored {len(result.inserted_ids)} objects")
            return [str(obj_id) for obj_id in result.inserted_ids]
            
        except pymongo.errors.BulkWriteError as e:
            # Handle partial success in bulk operations
            successful_inserts = len(e.details.get('writeErrors', []))
            logger.warning(f"Bulk insert partial success: {successful_inserts} errors")
            return []
        except Exception as e:
            logger.error(f"Error storing multiple objects: {e}")
            return []
    
    def find_object_by_persistent_id(self, persistent_id: int) -> Optional[Dict[str, Any]]:
        """Find object by persistent tracking ID"""
        if not self.connected:
            return None
        
        try:
            result = self.collection.find_one({"persistent_id": persistent_id})
            self.operations_count['queries'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error finding object by persistent_id {persistent_id}: {e}")
            return None
    
    def update_object_by_persistent_id(self, persistent_id: int, update_data: Dict[str, Any]) -> bool:
        """Update object by persistent tracking ID"""
        if not self.connected:
            return False
        
        try:
            # Add update timestamp
            update_data['updated_at'] = datetime.now()
            
            # Update document
            result = self.collection.update_one(
                {"persistent_id": persistent_id},
                {"$set": update_data}
            )
            
            self.operations_count['updates'] += 1
            
            if result.modified_count > 0:
                logger.debug(f"Updated object with persistent_id: {persistent_id}")
                return True
            else:
                logger.warning(f"No object found with persistent_id: {persistent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating object {persistent_id}: {e}")
            return False
    
    def upsert_object(self, object_data: Dict[str, Any]) -> bool:
        """Insert or update object based on persistent_id"""
        if not self.connected:
            return False
        
        try:
            persistent_id = object_data.get('persistent_id')
            if persistent_id is None:
                logger.error("Object data missing persistent_id for upsert")
                return False
            
            # Add/update timestamps
            timestamp = datetime.now()
            object_data['updated_at'] = timestamp
            
            # Upsert operation
            result = self.collection.update_one(
                {"persistent_id": persistent_id},
                {
                    "$set": object_data,
                    "$setOnInsert": {"created_at": timestamp}
                },
                upsert=True
            )
            
            if result.upserted_id:
                self.operations_count['inserts'] += 1
                logger.debug(f"Inserted new object with persistent_id: {persistent_id}")
            else:
                self.operations_count['updates'] += 1
                logger.debug(f"Updated existing object with persistent_id: {persistent_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error upserting object: {e}")
            return False
    
    def get_objects_by_timerange(self, start_time: datetime, end_time: datetime = None) -> List[Dict[str, Any]]:
        """Get objects within a time range"""
        if not self.connected:
            return []
        
        try:
            end_time = end_time or datetime.now()
            
            query = {
                "last_seen": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }
            
            results = list(self.collection.find(query).sort("last_seen", DESCENDING))
            self.operations_count['queries'] += 1
            
            logger.debug(f"Found {len(results)} objects in time range")
            return results
            
        except Exception as e:
            logger.error(f"Error querying objects by time range: {e}")
            return []
    
    def get_active_objects(self, max_age_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recently active objects"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        return self.get_objects_by_timerange(cutoff_time)
    
    def get_object_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if not self.connected:
            return {}
        
        try:
            # Basic counts
            total_objects = self.collection.count_documents({})
            
            # Recent activity (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_objects = self.collection.count_documents({
                "last_seen": {"$gte": one_hour_ago}
            })
            
            # Objects with persistent IDs
            tracked_objects = self.collection.count_documents({
                "persistent_id": {"$exists": True}
            })
            
            # Unique persistent IDs - fix the cursor issue
            try:
                distinct_ids = list(self.collection.distinct("persistent_id"))
                unique_persistent_ids = len(distinct_ids)
            except Exception as e:
                logger.warning(f"Error getting distinct persistent IDs: {e}")
                unique_persistent_ids = 0
            
            # Average tracking duration for active objects
            try:
                pipeline = [
                    {"$match": {"first_seen": {"$exists": True}, "last_seen": {"$exists": True}}},
                    {"$project": {
                        "duration": {"$subtract": ["$last_seen", "$first_seen"]}
                    }},
                    {"$group": {
                        "_id": None,
                        "avg_duration": {"$avg": "$duration"},
                        "max_duration": {"$max": "$duration"},
                        "min_duration": {"$min": "$duration"}
                    }}
                ]
                
                duration_stats = list(self.collection.aggregate(pipeline))
            except Exception as e:
                logger.warning(f"Error getting duration stats: {e}")
                duration_stats = []
            
            stats = {
                'total_detections': total_objects,
                'unique_objects': unique_persistent_ids,
                'tracked_objects': tracked_objects,
                'recent_detections': recent_objects,
                'database_operations': self.operations_count.copy(),
                'collection_size_mb': self._get_collection_size_mb(),
                'indexes_count': len(list(self.collection.list_indexes()))
            }
            
            # Add duration statistics if available
            if duration_stats and len(duration_stats) > 0:
                duration_data = duration_stats[0]
                stats.update({
                    'avg_tracking_duration_ms': duration_data.get('avg_duration', 0),
                    'max_tracking_duration_ms': duration_data.get('max_duration', 0),
                    'min_tracking_duration_ms': duration_data.get('min_duration', 0)
                })
            
            self.operations_count['queries'] += 1
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'error': str(e),
                'total_detections': 0,
                'unique_objects': 0,
                'tracked_objects': 0,
                'recent_detections': 0
            }
    
    def _get_collection_size_mb(self) -> float:
        """Get collection size in MB"""
        try:
            stats = self.database.command("collStats", self.collection_name)
            return stats.get('size', 0) / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def clear_old_objects(self, hours_old: int = 24) -> int:
        """Clear objects older than specified hours"""
        if not self.connected:
            return 0
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_old)
            
            result = self.collection.delete_many({
                "last_seen": {"$lt": cutoff_time}
            })
            
            self.operations_count['deletes'] += result.deleted_count
            logger.info(f"Cleared {result.deleted_count} old objects")
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing old objects: {e}")
            return 0
    
    def clear_duplicate_objects(self) -> int:
        """Remove duplicate objects based on persistent_id, keeping the most recent"""
        if not self.connected:
            return 0
        
        try:
            # Find duplicates
            pipeline = [
                {"$group": {
                    "_id": "$persistent_id",
                    "docs": {"$push": "$_id"},
                    "count": {"$sum": 1}
                }},
                {"$match": {"count": {"$gt": 1}}}
            ]
            
            duplicates = list(self.collection.aggregate(pipeline))
            deleted_count = 0
            
            for duplicate in duplicates:
                # Keep the first document, delete the rest
                docs_to_delete = duplicate['docs'][1:]
                result = self.collection.delete_many({
                    "_id": {"$in": docs_to_delete}
                })
                deleted_count += result.deleted_count
            
            self.operations_count['deletes'] += deleted_count
            logger.info(f"Removed {deleted_count} duplicate objects")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing duplicates: {e}")
            return 0
    
    def get_object_history(self, persistent_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history of a specific object"""
        if not self.connected:
            return []
        
        try:
            results = list(
                self.collection.find({"persistent_id": persistent_id})
                .sort("last_seen", DESCENDING)
                .limit(limit)
            )
            
            self.operations_count['queries'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error getting object history: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        health_status = {
            'connected': self.connected,
            'timestamp': datetime.now(),
            'database': self.database_name,
            'collection': self.collection_name
        }
        
        if self.connected:
            try:
                # Test basic operations
                test_start = datetime.now()
                self.collection.find_one()
                query_time = (datetime.now() - test_start).total_seconds()
                
                health_status.update({
                    'query_response_time_ms': query_time * 1000,
                    'server_status': 'healthy',
                    'operations_count': self.operations_count.copy()
                })
                
            except Exception as e:
                health_status.update({
                    'server_status': 'error',
                    'error': str(e)
                })
        
        return health_status
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()

# Test function
def test_database_handler():
    """Test database handler functionality"""
    print("Testing DatabaseHandler...")
    
    try:
        # Initialize database handler
        db = DatabaseHandler()
        
        # Test health check
        health = db.health_check()
        print(f"Health check: {health}")
        
        # Test storing an object
        test_object = {
            'persistent_id': 999,
            'center': (320, 240),
            'bbox': (300, 220, 340, 260),
            'confidence': 0.85,
            'age_seconds': 0,
            'times_seen': 1
        }
        
        result = db.store_single_object(test_object)
        print(f"Store result: {result}")
        
        # Test finding the object
        found_object = db.find_object_by_persistent_id(999)
        print(f"Found object: {found_object is not None}")
        
        # Test updating the object
        update_data = {
            'times_seen': 2,
            'age_seconds': 5.0
        }
        update_result = db.update_object_by_persistent_id(999, update_data)
        print(f"Update result: {update_result}")
        
        # Test statistics
        stats = db.get_object_statistics()
        print(f"Statistics: {stats}")
        
        # Cleanup test object
        db.collection.delete_one({'persistent_id': 999})
        
        db.close_connection()
        print("Database test completed successfully!")
        
    except Exception as e:
        print(f"Database test failed: {e}")

if __name__ == "__main__":
    test_database_handler()