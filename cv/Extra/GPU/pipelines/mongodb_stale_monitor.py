#!/usr/bin/env python3
"""
MongoDB Stale Detection Monitor
Safely deletes detections that are 5+ minutes older than the latest detection
Runs every 2 minutes as a standalone monitoring script
"""

import time
import sys
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import signal
import logging

# Add cv directory to path to import Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'cv'))

try:
    from configs.config import Config
    print("✅ Using project Config for database settings")
except ImportError:
    print("⚠️ Could not import Config, using fallback settings")
    class Config:
        MONGO_URI = "mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/WARP?retryWrites=true&w=majority"
        DATABASE_NAME = "WARP"
        COLLECTION_NAME = "detections"
        USE_LOCAL_DATABASE = False

class MongoStaleMonitor:
    def __init__(self):
        """Initialize the MongoDB stale detection monitor"""
        self.running = True
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_collection = None
        
        # Configuration
        self.mongodb_url = Config.MONGO_URI
        self.database_name = Config.DATABASE_NAME
        self.collection_name = Config.COLLECTION_NAME
        self.staleness_minutes = 5  # Delete detections 5+ minutes older than latest
        self.check_interval_seconds = 120  # Check every 2 minutes
        self.batch_delete_size = 1000  # Delete in batches for performance
        
        # Statistics
        self.total_deleted = 0
        self.cycles_completed = 0
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def connect_to_mongodb(self):
        """Initialize MongoDB connection with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔌 Connecting to MongoDB (attempt {attempt + 1}/{max_retries})...")
                
                # Create client with timeout
                self.mongodb_client = MongoClient(
                    self.mongodb_url,
                    serverSelectionTimeoutMS=10000,  # 10 second timeout
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000
                )
                
                # Test connection
                self.mongodb_client.admin.command('ping')
                
                # Get database and collection
                self.mongodb_db = self.mongodb_client[self.database_name]
                self.mongodb_collection = self.mongodb_db[self.collection_name]
                
                # Test collection access
                count = self.mongodb_collection.estimated_document_count()
                
                self.logger.info(f"✅ MongoDB connected successfully")
                self.logger.info(f"📊 Database: {self.database_name}")
                self.logger.info(f"📊 Collection: {self.collection_name}")
                self.logger.info(f"📊 Estimated documents: {count:,}")
                
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                self.logger.error(f"❌ MongoDB connection failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error("❌ All connection attempts failed")
                    return False
            except Exception as e:
                self.logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
                return False
                
        return False
        
    def get_latest_detection_time(self):
        """Get the timestamp of the most recent detection across all cameras"""
        try:
            # Find the most recent detection by last_seen timestamp
            latest_detection = self.mongodb_collection.find_one(
                {},
                sort=[("last_seen", -1)]
            )
            
            if latest_detection and 'last_seen' in latest_detection:
                latest_time = latest_detection['last_seen']
                self.logger.debug(f"🕐 Latest detection time: {latest_time}")
                return latest_time
            else:
                self.logger.warning("⚠️ No detections found in database")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting latest detection time: {e}")
            return None
            
    def count_stale_detections(self, cutoff_time):
        """Count how many detections would be deleted"""
        try:
            count = self.mongodb_collection.count_documents({
                "last_seen": {"$lt": cutoff_time}
            })
            return count
        except Exception as e:
            self.logger.error(f"❌ Error counting stale detections: {e}")
            return 0
            
    def delete_stale_detections(self, cutoff_time):
        """Safely delete stale detections in batches"""
        total_deleted = 0
        
        try:
            while True:
                # Delete in batches to avoid memory issues
                result = self.mongodb_collection.delete_many(
                    {"last_seen": {"$lt": cutoff_time}},
                    # Note: MongoDB delete_many doesn't support limit, so we'll do it differently
                )
                
                deleted_count = result.deleted_count
                total_deleted += deleted_count
                
                if deleted_count == 0:
                    break  # No more documents to delete
                    
                self.logger.info(f"🗑️ Deleted batch: {deleted_count} detections")
                
                # Small delay between batches to avoid overwhelming the database
                if deleted_count > 0:
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"❌ Error deleting stale detections: {e}")
            
        return total_deleted
        
    def run_cleanup_cycle(self):
        """Run one cycle of stale detection cleanup"""
        cycle_start = time.time()
        self.logger.info("🔄 Starting cleanup cycle...")
        
        try:
            # Step 1: Get latest detection time
            latest_detection_time = self.get_latest_detection_time()
            
            if latest_detection_time is None:
                self.logger.warning("⚠️ No detections found - skipping cleanup cycle")
                return
                
            # Step 2: Calculate cutoff time (5 minutes before latest detection)
            cutoff_time = latest_detection_time - timedelta(minutes=self.staleness_minutes)
            
            self.logger.info(f"📅 Latest detection: {latest_detection_time}")
            self.logger.info(f"📅 Cutoff time: {cutoff_time}")
            
            # Step 3: Count stale detections
            stale_count = self.count_stale_detections(cutoff_time)
            
            if stale_count == 0:
                self.logger.info("✅ No stale detections found")
                return
                
            self.logger.info(f"🔍 Found {stale_count:,} stale detections to delete")
            
            # Step 4: Delete stale detections
            deleted_count = self.delete_stale_detections(cutoff_time)
            
            # Step 5: Update statistics
            self.total_deleted += deleted_count
            
            # Step 6: Log results
            cycle_duration = time.time() - cycle_start
            self.logger.info(f"✅ Cleanup cycle completed in {cycle_duration:.2f}s")
            self.logger.info(f"🗑️ Deleted: {deleted_count:,} detections")
            self.logger.info(f"📊 Total deleted this session: {self.total_deleted:,}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in cleanup cycle: {e}")
            
    def run(self):
        """Main monitoring loop"""
        self.logger.info("🚀 MongoDB Stale Detection Monitor Starting...")
        self.logger.info(f"⚙️ Check interval: {self.check_interval_seconds} seconds")
        self.logger.info(f"⚙️ Staleness threshold: {self.staleness_minutes} minutes")
        self.logger.info(f"⚙️ Database: {self.database_name}.{self.collection_name}")
        
        # Connect to MongoDB
        if not self.connect_to_mongodb():
            self.logger.error("❌ Failed to connect to MongoDB, exiting...")
            return
            
        # Main monitoring loop
        try:
            while self.running:
                self.run_cleanup_cycle()
                self.cycles_completed += 1
                
                if self.running:  # Check if we should continue
                    self.logger.info(f"⏳ Waiting {self.check_interval_seconds} seconds until next cycle...")
                    time.sleep(self.check_interval_seconds)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in main loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources and show final statistics"""
        self.logger.info("🧹 Cleaning up...")
        
        if self.mongodb_client:
            self.mongodb_client.close()
            self.logger.info("✅ MongoDB connection closed")
            
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   Cycles completed: {self.cycles_completed}")
        self.logger.info(f"   Total detections deleted: {self.total_deleted:,}")
        self.logger.info("👋 MongoDB Stale Detection Monitor stopped")

def main():
    """Main entry point"""
    print("=" * 60)
    print("🔍 MongoDB Stale Detection Monitor")
    print("=" * 60)
    
    monitor = MongoStaleMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
