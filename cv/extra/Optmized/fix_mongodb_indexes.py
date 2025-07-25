#!/usr/bin/env python3
"""
🔧 FIX MONGODB INDEXES
Fix MongoDB index issues causing duplicate key errors
"""

from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_mongodb_indexes():
    """Fix MongoDB indexes to prevent duplicate key errors"""
    
    print("🔧 FIXING MONGODB INDEXES")
    print("=" * 60)
    
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["warehouse_tracking"]
        collection = db["tracked_objects"]
        
        # Test connection
        client.server_info()
        print("✅ MongoDB connection successful")
        
        # Check current indexes
        indexes = list(collection.list_indexes())
        print(f"📊 Current indexes: {len(indexes)}")
        
        for index in indexes:
            index_name = index.get('name')
            index_key = index.get('key')
            unique = index.get('unique', False)
            print(f"   {index_name}: {index_key} {'(UNIQUE)' if unique else ''}")
        
        # Drop problematic persistent_id index if it exists
        problematic_indexes = ['persistent_id_1']
        
        for index_name in problematic_indexes:
            try:
                collection.drop_index(index_name)
                print(f"✅ Dropped problematic index: {index_name}")
            except Exception as e:
                if "index not found" in str(e).lower():
                    print(f"ℹ️ Index {index_name} doesn't exist (OK)")
                else:
                    print(f"⚠️ Failed to drop index {index_name}: {e}")
        
        # Create proper indexes
        print("\n🔧 Creating proper indexes...")
        
        # Index for camera_id and timestamp (for queries)
        try:
            collection.create_index([("camera_id", 1), ("timestamp", -1)])
            print("✅ Created index: camera_id + timestamp")
        except Exception as e:
            print(f"ℹ️ Index already exists: camera_id + timestamp")
        
        # Index for global_id (non-unique, allows nulls)
        try:
            collection.create_index([("global_id", 1)], sparse=True)
            print("✅ Created index: global_id (sparse)")
        except Exception as e:
            print(f"ℹ️ Index already exists: global_id")
        
        # Index for physical coordinates (for spatial queries)
        try:
            collection.create_index([("physical_x_ft", 1), ("physical_y_ft", 1)], sparse=True)
            print("✅ Created index: physical coordinates")
        except Exception as e:
            print(f"ℹ️ Index already exists: physical coordinates")
        
        # Check final indexes
        print("\n📊 Final indexes:")
        final_indexes = list(collection.list_indexes())
        for index in final_indexes:
            index_name = index.get('name')
            index_key = index.get('key')
            unique = index.get('unique', False)
            sparse = index.get('sparse', False)
            flags = []
            if unique:
                flags.append("UNIQUE")
            if sparse:
                flags.append("SPARSE")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            print(f"   {index_name}: {index_key}{flag_str}")
        
        print("\n✅ MongoDB indexes fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix MongoDB indexes: {e}")
        return False

def clean_duplicate_data():
    """Clean up any duplicate or problematic data"""
    
    print("\n🧹 CLEANING DUPLICATE DATA")
    print("=" * 60)
    
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["warehouse_tracking"]
        collection = db["tracked_objects"]
        
        # Remove documents with null persistent_id that might cause conflicts
        result = collection.delete_many({"persistent_id": None})
        print(f"🗑️ Removed {result.deleted_count} documents with null persistent_id")
        
        # Remove documents with global_id = -1 (invalid detections)
        result = collection.delete_many({"global_id": -1})
        print(f"🗑️ Removed {result.deleted_count} documents with invalid global_id")
        
        # Count remaining documents
        total_count = collection.count_documents({})
        print(f"📊 Remaining documents: {total_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to clean duplicate data: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 MONGODB DATABASE REPAIR TOOL")
    print("=" * 80)
    
    # Fix indexes
    index_success = fix_mongodb_indexes()
    
    # Clean duplicate data
    clean_success = clean_duplicate_data()
    
    print("\n" + "=" * 80)
    if index_success and clean_success:
        print("🎉 DATABASE REPAIR COMPLETE!")
        print("✅ MongoDB indexes fixed")
        print("✅ Duplicate data cleaned")
        print("🚀 CV system should now work without duplicate key errors")
    else:
        print("⚠️ SOME REPAIRS FAILED!")
        print("❌ Check the errors above and try again")
    print("=" * 80)

if __name__ == "__main__":
    main()
