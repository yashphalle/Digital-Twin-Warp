#!/usr/bin/env python3
"""
Clean up old database files to start fresh with camera-prefixed Global IDs
"""

import os
import glob
from pymongo import MongoClient

def cleanup_databases():
    """Remove old database files and optionally clear MongoDB"""
    print("🧹 Cleaning up old databases for fresh start...")
    print("=" * 60)
    
    # 1. Remove old pickle database files
    db_files = glob.glob('cpu_camera_*_global_features.pkl')
    print(f"📁 Found {len(db_files)} database files to remove:")
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"   ✅ Removed: {db_file}")
            except Exception as e:
                print(f"   ❌ Failed to remove {db_file}: {e}")
        else:
            print(f"   ⚠️ Not found: {db_file}")
    
    print()
    
    # 2. Ask about MongoDB cleanup
    print("🗄️ MongoDB Cleanup Options:")
    print("   1. Keep MongoDB data (recommended for testing)")
    print("   2. Clear MongoDB detections collection (fresh start)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['warehouse_tracking']
            collection = db['detections']
            
            # Count existing documents
            count = collection.count_documents({})
            print(f"📊 Found {count} documents in MongoDB")
            
            if count > 0:
                confirm = input(f"⚠️ Really delete {count} documents? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    result = collection.delete_many({})
                    print(f"✅ Deleted {result.deleted_count} documents from MongoDB")
                else:
                    print("❌ MongoDB cleanup cancelled")
            else:
                print("✅ MongoDB collection is already empty")
                
        except Exception as e:
            print(f"❌ MongoDB cleanup failed: {e}")
    else:
        print("✅ Keeping existing MongoDB data")
    
    print()
    print("🎯 Next Steps:")
    print("   1. Start the CV system with new camera-prefixed IDs")
    print("   2. Each camera will now use unique ID ranges:")
    print("      - Camera 1: 1001, 1002, 1003...")
    print("      - Camera 2: 2001, 2002, 2003...")
    print("      - Camera 8: 8001, 8002, 8003...")
    print("      - Camera 11: 11001, 11002, 11003...")
    print("   3. No more Global ID conflicts!")
    print()
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    cleanup_databases()
