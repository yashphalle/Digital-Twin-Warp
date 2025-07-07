#!/usr/bin/env python3
"""
Inspect Camera 4 Database State
Check SIFT feature database and MongoDB for Camera 4 objects
"""

import os
import pickle
import sys
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np

def inspect_sift_database():
    """Inspect SIFT feature database for Camera 4"""
    print("🔍 SIFT Feature Database Inspection")
    print("=" * 50)
    
    # Look for Camera 4 database files in data folder
    data_folder = ""
    possible_files = [
        os.path.join(data_folder, "cpu_camera_4_global_features.pkl"),
        "cpu_camera_4_global_features.pkl",
        os.path.join(data_folder, "camera_4_features.pkl"),
        "camera_4_features.pkl"
    ]
    
    sift_db_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            sift_db_file = file_path
            break
    
    if not sift_db_file:
        print("❌ No SIFT database file found for Camera 4")
        print(f"   Searched in: {possible_files}")
        return None
    
    try:
        print(f"📁 Found SIFT database: {sift_db_file}")
        
        # Get file size
        file_size = os.path.getsize(sift_db_file)
        print(f"📊 File size: {file_size / 1024:.1f} KB ({file_size:,} bytes)")
        
        # Load and inspect the database
        with open(sift_db_file, 'rb') as f:
            sift_data = pickle.load(f)
        
        print(f"📦 Database type: {type(sift_data)}")
        
        if isinstance(sift_data, dict):
            print(f"🎯 Total top-level keys in SIFT database: {len(sift_data)}")
            print(f"📋 Top-level keys: {list(sift_data.keys())}")

            # Handle different possible structures
            total_objects = 0
            feature_counts = []
            times_seen_list = []
            recent_objects = 0
            old_objects = 0
            now = datetime.now()

            # Check if this is a nested structure or direct object storage
            for key, value in sift_data.items():
                print(f"\n🔍 Analyzing key '{key}' (type: {type(value)})")

                if isinstance(value, dict):
                    # This could be either a single object or a collection of objects
                    if 'features' in value:
                        # This is a single object
                        total_objects += 1
                        print(f"   📦 Single object with keys: {list(value.keys())}")

                        # Analyze features
                        if value['features'] is not None and isinstance(value['features'], np.ndarray):
                            feature_counts.append(len(value['features']))
                            print(f"   🔢 Features: {len(value['features'])}")

                        # Analyze times seen
                        if 'times_seen' in value:
                            times_seen_list.append(value['times_seen'])
                            print(f"   👁️  Times seen: {value['times_seen']}")

                        # Check age
                        if 'last_seen' in value:
                            try:
                                last_seen = datetime.fromisoformat(value['last_seen'])
                                age = now - last_seen
                                if age.total_seconds() < 3600:
                                    recent_objects += 1
                                else:
                                    old_objects += 1
                                print(f"   ⏰ Last seen: {value['last_seen']}")
                            except Exception as e:
                                print(f"   ⚠️  Could not parse last_seen: {e}")

                    else:
                        # This might be a collection of objects
                        print(f"   📁 Collection with {len(value)} items")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict) and 'features' in sub_value:
                                total_objects += 1

                                if sub_value['features'] is not None and isinstance(sub_value['features'], np.ndarray):
                                    feature_counts.append(len(sub_value['features']))

                                if 'times_seen' in sub_value:
                                    times_seen_list.append(sub_value['times_seen'])

                                if 'last_seen' in sub_value:
                                    try:
                                        last_seen = datetime.fromisoformat(sub_value['last_seen'])
                                        age = now - last_seen
                                        if age.total_seconds() < 3600:
                                            recent_objects += 1
                                        else:
                                            old_objects += 1
                                    except:
                                        pass

                else:
                    print(f"   ⚠️  Non-dict value: {type(value)}")

            print(f"\n📊 SIFT Database Analysis:")
            print(f"🎯 Total objects found: {total_objects}")

            if feature_counts:
                print(f"🔢 Features per object:")
                print(f"   Average: {np.mean(feature_counts):.1f}")
                print(f"   Range: {min(feature_counts)} - {max(feature_counts)}")
                print(f"   Total features: {sum(feature_counts):,}")

            if times_seen_list:
                print(f"👁️  Object tracking stats:")
                print(f"   Average times seen: {np.mean(times_seen_list):.1f}")
                print(f"   Range: {min(times_seen_list)} - {max(times_seen_list)}")

            print(f"⏰ Object age distribution:")
            print(f"   Recent (< 1 hour): {recent_objects}")
            print(f"   Old (> 1 hour): {old_objects}")

            return total_objects
        
        return sift_data
        
    except Exception as e:
        print(f"❌ Error reading SIFT database: {e}")
        return None

def inspect_mongodb():
    """Inspect MongoDB for Camera 4 objects"""
    print("\n🗄️  MongoDB Database Inspection")
    print("=" * 50)
    
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_tracking']
        collection = db['detections']
        
        print("✅ Connected to MongoDB")
        
        # Total documents
        total_docs = collection.count_documents({})
        print(f"📊 Total documents in database: {total_docs:,}")
        
        # Camera 4 specific documents
        camera4_docs = collection.count_documents({"camera_id": 4})
        print(f"🎥 Camera 4 documents: {camera4_docs:,}")
        
        if camera4_docs > 0:
            # Recent Camera 4 activity
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_camera4 = collection.count_documents({
                "camera_id": 4,
                "timestamp": {"$gte": one_hour_ago}
            })
            print(f"⏰ Camera 4 recent (1 hour): {recent_camera4:,}")
            
            # Unique global IDs for Camera 4
            try:
                unique_ids = collection.distinct("global_id", {"camera_id": 4})
                print(f"🆔 Unique global IDs for Camera 4: {len(unique_ids)}")
                
                if unique_ids:
                    camera4_ids = [id for id in unique_ids if isinstance(id, int) and 4000 <= id <= 4999]
                    print(f"🎯 Camera 4 prefixed IDs (4000-4999): {len(camera4_ids)}")
                    
                    if camera4_ids:
                        print(f"   ID range: {min(camera4_ids)} - {max(camera4_ids)}")
            except Exception as e:
                print(f"⚠️  Could not get unique IDs: {e}")
            
            # Sample recent document
            try:
                sample_doc = collection.find_one(
                    {"camera_id": 4}, 
                    sort=[("timestamp", -1)]
                )
                if sample_doc:
                    print(f"📋 Sample document keys: {list(sample_doc.keys())}")
                    if 'global_id' in sample_doc:
                        print(f"   Sample global_id: {sample_doc['global_id']}")
            except Exception as e:
                print(f"⚠️  Could not get sample document: {e}")
        
        # Database size
        try:
            stats = db.command("collStats", "detections")
            size_mb = stats.get('size', 0) / (1024 * 1024)
            print(f"💾 Collection size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"⚠️  Could not get collection size: {e}")
        
        client.close()
        return camera4_docs
        
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return 0

def main():
    """Main inspection function"""
    print("🔍 Camera 4 Database Inspection")
    print("=" * 60)
    print(f"📅 Inspection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # List data folder contents
    data_folder = "data"
    if os.path.exists(data_folder):
        print(f"📂 Data folder contents:")
        for file in os.listdir(data_folder):
            if "camera" in file.lower() or "4" in file:
                file_path = os.path.join(data_folder, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"   {file} ({size:,} bytes)")
    else:
        print(f"⚠️  Data folder not found: {data_folder}")
    
    print()
    
    # Inspect SIFT database
    sift_data = inspect_sift_database()
    
    # Inspect MongoDB
    mongo_count = inspect_mongodb()
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 30)

    sift_count = sift_data if isinstance(sift_data, int) else (len(sift_data) if sift_data else 0)
    print(f"🎯 SIFT objects: {sift_count}")
    print(f"🗄️  MongoDB objects: {mongo_count}")
    print(f"📊 Total tracked objects: {sift_count + mongo_count}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 30)

    if sift_count > 100:
        print(f"⚠️  SIFT database is large ({sift_count} objects)")
        print("   Consider setting MAX_OBJECTS_IN_DATABASE = 50")
    elif sift_count > 50:
        print(f"⚠️  SIFT database is moderate ({sift_count} objects)")
        print("   Consider setting MAX_OBJECTS_IN_DATABASE = 30")
    else:
        print(f"✅ SIFT database size is manageable ({sift_count} objects)")
        print("   Can safely re-enable SIFT with current size")

    if mongo_count > 1000:
        print(f"⚠️  MongoDB has many documents ({mongo_count})")
        print("   Consider periodic cleanup of old documents")

if __name__ == "__main__":
    main()
