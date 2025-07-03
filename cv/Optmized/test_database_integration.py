#!/usr/bin/env python3
"""
🔍 TEST DATABASE INTEGRATION
Verify that CV system, backend, and frontend are using the same MongoDB collections
"""

import requests
import time
from pymongo import MongoClient
from datetime import datetime

def test_mongodb_collections():
    """Test MongoDB collections used by different components"""
    
    print("🔍 TESTING MONGODB COLLECTIONS")
    print("=" * 60)
    
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["warehouse_tracking"]
        
        # Test connection
        client.server_info()
        print("✅ MongoDB connection successful")
        
        # Check collections
        collections = db.list_collection_names()
        print(f"📦 Available collections: {collections}")
        
        # Check specific collections
        collections_to_check = [
            ("tracked_objects", "Backend/Frontend reads from this"),
            ("detections", "Old CV system collection"),
            ("global_features", "New feature database collection")
        ]
        
        for collection_name, description in collections_to_check:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"   📊 {collection_name}: {count} documents - {description}")
            
            if count > 0:
                # Show sample document
                sample = collection.find_one()
                if sample:
                    keys = list(sample.keys())[:5]  # Show first 5 keys
                    print(f"      Sample keys: {keys}")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return False

def test_backend_api():
    """Test backend API endpoints"""
    
    print("\n🔍 TESTING BACKEND API")
    print("=" * 60)
    
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend API running - Status: {data.get('status')}")
            print(f"   Database: {data.get('database')}")
            print(f"   CV System: {data.get('cv_system')}")
        else:
            print(f"❌ Backend API error: {response.status_code}")
            return False
            
        # Test tracking objects endpoint
        response = requests.get("http://localhost:8000/api/tracking/objects", timeout=5)
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
            print(f"✅ Tracking API working - {len(objects)} objects")
            
            if objects:
                sample = objects[0]
                print(f"   Sample object: ID={sample.get('persistent_id')}")
                print(f"   Physical coords: {sample.get('real_center')}")
                print(f"   Camera ID: {sample.get('camera_id', 'N/A')}")
        else:
            print(f"❌ Tracking API error: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_cv_system_database_config():
    """Test CV system database configuration"""
    
    print("\n🔍 TESTING CV SYSTEM DATABASE CONFIG")
    print("=" * 60)
    
    try:
        # Import CV system config
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        from gpu_11camera_configurable import MONGODB_DATABASE, MONGODB_COLLECTION, MONGODB_URL
        
        print(f"📊 CV System MongoDB Config:")
        print(f"   URL: {MONGODB_URL}")
        print(f"   Database: {MONGODB_DATABASE}")
        print(f"   Collection: {MONGODB_COLLECTION}")
        
        # Check if it matches backend config
        backend_database = "warehouse_tracking"
        backend_collection = "tracked_objects"
        
        if MONGODB_DATABASE == backend_database:
            print(f"   ✅ Database matches backend: {MONGODB_DATABASE}")
        else:
            print(f"   ❌ Database mismatch! CV: {MONGODB_DATABASE}, Backend: {backend_database}")
            
        if MONGODB_COLLECTION == backend_collection:
            print(f"   ✅ Collection matches backend: {MONGODB_COLLECTION}")
        else:
            print(f"   ❌ Collection mismatch! CV: {MONGODB_COLLECTION}, Backend: {backend_collection}")
            
        return MONGODB_DATABASE == backend_database and MONGODB_COLLECTION == backend_collection
        
    except Exception as e:
        print(f"❌ CV system config test failed: {e}")
        return False

def test_data_flow():
    """Test complete data flow"""
    
    print("\n🔍 TESTING COMPLETE DATA FLOW")
    print("=" * 60)
    
    try:
        # Check if CV system is saving data
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["warehouse_tracking"]
        collection = db["tracked_objects"]
        
        # Get recent data (last 5 minutes)
        five_minutes_ago = datetime.now().timestamp() - 300
        
        recent_count = collection.count_documents({
            "$or": [
                {"timestamp": {"$gte": five_minutes_ago}},
                {"last_seen": {"$exists": True}}
            ]
        })
        
        print(f"📊 Recent detections (last 5 min): {recent_count}")
        
        if recent_count > 0:
            print("✅ CV system is actively saving data")
            
            # Check if data has physical coordinates
            coords_count = collection.count_documents({
                "physical_x_ft": {"$exists": True, "$ne": None}
            })
            print(f"📍 Objects with physical coordinates: {coords_count}")
            
            if coords_count > 0:
                print("✅ Physical coordinate transformation working")
                
                # Show sample coordinates
                sample = collection.find_one({
                    "physical_x_ft": {"$exists": True, "$ne": None}
                })
                if sample:
                    phys_x = sample.get('physical_x_ft')
                    phys_y = sample.get('physical_y_ft')
                    cam_id = sample.get('camera_id')
                    print(f"   Sample: Camera {cam_id} → ({phys_x}, {phys_y}) ft")
            else:
                print("⚠️ No objects with physical coordinates found")
        else:
            print("⚠️ No recent data found - CV system may not be running")
            
        return recent_count > 0
        
    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        return False

def main():
    """Run all database integration tests"""
    
    print("🔍 DATABASE INTEGRATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("MongoDB Collections", test_mongodb_collections),
        ("Backend API", test_backend_api),
        ("CV System Config", test_cv_system_database_config),
        ("Data Flow", test_data_flow)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 80)
    print("🎯 TEST RESULTS SUMMARY:")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Database integration is working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED! Check the issues above.")
    print("=" * 80)

if __name__ == "__main__":
    main()
