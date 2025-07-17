#!/usr/bin/env python3
"""
MongoDB Connection Test - Debug MongoDB connectivity issues
"""

import sys
import os

# Add cv directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cv'))

def test_direct_mongo():
    """Test direct MongoDB connection"""
    try:
        import pymongo
        
        print("🔍 Testing Direct MongoDB Connection...")
        
        # Test basic connection
        client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        # Test server info
        info = client.server_info()
        print(f"✅ MongoDB Version: {info['version']}")
        
        # Test database access
        db = client['warehouse_tracking']
        collection = db['tracked_objects']
        
        # Test count
        count = collection.count_documents({})
        print(f"✅ Total objects in DB: {count}")
        
        # Test recent objects
        recent_objects = list(collection.find().sort("last_seen", -1).limit(3))
        print(f"✅ Recent objects: {len(recent_objects)}")
        
        for i, obj in enumerate(recent_objects):
            real_center = obj.get('real_center', [None, None])
            source = obj.get('source', 'unknown')
            camera_id = obj.get('camera_id', 'unknown')
            print(f"   {i+1}. Camera {camera_id}, Coords: {real_center}, Source: {source}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct MongoDB test failed: {e}")
        return False

def test_config_mongo():
    """Test MongoDB connection using Config"""
    try:
        print("\n🔍 Testing Config-based MongoDB Connection...")
        
        from config import Config
        from database_handler import DatabaseHandler
        
        print(f"MongoDB URI from Config: {Config.MONGO_URI}")
        print(f"Database name: {Config.DATABASE_NAME}")
        print(f"Collection name: {Config.COLLECTION_NAME}")
        
        # Test database handler
        db_handler = DatabaseHandler()
        
        if db_handler.connected:
            print("✅ DatabaseHandler connected successfully")
            
            # Test health check
            health = db_handler.health_check()
            print(f"✅ Health check: {health.get('status', 'unknown')}")
            print(f"   Documents: {health.get('total_documents', 'unknown')}")
            print(f"   Collection size: {health.get('collection_size_mb', 'unknown')} MB")
            
        else:
            print("❌ DatabaseHandler connection failed")
            return False
            
        db_handler.close_connection()
        return True
        
    except Exception as e:
        print(f"❌ Config MongoDB test failed: {e}")
        return False

def test_backend_mongo():
    """Test MongoDB connection via backend API"""
    try:
        print("\n🔍 Testing Backend API MongoDB Connection...")
        
        import requests
        
        # Test API endpoints
        endpoints = [
            ('http://localhost:8000/', 'Root'),
            ('http://localhost:8000/api/tracking/objects', 'Objects'),
            ('http://localhost:8000/api/warehouse/config', 'Config'),
            ('http://localhost:8000/api/cameras/status', 'Cameras')
        ]
        
        for url, name in endpoints:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name} API: Working")
                    if 'objects' in url:
                        data = response.json()
                        print(f"   Objects returned: {len(data.get('objects', []))}")
                        print(f"   Source: {data.get('source', 'unknown')}")
                else:
                    print(f"❌ {name} API: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {name} API: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def diagnose_mongo_issue():
    """Run comprehensive MongoDB diagnostics"""
    print("🧪 MONGODB CONNECTION DIAGNOSTICS")
    print("=" * 50)
    
    # Test 1: Direct connection
    direct_ok = test_direct_mongo()
    
    # Test 2: Config-based connection
    config_ok = test_config_mongo()
    
    # Test 3: Backend API connection
    backend_ok = test_backend_mongo()
    
    print("\n📋 DIAGNOSTICS SUMMARY:")
    print(f"   Direct MongoDB:   {'✅ PASS' if direct_ok else '❌ FAIL'}")
    print(f"   Config MongoDB:   {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Backend API:      {'✅ PASS' if backend_ok else '❌ FAIL'}")
    
    # Overall assessment
    if all([direct_ok, config_ok, backend_ok]):
        print("\n🎯 RESULT: MongoDB connections are working properly")
        print("   The issue may be in the frontend or data flow")
    elif direct_ok and config_ok and not backend_ok:
        print("\n🎯 RESULT: MongoDB works, but backend API has issues")
        print("   Check if backend server is running properly")
    elif direct_ok and not config_ok:
        print("\n🎯 RESULT: MongoDB works, but Config/DatabaseHandler has issues")
        print("   Check Config.py settings and DatabaseHandler implementation")
    elif not direct_ok:
        print("\n🎯 RESULT: MongoDB is not accessible")
        print("   Check if MongoDB service is running: mongod")
    else:
        print("\n🎯 RESULT: Mixed results - need further investigation")

if __name__ == "__main__":
    diagnose_mongo_issue() 