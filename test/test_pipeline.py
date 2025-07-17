#!/usr/bin/env python3
"""
Test Pipeline Script - Verify full data flow from Camera 8 to Frontend
Tests: Smart Queue → Database → Backend API → Frontend Display
"""

import requests
import time
import json
from datetime import datetime

def test_database_connection():
    """Test if MongoDB is accessible"""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        
        # Check if tracking collection exists
        db = client["warehouse_tracking"]
        collection = db["tracked_objects"]
        count = collection.count_documents({})
        
        print(f"✅ MongoDB connected - {count} objects in database")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_backend_api():
    """Test if backend API is running and accessible"""
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API is running")
            
            # Test tracking objects endpoint
            response = requests.get("http://localhost:8000/api/tracking/objects", timeout=5)
            if response.status_code == 200:
                data = response.json()
                objects = data.get('objects', [])
                print(f"✅ Tracking API working - {len(objects)} objects returned")
                
                # Show sample object if available
                if objects:
                    sample = objects[0]
                    real_center = sample.get('real_center', [None, None])
                    print(f"   Sample object: ID={sample.get('persistent_id')}, "
                          f"Coords=({real_center[0]}, {real_center[1]}), "
                          f"Confidence={sample.get('confidence', 0):.2f}")
                
            # Test camera status
            response = requests.get("http://localhost:8000/api/cameras/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                cameras = data.get('cameras', [])
                print(f"✅ Camera API working - {len(cameras)} cameras configured")
                
                # Check Camera 8 status
                camera8 = next((c for c in cameras if c['camera_id'] == 8), None)
                if camera8:
                    print(f"   Camera 8 status: {camera8.get('status', 'unknown')}")
            
            return True
        else:
            print(f"❌ Backend API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_frontend_accessibility():
    """Test if frontend is accessible"""
    try:
        # Test if frontend dev server is running (common ports)
        for port in [3000, 5173, 3001]:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=3)
                if response.status_code == 200:
                    print(f"✅ Frontend accessible on port {port}")
                    return True
            except:
                continue
        
        print("⚠️  Frontend not accessible - may need to start dev server")
        return False
        
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def check_smart_queue_integration():
    """Check if smart queue script can access required components"""
    try:
        # Check if CV modules are available
        import sys
        import os
        
        # Add CV directory to path
        cv_path = os.path.join(os.path.dirname(__file__), 'cv')
        if cv_path not in sys.path:
            sys.path.append(cv_path)
        
        # Test imports
        from cv.config import Config
        from cv.database_handler import DatabaseHandler
        from cv.detector_tracker import DetectorTracker
        
        print("✅ Smart queue dependencies available")
        
        # Check Camera 8 configuration
        camera_url = Config.RTSP_CAMERA_URLS.get(8)
        active_cameras = Config.ACTIVE_CAMERAS
        
        print(f"   Camera 8 URL: {camera_url}")
        print(f"   Active cameras: {active_cameras}")
        print(f"   Camera 8 active: {8 in active_cameras}")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart queue integration check failed: {e}")
        return False

def monitor_data_flow():
    """Monitor data flow between components"""
    print("\n🔍 Monitoring data flow...")
    
    # Check database for recent objects
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["warehouse_tracking"]
        collection = db["tracked_objects"]
        
        # Get recent objects (last 5 minutes)
        from datetime import datetime, timedelta
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        
        recent_objects = list(collection.find({
            "$or": [
                {"last_seen": {"$gte": five_minutes_ago}},
                {"updated_at": {"$gte": five_minutes_ago}}
            ]
        }).sort("last_seen", -1).limit(10))
        
        print(f"📊 Recent objects in database: {len(recent_objects)}")
        
        for i, obj in enumerate(recent_objects):
            real_center = obj.get('real_center', [None, None])
            source = obj.get('source', 'unknown')
            last_seen = obj.get('last_seen', 'unknown')
            print(f"   {i+1}. ID={obj.get('persistent_id')}, "
                  f"Coords=({real_center[0]}, {real_center[1]}), "
                  f"Source={source}, Last={last_seen}")
        
        # Check API response
        response = requests.get("http://localhost:8000/api/tracking/objects", timeout=5)
        if response.status_code == 200:
            api_data = response.json()
            api_objects = api_data.get('objects', [])
            print(f"🌐 Objects via API: {len(api_objects)}")
            
            if len(api_objects) != len(recent_objects):
                print(f"⚠️  Mismatch: DB has {len(recent_objects)}, API returns {len(api_objects)}")
        
    except Exception as e:
        print(f"❌ Data flow monitoring failed: {e}")

def run_pipeline_test():
    """Run complete pipeline test"""
    print("🧪 DIGITAL TWIN PIPELINE TEST")
    print("=" * 50)
    print("Testing: Smart Queue Camera 8 → MongoDB → Backend API → Frontend")
    print("=" * 50)
    
    # Run tests
    db_ok = test_database_connection()
    api_ok = test_backend_api()
    frontend_ok = test_frontend_accessibility()
    smart_queue_ok = check_smart_queue_integration()
    
    print("\n📋 TEST SUMMARY:")
    print(f"   Database:     {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"   Backend API:  {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"   Frontend:     {'✅ PASS' if frontend_ok else '⚠️  WARN'}")
    print(f"   Smart Queue:  {'✅ PASS' if smart_queue_ok else '❌ FAIL'}")
    
    # Overall status
    critical_pass = db_ok and api_ok and smart_queue_ok
    print(f"\n🎯 Pipeline Status: {'✅ READY' if critical_pass else '❌ ISSUES DETECTED'}")
    
    if critical_pass:
        print("\n🚀 READY TO START:")
        print("   1. Run: python cv/smart_queue_camera8_test.py")
        print("   2. Run: python backend/live_server.py")
        print("   3. Start frontend: npm run dev (in frontend directory)")
        print("   4. Open: http://localhost:3000 or http://localhost:5173")
        
        monitor_data_flow()
    else:
        print("\n🔧 REQUIRED ACTIONS:")
        if not db_ok:
            print("   - Start MongoDB: mongod")
        if not api_ok:
            print("   - Start backend: python backend/live_server.py")
        if not smart_queue_ok:
            print("   - Check CV dependencies and configuration")

if __name__ == "__main__":
    run_pipeline_test() 