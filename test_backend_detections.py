#!/usr/bin/env python3
"""
🔍 TEST BACKEND WITH DETECTIONS COLLECTION
Test the updated backend API with the detections collection
"""

import requests
import json
from datetime import datetime

def test_backend_api():
    """Test the backend API endpoints"""
    
    print("🔍 TESTING BACKEND API WITH DETECTIONS COLLECTION")
    print("=" * 70)
    
    base_url = "http://localhost:8000"
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/api/tracking/objects", "Tracking objects"),
        ("/api/tracking/stats", "Tracking stats"),
        ("/api/warehouse/config", "Warehouse config"),
        ("/api/cameras/status", "Camera status")
    ]
    
    for endpoint, description in endpoints:
        print(f"\n🧪 Testing {description}: {endpoint}")
        
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ SUCCESS: {response.status_code}")
                
                if endpoint == "/api/tracking/objects":
                    objects = data.get('objects', [])
                    count = data.get('count', 0)
                    total_in_db = data.get('total_in_db', 0)
                    
                    print(f"   📊 Objects returned: {count}")
                    print(f"   📊 Total in DB: {total_in_db}")
                    print(f"   📊 Source: {data.get('source', 'unknown')}")
                    
                    if objects:
                        sample = objects[0]
                        print(f"   📍 Sample object:")
                        print(f"      ID: {sample.get('persistent_id')}")
                        print(f"      Camera: {sample.get('camera_id')}")
                        print(f"      Confidence: {sample.get('confidence', 0):.2f}")
                        print(f"      Physical coords: ({sample.get('physical_x_ft')}, {sample.get('physical_y_ft')}) ft")
                        print(f"      Real center: {sample.get('real_center')}")
                        print(f"      Timestamp: {sample.get('timestamp')}")
                    else:
                        print("   ⚠️ No objects returned")
                        
                elif endpoint == "/api/tracking/stats":
                    print(f"   📊 Total objects: {data.get('total_objects', 0)}")
                    print(f"   📊 Recent objects: {data.get('recent_objects', 0)}")
                    print(f"   📊 Unique IDs: {data.get('unique_ids', 0)}")
                    
                elif endpoint == "/api/warehouse/config":
                    print(f"   📐 Warehouse: {data.get('width_feet', 0)}ft x {data.get('length_feet', 0)}ft")
                    print(f"   📐 Calibrated: {data.get('calibrated', False)}")
                    
                elif endpoint == "/api/cameras/status":
                    cameras = data.get('cameras', [])
                    print(f"   📹 Cameras configured: {len(cameras)}")
                    active_count = sum(1 for cam in cameras if cam.get('status') == 'active')
                    print(f"   📹 Active cameras: {active_count}")
                    
            else:
                print(f"   ❌ ERROR: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"      Detail: {error_data.get('detail', 'No details')}")
                except:
                    print(f"      Response: {response.text[:100]}...")
                    
        except requests.exceptions.RequestException as e:
            print(f"   ❌ CONNECTION ERROR: {e}")
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("🎯 BACKEND API TEST COMPLETE")
    print("=" * 70)

def test_mongodb_direct():
    """Test MongoDB directly to see what data is available"""
    
    print("\n🔍 TESTING MONGODB DIRECTLY")
    print("=" * 70)
    
    try:
        from pymongo import MongoClient
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["warehouse_tracking"]
        collection = db["detections"]
        
        # Test connection
        client.server_info()
        print("✅ MongoDB connection successful")
        
        # Check total documents
        total_count = collection.count_documents({})
        print(f"📊 Total detections in DB: {total_count}")
        
        # Check recent documents (last 5 minutes)
        from datetime import datetime, timedelta
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        
        recent_count = collection.count_documents({
            "timestamp": {"$gte": five_minutes_ago}
        })
        print(f"📊 Recent detections (5 min): {recent_count}")
        
        # Check documents with physical coordinates
        coords_count = collection.count_documents({
            "$and": [
                {"physical_x_ft": {"$exists": True, "$ne": None}},
                {"physical_y_ft": {"$exists": True, "$ne": None}}
            ]
        })
        print(f"📊 Detections with coordinates: {coords_count}")
        
        # Show sample document
        if total_count > 0:
            sample = collection.find_one({}, {"_id": 0})
            print(f"📄 Sample detection:")
            for key, value in sample.items():
                if isinstance(value, datetime):
                    value = value.isoformat()
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 BACKEND & DATABASE INTEGRATION TEST")
    print("=" * 80)
    
    # Test MongoDB directly
    mongodb_ok = test_mongodb_direct()
    
    # Test backend API
    test_backend_api()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   MongoDB: {'✅ OK' if mongodb_ok else '❌ FAILED'}")
    print(f"   Backend API: Check results above")
    print("=" * 80)

if __name__ == "__main__":
    main()
