#!/usr/bin/env python3
"""
Check API Response - Debug script to see what data the backend is returning
"""

import requests
import json

def check_api():
    try:
        print("🔍 Checking Backend API...")
        
        # Test root endpoint
        response = requests.get('http://localhost:8000/')
        print(f"Root endpoint: {response.status_code}")
        
        # Test tracking objects endpoint
        response = requests.get('http://localhost:8000/api/tracking/objects')
        print(f"Tracking endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response successful")
            print(f"Objects count: {len(data.get('objects', []))}")
            print(f"Total in DB: {data.get('total_in_db', 'unknown')}")
            print(f"Source: {data.get('source', 'unknown')}")
            print(f"Timestamp: {data.get('timestamp', 'unknown')}")
            
            # Show first object structure
            objects = data.get('objects', [])
            if objects:
                print(f"\n📦 Sample Object Structure:")
                sample = objects[0]
                print(f"  Persistent ID: {sample.get('persistent_id')}")
                print(f"  Camera ID: {sample.get('camera_id')}")
                print(f"  Confidence: {sample.get('confidence')}")
                print(f"  Real Center: {sample.get('real_center')}")
                print(f"  Real Center X: {sample.get('real_center_x')}")
                print(f"  Real Center Y: {sample.get('real_center_y')}")
                print(f"  Source: {sample.get('source')}")
                print(f"  Class: {sample.get('object_class')}")
                
                # Check coordinate format
                real_center = sample.get('real_center')
                if real_center and len(real_center) >= 2:
                    x, y = real_center[0], real_center[1]
                    print(f"  Coordinates: ({x}, {y})")
                    
                    if x is not None and y is not None:
                        print(f"  ✅ Valid coordinates for frontend")
                    else:
                        print(f"  ❌ Invalid coordinates - None values")
                else:
                    print(f"  ❌ Missing or invalid real_center format")
                
                print(f"\n📄 Full Object JSON:")
                print(json.dumps(sample, indent=2, default=str))
            else:
                print("❌ No objects in response")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking API: {e}")

def check_warehouse_config():
    try:
        print("\n🏭 Checking Warehouse Config...")
        response = requests.get('http://localhost:8000/api/warehouse/config')
        
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Warehouse config loaded")
            print(f"  Width: {config.get('width_feet')}ft")
            print(f"  Length: {config.get('length_feet')}ft")
            print(f"  Calibrated: {config.get('calibrated')}")
            print(f"  Source: {config.get('source')}")
        else:
            print(f"❌ Config Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking config: {e}")

def check_cameras():
    try:
        print("\n📹 Checking Camera Status...")
        response = requests.get('http://localhost:8000/api/cameras/status')
        
        if response.status_code == 200:
            data = response.json()
            cameras = data.get('cameras', [])
            print(f"✅ Camera status loaded - {len(cameras)} cameras")
            
            # Find Camera 8
            camera8 = next((c for c in cameras if c.get('camera_id') == 8), None)
            if camera8:
                print(f"  Camera 8 Status: {camera8.get('status')}")
            else:
                print(f"  ❌ Camera 8 not found in response")
                
        else:
            print(f"❌ Camera Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking cameras: {e}")

if __name__ == "__main__":
    check_api()
    check_warehouse_config()
    check_cameras() 