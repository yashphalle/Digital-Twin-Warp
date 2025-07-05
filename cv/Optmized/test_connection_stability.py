#!/usr/bin/env python3
"""
Test connection stability between frontend, backend, and CV system
"""

import requests
import time
import json
from datetime import datetime

def test_api_stability():
    """Test API connection stability"""
    print("🔍 TESTING API CONNECTION STABILITY")
    print("=" * 50)
    
    errors = 0
    successes = 0
    object_counts = []
    
    for i in range(10):
        try:
            start_time = time.time()
            response = requests.get('http://localhost:8000/api/tracking/objects', timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                object_count = len(data.get('objects', []))
                object_counts.append(object_count)
                successes += 1
                
                print(f"Test {i+1:2d}: ✅ {response.status_code} | {object_count:2d} objects | {response_time:.1f}ms")
            else:
                errors += 1
                print(f"Test {i+1:2d}: ❌ {response.status_code} | Error response")
                
        except requests.exceptions.Timeout:
            errors += 1
            print(f"Test {i+1:2d}: ⏰ TIMEOUT | Request took >5s")
        except requests.exceptions.ConnectionError:
            errors += 1
            print(f"Test {i+1:2d}: 🔌 CONNECTION ERROR | Backend not responding")
        except Exception as e:
            errors += 1
            print(f"Test {i+1:2d}: ❌ ERROR | {e}")
        
        time.sleep(0.5)
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Successes: {successes}/10")
    print(f"❌ Errors: {errors}/10")
    
    if object_counts:
        print(f"📦 Object counts: {object_counts}")
        print(f"📈 Min/Max objects: {min(object_counts)}/{max(object_counts)}")
        
        # Check for rapid changes (objects appearing/disappearing)
        rapid_changes = 0
        for i in range(1, len(object_counts)):
            if abs(object_counts[i] - object_counts[i-1]) > 2:
                rapid_changes += 1
        
        if rapid_changes > 0:
            print(f"⚠️ Rapid object changes detected: {rapid_changes} times")
            print("   This could indicate objects appearing/disappearing quickly")
    
    return errors == 0

def test_database_consistency():
    """Test database consistency"""
    print(f"\n🗄️ TESTING DATABASE CONSISTENCY")
    print("=" * 40)
    
    try:
        response = requests.get('http://localhost:8000/api/tracking/objects')
        if response.status_code != 200:
            print("❌ Cannot connect to API")
            return False
        
        data = response.json()
        objects = data.get('objects', [])
        
        if not objects:
            print("⚠️ No objects in database")
            return True
        
        print(f"📦 Found {len(objects)} objects")
        
        # Check for required fields
        required_fields = ['persistent_id', 'camera_id', 'real_center', 'physical_corners']
        missing_fields = []
        
        for obj in objects[:3]:  # Check first 3 objects
            for field in required_fields:
                if field not in obj or obj[field] is None:
                    missing_fields.append(f"Object {obj.get('persistent_id', 'Unknown')}: missing {field}")
        
        if missing_fields:
            print("❌ Missing required fields:")
            for missing in missing_fields:
                print(f"   {missing}")
            return False
        else:
            print("✅ All required fields present")
        
        # Check for duplicate IDs
        ids = [obj.get('persistent_id') for obj in objects]
        duplicates = len(ids) - len(set(ids))
        
        if duplicates > 0:
            print(f"⚠️ Found {duplicates} duplicate object IDs")
        else:
            print("✅ No duplicate IDs")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_frontend_refresh_issue():
    """Test for frontend refresh issues"""
    print(f"\n🖥️ TESTING FRONTEND REFRESH PATTERNS")
    print("=" * 45)
    
    try:
        # Check if objects have timestamps that indicate rapid updates
        response = requests.get('http://localhost:8000/api/tracking/objects')
        data = response.json()
        objects = data.get('objects', [])
        
        if not objects:
            print("⚠️ No objects to analyze")
            return True
        
        # Check age_seconds for rapid changes
        ages = [obj.get('age_seconds', 0) for obj in objects]
        very_new_objects = [age for age in ages if age < 2]  # Less than 2 seconds old
        
        print(f"📊 Object ages: min={min(ages):.1f}s, max={max(ages):.1f}s")
        print(f"🆕 Very new objects (<2s): {len(very_new_objects)}")
        
        if len(very_new_objects) > len(objects) * 0.5:  # More than 50% are very new
            print("⚠️ Many objects are very new - possible rapid spawning/despawning")
            print("   This could cause the 'flashing' effect you're seeing")
        else:
            print("✅ Object ages look normal")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CONNECTION STABILITY DIAGNOSTIC")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    api_ok = test_api_stability()
    db_ok = test_database_consistency()
    frontend_ok = test_frontend_refresh_issue()
    
    print(f"\n🏁 FINAL DIAGNOSIS:")
    print("=" * 30)
    
    if api_ok and db_ok and frontend_ok:
        print("✅ All tests passed - connection looks stable")
        print("🔍 If you're still seeing issues, they might be:")
        print("   1. Frontend caching/refresh problems")
        print("   2. Browser-specific rendering issues")
        print("   3. Network latency spikes")
    else:
        print("❌ Issues detected:")
        if not api_ok:
            print("   - API connection instability")
        if not db_ok:
            print("   - Database consistency problems")
        if not frontend_ok:
            print("   - Frontend refresh issues")
        
        print("\n🔧 Recommended fixes:")
        print("   1. Restart the backend API server")
        print("   2. Clear browser cache and refresh")
        print("   3. Check CV system is running properly")
        print("   4. Verify MongoDB is stable")
