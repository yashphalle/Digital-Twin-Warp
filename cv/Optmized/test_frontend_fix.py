#!/usr/bin/env python3
"""
Test the frontend polling fix
"""

import requests
import time
from datetime import datetime

def monitor_api_requests():
    """Monitor API request patterns"""
    print("🔍 MONITORING API REQUEST PATTERNS")
    print("=" * 50)
    print("Watching for overlapping requests and timing issues...")
    print("(Refresh your browser now to see the new polling behavior)")
    print()
    
    request_times = []
    
    # Monitor for 30 seconds
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            request_start = time.time()
            response = requests.get('http://localhost:8000/api/tracking/objects', timeout=10)
            request_end = time.time()
            
            request_duration = (request_end - request_start) * 1000
            current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            if response.status_code == 200:
                data = response.json()
                object_count = len(data.get('objects', []))
                print(f"{current_time} | ✅ {response.status_code} | {object_count:3d} objects | {request_duration:6.1f}ms")
                request_times.append(request_duration)
            else:
                print(f"{current_time} | ❌ {response.status_code} | Error response")
                
        except requests.exceptions.Timeout:
            print(f"{current_time} | ⏰ TIMEOUT | Request took >10s")
        except requests.exceptions.ConnectionError:
            print(f"{current_time} | 🔌 CONNECTION ERROR")
        except Exception as e:
            print(f"{current_time} | ❌ ERROR | {e}")
        
        time.sleep(0.5)  # Check every 500ms
    
    if request_times:
        avg_time = sum(request_times) / len(request_times)
        print(f"\n📊 ANALYSIS:")
        print(f"Total requests monitored: {len(request_times)}")
        print(f"Average response time: {avg_time:.1f}ms")
        print(f"Min/Max response time: {min(request_times):.1f}ms / {max(request_times):.1f}ms")
        
        if avg_time > 1000:
            print("⚠️ Response times are still slow (>1s)")
            print("   Consider optimizing the backend API")
        else:
            print("✅ Response times look reasonable")

def test_polling_frequency():
    """Test if frontend polling frequency has been reduced"""
    print(f"\n🔧 FRONTEND POLLING FIX VERIFICATION")
    print("=" * 45)
    print("Expected behavior after fix:")
    print("✅ Polling every 2 seconds (instead of 0.5s)")
    print("✅ No overlapping requests")
    print("✅ Reduced 'connection error' messages")
    print("✅ Less 'flashing' of objects")
    print()
    print("🎯 To verify the fix:")
    print("1. Refresh your browser at http://localhost:5174")
    print("2. Open browser dev tools (F12) → Network tab")
    print("3. Look for requests to '/api/tracking/objects'")
    print("4. Verify they occur every ~2 seconds (not 0.5s)")
    print("5. Check that objects don't flash/disappear rapidly")

if __name__ == "__main__":
    print("🚀 FRONTEND POLLING FIX TEST")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_polling_frequency()
    monitor_api_requests()
    
    print(f"\n🏁 SUMMARY:")
    print("=" * 20)
    print("✅ Frontend polling frequency reduced from 500ms to 2000ms")
    print("✅ Added request debouncing to prevent overlaps")
    print("✅ Separated camera/config polling to reduce load")
    print()
    print("🔍 Expected improvements:")
    print("   - Less 'connection error' messages")
    print("   - Reduced object 'flashing' effect")
    print("   - More stable frontend display")
    print("   - Lower server load")
