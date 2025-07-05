#!/usr/bin/env python3
"""
Test script to verify timestamp display is working correctly in frontend
"""

import requests
import json
from datetime import datetime

def test_timestamp_display():
    """Test that timestamps are displayed correctly"""
    
    print("🕐 TESTING TIMESTAMP DISPLAY")
    print("=" * 50)
    
    try:
        # Get objects from API
        response = requests.get("http://localhost:8000/api/tracking/objects", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
            
            print(f"✅ Found {len(objects)} objects")
            
            if objects:
                # Show first object's timestamps
                obj = objects[0]
                print(f"\n📋 Sample Object Timestamps:")
                print(f"   Persistent ID: {obj.get('persistent_id')}")
                print(f"   Global ID: {obj.get('global_id')}")
                
                # Raw timestamps from API
                first_seen = obj.get('first_seen')
                last_seen = obj.get('last_seen')
                warp_linked = obj.get('warp_id_linked_at')
                
                print(f"\n🔍 Raw API Timestamps:")
                print(f"   first_seen: {first_seen}")
                print(f"   last_seen: {last_seen}")
                print(f"   warp_id_linked_at: {warp_linked}")
                
                # Format timestamps like frontend does
                print(f"\n📅 Formatted Timestamps (as shown in frontend):")
                
                if first_seen:
                    try:
                        dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                        formatted = dt.strftime('%m/%d/%Y, %I:%M:%S %p')
                        print(f"   Inbound Time: {formatted}")
                    except Exception as e:
                        print(f"   Inbound Time: Error formatting - {e}")
                
                if last_seen:
                    try:
                        dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        formatted = dt.strftime('%m/%d/%Y, %I:%M:%S %p')
                        print(f"   Last Seen: {formatted}")
                    except Exception as e:
                        print(f"   Last Seen: Error formatting - {e}")
                
                if warp_linked:
                    try:
                        dt = datetime.fromisoformat(warp_linked.replace('Z', '+00:00'))
                        formatted = dt.strftime('%m/%d/%Y, %I:%M:%S %p')
                        print(f"   Warp Linked: {formatted}")
                    except Exception as e:
                        print(f"   Warp Linked: Error formatting - {e}")
                
                print(f"\n✅ Frontend should now show exact timestamps instead of 'sec ago'")
                print(f"📋 Expected frontend display:")
                print(f"   Inbound Time: [exact date/time]")
                print(f"   Last Seen: [exact date/time]")
                
            else:
                print("⚠️ No objects found")
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_timestamp_display()
    
    print(f"\n📋 FRONTEND TESTING INSTRUCTIONS:")
    print("1. Refresh your frontend page")
    print("2. Click on any object to open details sidebar")
    print("3. Look for 'Timing Information' section")
    print("4. Should show:")
    print("   - Inbound Time: [exact timestamp]")
    print("   - Last Seen: [exact timestamp]")
    print("   - Warp Linked: [exact timestamp] (if applicable)")
    print("5. NO MORE negative seconds or 'sec ago' text!")
