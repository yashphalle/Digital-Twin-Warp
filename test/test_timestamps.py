#!/usr/bin/env python3
"""
Test script to verify first_seen and last_seen timestamps are working correctly
"""

import requests
import json
from datetime import datetime

def test_timestamps():
    """Test that timestamps are properly managed"""
    
    print("🕐 TESTING FIRST_SEEN & LAST_SEEN TIMESTAMPS")
    print("=" * 60)
    
    try:
        # Get current objects
        response = requests.get("http://localhost:8000/api/tracking/objects", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
            
            print(f"✅ Found {len(objects)} objects")
            
            if objects:
                # Check first few objects for timestamp fields
                print(f"\n📋 Checking timestamp fields in first 3 objects:")
                
                for i, obj in enumerate(objects[:3]):
                    print(f"\n🔍 Object {i+1}:")
                    print(f"   Persistent ID: {obj.get('persistent_id')}")
                    print(f"   Global ID: {obj.get('global_id')}")
                    print(f"   Warp ID: {obj.get('warp_id', 'None')}")
                    
                    # Check timestamp fields
                    first_seen = obj.get('first_seen')
                    last_seen = obj.get('last_seen')
                    warp_linked = obj.get('warp_id_linked_at')
                    
                    print(f"   First Seen: {first_seen}")
                    print(f"   Last Seen: {last_seen}")
                    print(f"   Warp Linked: {warp_linked}")
                    
                    # Validate timestamps
                    if first_seen and last_seen:
                        try:
                            first_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                            last_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                            
                            if last_dt >= first_dt:
                                print(f"   ✅ Timestamps are logical (last_seen >= first_seen)")
                            else:
                                print(f"   ❌ Timestamps are illogical (last_seen < first_seen)")
                                
                        except Exception as e:
                            print(f"   ⚠️ Error parsing timestamps: {e}")
                    else:
                        print(f"   ⚠️ Missing timestamp fields")
                
                # Test linking a Warp ID to see if timestamps are preserved
                print(f"\n🔗 Testing Warp ID linking (preserves timestamps)...")
                
                # Find an object without Warp ID
                unlinked_obj = next((obj for obj in objects if not obj.get('warp_id')), None)
                
                if unlinked_obj:
                    global_id = unlinked_obj.get('global_id')
                    original_first_seen = unlinked_obj.get('first_seen')
                    original_last_seen = unlinked_obj.get('last_seen')
                    
                    print(f"   Target: Global ID {global_id}")
                    print(f"   Original first_seen: {original_first_seen}")
                    print(f"   Original last_seen: {original_last_seen}")
                    
                    # Link Warp ID
                    link_response = requests.post(
                        "http://localhost:8000/api/link-warp-id",
                        json={
                            "pallet_id": global_id,
                            "warp_id": f"TEST-TIMESTAMP-{int(datetime.now().timestamp())}"
                        },
                        timeout=10
                    )
                    
                    if link_response.status_code == 200:
                        result = link_response.json()
                        if result.get('success'):
                            print(f"   ✅ Warp ID linked successfully")
                            
                            # Check if timestamps were preserved
                            updated_response = requests.get("http://localhost:8000/api/tracking/objects", timeout=10)
                            if updated_response.status_code == 200:
                                updated_objects = updated_response.json().get('objects', [])
                                updated_obj = next((obj for obj in updated_objects if obj.get('global_id') == global_id), None)
                                
                                if updated_obj:
                                    new_first_seen = updated_obj.get('first_seen')
                                    new_last_seen = updated_obj.get('last_seen')
                                    warp_linked_at = updated_obj.get('warp_id_linked_at')
                                    
                                    print(f"   After linking:")
                                    print(f"     first_seen: {new_first_seen}")
                                    print(f"     last_seen: {new_last_seen}")
                                    print(f"     warp_id_linked_at: {warp_linked_at}")
                                    
                                    if new_first_seen == original_first_seen and new_last_seen == original_last_seen:
                                        print(f"   ✅ Timestamps preserved correctly!")
                                    else:
                                        print(f"   ❌ Timestamps were modified during Warp ID linking")
                                        
                                    if warp_linked_at:
                                        print(f"   ✅ warp_id_linked_at timestamp added")
                                    else:
                                        print(f"   ❌ warp_id_linked_at timestamp missing")
                        else:
                            print(f"   ❌ Warp ID linking failed: {result.get('message')}")
                    else:
                        print(f"   ❌ Warp ID linking request failed: {link_response.status_code}")
                else:
                    print(f"   ⚠️ No objects without Warp ID found for testing")
                    
            else:
                print("⚠️ No objects found in system")
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_timestamps()
