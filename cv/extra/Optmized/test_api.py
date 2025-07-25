#!/usr/bin/env python3
"""
Test the backend API to see if color data is being returned
"""

import requests
import json

def test_api():
    """Test the backend API"""
    print("🌐 Testing backend API...")

    try:
        response = requests.get('http://localhost:8000/api/tracking/objects')
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API responded with {len(data)} objects")

            if len(data) > 0:
                # Check first object for color data
                first_obj = data[0]
                print(f"\n🔍 First object data:")
                print(json.dumps(first_obj, indent=2))

                # Check for color fields specifically
                color_fields = ['color_rgb', 'color_hsv', 'color_hex', 'color_name', 'color_confidence']
                print(f"\n🎨 Color field check:")
                for field in color_fields:
                    value = first_obj.get(field)
                    print(f"  {field}: {value}")

            else:
                print("📭 No objects returned from API")

        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Cannot connect to backend server")
        print("Make sure the backend server is running on port 8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_api()
