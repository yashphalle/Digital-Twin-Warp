#!/usr/bin/env python3
"""
Test the backend API after removing time filter
"""

import requests
import json

def test_backend_fix():
    """Test backend API after removing time filter"""
    print('🔍 Testing Backend API after removing time filter...')
    print('=' * 60)

    try:
        response = requests.get('http://localhost:8000/api/tracking/objects')
        
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
            
            print(f'✅ API Response:')
            print(f'   Objects returned: {len(objects)}')
            print(f'   Total in DB: {data.get("total_in_db", "N/A")}')
            print(f'   Source: {data.get("source", "N/A")}')
            
            if objects:
                print(f'\n📊 Sample Objects:')
                for i, obj in enumerate(objects[:5], 1):
                    obj_id = obj.get('persistent_id', 'N/A')
                    camera = obj.get('camera_id', 'N/A')
                    x = obj.get('physical_x_ft', 'N/A')
                    y = obj.get('physical_y_ft', 'N/A')
                    print(f'   {i}. ID: {obj_id}, Camera: {camera}, Position: ({x}, {y})')
            
            # Check for camera distribution
            camera_counts = {}
            for obj in objects:
                camera = obj.get('camera_id', 'Unknown')
                camera_counts[camera] = camera_counts.get(camera, 0) + 1
            
            print(f'\n📹 Objects by Camera:')
            for camera, count in sorted(camera_counts.items()):
                print(f'   Camera {camera}: {count} objects')
                
        else:
            print(f'❌ API Error: {response.status_code}')
            print(response.text)
            
    except Exception as e:
        print(f'❌ Connection error: {e}')

if __name__ == "__main__":
    test_backend_fix()
