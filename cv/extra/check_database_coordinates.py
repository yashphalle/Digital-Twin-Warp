#!/usr/bin/env python3
"""
Check what coordinates are actually being stored in the database
"""

from database_handler import DatabaseHandler
import json
from datetime import datetime, timedelta

def check_recent_objects():
    """Check recent objects in database"""
    print("🔍 CHECKING DATABASE COORDINATES")
    print("=" * 50)
    
    db = DatabaseHandler()
    if not db.connected:
        print("❌ Failed to connect to database")
        return
    
    # Get recent objects (last 10 minutes)
    recent_time = datetime.now() - timedelta(minutes=10)
    recent_objects = list(db.collection.find({
        'created_at': {'$gte': recent_time}
    }).sort('created_at', -1).limit(10))
    
    if not recent_objects:
        print("❌ No recent objects found in database")
        print("💡 Make sure the multi-camera system is running and detecting objects")
        return
    
    print(f"✅ Found {len(recent_objects)} recent objects:")
    print()
    
    for i, obj in enumerate(recent_objects, 1):
        print(f"Object #{i}:")
        print(f"   ID: {obj.get('persistent_id')}")
        print(f"   Camera: {obj.get('camera_id')} ({obj.get('camera_source', 'Unknown')})")
        print(f"   Real Center: {obj.get('real_center')}")
        print(f"   Pixel Center: {obj.get('center')}")
        print(f"   Confidence: {obj.get('confidence', 0):.2f}")
        print(f"   Created: {obj.get('created_at')}")
        print(f"   Coverage Zone: {obj.get('coverage_zone', 'Unknown')}")
        print()
    
    # Analyze coordinate ranges
    print("📊 COORDINATE ANALYSIS:")
    print("-" * 30)
    
    camera_coords = {}
    for obj in recent_objects:
        camera_id = obj.get('camera_id')
        real_center = obj.get('real_center')
        
        if camera_id and real_center and len(real_center) >= 2:
            if camera_id not in camera_coords:
                camera_coords[camera_id] = {'x_coords': [], 'y_coords': []}
            
            camera_coords[camera_id]['x_coords'].append(real_center[0])
            camera_coords[camera_id]['y_coords'].append(real_center[1])
    
    for camera_id, coords in camera_coords.items():
        x_coords = coords['x_coords']
        y_coords = coords['y_coords']
        
        print(f"Camera {camera_id}:")
        print(f"   X range: {min(x_coords):.1f} - {max(x_coords):.1f}ft")
        print(f"   Y range: {min(y_coords):.1f} - {max(y_coords):.1f}ft")
        print(f"   Objects: {len(x_coords)}")
        
        # Expected ranges for Column 3 cameras
        expected_ranges = {
            8: {'x': (120, 180), 'y': (0, 25)},
            9: {'x': (120, 180), 'y': (25, 50)},
            10: {'x': (120, 180), 'y': (50, 75)},
            11: {'x': (120, 180), 'y': (75, 90)}
        }
        
        if camera_id in expected_ranges:
            expected = expected_ranges[camera_id]
            x_correct = expected['x'][0] <= min(x_coords) and max(x_coords) <= expected['x'][1]
            y_correct = expected['y'][0] <= min(y_coords) and max(y_coords) <= expected['y'][1]
            
            print(f"   Expected: {expected['x'][0]}-{expected['x'][1]}ft × {expected['y'][0]}-{expected['y'][1]}ft")
            
            if x_correct and y_correct:
                print(f"   ✅ Coordinates are in expected range")
            else:
                print(f"   ❌ Coordinates are outside expected range")
                if not x_correct:
                    print(f"      X coordinates outside {expected['x'][0]}-{expected['x'][1]}ft")
                if not y_correct:
                    print(f"      Y coordinates outside {expected['y'][0]}-{expected['y'][1]}ft")
        print()

def main():
    check_recent_objects()

if __name__ == "__main__":
    main()
