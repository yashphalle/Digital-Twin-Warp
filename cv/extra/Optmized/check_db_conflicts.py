#!/usr/bin/env python3
"""
Check for Global ID conflicts in MongoDB
"""

from pymongo import MongoClient
import json

def check_db_conflicts():
    """Check MongoDB for Global ID conflicts across cameras"""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['warehouse_tracking']
    collection = db['detections']

    print('🔍 Checking MongoDB for Global ID conflicts...')
    print('=' * 60)

    # Check recent objects by camera
    cameras = list(range(1, 12))
    for camera_id in cameras:
        recent = list(collection.find({'camera_id': camera_id}).sort('_id', -1).limit(5))
        if recent:
            print(f'📹 Camera {camera_id}: {len(recent)} recent objects')
            for obj in recent[:3]:
                global_id = obj.get('global_id', 'N/A')
                phys_x = obj.get('physical_x_ft', 'N/A')
                phys_y = obj.get('physical_y_ft', 'N/A')
                print(f'   Global ID: {global_id}, Physical: ({phys_x}, {phys_y})')
            print()

    # Check for duplicate global IDs across cameras
    print('🔍 Checking for Global ID conflicts across cameras...')
    pipeline = [
        {'$group': {
            '_id': '$global_id',
            'cameras': {'$addToSet': '$camera_id'},
            'count': {'$sum': 1}
        }},
        {'$match': {'count': {'$gt': 1}}}
    ]

    conflicts = list(collection.aggregate(pipeline))
    if conflicts:
        print(f'❌ Found {len(conflicts)} Global ID conflicts:')
        for conflict in conflicts[:10]:
            global_id = conflict['_id']
            cameras = conflict['cameras']
            print(f'   Global ID {global_id} used by cameras: {cameras}')
    else:
        print('✅ No Global ID conflicts found')

    # Check coordinate ranges by camera
    print('\n🗺️ Physical coordinate ranges by camera:')
    for camera_id in cameras:
        coords = list(collection.find(
            {'camera_id': camera_id, 'physical_x_ft': {'$ne': None}, 'physical_y_ft': {'$ne': None}},
            {'physical_x_ft': 1, 'physical_y_ft': 1}
        ))
        
        if coords:
            x_coords = [c['physical_x_ft'] for c in coords]
            y_coords = [c['physical_y_ft'] for c in coords]
            
            print(f'Camera {camera_id}: X: {min(x_coords):.1f}-{max(x_coords):.1f}ft, Y: {min(y_coords):.1f}-{max(y_coords):.1f}ft ({len(coords)} objects)')

if __name__ == "__main__":
    check_db_conflicts()
