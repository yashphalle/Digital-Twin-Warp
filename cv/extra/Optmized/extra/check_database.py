#!/usr/bin/env python3
"""
Check what's in the database
"""

from pymongo import MongoClient
import json
from datetime import datetime

def check_database():
    """Check recent database entries"""
    print("🔍 Checking database for color data...")
    
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_tracking']
        collection = db['detections']
        
        # Get recent entries
        recent = list(collection.find().sort('_id', -1).limit(3))
        
        print(f"📊 Found {len(recent)} recent entries")
        
        for i, doc in enumerate(recent, 1):
            print(f"\n--- Entry {i} ---")
            doc['_id'] = str(doc['_id'])
            
            # Check for color fields
            color_fields = ['color_rgb', 'color_hsv', 'color_hex', 'color_name', 'color_confidence']
            has_color = any(field in doc for field in color_fields)
            
            print(f"Has color data: {has_color}")
            
            if has_color:
                for field in color_fields:
                    if field in doc:
                        print(f"  {field}: {doc[field]}")
            
            # Show basic info
            print(f"Global ID: {doc.get('global_id', 'N/A')}")
            print(f"Camera: {doc.get('camera_id', 'N/A')}")
            print(f"Last seen: {doc.get('last_seen', doc.get('timestamp', 'N/A'))}")
            
        # Count total entries
        total = collection.count_documents({})
        print(f"\n📈 Total entries in database: {total}")
        
        # Count entries with color data
        color_count = collection.count_documents({'color_rgb': {'$exists': True}})
        print(f"🎨 Entries with color data: {color_count}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_database()
