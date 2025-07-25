#!/usr/bin/env python3
"""
Debug the display logic to understand what objects are being shown
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import json

def debug_display_logic():
    """Debug what objects the backend is returning"""
    print("🔍 Debugging Backend Display Logic...")
    print("=" * 60)
    
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_tracking']
        collection = db['detections']
        
        # 1. Check total objects in database
        total_objects = collection.count_documents({})
        print(f"📊 Total objects in database: {total_objects}")
        
        # 2. Check objects with coordinates
        with_coords = collection.count_documents({
            "physical_x_ft": {"$exists": True, "$ne": None},
            "physical_y_ft": {"$exists": True, "$ne": None}
        })
        print(f"📍 Objects with coordinates: {with_coords}")
        
        # 3. Check time-based filtering (last 5 minutes)
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        print(f"⏰ Looking for objects newer than: {five_minutes_ago}")
        
        recent_query = {
            "$and": [
                {
                    "$or": [
                        {"timestamp": {"$gte": five_minutes_ago}},
                        {"last_seen": {"$gte": five_minutes_ago}},
                        {"first_seen": {"$gte": five_minutes_ago}}
                    ]
                },
                {"physical_x_ft": {"$exists": True, "$ne": None}},
                {"physical_y_ft": {"$exists": True, "$ne": None}}
            ]
        }
        
        recent_objects = list(collection.find(recent_query).limit(100))
        print(f"🕐 Recent objects (last 5 min): {len(recent_objects)}")
        
        # 4. Check what timestamps exist
        print(f"\n📅 Timestamp Analysis:")
        
        # Check different timestamp fields
        timestamp_fields = ["timestamp", "last_seen", "first_seen"]
        for field in timestamp_fields:
            count = collection.count_documents({field: {"$exists": True}})
            print(f"   {field}: {count} objects")
            
            if count > 0:
                # Get latest timestamp for this field
                latest = list(collection.find({field: {"$exists": True}}).sort(field, -1).limit(1))
                if latest:
                    latest_time = latest[0].get(field)
                    if isinstance(latest_time, datetime):
                        time_diff = datetime.now() - latest_time
                        print(f"      Latest: {latest_time} ({time_diff.total_seconds():.0f} seconds ago)")
                    else:
                        print(f"      Latest: {latest_time} (not datetime)")
        
        # 5. Sample recent objects
        print(f"\n🔍 Sample Recent Objects:")
        sample_objects = list(collection.find(recent_query).limit(5))
        
        for i, obj in enumerate(sample_objects, 1):
            global_id = obj.get('global_id', 'N/A')
            camera_id = obj.get('camera_id', 'N/A')
            phys_x = obj.get('physical_x_ft', 'N/A')
            phys_y = obj.get('physical_y_ft', 'N/A')
            
            # Check timestamps
            timestamps = {}
            for field in timestamp_fields:
                if field in obj:
                    timestamps[field] = obj[field]
            
            print(f"   {i}. Global ID: {global_id}, Camera: {camera_id}")
            print(f"      Physical: ({phys_x}, {phys_y})")
            print(f"      Timestamps: {timestamps}")
            print()
        
        # 6. Check if we're hitting the 100 object limit
        all_recent = collection.count_documents(recent_query)
        print(f"📈 Total recent objects: {all_recent}")
        print(f"📊 Limit applied: 100")
        print(f"🎯 Objects returned: {min(all_recent, 100)}")
        
        if all_recent > 100:
            print(f"⚠️ WARNING: {all_recent - 100} recent objects are being excluded due to 100 object limit!")
        
        # 7. Check for duplicate Global IDs in recent objects
        print(f"\n🔍 Checking for duplicate Global IDs in recent objects:")
        pipeline = [
            {"$match": recent_query},
            {"$group": {
                "_id": "$global_id",
                "count": {"$sum": 1},
                "cameras": {"$addToSet": "$camera_id"}
            }},
            {"$match": {"count": {"$gt": 1}}}
        ]
        
        duplicates = list(collection.aggregate(pipeline))
        if duplicates:
            print(f"❌ Found {len(duplicates)} duplicate Global IDs in recent objects:")
            for dup in duplicates[:5]:
                print(f"   Global ID {dup['_id']}: {dup['count']} objects from cameras {dup['cameras']}")
        else:
            print("✅ No duplicate Global IDs in recent objects")
        
        # 8. Age analysis
        print(f"\n⏰ Object Age Analysis:")
        now = datetime.now()
        
        age_ranges = [
            ("< 1 minute", timedelta(minutes=1)),
            ("1-5 minutes", timedelta(minutes=5)),
            ("5-30 minutes", timedelta(minutes=30)),
            ("30+ minutes", timedelta(days=1))
        ]
        
        for label, age_limit in age_ranges:
            cutoff = now - age_limit
            count = collection.count_documents({
                "$and": [
                    {
                        "$or": [
                            {"timestamp": {"$gte": cutoff}},
                            {"last_seen": {"$gte": cutoff}},
                            {"first_seen": {"$gte": cutoff}}
                        ]
                    },
                    {"physical_x_ft": {"$exists": True, "$ne": None}},
                    {"physical_y_ft": {"$exists": True, "$ne": None}}
                ]
            })
            print(f"   {label}: {count} objects")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_display_logic()
