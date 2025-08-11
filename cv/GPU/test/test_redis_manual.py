#!/usr/bin/env python3
"""
Manual Redis Test - Test saving tracks to Redis manually
"""

import redis
import pickle
import time

def test_manual_redis_save():
    """Manually save some test tracks to Redis"""
    print("🧪 Testing manual Redis save...")
    
    try:
        # Connect to Redis
        redis_client = redis.Redis(
            host='localhost', 
            port=6379,
            decode_responses=False,
            socket_keepalive=True,
            health_check_interval=30
        )
        
        # Test connection
        redis_client.ping()
        print("✅ Connected to Redis successfully!")
        
        # Create test tracks
        test_tracks = [
            {
                'global_id': 8001,
                'camera_id': 8,
                'last_bbox': [100, 100, 200, 200],
                'confidence': 0.85,
                'class': 0,
                'track_age': 5,
                'last_seen': time.time(),
                'has_features': True
            },
            {
                'global_id': 8002,
                'camera_id': 8,
                'last_bbox': [300, 150, 400, 250],
                'confidence': 0.92,
                'class': 1,
                'track_age': 12,
                'last_seen': time.time(),
                'has_features': True
            },
            {
                'global_id': 9001,
                'camera_id': 9,
                'last_bbox': [150, 120, 250, 220],
                'confidence': 0.88,
                'class': 0,
                'track_age': 8,
                'last_seen': time.time(),
                'has_features': True
            }
        ]
        
        # Save tracks to Redis (same format as Smart Persistence)
        for track in test_tracks:
            redis_key = f"camera_{track['camera_id']}:track_{track['global_id']}"
            
            # Store each field individually (Redis 3.0 compatible)
            for field, value in track.items():
                redis_client.hset(redis_key, field, pickle.dumps(value))
            
            # Set expiration (10 minutes)
            redis_client.expire(redis_key, 600)
            
            print(f"💾 Saved track {track['global_id']} for Camera {track['camera_id']}")
        
        print(f"✅ Successfully saved {len(test_tracks)} test tracks to Redis!")
        
        # Verify tracks are saved
        all_keys = redis_client.keys("camera_*:track_*")
        print(f"🔍 Found {len(all_keys)} track keys in Redis:")
        for key in all_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            print(f"   📦 {key_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual Redis test failed: {e}")
        return False

if __name__ == "__main__":
    test_manual_redis_save()
