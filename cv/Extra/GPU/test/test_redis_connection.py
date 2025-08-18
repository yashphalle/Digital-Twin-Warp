#!/usr/bin/env python3
"""
Test Redis Connection and Manual Operations
"""

import redis
import json
import time
from datetime import datetime

def test_redis_connection():
    """Test Redis connection and perform manual operations"""
    print("🔍 Testing Redis connection...")
    
    try:
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connection
        print("📡 Attempting to connect to Redis...")
        response = r.ping()
        print(f"✅ Redis connection successful! Ping response: {response}")
        
        # Test basic operations
        print("\n🧪 Testing basic Redis operations...")
        
        # Set a test key
        test_key = "test:connection"
        test_value = f"Connected at {datetime.now()}"
        r.set(test_key, test_value)
        print(f"✅ SET operation successful: {test_key} = {test_value}")
        
        # Get the test key
        retrieved_value = r.get(test_key)
        print(f"✅ GET operation successful: {test_key} = {retrieved_value}")
        
        # Test JSON operations (for Smart Persistence)
        print("\n📊 Testing JSON operations for Smart Persistence...")
        
        # Create a sample track object
        sample_track = {
            "track_id": "8001",
            "camera_id": 8,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.95,
            "class": "Pallet",
            "timestamp": time.time(),
            "features": [0.1, 0.2, 0.3, 0.4, 0.5]  # Sample feature vector
        }
        
        # Store track in Redis
        track_key = f"track:camera_8:8001"
        r.set(track_key, json.dumps(sample_track))
        print(f"✅ Stored sample track: {track_key}")
        
        # Retrieve track from Redis
        retrieved_track = json.loads(r.get(track_key))
        print(f"✅ Retrieved sample track: {retrieved_track}")
        
        # Test Redis Hash operations (alternative storage method)
        print("\n🗂️ Testing Redis Hash operations...")
        
        hash_key = "tracks:camera_8"
        r.hset(hash_key, "8001", json.dumps(sample_track))
        print(f"✅ Stored track in hash: {hash_key}")
        
        # Get all tracks for camera
        all_tracks = r.hgetall(hash_key)
        print(f"✅ Retrieved all tracks for camera 8: {len(all_tracks)} tracks")
        
        # Test expiration (for automatic cleanup)
        print("\n⏰ Testing key expiration...")
        expire_key = "test:expire"
        r.set(expire_key, "This will expire in 10 seconds")
        r.expire(expire_key, 10)
        ttl = r.ttl(expire_key)
        print(f"✅ Set expiration: {expire_key} will expire in {ttl} seconds")
        
        # Clean up test keys
        print("\n🧹 Cleaning up test keys...")
        r.delete(test_key, track_key, hash_key, expire_key)
        print("✅ Test keys cleaned up")
        
        # Show Redis info
        print("\n📈 Redis Server Info:")
        info = r.info()
        print(f"Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"Used Memory: {info.get('used_memory_human', 'Unknown')}")
        print(f"Connected Clients: {info.get('connected_clients', 'Unknown')}")
        print(f"Total Commands Processed: {info.get('total_commands_processed', 'Unknown')}")
        
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Possible solutions:")
        print("   1. Install Redis: https://redis.io/download")
        print("   2. Start Redis server: redis-server")
        print("   3. Check if Redis is running on port 6379")
        return False
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_redis_for_smart_persistence():
    """Test Redis operations specifically for Smart Persistence"""
    print("\n🎯 Testing Redis for Smart Persistence use case...")
    
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Simulate Smart Persistence operations
        camera_id = 8
        tracks = [
            {
                "track_id": "8001",
                "camera_id": camera_id,
                "bbox": [100, 100, 200, 200],
                "confidence": 0.95,
                "class": "Pallet",
                "timestamp": time.time(),
                "features": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "track_id": "8002", 
                "camera_id": camera_id,
                "bbox": [300, 150, 400, 250],
                "confidence": 0.87,
                "class": "Forklift",
                "timestamp": time.time(),
                "features": [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ]
        
        # Store tracks using Smart Persistence pattern
        for track in tracks:
            track_key = f"track:camera_{camera_id}:{track['track_id']}"
            r.set(track_key, json.dumps(track), ex=300)  # Expire in 5 minutes
            print(f"✅ Stored track {track['track_id']} for camera {camera_id}")
        
        # Retrieve all tracks for camera (Smart Persistence query pattern)
        pattern = f"track:camera_{camera_id}:*"
        track_keys = r.keys(pattern)
        print(f"✅ Found {len(track_keys)} tracks for camera {camera_id}")
        
        for key in track_keys:
            track_data = json.loads(r.get(key))
            print(f"   Track {track_data['track_id']}: {track_data['class']} (conf: {track_data['confidence']:.2f})")
        
        # Test cross-camera feature matching
        print("\n🔄 Testing cross-camera feature matching...")
        
        # Store features for cross-camera matching
        for track in tracks:
            feature_key = f"features:camera_{camera_id}:{track['track_id']}"
            r.set(feature_key, json.dumps(track['features']), ex=300)
            print(f"✅ Stored features for track {track['track_id']}")
        
        # Simulate feature retrieval for matching
        all_feature_keys = r.keys("features:*")
        print(f"✅ Found {len(all_feature_keys)} feature vectors across all cameras")
        
        # Clean up
        cleanup_keys = r.keys("track:camera_8:*") + r.keys("features:camera_8:*")
        if cleanup_keys:
            r.delete(*cleanup_keys)
            print(f"✅ Cleaned up {len(cleanup_keys)} test keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart Persistence Redis test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Redis Connection and Manual Operations Test")
    print("=" * 50)
    
    # Test basic connection
    connection_success = test_redis_connection()
    
    if connection_success:
        # Test Smart Persistence specific operations
        test_redis_for_smart_persistence()
        print("\n🎉 All Redis tests completed successfully!")
    else:
        print("\n💥 Redis is not available - Smart Persistence will not work")
        print("\n📋 To install Redis on Windows:")
        print("1. Download Redis from: https://github.com/microsoftarchive/redis/releases")
        print("2. Or use WSL: wsl --install, then sudo apt install redis-server")
        print("3. Or use Docker: docker run -d -p 6379:6379 redis:latest")
