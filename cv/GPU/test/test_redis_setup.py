#!/usr/bin/env python3
"""
Redis Setup and Test Script for Smart Persistence BoT-SORT
This script helps install Redis and test the Smart Persistence functionality
"""

import subprocess
import sys
import time
import logging
import redis
import pickle
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_redis_windows():
    """Install Redis on Windows using chocolatey or manual instructions"""
    logger.info("🔧 Installing Redis on Windows...")
    
    try:
        # Try chocolatey first
        result = subprocess.run(['choco', 'install', 'redis-64', '-y'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Redis installed successfully via Chocolatey")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️ Chocolatey not found or failed")
    
    # Manual installation instructions
    logger.info("""
    📋 Manual Redis Installation for Windows:
    
    1. Download Redis from: https://github.com/microsoftarchive/redis/releases
    2. Extract to C:\\Redis
    3. Run: C:\\Redis\\redis-server.exe
    4. Or use WSL: wsl --install, then: sudo apt install redis-server
    
    Alternative - Docker:
    docker run -d -p 6379:6379 --name redis redis:latest
    """)
    return False

def install_redis_python_client():
    """Install Redis Python client"""
    logger.info("📦 Installing Redis Python client...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'redis'], 
                      check=True, timeout=120)
        logger.info("✅ Redis Python client installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Redis Python client: {e}")
        return False

def test_redis_connection():
    """Test Redis connection and basic operations"""
    logger.info("🔍 Testing Redis connection...")
    
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
        logger.info("✅ Redis connection successful!")
        
        # Test basic operations
        test_key = "test_smart_persistence"
        test_data = {
            'camera_id': 8,
            'track_id': 1001,
            'features': np.random.rand(128).astype(np.float32),
            'timestamp': time.time()
        }
        
        # Store test data
        serialized = pickle.dumps(test_data)
        redis_client.set(test_key, serialized)
        redis_client.expire(test_key, 60)  # 1 minute expiration
        
        # Retrieve test data
        retrieved = redis_client.get(test_key)
        if retrieved:
            deserialized = pickle.loads(retrieved)
            logger.info(f"✅ Redis data storage test successful!")
            logger.info(f"   Stored camera_id: {deserialized['camera_id']}")
            logger.info(f"   Stored track_id: {deserialized['track_id']}")
            logger.info(f"   Features shape: {deserialized['features'].shape}")
        
        # Test hash operations (used by Smart Persistence) - Compatible with Redis 3.0
        hash_key = "camera_8:track_1001"
        hash_data = {
            'global_id': pickle.dumps(1001),
            'camera_id': pickle.dumps(8),
            'confidence': pickle.dumps(0.85),
            'last_seen': pickle.dumps(time.time())
        }

        # Use individual hset calls for Redis 3.0 compatibility
        for field, value in hash_data.items():
            redis_client.hset(hash_key, field, value)
        redis_client.expire(hash_key, 60)
        
        retrieved_hash = redis_client.hgetall(hash_key)
        if retrieved_hash:
            logger.info("✅ Redis hash operations test successful!")
            for k, v in retrieved_hash.items():
                logger.info(f"   {k.decode()}: {pickle.loads(v)}")
        
        # Clean up test data
        redis_client.delete(test_key, hash_key)
        
        return True
        
    except redis.ConnectionError:
        logger.error("❌ Redis connection failed! Make sure Redis server is running.")
        logger.info("💡 Start Redis server with: redis-server")
        return False
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        return False

def test_smart_persistence_simulation():
    """Simulate Smart Persistence operations"""
    logger.info("🧪 Testing Smart Persistence simulation...")
    
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
        
        # Simulate tracks from multiple cameras
        cameras = [8, 9, 10, 11]
        tracks_per_camera = 3
        
        logger.info(f"📊 Simulating {tracks_per_camera} tracks per camera for cameras: {cameras}")
        
        for camera_id in cameras:
            for track_num in range(1, tracks_per_camera + 1):
                track_id = camera_id * 1000 + track_num  # e.g., 8001, 8002, 8003
                
                # Simulate track data
                track_data = {
                    'global_id': track_id,
                    'camera_id': camera_id,
                    'last_bbox': [100 + track_num * 50, 100, 200 + track_num * 50, 200],
                    'confidence': 0.8 + (track_num * 0.05),
                    'class': 0,  # Assuming class 0 for all
                    'track_age': track_num * 5,
                    'last_seen': time.time() - (track_num * 2),  # Staggered timestamps
                    'has_features': True
                }
                
                # Store in Redis (Redis 3.0 compatible)
                redis_key = f"camera_{camera_id}:track_{track_id}"
                for field, value in track_data.items():
                    redis_client.hset(redis_key, field, pickle.dumps(value))
                redis_client.expire(redis_key, 600)  # 10 minutes
                
                logger.info(f"   📦 Stored track {track_id} for Camera {camera_id}")
        
        # Test cross-camera lookup simulation
        logger.info("🔄 Testing cross-camera lookup...")
        
        # Simulate looking for tracks from Camera 9 when processing Camera 8
        pattern = "camera_9:track_*"
        neighbor_tracks = redis_client.keys(pattern)
        
        logger.info(f"🔍 Found {len(neighbor_tracks)} tracks from Camera 9:")
        for key in neighbor_tracks[:2]:  # Show first 2
            track_data = redis_client.hgetall(key)
            if track_data:
                deserialized = {k.decode(): pickle.loads(v) for k, v in track_data.items()}
                logger.info(f"   Track {deserialized['global_id']}: bbox={deserialized['last_bbox']}, conf={deserialized['confidence']:.3f}")
        
        # Clean up
        all_keys = redis_client.keys("camera_*:track_*")
        if all_keys:
            redis_client.delete(*all_keys)
            logger.info(f"🧹 Cleaned up {len(all_keys)} test tracks")
        
        logger.info("✅ Smart Persistence simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smart Persistence simulation failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Redis Setup and Smart Persistence Test")
    
    # Step 1: Install Redis Python client
    if not install_redis_python_client():
        logger.error("❌ Failed to install Redis Python client")
        return False
    
    # Step 2: Test Redis connection
    if not test_redis_connection():
        logger.error("❌ Redis connection test failed")
        logger.info("💡 Please install and start Redis server:")
        logger.info("   Windows: Download from https://github.com/microsoftarchive/redis/releases")
        logger.info("   Or use Docker: docker run -d -p 6379:6379 redis:latest")
        logger.info("   Or use WSL: sudo apt install redis-server && redis-server")
        return False
    
    # Step 3: Test Smart Persistence simulation
    if not test_smart_persistence_simulation():
        logger.error("❌ Smart Persistence simulation failed")
        return False
    
    logger.info("🎉 All tests passed! Smart Persistence system is ready!")
    logger.info("📋 Next steps:")
    logger.info("   1. Make sure Redis server is running: redis-server")
    logger.info("   2. Run your BoT-SORT pipeline: python test_pipline_with_BoT_persistence.py")
    logger.info("   3. Check logs for cross-camera tracking messages")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
