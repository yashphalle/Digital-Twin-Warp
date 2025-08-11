#!/usr/bin/env python3
"""
Mock Test for Smart Persistence System (without Redis server)
This tests the Smart Persistence logic without requiring a running Redis server
"""

import sys
import os
import time
import logging
import numpy as np
from unittest.mock import Mock, MagicMock
import pickle

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockRedisClient:
    """Mock Redis client for testing without Redis server"""
    
    def __init__(self):
        self.data = {}
        self.hash_data = {}
        self.expiry = {}
    
    def ping(self):
        return True
    
    def set(self, key, value):
        self.data[key] = value
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def hset(self, key, mapping=None, **kwargs):
        if key not in self.hash_data:
            self.hash_data[key] = {}
        if mapping:
            self.hash_data[key].update(mapping)
        if kwargs:
            self.hash_data[key].update(kwargs)
        return True
    
    def hgetall(self, key):
        return self.hash_data.get(key, {})
    
    def expire(self, key, seconds):
        self.expiry[key] = time.time() + seconds
        return True
    
    def keys(self, pattern):
        # Simple pattern matching for camera_*:track_* patterns
        if pattern == "camera_*:track_*":
            return [k.encode() for k in self.hash_data.keys() if k.startswith("camera_") and ":track_" in k]
        elif pattern.startswith("camera_") and pattern.endswith(":track_*"):
            camera_prefix = pattern.replace(":track_*", "")
            return [k.encode() for k in self.hash_data.keys() if k.startswith(camera_prefix)]
        return []
    
    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)
            self.hash_data.pop(key, None)
            self.expiry.pop(key, None)
        return len(keys)

def test_smart_persistence_logic():
    """Test Smart Persistence logic with mock Redis"""
    logger.info("🧪 Testing Smart Persistence Logic (Mock Mode)")
    
    # Import the SmartPersistenceManager from our modified file
    try:
        # Mock redis module
        import sys
        sys.modules['redis'] = Mock()
        
        # Import our SmartPersistenceManager
        from test_pipline_with_BoT_persistence import SmartPersistenceManager
        
        # Create instance with mock Redis
        smart_persistence = SmartPersistenceManager([8, 9, 10, 11])
        smart_persistence.redis_client = MockRedisClient()
        
        logger.info("✅ SmartPersistenceManager created successfully")
        
        # Test 1: Save tracks to Redis
        logger.info("📊 Test 1: Saving tracks to Redis...")
        
        test_tracks = [
            {
                'track_id': 8001,
                'bbox': [100, 100, 200, 200],
                'confidence': 0.85,
                'class': 0,
                'track_age': 5
            },
            {
                'track_id': 8002,
                'bbox': [300, 150, 400, 250],
                'confidence': 0.92,
                'class': 0,
                'track_age': 12
            }
        ]
        
        smart_persistence.save_tracks_to_redis(test_tracks, 8)
        logger.info(f"✅ Saved {len(test_tracks)} tracks for Camera 8")
        
        # Test 2: Cross-camera neighbor mapping
        logger.info("📊 Test 2: Testing camera neighbor mapping...")
        
        neighbors_8 = smart_persistence.camera_neighbors.get(8, [])
        neighbors_9 = smart_persistence.camera_neighbors.get(9, [])
        neighbors_10 = smart_persistence.camera_neighbors.get(10, [])
        
        logger.info(f"Camera 8 neighbors: {neighbors_8}")
        logger.info(f"Camera 9 neighbors: {neighbors_9}")
        logger.info(f"Camera 10 neighbors: {neighbors_10}")
        
        assert 9 in neighbors_8, "Camera 8 should have Camera 9 as neighbor"
        assert 8 in neighbors_9, "Camera 9 should have Camera 8 as neighbor"
        assert 6 in neighbors_9, "Camera 9 should have Camera 6 as neighbor"
        
        logger.info("✅ Camera neighbor mapping is correct")
        
        # Test 3: Feature similarity calculation
        logger.info("📊 Test 3: Testing feature similarity calculation...")
        
        detection = {
            'bbox': [105, 105, 205, 205],  # Similar to track 8001
            'confidence': 0.80,
            'class': 0
        }
        
        track_data = {
            'global_id': 8001,
            'camera_id': 8,
            'last_bbox': [100, 100, 200, 200],
            'confidence': 0.85,
            'class': 0,
            'track_age': 5,
            'last_seen': time.time() - 5  # 5 seconds ago
        }
        
        similarity = smart_persistence._calculate_feature_similarity(
            detection, track_data, 9, 8
        )
        
        logger.info(f"Similarity score: {similarity:.3f}")
        assert similarity > 0.5, f"Similarity should be > 0.5, got {similarity}"
        logger.info("✅ Feature similarity calculation working")
        
        # Test 4: Cross-camera matching simulation
        logger.info("📊 Test 4: Testing cross-camera matching...")
        
        # Add tracks for Camera 9
        camera_9_tracks = [
            {
                'track_id': 9001,
                'bbox': [150, 120, 250, 220],
                'confidence': 0.88,
                'class': 0,
                'track_age': 8
            }
        ]
        
        smart_persistence.save_tracks_to_redis(camera_9_tracks, 9)
        
        # Simulate unmatched detection on Camera 8
        unmatched_detections = [
            {
                'bbox': [155, 125, 255, 225],  # Similar to track 9001
                'confidence': 0.82,
                'class': 0
            }
        ]
        
        matches = smart_persistence.check_cross_camera_matches(unmatched_detections, 8)
        
        logger.info(f"Found {len(matches)} cross-camera matches")
        if matches:
            for match in matches:
                logger.info(f"  Match: Track {match['global_id']} from Camera {match['source_camera']} (similarity: {match['similarity']:.3f})")
        
        logger.info("✅ Cross-camera matching logic working")
        
        # Test 5: Background persistence simulation
        logger.info("📊 Test 5: Testing background persistence...")
        
        # This would normally run in a background thread
        smart_persistence.background_persistence(test_tracks, 8)
        logger.info("✅ Background persistence logic working")
        
        logger.info("🎉 All Smart Persistence tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smart Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_readiness():
    """Test if the integration is ready for real Redis"""
    logger.info("🔧 Testing integration readiness...")
    
    try:
        # Test imports
        from test_pipline_with_BoT_persistence import GPUBatchProcessorBoTSORT, SmartPersistenceManager
        logger.info("✅ All imports successful")
        
        # Test class initialization (without Redis)
        active_cameras = [8, 9, 10, 11]
        
        # Mock the Redis connection to avoid connection error
        import sys
        sys.modules['redis'] = Mock()
        
        smart_persistence = SmartPersistenceManager(active_cameras)
        smart_persistence.redis_client = None  # Simulate Redis connection failure
        
        logger.info("✅ SmartPersistenceManager initialization successful")
        
        # Test camera neighbor configuration
        expected_neighbors = {
            1: [2, 5], 2: [1, 3, 6], 3: [2, 4, 7], 4: [3],
            5: [1, 6, 8], 6: [5, 7, 2, 9], 7: [6, 3, 10],
            8: [5, 9], 9: [8, 10, 6], 10: [9, 11, 7], 11: [10]
        }
        
        for camera_id, expected in expected_neighbors.items():
            actual = smart_persistence.camera_neighbors.get(camera_id, [])
            assert set(actual) == set(expected), f"Camera {camera_id} neighbors mismatch: expected {expected}, got {actual}"
        
        logger.info("✅ Camera neighbor configuration is correct")
        
        logger.info("🎉 Integration readiness test passed!")
        logger.info("📋 Ready for Redis server installation and real testing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration readiness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Smart Persistence Mock Tests")
    
    success = True
    
    # Test 1: Smart Persistence Logic
    if not test_smart_persistence_logic():
        success = False
    
    # Test 2: Integration Readiness
    if not test_integration_readiness():
        success = False
    
    if success:
        logger.info("🎉 All mock tests passed!")
        logger.info("📋 Next steps:")
        logger.info("   1. Install Redis server:")
        logger.info("      - Docker: docker run -d -p 6379:6379 redis:latest")
        logger.info("      - Windows: Download from https://github.com/microsoftarchive/redis/releases")
        logger.info("      - WSL: sudo apt install redis-server && redis-server")
        logger.info("   2. Run: python test_redis_setup.py")
        logger.info("   3. Run: python test_pipline_with_BoT_persistence.py")
    else:
        logger.error("❌ Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
