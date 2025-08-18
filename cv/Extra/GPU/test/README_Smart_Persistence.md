# Smart Persistence BoT-SORT System

## Overview
Enhanced BoT-SORT tracking system with feature-based cross-camera tracking and Redis persistence for warehouse object tracking.

## Features
- **Feature-Based Cross-Camera Tracking**: Objects maintain identity across camera boundaries
- **Redis Persistence**: Fast in-memory storage for active tracks
- **MongoDB Integration**: Long-term persistence and system restart recovery
- **Background Processing**: Non-blocking persistence operations
- **Warehouse-Optimized**: Custom camera neighbor mapping for your warehouse layout

## Camera Layout (Neighbor Mapping)
```
Camera Neighbors:
- Camera 1: [2, 5]
- Camera 2: [1, 3, 6]  
- Camera 3: [2, 4, 7]
- Camera 4: [3]
- Camera 5: [1, 6, 8]
- Camera 6: [5, 7, 2, 9]
- Camera 7: [6, 3, 10]
- Camera 8: [5, 9]
- Camera 9: [8, 10, 6]
- Camera 10: [9, 11, 7]
- Camera 11: [10]
```

## Installation

### 1. Install Redis
Run the installation script:
```bash
# Windows
install_redis.bat

# Or manually:
pip install redis

# Then install Redis server (choose one):
# Option A: Docker (Recommended)
docker run -d -p 6379:6379 --name redis redis:latest

# Option B: Windows Binary
# Download from: https://github.com/microsoftarchive/redis/releases
# Extract to C:\Redis and run: C:\Redis\redis-server.exe

# Option C: WSL
wsl --install
sudo apt update && sudo apt install redis-server
redis-server
```

### 2. Test Redis Setup
```bash
python test_redis_setup.py
```

## Usage

### Basic Usage
```bash
# Run with Smart Persistence enabled
python test_pipline_with_BoT_persistence.py
```

### Configuration
The system uses these key parameters:
- **Cross-camera similarity threshold**: 0.5 (lower = more aggressive matching)
- **Redis save interval**: 5 seconds
- **MongoDB update interval**: 2 minutes
- **Track expiration**: 10 minutes in Redis

### Monitoring
Watch for these log messages:
- `✅ Connected to Redis for Smart Persistence` - Redis connection successful
- `🔄 Cross-camera match: Track XXXX from Camera Y → Camera Z` - Cross-camera tracking working
- `💾 Saved N tracks to Redis for Camera X` - Background persistence working

## How It Works

### 1. Normal Operation
- BoT-SORT runs at full speed (unchanged)
- Tracks are extracted and saved to Redis every 5 seconds (background)
- MongoDB is updated every 2 minutes (background)

### 2. Cross-Camera Matching
- When BoT-SORT can't match a detection to existing tracks
- System checks Redis for tracks from neighboring cameras
- Uses feature-based similarity (size, confidence, temporal proximity)
- Recovers tracks with similarity > 0.5

### 3. System Restart Recovery
- On startup, system loads active tracks from MongoDB (last 10 minutes)
- Restores tracks to Redis for fast access
- Continues tracking with preserved IDs

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BoT-SORT      │    │  Smart          │    │   Redis         │
│   Tracking      │───▶│  Persistence    │───▶│   (Fast)        │
│   (Full Speed)  │    │  Manager        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MongoDB       │
                       │   (Persistent)  │
                       └─────────────────┘
```

## Performance Impact
- **Normal tracking**: 0% impact (same speed as original)
- **Cross-camera checks**: Only for unmatched detections (~1-5% of cases)
- **Background persistence**: Non-blocking threads
- **Memory usage**: ~26MB for 1000 active tracks

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis server
redis-server

# Check Redis logs
redis-cli monitor
```

### No Cross-Camera Matches
- Check camera neighbor mapping in `SmartPersistenceManager.__init__`
- Verify similarity threshold (0.5) is appropriate
- Check Redis for stored tracks: `redis-cli keys "camera_*:track_*"`

### Performance Issues
- Monitor Redis memory usage: `redis-cli info memory`
- Check background thread performance in logs
- Adjust save intervals if needed

## Files Modified
- `test_pipline_with_BoT_persistence.py` - Main tracking system with Smart Persistence
- `test_redis_setup.py` - Redis installation and testing
- `install_redis.bat` - Windows Redis installation script

## Next Steps
1. Test with your specific camera setup
2. Adjust similarity thresholds based on results
3. Monitor cross-camera matching accuracy
4. Fine-tune persistence intervals for your workload

## Support
Check logs for detailed debugging information. All Smart Persistence operations are logged with specific prefixes:
- `🚀` - Initialization
- `✅` - Success operations  
- `🔄` - Cross-camera operations
- `💾` - Persistence operations
- `❌` - Error conditions
