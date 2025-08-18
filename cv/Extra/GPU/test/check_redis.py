#!/usr/bin/env python3
"""
Redis Monitoring Script for Smart Persistence System
Shows all stored tracks, cross-camera data, and system status
"""

import redis
import pickle
import time
import json
from datetime import datetime

def connect_redis():
    """Connect to Redis server"""
    try:
        client = redis.Redis(
            host='localhost', 
            port=6379,
            decode_responses=False,
            socket_keepalive=True,
            health_check_interval=30
        )
        client.ping()
        print("✅ Connected to Redis successfully!")
        return client
    except redis.ConnectionError:
        print("❌ Redis connection failed! Make sure Redis server is running.")
        print("💡 Start Redis server with: Redis\\redis-server.exe Redis\\redis.windows.conf")
        return None
    except Exception as e:
        print(f"❌ Redis connection error: {e}")
        return None

def check_redis_info(client):
    """Display Redis server information"""
    print("\n" + "="*60)
    print("📊 REDIS SERVER INFORMATION")
    print("="*60)
    
    info = client.info()
    print(f"Redis Version: {info.get('redis_version', 'Unknown')}")
    print(f"Connected Clients: {info.get('connected_clients', 0)}")
    print(f"Used Memory: {info.get('used_memory_human', 'Unknown')}")
    print(f"Total Keys: {info.dbsize()}")
    print(f"Uptime: {info.get('uptime_in_seconds', 0)} seconds")

def show_all_tracks(client):
    """Display all stored tracks"""
    print("\n" + "="*60)
    print("🎯 ALL STORED TRACKS")
    print("="*60)
    
    # Get all track keys
    track_keys = client.keys("camera_*:track_*")
    
    if not track_keys:
        print("📭 No tracks found in Redis")
        return
    
    print(f"📊 Found {len(track_keys)} stored tracks:")
    
    tracks_by_camera = {}
    
    for key in track_keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        
        # Parse camera and track ID from key
        try:
            parts = key_str.split(':')
            camera_part = parts[0].replace('camera_', '')
            track_part = parts[1].replace('track_', '')
            
            camera_id = int(camera_part)
            track_id = int(track_part)
            
            if camera_id not in tracks_by_camera:
                tracks_by_camera[camera_id] = []
            
            # Get track data
            track_data = client.hgetall(key)
            if track_data:
                decoded_data = {}
                for field, value in track_data.items():
                    field_str = field.decode() if isinstance(field, bytes) else field
                    try:
                        decoded_data[field_str] = pickle.loads(value)
                    except:
                        decoded_data[field_str] = value
                
                tracks_by_camera[camera_id].append({
                    'track_id': track_id,
                    'data': decoded_data
                })
        except Exception as e:
            print(f"⚠️ Error parsing key {key_str}: {e}")
    
    # Display tracks organized by camera
    for camera_id in sorted(tracks_by_camera.keys()):
        tracks = tracks_by_camera[camera_id]
        print(f"\n📹 Camera {camera_id}: {len(tracks)} tracks")
        
        for track in sorted(tracks, key=lambda x: x['track_id']):
            data = track['data']
            track_id = track['track_id']
            
            # Format track info
            bbox = data.get('last_bbox', 'Unknown')
            confidence = data.get('confidence', 'Unknown')
            age = data.get('track_age', 'Unknown')
            last_seen = data.get('last_seen', 'Unknown')
            
            if isinstance(last_seen, (int, float)):
                last_seen_str = f"{time.time() - last_seen:.1f}s ago"
            else:
                last_seen_str = str(last_seen)
            
            print(f"   🎯 Track {track_id}: bbox={bbox}, conf={confidence}, age={age}, seen={last_seen_str}")

def show_cross_camera_potential(client):
    """Show potential cross-camera matches"""
    print("\n" + "="*60)
    print("🔄 CROSS-CAMERA ANALYSIS")
    print("="*60)
    
    # Camera neighbor mapping (same as in Smart Persistence Manager)
    camera_neighbors = {
        1: [2, 5], 2: [1, 3, 6], 3: [2, 4, 7], 4: [3],
        5: [1, 6, 8], 6: [5, 7, 2, 9], 7: [6, 3, 10],
        8: [5, 9], 9: [8, 10, 6], 10: [9, 11, 7], 11: [10]
    }
    
    # Get all cameras with tracks
    track_keys = client.keys("camera_*:track_*")
    active_cameras = set()
    
    for key in track_keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        try:
            camera_id = int(key_str.split(':')[0].replace('camera_', ''))
            active_cameras.add(camera_id)
        except:
            continue
    
    print(f"📊 Active cameras with tracks: {sorted(active_cameras)}")
    
    # Show neighbor relationships for active cameras
    for camera_id in sorted(active_cameras):
        neighbors = camera_neighbors.get(camera_id, [])
        active_neighbors = [n for n in neighbors if n in active_cameras]
        
        if active_neighbors:
            print(f"📹 Camera {camera_id} → Neighbors with tracks: {active_neighbors}")
            
            # Count tracks in neighboring cameras
            for neighbor_id in active_neighbors:
                neighbor_keys = client.keys(f"camera_{neighbor_id}:track_*")
                print(f"   🔗 Camera {neighbor_id}: {len(neighbor_keys)} tracks available for cross-camera matching")

def monitor_redis_activity(client, duration=30):
    """Monitor Redis activity in real-time"""
    print(f"\n" + "="*60)
    print(f"📡 MONITORING REDIS ACTIVITY ({duration}s)")
    print("="*60)
    print("Press Ctrl+C to stop monitoring...")
    
    try:
        # Use Redis MONITOR command to see all operations
        print("🔍 Starting Redis monitor...")
        print("💡 This will show all Redis commands in real-time")
        
        start_time = time.time()
        initial_keys = client.dbsize()
        
        while time.time() - start_time < duration:
            current_keys = client.dbsize()
            if current_keys != initial_keys:
                print(f"📊 Key count changed: {initial_keys} → {current_keys}")
                initial_keys = current_keys
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")

def clear_all_tracks(client):
    """Clear all stored tracks (use with caution!)"""
    print("\n" + "="*60)
    print("🗑️ CLEAR ALL TRACKS")
    print("="*60)
    
    track_keys = client.keys("camera_*:track_*")
    
    if not track_keys:
        print("📭 No tracks to clear")
        return
    
    print(f"⚠️ Found {len(track_keys)} tracks to delete")
    confirm = input("Are you sure you want to delete ALL tracks? (yes/no): ")
    
    if confirm.lower() == 'yes':
        deleted = client.delete(*track_keys)
        print(f"✅ Deleted {deleted} tracks from Redis")
    else:
        print("❌ Operation cancelled")

def main():
    """Main function"""
    print("🚀 Redis Smart Persistence Monitor")
    print("="*60)
    
    # Connect to Redis
    client = connect_redis()
    if not client:
        return
    
    while True:
        print("\n" + "="*60)
        print("📋 REDIS MONITORING OPTIONS")
        print("="*60)
        print("1. Show Redis server info")
        print("2. Show all stored tracks")
        print("3. Show cross-camera analysis")
        print("4. Monitor Redis activity (30s)")
        print("5. Clear all tracks (DANGER!)")
        print("6. Refresh/Check again")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                check_redis_info(client)
            elif choice == '2':
                show_all_tracks(client)
            elif choice == '3':
                show_cross_camera_potential(client)
            elif choice == '4':
                monitor_redis_activity(client)
            elif choice == '5':
                clear_all_tracks(client)
            elif choice == '6':
                print("🔄 Refreshing...")
                continue
            else:
                print("❌ Invalid option. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
