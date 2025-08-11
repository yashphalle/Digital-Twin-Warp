#!/usr/bin/env python3
"""
Test script to verify the numpy hashable fix works
"""

import numpy as np

def test_numpy_hashable_conversion():
    """Test converting numpy arrays to hashable types"""
    print("🧪 Testing Numpy Hashable Conversion Fix")
    print("=" * 50)
    
    # Simulate the problematic scenario
    tracked_objects = [
        {'track_id': np.array(8001)},  # This was causing the error
        {'track_id': np.array(8002)},
        {'track_id': 8003},            # Regular int
        {'track_id': None},            # None value
    ]
    
    print("📊 Original tracked objects:")
    for i, track in enumerate(tracked_objects):
        track_id = track.get('track_id')
        print(f"  Track {i}: {track_id} (type: {type(track_id)})")
    
    print("\n🔧 Applying hashable conversion fix:")
    
    # Apply the fix
    tracked_ids = set()
    for track in tracked_objects:
        track_id = track.get('track_id')
        if track_id is not None:
            # Convert numpy arrays to Python int/float for hashability
            if hasattr(track_id, 'item'):
                track_id = track_id.item()
                print(f"  ✅ Converted numpy array to: {track_id} (type: {type(track_id)})")
            else:
                print(f"  ✅ Already hashable: {track_id} (type: {type(track_id)})")
            tracked_ids.add(track_id)
        else:
            print(f"  ⚠️ Skipped None track_id")
    
    print(f"\n📋 Final tracked_ids set: {tracked_ids}")
    
    # Test the comparison logic
    print("\n🔍 Testing track ID comparisons:")
    test_track_ids = [
        np.array(8001),  # Should match
        np.array(8005),  # Should not match
        8002,            # Should match
        8006,            # Should not match
    ]
    
    for test_id in test_track_ids:
        # Apply same conversion
        if hasattr(test_id, 'item'):
            converted_id = test_id.item()
        else:
            converted_id = test_id
            
        is_tracked = converted_id in tracked_ids
        status = "✅ TRACKED" if is_tracked else "❌ UNMATCHED"
        print(f"  ID {test_id} → {converted_id}: {status}")
    
    print("\n✅ Numpy Hashable Conversion Test Complete!")
    print("The 'unhashable type: numpy.ndarray' error should now be fixed!")

if __name__ == "__main__":
    test_numpy_hashable_conversion()
