#!/usr/bin/env python3
"""
Test script to verify consecutive detection logic
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from test_pipline_with_BoT_persistence import enrich_detection_fast, CONSECUTIVE_DETECTION_THRESHOLD

def test_consecutive_detection_logic():
    """Test the consecutive detection threshold logic"""
    print("🧪 Testing Consecutive Detection Logic")
    print(f"📊 Threshold: {CONSECUTIVE_DETECTION_THRESHOLD} consecutive detections")
    print("=" * 60)
    
    # Mock detection data
    base_detection = {
        'bbox': [100, 100, 200, 200],
        'confidence': 0.85,
        'class': 'Pallet',
        'track_id': 8001,
        'camera_id': 8
    }
    
    # Test different track ages
    test_cases = [
        (1, "First detection"),
        (2, "Second detection"), 
        (3, "Third detection"),
        (4, "Fourth detection"),
        (5, "Fifth detection - Should be ready for DB"),
        (6, "Sixth detection - Should be existing"),
        (10, "Tenth detection - Should be existing")
    ]
    
    print("🔍 Testing tracking status for different consecutive detection counts:")
    print()
    
    for track_age, description in test_cases:
        # Create test detection
        test_detection = base_detection.copy()
        test_detection['track_age'] = track_age
        
        # Test enrichment (this applies the consecutive detection logic)
        enriched = enrich_detection_fast(
            detection=test_detection,
            coordinate_mappers={},  # Empty for test
            color_info={}  # Empty for test
        )
        
        tracking_status = enriched['tracking_status']
        consecutive_detections = enriched['consecutive_detections']
        
        # Determine expected behavior
        if track_age < CONSECUTIVE_DETECTION_THRESHOLD:
            expected_status = "pending"
            db_action = "❌ Skip DB (building consecutive detections)"
        elif track_age == CONSECUTIVE_DETECTION_THRESHOLD:
            expected_status = "new"
            db_action = "✅ INSERT to DB (threshold reached)"
        else:
            expected_status = "existing"
            db_action = "✅ UPDATE in DB"
            
        # Check if logic is correct
        status_correct = tracking_status == expected_status
        status_icon = "✅" if status_correct else "❌"
        
        print(f"{status_icon} Track Age {track_age:2d}: {description}")
        print(f"   Status: {tracking_status:8s} | Consecutive: {consecutive_detections:2d}/{CONSECUTIVE_DETECTION_THRESHOLD} | {db_action}")
        
        if not status_correct:
            print(f"   ❌ ERROR: Expected '{expected_status}', got '{tracking_status}'")
        print()
    
    print("🎨 Visual Indicators:")
    print("   Gray   (pending)  : Building consecutive detections")
    print("   Yellow (new)      : Ready for database insertion") 
    print("   Orange (existing) : Already in database")
    print()
    
    print("📊 Database Behavior:")
    print(f"   Detections 1-{CONSECUTIVE_DETECTION_THRESHOLD-1}: Tracked but NOT saved to database")
    print(f"   Detection {CONSECUTIVE_DETECTION_THRESHOLD}:     INSERT new record to database")
    print(f"   Detections {CONSECUTIVE_DETECTION_THRESHOLD+1}+:      UPDATE existing record in database")
    print()
    
    print("✅ Consecutive Detection Logic Test Complete!")

if __name__ == "__main__":
    test_consecutive_detection_logic()
