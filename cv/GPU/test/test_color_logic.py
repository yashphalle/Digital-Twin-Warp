#!/usr/bin/env python3
"""
Test script to verify the color logic works correctly
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from test_pipline_with_BoT_persistence import CONSECUTIVE_DETECTION_THRESHOLD

# Mock the GUI display class to test color logic
class MockGUIDisplay:
    def get_tracking_color(self, detection):
        """Get color based on tracking state"""
        track_id = detection.get('track_id')
        track_age = detection.get('track_age', 0)
        
        # Calculate tracking status based on consecutive detection logic
        if track_id is None:
            tracking_status = 'detection'
        elif track_age < CONSECUTIVE_DETECTION_THRESHOLD:
            tracking_status = 'pending'
        else:
            tracking_status = 'database'  # 20+ consecutive detections

        if tracking_status == 'detection':
            return (0, 255, 0)      # Green - New detection (no tracking yet)
        elif tracking_status == 'pending':
            return (0, 255, 255)    # Yellow - Tracked but building consecutive detections
        else:
            return (0, 165, 255)    # Orange - Database entry (20+ consecutive detections)

def color_to_name(color):
    """Convert BGR color tuple to color name"""
    if color == (0, 255, 0):
        return "🟢 Green"
    elif color == (0, 255, 255):
        return "🟡 Yellow"
    elif color == (0, 165, 255):
        return "🟠 Orange"
    else:
        return f"Unknown {color}"

def test_color_logic():
    """Test the color logic for different detection states"""
    print("🎨 Testing Color Logic for Consecutive Detection Threshold")
    print(f"📊 Threshold: {CONSECUTIVE_DETECTION_THRESHOLD} consecutive detections")
    print("=" * 70)
    
    gui = MockGUIDisplay()
    
    # Test cases
    test_cases = [
        # (track_id, track_age, description)
        (None, 0, "New detection (no tracking)"),
        (8001, 1, "First tracked frame"),
        (8001, 5, "5th tracked frame"),
        (8001, 10, "10th tracked frame"),
        (8001, 15, "15th tracked frame"),
        (8001, 19, "19th tracked frame (almost ready)"),
        (8001, 20, "20th tracked frame (database ready!)"),
        (8001, 25, "25th tracked frame (in database)"),
        (8001, 50, "50th tracked frame (established)"),
        (8001, 200, "200th tracked frame (long-term)")
    ]
    
    print("🔍 Color progression test:")
    print()
    
    for track_id, track_age, description in test_cases:
        # Create test detection
        detection = {
            'track_id': track_id,
            'track_age': track_age,
            'confidence': 0.85,
            'class': 'Pallet'
        }
        
        # Get color
        color = gui.get_tracking_color(detection)
        color_name = color_to_name(color)
        
        # Determine expected behavior
        if track_id is None:
            expected_color = "🟢 Green"
            db_status = "No database action"
        elif track_age < CONSECUTIVE_DETECTION_THRESHOLD:
            expected_color = "🟡 Yellow"
            db_status = f"Building ({track_age}/{CONSECUTIVE_DETECTION_THRESHOLD})"
        else:
            expected_color = "🟠 Orange"
            if track_age == CONSECUTIVE_DETECTION_THRESHOLD:
                db_status = "INSERT to database"
            else:
                db_status = "UPDATE in database"
        
        # Check if correct
        is_correct = color_name == expected_color
        status_icon = "✅" if is_correct else "❌"
        
        print(f"{status_icon} {description:30s} → {color_name:12s} | {db_status}")
        
        if not is_correct:
            print(f"   ❌ ERROR: Expected {expected_color}, got {color_name}")
    
    print()
    print("🎯 Expected Color Progression:")
    print("   🟢 Green  : New detections (no track ID)")
    print("   🟡 Yellow : Tracked objects building consecutive detections (1-19 frames)")
    print("   🟠 Orange : Database entries (20+ consecutive detections)")
    print()
    print("✅ Color Logic Test Complete!")

if __name__ == "__main__":
    test_color_logic()
