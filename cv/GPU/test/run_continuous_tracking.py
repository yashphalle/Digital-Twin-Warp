#!/usr/bin/env python3
"""
Continuous Warehouse Tracking System
====================================

Simple script to run the warehouse tracking system continuously.
This script provides easy-to-use functions for different tracking modes.

Usage Examples:
    # Run continuous tracking on 3 cameras
    python run_continuous_tracking.py

    # Run with all 11 cameras (modify CAMERA_IDS below)
    # CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Run timed test for 5 minutes
    # run_timed_test(duration=300)
"""

import sys
import os
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from test_pipeline_with_database import run_continuous_tracking, run_timed_test, test_detection_speed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
CAMERA_IDS = [8, 9, 10]  # Default cameras - modify as needed
CONFIDENCE = 0.5         # Detection confidence threshold
ENABLE_GUI = True        # Show GUI display

def main():
    """Main function - choose your tracking mode"""
    
    print("=" * 60)
    print("🚀 WAREHOUSE TRACKING SYSTEM")
    print("=" * 60)
    print("Available modes:")
    print("1. Continuous tracking (runs until stopped)")
    print("2. Timed test (runs for specific duration)")
    print("3. Custom configuration")
    print("=" * 60)
    
    # Get user choice
    try:
        choice = input("Enter your choice (1-3) or press Enter for continuous mode: ").strip()
        
        if choice == "2":
            # Timed test mode
            duration = input("Enter test duration in seconds (default: 60): ").strip()
            duration = int(duration) if duration else 60
            
            print(f"\n🚀 Starting {duration}-second tracking test...")
            print("💡 Press Ctrl+C to stop early")
            
            run_timed_test(
                camera_ids=CAMERA_IDS,
                duration=duration,
                confidence=CONFIDENCE,
                enable_gui=ENABLE_GUI
            )
            
        elif choice == "3":
            # Custom configuration mode
            print("\n🔧 Custom Configuration Mode")
            
            # Get camera IDs
            camera_input = input(f"Enter camera IDs (default: {CAMERA_IDS}): ").strip()
            if camera_input:
                try:
                    camera_ids = [int(x.strip()) for x in camera_input.split(',')]
                except ValueError:
                    print("Invalid camera IDs, using default")
                    camera_ids = CAMERA_IDS
            else:
                camera_ids = CAMERA_IDS
            
            # Get confidence
            conf_input = input(f"Enter confidence threshold (default: {CONFIDENCE}): ").strip()
            confidence = float(conf_input) if conf_input else CONFIDENCE
            
            # Get duration
            duration_input = input("Enter duration in seconds (Enter for continuous): ").strip()
            duration = int(duration_input) if duration_input else None
            
            print(f"\n🚀 Starting custom tracking...")
            print(f"📹 Cameras: {camera_ids}")
            print(f"🎯 Confidence: {confidence}")
            print(f"⏱️ Duration: {'Continuous' if duration is None else f'{duration}s'}")
            print("💡 Press 'q' in GUI or Ctrl+C to stop")
            
            test_detection_speed(
                camera_ids=camera_ids,
                duration=duration,
                confidence=confidence,
                enable_gui=ENABLE_GUI,
                enable_tracking=True,
                enable_database=True,
                continuous=(duration is None)
            )
            
        else:
            # Default: Continuous mode
            print(f"\n🚀 Starting CONTINUOUS tracking...")
            print(f"📹 Cameras: {CAMERA_IDS}")
            print(f"🎯 Confidence: {CONFIDENCE}")
            print("💡 Press 'q' in GUI window or Ctrl+C to stop")
            
            run_continuous_tracking(
                camera_ids=CAMERA_IDS,
                confidence=CONFIDENCE,
                enable_gui=ENABLE_GUI
            )
            
    except KeyboardInterrupt:
        print("\n🛑 Tracking stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"\n❌ Error occurred: {e}")

def quick_start_continuous():
    """Quick start function for continuous tracking with default settings"""
    print("🚀 Quick Start: Continuous Tracking")
    print(f"📹 Cameras: {CAMERA_IDS}")
    print("💡 Press 'q' in GUI window or Ctrl+C to stop")
    
    run_continuous_tracking(
        camera_ids=CAMERA_IDS,
        confidence=CONFIDENCE,
        enable_gui=ENABLE_GUI
    )

def quick_start_timed(duration=300):
    """Quick start function for timed tracking (default: 5 minutes)"""
    print(f"🚀 Quick Start: {duration}-second Tracking Test")
    print(f"📹 Cameras: {CAMERA_IDS}")
    print("💡 Press Ctrl+C to stop early")
    
    run_timed_test(
        camera_ids=CAMERA_IDS,
        duration=duration,
        confidence=CONFIDENCE,
        enable_gui=ENABLE_GUI
    )

if __name__ == "__main__":
    """
    🚀 WAREHOUSE TRACKING SYSTEM - EASY LAUNCHER
    
    This script provides an easy way to run the warehouse tracking system
    in different modes without modifying the main test file.
    
    Features:
    - Interactive mode selection
    - Continuous tracking (production mode)
    - Timed testing (development/testing)
    - Custom configuration
    - Real-time database integration
    - GUI display with tracking visualization
    """
    
    # Uncomment one of these for direct execution:
    
    # Option 1: Interactive mode (default)
    main()
    
    # Option 2: Direct continuous mode
    # quick_start_continuous()
    
    # Option 3: Direct timed mode (5 minutes)
    # quick_start_timed(duration=300)
