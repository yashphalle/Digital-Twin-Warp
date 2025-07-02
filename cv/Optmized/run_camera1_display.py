#!/usr/bin/env python3
"""
Run Camera 1 Display
Simple script to launch the Camera 1 GUI display
"""

import sys
import os

# Add the cv directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_camera_display import SingleCameraDisplay

def main():
    """Run the Camera 1 display"""
    print("LAUNCHING CAMERA 1 DISPLAY")
    print("=" * 50)
    
    try:
        # Create and start the camera display
        camera_display = SingleCameraDisplay(camera_id=1)
        camera_display.start_display()
        
        # Keep the main thread alive
        while camera_display.is_running():
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down Camera 1 display...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'camera_display' in locals():
            camera_display.stop_display()

if __name__ == "__main__":
    main()
