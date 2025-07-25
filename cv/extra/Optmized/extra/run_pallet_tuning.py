#!/usr/bin/env python3
"""
Launch Interactive Pallet Detection Tuning
Simple launcher for prompt and threshold experimentation
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_pallet_tuning import InteractivePalletTuning

def main():
    """Launch the interactive pallet detection tuning"""
    print("LAUNCHING INTERACTIVE PALLET DETECTION TUNING")
    print("=" * 60)
    print("Real-time prompt and threshold experimentation")
    print("")
    print("Sample Prompts Available:")
    print("- wooden pallet")
    print("- shipping pallet") 
    print("- warehouse pallet")
    print("- pallet on floor")
    print("- wooden platform")
    print("- freight pallet")
    print("- wooden skid")
    print("- rectangular wooden structure")
    print("- pallet on concrete")
    print("- industrial pallet")
    print("- storage pallet")
    print("- wooden shipping platform")
    print("")
    print("Controls:")
    print("  'n' - Next prompt")
    print("  'p' - Previous prompt") 
    print("  '+' - Increase confidence threshold")
    print("  '-' - Decrease confidence threshold")
    print("  'd' - Toggle detection on/off")
    print("  's' - Save current optimal settings")
    print("  'q' - Quit and save")
    print("=" * 60)
    
    try:
        # Create and start the tuning interface
        tuning_interface = InteractivePalletTuning(camera_id=1)
        tuning_interface.start_tuning()
        
        # Keep the main thread alive
        while tuning_interface.is_running():
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down pallet detection tuning...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tuning_interface' in locals():
            tuning_interface.stop_tuning()

if __name__ == "__main__":
    main()
