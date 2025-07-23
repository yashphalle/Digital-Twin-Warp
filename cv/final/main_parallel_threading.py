#!/usr/bin/env python3
"""
Parallel Threading System Test Entry Point
Tests the new parallel pipeline system alongside the existing optimized system
"""

import logging
import argparse
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging with UTF-8 encoding for Windows compatibility (SAME as main_optimized_threading.py)
import io

# Create UTF-8 compatible stream handler
utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(utf8_stdout),
        logging.FileHandler('parallel_pipeline_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for parallel pipeline testing"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parallel Pipeline System Test')
    parser.add_argument('--cameras', nargs='+', type=int, default=[1], 
                       help='Camera IDs to test (default: [1] for single camera test)')
    parser.add_argument('--no-gui', action='store_true', 
                       help='Run in headless mode (no GUI windows)')
    parser.add_argument('--test-mode', choices=['single', 'all'], default='single',
                       help='Test mode: single camera or all cameras')
    parser.add_argument('--tracking', choices=['sift', 'deepsort'], default='sift',
                       help='Tracking method: sift (original) or deepsort (new)')

    args = parser.parse_args()
    
    # Determine cameras to test
    if args.test_mode == 'single':
        ACTIVE_CAMERAS = [1]  # Start with single camera for safety
        logger.info("SINGLE CAMERA TEST MODE")
    elif args.test_mode == 'all':
        ACTIVE_CAMERAS = [1,2,3,4,5,6,7,8,9,10,11]  # All cameras
        logger.info("ALL CAMERAS TEST MODE")
    else:
        ACTIVE_CAMERAS = args.cameras
        logger.info(f"CUSTOM CAMERAS TEST MODE: {ACTIVE_CAMERAS}")
    
    # GUI Configuration
    ENABLE_GUI = not args.no_gui
    GUI_CAMERAS = ACTIVE_CAMERAS[:2] if ENABLE_GUI else []  # Limit GUI to 2 cameras for testing

    # Tracking Configuration
    USE_DEEPSORT = args.tracking == 'deepsort'

    logger.info("PARALLEL PIPELINE SYSTEM TEST")
    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    logger.info(f"GUI Cameras: {GUI_CAMERAS} ({len(GUI_CAMERAS)} windows)")
    logger.info(f"GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    logger.info(f"Tracking Method: {'DeepSORT (NEW)' if USE_DEEPSORT else 'SIFT (Original)'}")
    logger.info("=" * 80)
    logger.info("PARALLEL PROCESSING ARCHITECTURE:")
    logger.info("1) Camera Threads (same as optimized)")
    logger.info("2) GPU Detection Workers (same as optimized)")
    logger.info("3) NEW: PARALLEL Processing Threads (11 threads instead of 1)")
    logger.info("4) NEW: ASYNC Database Workers (11 workers instead of blocking)")
    logger.info("5) Per-Camera Performance Monitoring")
    if USE_DEEPSORT:
        logger.info("6) NEW: DeepSORT Object Tracking (GPU-accelerated)")
    else:
        logger.info("6) SIFT Object Tracking (original system)")
    logger.info("=" * 80)

    if ENABLE_GUI:
        logger.info("\nGUI Mode:")
        logger.info("- Green: New objects")
        logger.info("- Orange: CPU-tracked existing objects")
        logger.info("- Red: Failed tracking")
        logger.info("- Cyan: Physical coordinate labels")
        logger.info("- Press 'q' to quit")
    else:
        logger.info("\nHEADLESS Mode: No GUI windows, console logging only")

    logger.info("=" * 80)

    # Performance monitoring instructions
    logger.info("PERFORMANCE COMPARISON INSTRUCTIONS:")
    logger.info("   [1] Run this parallel system")
    logger.info("   [2] Compare FPS with main_optimized_threading.py")
    logger.info("   [3] Expected improvement: 7 FPS → 25-35 FPS")
    logger.info("   [4] Monitor per-camera FPS in logs")
    logger.info("   [5] Check database save performance")
    logger.info("=" * 80)

    try:
        logger.info(f"[INIT] Initializing parallel system for cameras: {ACTIVE_CAMERAS}")

        # Import and create parallel system
        from warehouse_threading.parallel_pipeline_system import ParallelPipelineSystem
        
        system = ParallelPipelineSystem(
            active_cameras=ACTIVE_CAMERAS,
            enable_gui=ENABLE_GUI,
            gui_cameras=GUI_CAMERAS,
            use_deepsort=USE_DEEPSORT
        )

        logger.info("[INIT] Parallel system initialized successfully")
        
        # Run the system
        system.run()

    except KeyboardInterrupt:
        logger.info("\n[EXIT] Keyboard interrupt received")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure you're running from the correct directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Parallel pipeline test completed")

if __name__ == "__main__":
    main()
