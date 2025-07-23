#!/usr/bin/env python3

import logging
import sys
import os
import argparse

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from warehouse_threading.per_camera_optimized_pipeline_system import PerCameraOptimizedPipelineSystem
from configs.config import Config

# Configure logging with UTF-8 encoding for Windows compatibility
import io

# Create UTF-8 compatible stream handler
utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(utf8_stdout),
        logging.FileHandler('per_camera_optimized.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Per-Camera Optimized Warehouse Threading System')
    parser.add_argument('--local', action='store_true',
                       help='Use local camera URLs (192.168.x.x) instead of remote URLs')
    parser.add_argument('--remote', action='store_true',
                       help='Use remote camera URLs (104.181.138.5) instead of local URLs')
    parser.add_argument('--local-db', action='store_true',
                       help='Use local MongoDB database (localhost:27017)')
    parser.add_argument('--online-db', action='store_true',
                       help='Use online MongoDB database (MongoDB Atlas)')
    parser.add_argument('--camera', type=int, default=8,
                       help='Single camera ID to process (default: 8)')
    parser.add_argument('--cameras', type=str, default='1,2,3,4,5,6,7,8,9,10,11',
                       help='Comma-separated list of camera IDs (default: 8,9,10)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run in headless mode without GUI')
    return parser.parse_args()

def main():

    # Parse command line arguments
    args = parse_arguments()

    # Configure camera URLs based on arguments
    if args.local:
        Config.switch_to_local_cameras()
    elif args.remote:
        Config.switch_to_remote_cameras()
    # If neither specified, use default from config (currently LOCAL)

    # Configure database based on arguments
    if args.local_db:
        Config.switch_to_local_database()
    elif args.online_db:
        Config.switch_to_online_database()
    # If neither specified, use default from config (currently LOCAL)

    # Parse camera list - support both --camera and --cameras
    if hasattr(args, 'cameras') and args.cameras:
        try:
            ACTIVE_CAMERAS = [int(x.strip()) for x in args.cameras.split(',')]
        except ValueError:
            logger.error(f"Invalid camera list: {args.cameras}")
            return
    else:
        ACTIVE_CAMERAS = [args.camera]  # Single camera mode

    # Show current camera configuration
    logger.info(f"📡 Camera URL Mode: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    for camera_id in ACTIVE_CAMERAS[:3]:  # Show first 3 cameras
        camera_info = Config.get_camera_info(camera_id)
        logger.info(f"📹 Camera {camera_id} URL: {camera_info['current_url']}")

    # Show current database configuration
    database_info = Config.get_database_info()
    logger.info(f"🗄️ Database Mode: {'LOCAL' if Config.USE_LOCAL_DATABASE else 'ONLINE'}")
    logger.info(f"💾 Database URI: {database_info['current_uri'][:50]}..." if len(database_info['current_uri']) > 50 else f"💾 Database URI: {database_info['current_uri']}")

    # GUI Configuration
    ENABLE_GUI = not args.no_gui
    GUI_CAMERAS = ACTIVE_CAMERAS if ENABLE_GUI else []

    logger.info("🚀 PER-CAMERA OPTIMIZED WAREHOUSE THREADING SYSTEM")
    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"📹 Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    logger.info(f"🖥️ GUI Cameras: {GUI_CAMERAS} ({len(GUI_CAMERAS)} windows)")
    logger.info(f"🎛️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    logger.info("=" * 80)
    logger.info("PER-CAMERA OPTIMIZED PROCESSING:")
    logger.info("1) 🔄 Separate Detection Queue Per Camera (SOLVES FRAME ORDERING)")
    logger.info("2) 🎯 Round-Robin Frame Selection (FAIR PROCESSING)")
    logger.info("3) 📊 Per-Camera Statistics Tracking")
    logger.info("4) ⚖️ Fair Processing Across All Cameras")
    logger.info("5) 🚀 Smart Frame Skipping (95% CPU savings)")
    logger.info("6) 🚀 Enhanced Queue Management")
    logger.info("7) 🚀 GPU Detection (Grounding DINO + post_process_grounded_object_detection)")
    logger.info("8) 🚀 CPU Area + Grid Cell Filtering")
    logger.info("9) 🚀 CPU Physical Coordinate Translation")
    logger.info("10) 🚀 CPU SIFT Feature Matching")
    logger.info("11) 🚀 CPU Persistent Object IDs")
    logger.info("12) 🚀 CPU Cross-Frame Tracking & Database")
    logger.info("=" * 80)

    if ENABLE_GUI:
        logger.info("\nGUI Mode:")
        logger.info("- Green: New objects")
        logger.info("- Orange: CPU-tracked existing objects")
        logger.info("- Red: Failed tracking")
        logger.info("- Cyan: Physical coordinate labels")
        logger.info("- Press 'q' to quit")
        logger.info("- Press 'n'/'p' to change detection prompts")
    else:
        logger.info("\nHEADLESS Mode: No GUI windows, console logging only")
    logger.info("=" * 80)

    # Performance monitoring instructions
    logger.info("PER-CAMERA SYSTEM BENEFITS:")
    logger.info("   [1] Fair Processing: Each camera gets equal detection time")
    logger.info("   [2] No Frame Ordering Issues: Camera 1 won't dominate processing")
    logger.info("   [3] Better Load Balancing: Round-robin ensures all cameras processed")
    logger.info("   [4] Detailed Statistics: Per-camera performance tracking")
    logger.info("   [5] Improved Throughput: Better GPU utilization across cameras")
    logger.info("   [6] COMPLETE PIPELINE: All processing modules included")
    logger.info("=" * 80)

    # Frame ordering solution explanation
    logger.info("FRAME ORDERING SOLUTION:")
    logger.info("   ❌ OLD: Single shared queue → Camera 1 frames processed multiple times")
    logger.info("   ✅ NEW: Per-camera queues → Fair round-robin processing")
    logger.info("   📊 MONITORING: Balance ratio shows processing fairness (1.0 = perfect)")
    logger.info("   🔧 COMPLETE: Uses SAME processing pipeline as working system")
    logger.info("=" * 80)

    try:
        logger.info(f"[INIT] Initializing per-camera optimized system for cameras: {ACTIVE_CAMERAS}")

        system = PerCameraOptimizedPipelineSystem(
            active_cameras=ACTIVE_CAMERAS,
            enable_gui=ENABLE_GUI,
            gui_cameras=GUI_CAMERAS
        )

        logger.info("[START] Starting per-camera optimized threading system...")
        logger.info("[INFO] System will run until Ctrl+C is pressed or 'q' key in GUI")
        logger.info("[MONITOR] Performance stats will be logged every 30 seconds")
        logger.info("[BALANCE] Watch for camera balance ratio in stats (1.0 = perfect balance)")
        logger.info("[COMPLETE] Full processing pipeline: Detection → Filtering → Coordinates → SIFT → Database")

        # Start the per-camera optimized system
        system.start()
        
    except KeyboardInterrupt:
        logger.info("[STOP] Keyboard interrupt received - shutting down gracefully")
    except Exception as e:
        logger.error(f"[ERROR] Error in per-camera optimized system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("[COMPLETE] Per-camera optimized system shutdown complete")

if __name__ == "__main__":
    main()
