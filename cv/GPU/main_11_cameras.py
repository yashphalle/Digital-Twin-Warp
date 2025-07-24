#!/usr/bin/env python3
"""
11-Camera Display System
Simple display-only system for warehouse monitoring
"""

import logging
import sys
import os
import time
import signal

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from per_camera_queue_manager import PerCameraQueueManager
from camera_threads import CameraThreadManager
from multi_camera_display import MultiCameraDisplayManager
from configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ElevenCameraDisplaySystem:
    """
    11-Camera Display System
    Simplified system focused on display only (no detection/database)
    """

    def __init__(self):
        self.active_cameras = Config.ACTIVE_CAMERAS  # All 11 cameras
        self.queue_manager = None
        self.camera_thread_manager = None
        self.display_manager = None
        self.running = False

        logger.info("🏭 11-Camera Display System Initializing...")
        logger.info(f"📹 Active cameras: {self.active_cameras}")

    def initialize_components(self):
        """Initialize all system components"""
        try:
            # 1. Initialize queue manager for 11 cameras
            self.queue_manager = PerCameraQueueManager(
                max_cameras=11,
                active_cameras=self.active_cameras
            )

            # 2. Initialize camera thread manager
            self.camera_thread_manager = CameraThreadManager(
                active_cameras=self.active_cameras,
                queue_manager=self.queue_manager
            )

            # 3. Initialize display manager
            self.display_manager = MultiCameraDisplayManager(
                active_cameras=self.active_cameras,
                queue_manager=self.queue_manager
            )

            logger.info("✅ All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    parser.add_argument('--online-db', action='store_true',
                       help='Use online MongoDB database (MongoDB Atlas)')
    parser.add_argument('--camera', type=int, default=7,
                       help='Camera ID to use (default: 7)')
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

    # Show current camera configuration
    camera_info = Config.get_camera_info(args.camera)
    logger.info(f"📡 Camera URL Mode: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    logger.info(f"📹 Camera {args.camera} URL: {camera_info['current_url']}")

    # Show current database configuration
    database_info = Config.get_database_info()
    logger.info(f"🗄️ Database Mode: {'LOCAL' if Config.USE_LOCAL_DATABASE else 'ONLINE'}")
    logger.info(f"💾 Database URI: {database_info['current_uri'][:50]}..." if len(database_info['current_uri']) > 50 else f"💾 Database URI: {database_info['current_uri']}")

    # GUI Configuration
    ENABLE_GUI = not args.no_gui
    ACTIVE_CAMERAS = [1,2,3,4,5,6,7,8,9,10,11]
    GUI_CAMERAS = [] if ENABLE_GUI else []

    logger.info("🚀 OPTIMIZED WAREHOUSE THREADING SYSTEM WITH GUI")
    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"📹 Active Cameras: {ACTIVE_CAMERAS} ({len(ACTIVE_CAMERAS)} cameras)")
    logger.info(f"🖥️ GUI Cameras: {GUI_CAMERAS} ({len(GUI_CAMERAS)} windows)")
    logger.info(f"🎛️ GUI Mode: {'ENABLED' if ENABLE_GUI else 'HEADLESS'}")
    logger.info("=" * 80)
    logger.info("OPTIMIZED PROCESSING - All operations with threading optimization:")
    logger.info("1) 🚀 Smart Frame Skipping (95% CPU savings)")
    logger.info("2) 🚀 Enhanced Queue Management")
    logger.info("3) 🚀 GPU Detection (Grounding DINO + post_process_grounded_object_detection)")
    logger.info("4) 🚀 CPU Area + Grid Cell Filtering")
    logger.info("5) 🚀 CPU Physical Coordinate Translation")
    logger.info("6) 🚀 CPU SIFT Feature Matching")
    logger.info("7) 🚀 CPU Persistent Object IDs")
    logger.info("8) 🚀 CPU Cross-Frame Tracking & Database")
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
    logger.info("PERFORMANCE MONITORING INSTRUCTIONS:")
    logger.info("   [1] Open Task Manager / System Monitor")
    logger.info("   [2] Watch CPU usage - should drop from 100% to ~20-30%")
    logger.info("   [3] Watch GPU usage - should increase from 20% to 80-90%")
    logger.info("   [4] Watch Memory usage - should be more stable")
    logger.info("   [5] Monitor for ~2-3 minutes to see the difference")
    logger.info("=" * 80)

    try:
        logger.info(f"[INIT] Initializing optimized system for cameras: {ACTIVE_CAMERAS}")

        system = OptimizedPipelineSystem(
            active_cameras=ACTIVE_CAMERAS,
            enable_gui=ENABLE_GUI,
            gui_cameras=GUI_CAMERAS
        )

        logger.info("[START] Starting optimized threading system with GUI...")
        logger.info("[INFO] System will run until Ctrl+C is pressed or 'q' key in GUI")
        logger.info("[MONITOR] Performance stats will be logged every 30 seconds")

        # Start the optimized system
        system.start()
        
    except KeyboardInterrupt:
        logger.info("[STOP] Keyboard interrupt received - shutting down gracefully")
    except Exception as e:
        logger.error(f"[ERROR] Error in optimized system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("[COMPLETE] OPTIMIZED system shutdown complete")

if __name__ == "__main__":
    main()
