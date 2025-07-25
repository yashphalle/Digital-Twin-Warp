#!/usr/bin/env python3
"""
Per-Camera Pipeline System
Uses separate queues for each camera to solve frame ordering issues
"""

import logging
import threading
import time
import signal
import sys
from typing import List, Dict, Any
from .per_camera_queue_manager import PerCameraQueueManager
from .per_camera_detection_pool import PerCameraDetectionThreadPool
from .optimized_camera_threads import OptimizedCameraThreadManager

logger = logging.getLogger(__name__)

class PerCameraPipelineSystem:
    """
    Complete pipeline system with per-camera queues
    Solves frame ordering issues by ensuring fair processing across all cameras
    """

    def __init__(self, active_cameras: List[int] = [1, 2], enable_gui: bool = False, gui_cameras: List[int] = None):
        self.active_cameras = active_cameras
        self.enable_gui = enable_gui
        self.gui_cameras = gui_cameras or []
        self.running = False

        # Initialize PER-CAMERA threading components
        self.queue_manager = PerCameraQueueManager(max_cameras=len(active_cameras), active_cameras=active_cameras)
        self.camera_manager = OptimizedCameraThreadManager(active_cameras, self.queue_manager)
        self.detection_pool = PerCameraDetectionThreadPool(num_workers=5, queue_manager=self.queue_manager)

        # Initialize GUI display managers if enabled (same as optimized system)
        self.display_managers = {}
        if self.enable_gui:
            logger.info(f"[GUI] Initializing display managers for cameras: {self.gui_cameras}")
            try:
                from modules.gui_display import CPUDisplayManager
                for cam_id in self.gui_cameras:
                    camera_name = f"Camera {cam_id}"
                    self.display_managers[cam_id] = CPUDisplayManager(cam_id, camera_name)
                    logger.info(f"[GUI] Display manager initialized for {camera_name}")
            except Exception as e:
                logger.error(f"[GUI] Failed to initialize display managers: {e}")
                self.enable_gui = False

        # Performance monitoring
        self.stats_thread = None
        self.last_stats_time = time.time()

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"✅ Per-Camera Pipeline System initialized")
        logger.info(f"📹 Active Cameras: {active_cameras}")
        logger.info(f"🖥️ GUI Enabled: {enable_gui}")
        logger.info(f"🔄 Per-Camera Queues: {len(self.queue_manager.camera_detection_queues)}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def start(self):
        """Start the complete per-camera pipeline system"""
        try:
            logger.info("🚀 Starting Per-Camera Pipeline System...")
            self.running = True

            # Start components (simplified like optimized system)
            logger.info("📹 Starting camera threads...")
            self.camera_manager.start_camera_threads()

            logger.info("🔍 Starting per-camera detection pool...")
            self.detection_pool.start_detection_workers()

            # Start performance monitoring
            logger.info("📊 Starting performance monitoring...")
            self._start_performance_monitoring()

            logger.info("✅ Per-Camera Pipeline System started successfully!")
            logger.info("=" * 80)
            logger.info("SYSTEM STATUS: RUNNING")
            logger.info("Press Ctrl+C to stop the system gracefully")
            logger.info("=" * 80)

            # Keep main thread alive and monitor system
            self._monitor_system()

        except Exception as e:
            logger.error(f"❌ Failed to start per-camera pipeline system: {e}")
            self.stop()
            raise

    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        self.stats_thread = threading.Thread(
            target=self._performance_monitor_worker,
            name="PerCameraPerformanceMonitor",
            daemon=True
        )
        self.stats_thread.start()

    def _performance_monitor_worker(self):
        """Performance monitoring worker with per-camera stats"""
        logger.info("📊 Performance monitoring started")
        
        while self.running:
            try:
                time.sleep(30)  # Log stats every 30 seconds
                
                if not self.running:
                    break
                
                logger.info("\n" + "=" * 80)
                logger.info("PER-CAMERA SYSTEM PERFORMANCE REPORT")
                logger.info("=" * 80)
                
                # Queue statistics
                self.queue_manager.log_camera_stats()
                
                # Detection statistics
                self.detection_pool.log_detection_stats()
                
                # System uptime
                uptime = time.time() - self.last_stats_time
                logger.info(f"System Uptime: {uptime:.1f} seconds")
                
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")

    def _monitor_system(self):
        """Monitor system health and handle GUI events"""
        try:
            while self.running:
                # Check if GUI is still running (if enabled)
                if self.enable_gui and self.display_managers:
                    # Simple monitoring - just keep running
                    pass

                # Brief sleep to prevent busy waiting
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self):
        """Stop the complete per-camera pipeline system"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping Per-Camera Pipeline System...")
        self.running = False

        try:
            # Stop components in reverse order
            logger.info("🔍 Stopping per-camera detection pool...")
            self.detection_pool.stop()

            logger.info("📹 Stopping camera threads...")
            self.camera_manager.stop()

            # Wait for stats thread to finish
            if self.stats_thread and self.stats_thread.is_alive():
                logger.info("📊 Stopping performance monitoring...")
                self.stats_thread.join(timeout=2)

            logger.info("✅ Per-Camera Pipeline System stopped successfully")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'system': {
                'running': self.running,
                'active_cameras': self.active_cameras,
                'gui_enabled': self.enable_gui,
                'uptime': time.time() - self.last_stats_time
            },
            'queues': self.queue_manager.get_per_camera_stats(),
            'detection': self.detection_pool.get_detection_stats(),
        }
        
        return stats

    def log_system_summary(self):
        """Log a comprehensive system summary"""
        stats = self.get_system_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("PER-CAMERA SYSTEM SUMMARY")
        logger.info("=" * 80)
        
        # System info
        system = stats['system']
        logger.info(f"Status: {'RUNNING' if system['running'] else 'STOPPED'}")
        logger.info(f"Active Cameras: {system['active_cameras']}")
        logger.info(f"GUI Enabled: {system['gui_enabled']}")
        logger.info(f"Uptime: {system['uptime']:.1f} seconds")
        
        # Queue balance
        queue_stats = stats['queues']
        logger.info(f"\nQueue Balance:")
        for camera_id in self.active_cameras:
            camera_stats = queue_stats['per_camera'][camera_id]
            utilization = queue_stats['queue_utilization'][camera_id]
            logger.info(f"  Camera {camera_id}: {camera_stats['frames_processed']} processed, "
                       f"queue {utilization:.1%} full")
        
        # Detection balance
        detection_stats = stats['detection']
        if 'camera_balance' in detection_stats:
            balance = detection_stats['camera_balance']
            logger.info(f"\nDetection Balance Ratio: {balance['balance_ratio']:.3f}")
            logger.info("(1.0 = perfect balance, <0.5 = significant imbalance)")
        
        logger.info("=" * 80)

def main():
    """Test the per-camera pipeline system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    ACTIVE_CAMERAS = [8, 9, 10]  # Test with 3 cameras
    ENABLE_GUI = False
    
    logger.info("🧪 Testing Per-Camera Pipeline System")
    logger.info(f"📹 Test Cameras: {ACTIVE_CAMERAS}")
    
    try:
        system = PerCameraPipelineSystem(
            active_cameras=ACTIVE_CAMERAS,
            enable_gui=ENABLE_GUI
        )
        
        system.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
