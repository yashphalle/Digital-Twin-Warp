#!/usr/bin/env python3
"""
BASIC THREADED WAREHOUSE TRACKING SYSTEM - PHASE 1
Safe testing of camera preprocessing + detection threading
Original main.py remains untouched
"""

import logging
import time
import signal
import sys

# Import our new threading components
from warehouse_threading.pipeline_system import PipelineThreadingSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for basic threaded testing"""
    print("🚀 BASIC THREADED WAREHOUSE TRACKING SYSTEM - PHASE 1")
    print("=" * 80)
    print("SAFE TESTING: Original main.py remains completely untouched")
    print("ARCHITECTURE: Camera Threads + Detection Thread Pool")
    print("=" * 80)
    print("CONFIGURATION:")
    print("📹 Test Cameras: [1, 2, 3, 4] (4 cameras)")
    print("🧵 Camera Threads: 4 (one per camera)")
    print("🔍 Detection Threads: 3 (GPU thread pool)")
    print("📊 Queue Management: Automated")
    print("⚡ Frame Skipping: Every 20th frame")
    print("=" * 80)
    print("THREADING PIPELINE:")
    print("1) 📹 Camera Threads: RTSP → Fisheye Correction → Frame Skip → Queue")
    print("2) � Detection Pool: Queue → GPU Detection → Queue")
    print("3) � Processing Consumer: Queue → Detection Results → Log")
    print("4) 📊 Statistics: Real-time performance monitoring")
    print("=" * 80)
    print("BENEFITS EXPECTED:")
    print("✅ Parallel camera preprocessing (4 cameras)")
    print("✅ GPU utilization with 3 detection threads")
    print("✅ 20x speed boost from frame skipping")
    print("✅ Queue-based load balancing")
    print("✅ Real-time detection results")
    print("=" * 80)
    print("SAFETY GUARANTEES:")
    print("🛡️ Original main.py: COMPLETELY UNTOUCHED")
    print("🛡️ Original modules/: NO MODIFICATIONS")
    print("🛡️ Easy rollback: Delete threading/ folder")
    print("🛡️ Side-by-side testing: Run both systems")
    print("=" * 80)
    print("Press Ctrl+C to stop gracefully")
    print("=" * 80)
    
    # Test with just 2 cameras initially for safety
    system = PipelineThreadingSystem(active_cameras=[1, 2, 3, 4])
    
    # Record start time for FPS calculation
    system.start_time = time.time()
    
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.stop()
        print("\n" + "=" * 80)
        print("✅ BASIC THREADED SYSTEM TEST COMPLETE")
        print("🔄 Original main.py still available: python main.py")
        print("📊 Compare performance between threaded and sequential")
        print("=" * 80)

if __name__ == "__main__":
    main()
