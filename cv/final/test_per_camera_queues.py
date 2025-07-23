#!/usr/bin/env python3
"""
Test script to demonstrate the difference between shared and per-camera queues
Shows how per-camera queues solve the frame ordering issue
"""

import logging
import threading
import time
import queue
from typing import List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MockFrameData:
    """Mock frame data for testing"""
    camera_id: int
    frame_number: int
    timestamp: float

class SharedQueueTest:
    """Test with single shared queue (current problematic system)"""
    
    def __init__(self, cameras: List[int]):
        self.cameras = cameras
        self.shared_queue = queue.Queue(maxsize=20)
        self.processing_stats = {camera_id: 0 for camera_id in cameras}
        self.running = False
        
    def camera_producer(self, camera_id: int):
        """Simulate camera producing frames"""
        frame_number = 0
        while self.running:
            frame_data = MockFrameData(
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=time.time()
            )
            
            try:
                self.shared_queue.put(frame_data, timeout=0.1)
                frame_number += 1
                time.sleep(0.05)  # 20 FPS
            except queue.Full:
                pass  # Skip frame if queue full
                
    def detection_worker(self, worker_id: int):
        """Simulate detection worker processing frames"""
        while self.running:
            try:
                frame_data = self.shared_queue.get(timeout=1.0)
                
                # Simulate detection processing time
                time.sleep(0.1)  # 100ms processing time
                
                # Track which camera was processed
                self.processing_stats[frame_data.camera_id] += 1
                
                logger.debug(f"Worker {worker_id}: Processed Camera {frame_data.camera_id} frame {frame_data.frame_number}")
                
            except queue.Empty:
                continue
                
    def run_test(self, duration: int = 10):
        """Run the shared queue test"""
        logger.info("🧪 Testing SHARED QUEUE system...")
        self.running = True
        
        # Start camera producers
        camera_threads = []
        for camera_id in self.cameras:
            thread = threading.Thread(target=self.camera_producer, args=(camera_id,))
            thread.start()
            camera_threads.append(thread)
            
        # Start detection workers
        worker_threads = []
        for worker_id in range(2):  # 2 workers
            thread = threading.Thread(target=self.detection_worker, args=(worker_id,))
            thread.start()
            worker_threads.append(thread)
            
        # Run for specified duration
        time.sleep(duration)
        
        # Stop all threads
        self.running = False
        for thread in camera_threads + worker_threads:
            thread.join(timeout=1)
            
        return self.processing_stats

class PerCameraQueueTest:
    """Test with per-camera queues (new solution)"""
    
    def __init__(self, cameras: List[int]):
        self.cameras = cameras
        self.camera_queues = {
            camera_id: queue.Queue(maxsize=5) for camera_id in cameras
        }
        self.processing_stats = {camera_id: 0 for camera_id in cameras}
        self.running = False
        self.round_robin_index = 0
        
    def camera_producer(self, camera_id: int):
        """Simulate camera producing frames"""
        frame_number = 0
        while self.running:
            frame_data = MockFrameData(
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=time.time()
            )
            
            try:
                camera_queue = self.camera_queues[camera_id]
                
                # Replace old frame if queue is full
                if camera_queue.full():
                    try:
                        camera_queue.get_nowait()  # Remove old frame
                    except queue.Empty:
                        pass
                        
                camera_queue.put(frame_data, timeout=0.1)
                frame_number += 1
                time.sleep(0.05)  # 20 FPS
            except queue.Full:
                pass  # Skip frame if queue full
                
    def detection_worker_round_robin(self, worker_id: int):
        """Simulate detection worker with round-robin selection"""
        while self.running:
            frame_data = None
            
            # Try each camera in round-robin order
            for _ in range(len(self.cameras)):
                camera_id = self.cameras[self.round_robin_index]
                self.round_robin_index = (self.round_robin_index + 1) % len(self.cameras)
                
                try:
                    camera_queue = self.camera_queues[camera_id]
                    frame_data = camera_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
                    
            if frame_data:
                # Simulate detection processing time
                time.sleep(0.1)  # 100ms processing time
                
                # Track which camera was processed
                self.processing_stats[frame_data.camera_id] += 1
                
                logger.debug(f"Worker {worker_id}: Processed Camera {frame_data.camera_id} frame {frame_data.frame_number}")
            else:
                time.sleep(0.01)  # Brief pause if no frames available
                
    def run_test(self, duration: int = 10):
        """Run the per-camera queue test"""
        logger.info("🧪 Testing PER-CAMERA QUEUE system...")
        self.running = True
        
        # Start camera producers
        camera_threads = []
        for camera_id in self.cameras:
            thread = threading.Thread(target=self.camera_producer, args=(camera_id,))
            thread.start()
            camera_threads.append(thread)
            
        # Start detection workers
        worker_threads = []
        for worker_id in range(2):  # 2 workers
            thread = threading.Thread(target=self.detection_worker_round_robin, args=(worker_id,))
            thread.start()
            worker_threads.append(thread)
            
        # Run for specified duration
        time.sleep(duration)
        
        # Stop all threads
        self.running = False
        for thread in camera_threads + worker_threads:
            thread.join(timeout=1)
            
        return self.processing_stats

def calculate_balance_metrics(stats: Dict[int, int]) -> Dict[str, float]:
    """Calculate balance metrics from processing stats"""
    values = list(stats.values())
    if not values or max(values) == 0:
        return {'balance_ratio': 1.0, 'std_dev': 0.0, 'max_min_ratio': 1.0}
        
    total = sum(values)
    avg = total / len(values)
    
    # Standard deviation
    std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
    
    # Balance ratio (min/max)
    balance_ratio = min(values) / max(values) if max(values) > 0 else 1.0
    
    # Max/min ratio
    max_min_ratio = max(values) / min(values) if min(values) > 0 else float('inf')
    
    return {
        'balance_ratio': balance_ratio,
        'std_dev': std_dev,
        'max_min_ratio': max_min_ratio,
        'total_processed': total
    }

def main():
    """Run comparison test between shared and per-camera queues"""
    
    # Test configuration
    TEST_CAMERAS = [1, 2, 3, 4, 5]  # 5 cameras
    TEST_DURATION = 15  # 15 seconds
    
    logger.info("🚀 QUEUE SYSTEM COMPARISON TEST")
    logger.info("=" * 60)
    logger.info(f"📹 Test Cameras: {TEST_CAMERAS}")
    logger.info(f"⏱️ Test Duration: {TEST_DURATION} seconds")
    logger.info(f"👥 Detection Workers: 2")
    logger.info(f"📊 Expected: Per-camera queues should show better balance")
    logger.info("=" * 60)
    
    # Test 1: Shared Queue System
    logger.info("\n🧪 TEST 1: SHARED QUEUE SYSTEM (Current)")
    logger.info("-" * 40)
    
    shared_test = SharedQueueTest(TEST_CAMERAS)
    shared_stats = shared_test.run_test(TEST_DURATION)
    shared_metrics = calculate_balance_metrics(shared_stats)
    
    logger.info("📊 Shared Queue Results:")
    for camera_id, count in shared_stats.items():
        percentage = (count / shared_metrics['total_processed'] * 100) if shared_metrics['total_processed'] > 0 else 0
        logger.info(f"  Camera {camera_id}: {count:3d} frames ({percentage:.1f}%)")
    
    logger.info(f"⚖️ Balance Ratio: {shared_metrics['balance_ratio']:.3f} (1.0 = perfect)")
    logger.info(f"📈 Max/Min Ratio: {shared_metrics['max_min_ratio']:.2f} (1.0 = perfect)")
    
    # Brief pause between tests
    time.sleep(2)
    
    # Test 2: Per-Camera Queue System
    logger.info("\n🧪 TEST 2: PER-CAMERA QUEUE SYSTEM (New)")
    logger.info("-" * 40)
    
    per_camera_test = PerCameraQueueTest(TEST_CAMERAS)
    per_camera_stats = per_camera_test.run_test(TEST_DURATION)
    per_camera_metrics = calculate_balance_metrics(per_camera_stats)
    
    logger.info("📊 Per-Camera Queue Results:")
    for camera_id, count in per_camera_stats.items():
        percentage = (count / per_camera_metrics['total_processed'] * 100) if per_camera_metrics['total_processed'] > 0 else 0
        logger.info(f"  Camera {camera_id}: {count:3d} frames ({percentage:.1f}%)")
    
    logger.info(f"⚖️ Balance Ratio: {per_camera_metrics['balance_ratio']:.3f} (1.0 = perfect)")
    logger.info(f"📈 Max/Min Ratio: {per_camera_metrics['max_min_ratio']:.2f} (1.0 = perfect)")
    
    # Comparison Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Shared Queue Balance Ratio:     {shared_metrics['balance_ratio']:.3f}")
    logger.info(f"Per-Camera Queue Balance Ratio: {per_camera_metrics['balance_ratio']:.3f}")
    
    improvement = per_camera_metrics['balance_ratio'] - shared_metrics['balance_ratio']
    logger.info(f"Balance Improvement: {improvement:+.3f}")
    
    if per_camera_metrics['balance_ratio'] > shared_metrics['balance_ratio']:
        logger.info("✅ Per-camera queues show BETTER balance!")
    else:
        logger.info("❌ Shared queue performed better (unexpected)")
    
    logger.info("\n💡 INTERPRETATION:")
    logger.info("- Balance Ratio closer to 1.0 = more fair processing")
    logger.info("- Per-camera queues should prevent one camera from dominating")
    logger.info("- Round-robin ensures each camera gets processing time")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
