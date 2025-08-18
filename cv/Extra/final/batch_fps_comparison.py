#!/usr/bin/env python3
"""
Batch FPS Comparison Script
Uses debug_model.py functions to compare FPS performance across different batch sizes
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Tuple
import logging
from debug_model import get_model_instance, load_test_frames, API_KEY, MODEL_ID, CONFIDENCE_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchFPSComparator:
    """Compare FPS performance across different batch sizes"""
    
    def __init__(self, video_path: str = "cam7.mp4"):
        self.video_path = video_path
        self.model = None
        self.test_frames = []
        
    def initialize(self, max_test_frames: int = 100):
        """Initialize model and load test frames"""
        logger.info("🔑 Initializing Batch FPS Comparator...")
        
        # Load model
        logger.info(f"📥 Loading model: {MODEL_ID}")
        self.model = get_model_instance()
        logger.info("✅ Model loaded successfully!")
        
        # Load test frames
        logger.info(f"🎬 Loading test frames from: {self.video_path}")
        self.test_frames = load_test_frames(self.video_path, max_test_frames)
        logger.info(f"✅ Loaded {len(self.test_frames)} test frames")
        
        if len(self.test_frames) == 0:
            raise ValueError(f"No frames loaded from {self.video_path}")
    
    def test_single_frame_fps(self, num_iterations: int = 50) -> Dict:
        """Test FPS for single frame processing"""
        logger.info(f"🔄 Testing single frame FPS ({num_iterations} iterations)...")
        
        inference_times = []
        total_detections = 0
        
        # Use first frame for consistent testing
        test_frame = self.test_frames[0]
        
        # Warmup
        for _ in range(3):
            self.model.infer(test_frame)
        
        # Actual test
        for i in range(num_iterations):
            start_time = time.time()
            results = self.model.infer(test_frame)[0]
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Count detections
            detections = [pred for pred in results.predictions if pred.confidence >= CONFIDENCE_THRESHOLD]
            total_detections += len(detections)
            
            if (i + 1) % 10 == 0:
                avg_fps = 1 / (sum(inference_times[-10:]) / 10)
                logger.info(f"   Progress: {i+1}/{num_iterations} | Recent FPS: {avg_fps:.2f}")
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_fps = 1 / avg_inference_time
        
        return {
            'batch_size': 1,
            'avg_fps': avg_fps,
            'avg_inference_time': avg_inference_time,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / num_iterations,
            'inference_times': inference_times
        }
    
    def test_batch_fps(self, batch_size: int, num_batches: int = 20) -> Dict:
        """Test FPS for batch processing"""
        logger.info(f"🔄 Testing batch size {batch_size} FPS ({num_batches} batches)...")
        
        if batch_size > len(self.test_frames):
            logger.warning(f"⚠️ Batch size {batch_size} > available frames {len(self.test_frames)}")
            batch_size = len(self.test_frames)
        
        batch_times = []
        total_detections = 0
        total_frames_processed = 0
        
        # Warmup
        test_batch = self.test_frames[:batch_size]
        for _ in range(2):
            for frame in test_batch:
                self.model.infer(frame)
        
        # Actual test
        for batch_idx in range(num_batches):
            # Create batch from available frames (cycle through if needed)
            batch_frames = []
            for i in range(batch_size):
                frame_idx = (batch_idx * batch_size + i) % len(self.test_frames)
                batch_frames.append(self.test_frames[frame_idx])
            
            # Time batch processing
            batch_start = time.time()
            
            # Process each frame in the batch (simulating batch processing)
            batch_detections = 0
            for frame in batch_frames:
                results = self.model.infer(frame)[0]
                detections = [pred for pred in results.predictions if pred.confidence >= CONFIDENCE_THRESHOLD]
                batch_detections += len(detections)
            
            batch_end = time.time()
            
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            total_detections += batch_detections
            total_frames_processed += len(batch_frames)
            
            if (batch_idx + 1) % 5 == 0:
                recent_avg_time = sum(batch_times[-5:]) / 5
                recent_fps = batch_size / recent_avg_time
                logger.info(f"   Progress: {batch_idx+1}/{num_batches} | Recent Batch FPS: {recent_fps:.2f}")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_fps_per_frame = batch_size / avg_batch_time  # FPS per frame in batch
        
        return {
            'batch_size': batch_size,
            'avg_fps': avg_fps_per_frame,
            'avg_batch_time': avg_batch_time,
            'total_detections': total_detections,
            'total_frames_processed': total_frames_processed,
            'avg_detections_per_frame': total_detections / total_frames_processed,
            'batch_times': batch_times
        }
    
    def run_comparison(self, batch_sizes: List[int] = [1, 2, 4, 8, 16]) -> List[Dict]:
        """Run FPS comparison across different batch sizes"""
        logger.info("🚀 Starting Batch FPS Comparison...")
        logger.info("=" * 60)
        
        results = []
        
        # Test each batch size
        for batch_size in batch_sizes:
            try:
                if batch_size == 1:
                    result = self.test_single_frame_fps(num_iterations=50)
                else:
                    result = self.test_batch_fps(batch_size=batch_size, num_batches=20)
                
                results.append(result)
                
                logger.info(f"✅ Batch size {batch_size}: {result['avg_fps']:.2f} FPS")
                
            except Exception as e:
                logger.error(f"❌ Failed to test batch size {batch_size}: {e}")
                continue
        
        return results
    
    def print_comparison_results(self, results: List[Dict]):
        """Print detailed comparison results"""
        logger.info("\n🎉 Batch FPS Comparison Results")
        logger.info("=" * 60)
        
        # Sort by batch size
        results.sort(key=lambda x: x['batch_size'])
        
        # Print header
        print(f"{'Batch Size':<12} {'Avg FPS':<10} {'Avg Time (s)':<15} {'Detections/Frame':<18}")
        print("-" * 60)
        
        best_fps = 0
        best_batch_size = 1
        
        for result in results:
            batch_size = result['batch_size']
            avg_fps = result['avg_fps']
            avg_time = result.get('avg_inference_time', result.get('avg_batch_time', 0))
            detections_per_frame = result['avg_detections_per_frame']
            
            print(f"{batch_size:<12} {avg_fps:<10.2f} {avg_time:<15.4f} {detections_per_frame:<18.2f}")
            
            if avg_fps > best_fps:
                best_fps = avg_fps
                best_batch_size = batch_size
        
        print("-" * 60)
        print(f"🏆 Best Performance: Batch size {best_batch_size} with {best_fps:.2f} FPS")
        
        # Performance analysis
        logger.info(f"\n📊 Performance Analysis:")
        single_fps = next((r['avg_fps'] for r in results if r['batch_size'] == 1), 0)
        
        for result in results:
            if result['batch_size'] > 1:
                speedup = result['avg_fps'] / single_fps if single_fps > 0 else 0
                efficiency = speedup / result['batch_size'] * 100
                logger.info(f"   Batch {result['batch_size']}: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")

def main():
    """Main function to run batch FPS comparison"""
    try:
        # Initialize comparator
        comparator = BatchFPSComparator(video_path="cam7.mp4")
        comparator.initialize(max_test_frames=100)
        
        # Run comparison with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = comparator.run_comparison(batch_sizes)
        
        # Print results
        comparator.print_comparison_results(results)
        
        logger.info("\n✅ Batch FPS comparison completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Comparison stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
