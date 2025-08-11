#!/usr/bin/env python3

import time
import numpy as np
import cv2
from typing import List, Dict, Tuple
import logging
import statistics
import torch
from pathlib import Path
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrueBatchFPSTester:
    """Test true batch processing FPS with YOLOv8/Custom models"""
    
    def __init__(self, model_path: str = "Custom-4.pt", video_path: str = "cam7.mp4"):
        self.model_path = model_path
        self.video_path = video_path
        self.model = None
        self.test_frames = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize(self, max_test_frames: int = 200):
        """Initialize model and load test frames"""
        logger.info("🔑 Initializing True Batch FPS Tester...")
        logger.info(f"🎯 Device: {self.device}")

        # Load model using the same method as yolo_fps_test.py
        logger.info(f"📥 Loading model: {self.model_path}")
        try:
            self._initialize_model()
            if self.model is None:
                raise ValueError("Failed to load model")
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Load test frames
        logger.info(f"🎬 Loading test frames from: {self.video_path}")
        self.test_frames = self._load_test_frames(max_test_frames)
        logger.info(f"✅ Loaded {len(self.test_frames)} test frames")
        
        if len(self.test_frames) == 0:
            raise ValueError(f"No frames loaded from {self.video_path}")
    
    def _load_test_frames(self, max_frames: int) -> List[np.ndarray]:
        """Load frames from video for testing"""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames

    def _initialize_model(self):
        """Initialize YOLOv8 model for batch processing"""
        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                logger.info("💡 Available models in cv/final directory:")
                for pt_file in Path("cv/final").glob("*.pt"):
                    logger.info(f"   - {pt_file}")
                self.model = None
                return

            logger.info(f"📥 Loading YOLOv8 model: {self.model_path}")

            # Use standard YOLO loading (works with yolov8m.pt)
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Model loaded successfully")

            # Set device for inference
            if torch.cuda.is_available() and "cuda" in self.device:
                logger.info(f"🚀 Model will use device: {self.device}")
            else:
                logger.warning(f"⚠️ CUDA not available, using CPU")
                self.device = "cpu"

        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            self.model = None

    def warmup_model(self, warmup_iterations: int = 5):
        """Warm up the model for consistent performance"""
        logger.info(f"🔥 Warming up model ({warmup_iterations} iterations)...")
        
        test_frame = self.test_frames[0]
        for i in range(warmup_iterations):
            # Single frame warmup
            _ = self.model(test_frame, device=self.device, verbose=False)
            
        # Batch warmup
        if len(self.test_frames) >= 4:
            batch_frames = self.test_frames[:4]
            _ = self.model(batch_frames, device=self.device, verbose=False)
            
        logger.info("✅ Model warmed up")
    
    def test_sequential_processing(self, batch_size: int, num_tests: int = 30) -> Dict:
        """Test sequential frame processing (one by one) - OLD WAY"""
        logger.info(f"🔄 Testing SEQUENTIAL processing - Batch size: {batch_size}")
        
        processing_times = []
        total_detections = 0
        total_frames = 0
        
        for test_idx in range(num_tests):
            # Select frames for this test
            start_idx = (test_idx * batch_size) % len(self.test_frames)
            batch_frames = []
            for i in range(batch_size):
                frame_idx = (start_idx + i) % len(self.test_frames)
                batch_frames.append(self.test_frames[frame_idx])
            
            # Time sequential processing - EACH FRAME INDIVIDUALLY
            batch_start = time.time()
            batch_detections = 0
            
            for frame in batch_frames:
                # SEQUENTIAL: One frame at a time
                results = self.model(frame, device=self.device, verbose=False)
                batch_detections += len(results[0].boxes) if results[0].boxes is not None else 0
            
            batch_end = time.time()
            
            processing_time = batch_end - batch_start
            processing_times.append(processing_time)
            total_detections += batch_detections
            total_frames += len(batch_frames)
            
            if (test_idx + 1) % 10 == 0:
                recent_avg = sum(processing_times[-10:]) / 10
                recent_fps = batch_size / recent_avg
                logger.info(f"   Progress: {test_idx+1}/{num_tests} | Recent FPS: {recent_fps:.2f}")
        
        avg_processing_time = statistics.mean(processing_times)
        std_processing_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        avg_fps = batch_size / avg_processing_time
        
        return {
            'method': 'sequential',
            'batch_size': batch_size,
            'avg_fps': avg_fps,
            'avg_processing_time': avg_processing_time,
            'std_processing_time': std_processing_time,
            'total_detections': total_detections,
            'total_frames': total_frames,
            'avg_detections_per_frame': total_detections / total_frames,
            'processing_times': processing_times
        }
    
    def test_true_batch_processing(self, batch_size: int, num_tests: int = 30) -> Dict:
        """Test TRUE batch processing - ALL FRAMES AT ONCE"""
        logger.info(f"🚀 Testing TRUE BATCH processing - Batch size: {batch_size}")
        
        processing_times = []
        total_detections = 0
        total_frames = 0
        
        for test_idx in range(num_tests):
            # Select frames for this test
            start_idx = (test_idx * batch_size) % len(self.test_frames)
            batch_frames = []
            for i in range(batch_size):
                frame_idx = (start_idx + i) % len(self.test_frames)
                batch_frames.append(self.test_frames[frame_idx])
            
            # Time TRUE batch processing - ALL FRAMES AT ONCE
            batch_start = time.time()
            
            # TRUE BATCH: All frames in single model call
            results = self.model(batch_frames, device=self.device, verbose=False)
            
            batch_end = time.time()
            
            # Count detections from batch results
            batch_detections = 0
            for result in results:
                batch_detections += len(result.boxes) if result.boxes is not None else 0
            
            processing_time = batch_end - batch_start
            processing_times.append(processing_time)
            total_detections += batch_detections
            total_frames += len(batch_frames)
            
            if (test_idx + 1) % 10 == 0:
                recent_avg = sum(processing_times[-10:]) / 10
                recent_fps = batch_size / recent_avg
                logger.info(f"   Progress: {test_idx+1}/{num_tests} | Recent FPS: {recent_fps:.2f}")
        
        avg_processing_time = statistics.mean(processing_times)
        std_processing_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        avg_fps = batch_size / avg_processing_time
        
        return {
            'method': 'true_batch',
            'batch_size': batch_size,
            'avg_fps': avg_fps,
            'avg_processing_time': avg_processing_time,
            'std_processing_time': std_processing_time,
            'total_detections': total_detections,
            'total_frames': total_frames,
            'avg_detections_per_frame': total_detections / total_frames,
            'processing_times': processing_times
        }
    
    def run_comprehensive_test(self, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> List[Dict]:
        """Run comprehensive batch size comparison with TRUE batch processing"""
        logger.info("🚀 Starting TRUE Batch FPS Test...")
        logger.info("=" * 70)
        
        # Warm up model
        self.warmup_model()
        
        all_results = []
        
        for batch_size in batch_sizes:
            if batch_size > len(self.test_frames):
                logger.warning(f"⚠️ Skipping batch size {batch_size} (> available frames)")
                continue
            
            try:
                logger.info(f"\n📊 Testing batch size: {batch_size}")
                
                # Test sequential processing (old way)
                seq_result = self.test_sequential_processing(batch_size, num_tests=25)
                all_results.append(seq_result)
                
                # Test TRUE batch processing (new way)
                batch_result = self.test_true_batch_processing(batch_size, num_tests=25)
                all_results.append(batch_result)
                
                # Calculate speedup
                speedup = batch_result['avg_fps'] / seq_result['avg_fps']
                logger.info(f"   Sequential: {seq_result['avg_fps']:.2f} FPS")
                logger.info(f"   True Batch: {batch_result['avg_fps']:.2f} FPS")
                logger.info(f"   🚀 Speedup: {speedup:.2f}x")
                
            except Exception as e:
                logger.error(f"❌ Failed to test batch size {batch_size}: {e}")
                continue
        
        return all_results
    
    def print_detailed_results(self, results: List[Dict]):
        """Print detailed comparison results"""
        logger.info("\n🎉 TRUE Batch FPS Test Results")
        logger.info("=" * 80)
        
        # Group results by batch size
        batch_groups = {}
        for result in results:
            batch_size = result['batch_size']
            if batch_size not in batch_groups:
                batch_groups[batch_size] = {}
            batch_groups[batch_size][result['method']] = result
        
        # Print header
        print(f"{'Batch':<6} {'Method':<15} {'Avg FPS':<10} {'Speedup':<10} {'Det/Frame':<12} {'GPU Util':<12}")
        print("-" * 80)
        
        baseline_fps = None
        
        for batch_size in sorted(batch_groups.keys()):
            group = batch_groups[batch_size]
            
            seq_fps = group.get('sequential', {}).get('avg_fps', 0)
            batch_fps = group.get('true_batch', {}).get('avg_fps', 0)
            
            if baseline_fps is None and batch_size == 1:
                baseline_fps = seq_fps
            
            for method in ['sequential', 'true_batch']:
                if method in group:
                    result = group[method]
                    avg_fps = result['avg_fps']
                    det_per_frame = result['avg_detections_per_frame']
                    
                    # Calculate speedup vs sequential
                    if method == 'sequential':
                        speedup = 1.0
                        gpu_util = f"{batch_size * 100 / batch_size:.0f}%"  # Sequential = 100%
                    else:
                        speedup = avg_fps / seq_fps if seq_fps > 0 else 0
                        gpu_util = f"{speedup * 100:.0f}%"  # Batch utilization
                    
                    method_display = method.replace('_', ' ').title()
                    print(f"{batch_size:<6} {method_display:<15} {avg_fps:<10.2f} {speedup:<10.2f}x {det_per_frame:<12.2f} {gpu_util:<12}")
            
            print()  # Empty line between batch sizes
        
        # Find best performance
        best_result = max(results, key=lambda x: x['avg_fps'])
        logger.info(f"🏆 Best Performance: {best_result['method']} with batch size {best_result['batch_size']} - {best_result['avg_fps']:.2f} FPS")
        
        # Performance insights
        logger.info(f"\n📈 Batch Processing Benefits:")
        for batch_size in sorted(batch_groups.keys()):
            if batch_size in batch_groups and len(batch_groups[batch_size]) == 2:
                seq = batch_groups[batch_size].get('sequential', {})
                batch = batch_groups[batch_size].get('true_batch', {})
                
                if seq and batch:
                    speedup = batch['avg_fps'] / seq['avg_fps']
                    efficiency = speedup / batch_size * 100
                    logger.info(f"   Batch {batch_size}: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")

def main():
    """Main function to run true batch FPS test"""
    try:
        # Use yolov8m.pt which is known to work with batch processing
        model_path = "cv/final/yolov8m.pt"
        logger.info(f"🎯 Using model: {model_path}")

        # Initialize tester
        tester = TrueBatchFPSTester(model_path=model_path, video_path="cv/final/cam7.mp4")
        tester.initialize(max_test_frames=150)
        
        # Run comprehensive test
        batch_sizes = [1, 2, 4, 8, 16, 24, 32]
        results = tester.run_comprehensive_test(batch_sizes)
        
        # Print detailed results
        tester.print_detailed_results(results)
        
        logger.info("\n✅ True batch FPS test completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Test stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
