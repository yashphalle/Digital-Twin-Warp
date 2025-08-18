#!/usr/bin/env python3
"""
RF-DETR Video Processor - Custom Model FPS Testing with Batch Size Comparison
Based on dino.py but adapted for RF-DETR models including Custom-4.pt
Tests different batch sizes for optimal performance
"""

import cv2
import numpy as np
import logging
import torch
import time
import threading
import argparse
from typing import List, Dict, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RFDETRDetector:
    """Production-ready RF-DETR detector optimized for batch processing and FPS testing"""
    
    def __init__(self, model_path="Custom-4.pt", gpu_id=None, confidence_threshold=0.25, batch_size=1):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Device selection with explicit GPU assignment
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            device_name = torch.cuda.get_device_name(gpu_id)
            memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            logger.info(f"🚀 Using GPU {gpu_id}: {device_name}")
            logger.info(f"📊 GPU Memory: {memory_gb:.1f}GB")
        elif torch.cuda.is_available():
            # Auto-select best NVIDIA GPU
            nvidia_device = 0
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                if "NVIDIA" in device_name:
                    nvidia_device = i
                    break
            self.device = torch.device(f"cuda:{nvidia_device}")
            torch.cuda.set_device(nvidia_device)
            logger.info(f"🚀 Auto-selected GPU {nvidia_device}: {torch.cuda.get_device_name(nvidia_device)}")
        else:
            self.device = torch.device("cpu")
            logger.info("⚠️ GPU not available, using CPU")

        logger.info(f"🔍 Initializing RF-DETR detector on {self.device}")
        logger.info(f"📦 Model: {self.model_path}")
        logger.info(f"🎯 Confidence: {self.confidence_threshold}")
        logger.info(f"📊 Batch Size: {self.batch_size}")
        
        # Thread ID for unique identification
        self.thread_id = threading.current_thread().ident
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize RF-DETR custom model using the correct loading method"""
        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                logger.info("💡 Available models in current directory:")
                for pt_file in Path(".").glob("*.pt"):
                    logger.info(f"   - {pt_file}")
                self.model = None
                return

            logger.info(f"📥 Loading RF-DETR custom model: {self.model_path}")

            # Load the custom RF-DETR model using the correct method
            try:
                from ultralytics import RTDETR

                # Load checkpoint
                logger.info("📥 Loading checkpoint...")
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

                # Extract model components
                if 'args' in checkpoint:
                    args = checkpoint['args']
                    logger.info(f"📋 Model args found - num_classes: {getattr(args, 'num_classes', 'unknown')}")
                else:
                    logger.warning("⚠️ No args found in checkpoint")

                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    logger.info("📋 Model state dict found")
                else:
                    logger.error("❌ No model state dict found in checkpoint")
                    self.model = None
                    return

                # Create and load the model
                logger.info("📥 Creating RT-DETR model...")
                self.model = RTDETR()

                # Load state dict
                logger.info("📥 Loading state dict...")
                self.model.model.load_state_dict(state_dict, strict=False)
                self.model.model.eval()

                # Store model info
                if 'args' in checkpoint and hasattr(checkpoint['args'], 'num_classes'):
                    self.num_classes = checkpoint['args'].num_classes
                    logger.info(f"✅ RF-DETR model loaded for {self.num_classes} classes")
                else:
                    self.num_classes = None
                    logger.info("✅ RF-DETR model loaded (num_classes unknown)")

                logger.info(f"✅ Model will use device: {self.device}")

            except Exception as rtdetr_error:
                logger.error(f"❌ RF-DETR custom loading failed: {rtdetr_error}")

                # Fallback: Try standard ultralytics loading
                try:
                    from ultralytics import YOLO
                    logger.info("📥 Falling back to standard YOLO loading...")
                    self.model = YOLO(self.model_path)
                    logger.info(f"✅ Model loaded as YOLO fallback")

                except Exception as yolo_error:
                    logger.error(f"❌ All loading methods failed. YOLO error: {yolo_error}")
                    self.model = None
                    return

            # Log GPU memory usage
            if self.device.type == 'cuda':
                try:
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    logger.info(f"📊 GPU Memory allocated: {memory_allocated:.2f}GB")
                except:
                    logger.info("📊 GPU memory info not available")

        except ImportError as ie:
            logger.error(f"❌ Required package not installed: {ie}")
            logger.info("💡 Install with: pip install ultralytics torch")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def detect_objects_single(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """Detect objects in single frame with precise timing"""
        if self.model is None:
            return [], 0.0
        
        try:
            # Precise inference timing
            inference_start = time.time()
            
            # Run model inference (works for both YOLO and RT-DETR)
            results = self.model(frame, device=str(self.device), conf=self.confidence_threshold, verbose=False)
            
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_fps = 1.0 / inference_time
            
            # Convert results to structured format
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)

                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            'confidence': float(score),
                            'class_id': int(cls),
                            'area': area,
                            'shape_type': 'quadrangle'
                        }
                        detections.append(detection)
            
            # Clear GPU cache to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return detections, inference_fps
            
        except Exception as e:
            logger.error(f"❌ Single detection failed: {e}")
            return [], 0.0

    def detect_objects_batch(self, frames: List[np.ndarray]) -> Tuple[List[List[Dict]], float]:
        """Detect objects in batch of frames with precise timing"""
        if self.model is None:
            return [[] for _ in frames], 0.0
        
        try:
            # Precise inference timing
            inference_start = time.time()
            
            # Run model batch inference (works for both YOLO and RT-DETR)
            results = self.model(frames, device=str(self.device), conf=self.confidence_threshold, verbose=False)
            
            inference_end = time.time()
            inference_time = inference_end - inference_start
            batch_fps = len(frames) / inference_time  # FPS per frame in batch
            
            # Convert all results to structured format
            all_detections = []
            for i, result in enumerate(results):
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)

                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            'confidence': float(score),
                            'class_id': int(cls),
                            'area': area,
                            'shape_type': 'quadrangle'
                        }
                        detections.append(detection)
                
                all_detections.append(detections)
            
            # Clear GPU cache to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return all_detections, batch_fps
            
        except Exception as e:
            logger.error(f"❌ Batch detection failed: {e}")
            return [[] for _ in frames], 0.0

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection.get('class_id', 0)
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Class {class_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame

def test_batch_sizes(video_path: str, model_path: str, gpu_id=None, confidence=0.25, max_frames=100):
    """Test different batch sizes to find optimal performance"""

    logger.info("🧪 BATCH SIZE PERFORMANCE TEST")
    logger.info("=" * 60)

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}

    # Load video frames for testing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return

    # Read frames for testing
    test_frames = []
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        test_frames.append(frame)
        frame_count += 1

    cap.release()
    logger.info(f"📹 Loaded {len(test_frames)} frames for batch testing")

    # Test each batch size
    for batch_size in batch_sizes:
        logger.info(f"\n🔄 Testing batch size: {batch_size}")

        # Initialize detector for this batch size
        detector = RFDETRDetector(
            model_path=model_path,
            gpu_id=gpu_id,
            confidence_threshold=confidence,
            batch_size=batch_size
        )

        if detector.model is None:
            logger.error(f"❌ Failed to initialize detector for batch size {batch_size}")
            continue

        # Warm up
        logger.info("   🔥 Warming up...")
        for i in range(3):
            if batch_size == 1:
                detector.detect_objects_single(test_frames[0])
            else:
                batch_frames = test_frames[:min(batch_size, len(test_frames))]
                detector.detect_objects_batch(batch_frames)

        # Performance test
        total_frames_processed = 0
        total_detections = 0
        total_inference_time = 0
        test_start = time.time()

        # Process frames in batches
        for i in range(0, len(test_frames), batch_size):
            batch_frames = test_frames[i:i+batch_size]

            if batch_size == 1:
                # Single frame processing
                for frame in batch_frames:
                    detections, fps = detector.detect_objects_single(frame)
                    total_detections += len(detections)
                    total_inference_time += 1.0 / fps if fps > 0 else 0
                    total_frames_processed += 1
            else:
                # Batch processing
                all_detections, batch_fps = detector.detect_objects_batch(batch_frames)
                for detections in all_detections:
                    total_detections += len(detections)
                total_inference_time += len(batch_frames) / batch_fps if batch_fps > 0 else 0
                total_frames_processed += len(batch_frames)

        test_end = time.time()
        total_test_time = test_end - test_start

        # Calculate metrics
        avg_fps = total_frames_processed / total_test_time if total_test_time > 0 else 0
        avg_inference_fps = total_frames_processed / total_inference_time if total_inference_time > 0 else 0
        avg_detections_per_frame = total_detections / total_frames_processed if total_frames_processed > 0 else 0

        # Store results
        results[batch_size] = {
            'avg_fps': avg_fps,
            'avg_inference_fps': avg_inference_fps,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'total_time': total_test_time,
            'frames_processed': total_frames_processed
        }

        logger.info(f"   📊 Results for batch size {batch_size}:")
        logger.info(f"      Average FPS: {avg_fps:.2f}")
        logger.info(f"      Inference FPS: {avg_inference_fps:.2f}")
        logger.info(f"      Total detections: {total_detections}")
        logger.info(f"      Avg detections/frame: {avg_detections_per_frame:.1f}")

    # Print comparison summary
    logger.info("\n🏆 BATCH SIZE COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Batch Size':<12} {'Avg FPS':<10} {'Inf FPS':<10} {'Detections':<12} {'Avg Det/Frame':<15}")
    logger.info("-" * 60)

    best_fps = 0
    best_batch_size = 1

    for batch_size in batch_sizes:
        if batch_size in results:
            result = results[batch_size]
            logger.info(f"{batch_size:<12} {result['avg_fps']:<10.2f} {result['avg_inference_fps']:<10.2f} "
                       f"{result['total_detections']:<12} {result['avg_detections_per_frame']:<15.1f}")

            if result['avg_fps'] > best_fps:
                best_fps = result['avg_fps']
                best_batch_size = batch_size

    logger.info("=" * 60)
    logger.info(f"🥇 Best performing batch size: {best_batch_size} ({best_fps:.2f} FPS)")

    return results

def process_video(video_path: str, model_path: str, gpu_id=None, no_gui=False, confidence=0.25, batch_size=1):
    """Process single video with YOLOv8 detection"""

    # Initialize detector
    detector = RFDETRDetector(
        model_path=model_path,
        gpu_id=gpu_id,
        confidence_threshold=confidence,
        batch_size=batch_size
    )

    if detector.model is None:
        logger.error("❌ Failed to initialize detector")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"📹 Video: {video_path}")
    logger.info(f"📊 Total frames: {total_frames}, Original FPS: {video_fps:.2f}")
    logger.info(f"🎯 Model: {model_path}")
    logger.info(f"📦 Batch Size: {batch_size}")
    logger.info("🔄 Press 'q' to quit")

    # Processing variables
    frame_count = 0
    start_time = time.time()
    detection_count = 0
    total_inference_fps = 0
    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in buffer
            if frame_buffer and batch_size > 1:
                all_detections, batch_fps = detector.detect_objects_batch(frame_buffer)
                total_inference_fps += batch_fps
                for detections in all_detections:
                    detection_count += len(detections)
            break

        frame_count += 1
        frame_buffer.append(frame)

        # Process when buffer is full or using single frame mode
        if len(frame_buffer) >= batch_size or batch_size == 1:
            frame_start = time.time()

            if batch_size == 1:
                # Single frame processing
                detections, inference_fps = detector.detect_objects_single(frame_buffer[0])
                total_inference_fps += inference_fps
                detection_count += len(detections)

                # Draw detections for display
                if detections:
                    annotated_frame = detector.draw_detections(frame_buffer[0], detections)
                    logger.info(f"🎯 Frame {frame_count}: Found {len(detections)} objects")
                else:
                    annotated_frame = frame_buffer[0]

                # Display frame (if GUI enabled)
                if not no_gui:
                    cv2.imshow('YOLOv8 Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            else:
                # Batch processing
                all_detections, batch_fps = detector.detect_objects_batch(frame_buffer)
                total_inference_fps += batch_fps

                batch_detection_count = 0
                for detections in all_detections:
                    batch_detection_count += len(detections)
                detection_count += batch_detection_count

                logger.info(f"🎯 Batch {frame_count//batch_size}: Processed {len(frame_buffer)} frames, "
                           f"Found {batch_detection_count} objects")

                # Display last frame from batch (if GUI enabled)
                if not no_gui and frame_buffer:
                    last_frame = frame_buffer[-1]
                    last_detections = all_detections[-1] if all_detections else []
                    if last_detections:
                        annotated_frame = detector.draw_detections(last_frame, last_detections)
                    else:
                        annotated_frame = last_frame

                    cv2.imshow('YOLOv8 Detection (Batch)', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

            # Clear buffer
            frame_buffer = []

        # Calculate and display FPS every 50 frames
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            processing_fps = frame_count / elapsed
            avg_inference_fps = total_inference_fps / (frame_count // batch_size) if frame_count > 0 else 0

            logger.info(f"📊 Frame {frame_count}/{total_frames}")
            logger.info(f"⚡ Processing FPS: {processing_fps:.2f}")
            logger.info(f"🧠 Avg Inference FPS: {avg_inference_fps:.2f}")
            logger.info(f"🎯 Total detections so far: {detection_count}")

    # Final statistics
    total_time = time.time() - start_time
    final_fps = frame_count / total_time
    avg_inference_fps = total_inference_fps / max(1, frame_count // batch_size) if frame_count > 0 else 0

    logger.info("🏁 Processing Complete!")
    logger.info(f"📊 Processed {frame_count} frames in {total_time:.2f}s")
    logger.info(f"⚡ Average Processing FPS: {final_fps:.2f}")
    logger.info(f"🧠 Average Inference FPS: {avg_inference_fps:.2f}")
    logger.info(f"🎯 Total detections: {detection_count}")
    logger.info(f"📦 Batch Size Used: {batch_size}")

    cap.release()
    if not no_gui:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='RF-DETR Video Processor - Custom Model FPS Testing with Batch Size Comparison')
    parser.add_argument('--video', '-v', required=True, help='Input video path (e.g., cam7.mp4)')
    parser.add_argument('--model', '-m', default='Custom-4.pt', help='RF-DETR model path (default: Custom-4.pt)')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID to use')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing (default: 1)')
    parser.add_argument('--test-batches', action='store_true', help='Test different batch sizes for optimal performance')
    parser.add_argument('--max-test-frames', type=int, default=100, help='Max frames for batch testing (default: 100)')

    args = parser.parse_args()

    logger.info("🚀 Starting RF-DETR Video Processor")
    logger.info(f"📹 Video: {args.video}")
    logger.info(f"🤖 Model: {args.model}")

    # Check if video file exists
    if not Path(args.video).exists():
        logger.error(f"❌ Video file not found: {args.video}")
        logger.info("💡 Available video files in current directory:")
        for video_file in Path(".").glob("*.mp4"):
            logger.info(f"   - {video_file}")
        return

    # Check if model file exists
    if not Path(args.model).exists():
        logger.error(f"❌ Model file not found: {args.model}")
        logger.info("💡 Available model files in current directory:")
        for model_file in Path(".").glob("*.pt"):
            logger.info(f"   - {model_file}")
        return

    if args.test_batches:
        # Run batch size comparison test
        logger.info("🧪 Running batch size comparison test...")
        test_batch_sizes(
            video_path=args.video,
            model_path=args.model,
            gpu_id=args.gpu_id,
            confidence=args.confidence,
            max_frames=args.max_test_frames
        )
    else:
        # Run single batch size test
        logger.info(f"🎯 Running single batch size test (batch_size={args.batch_size})...")
        process_video(
            video_path=args.video,
            model_path=args.model,
            gpu_id=args.gpu_id,
            no_gui=args.no_gui,
            confidence=args.confidence,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
