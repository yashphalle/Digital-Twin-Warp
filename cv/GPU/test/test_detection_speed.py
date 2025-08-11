# cv/GPU/pipelines/test_detection_speed.py

import sys
import os
import time
import logging
import cv2
import numpy as np
import threading
from typing import Dict, List

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from GPU.pipelines.camera_worker_parallel import ParallelCameraWorker
from GPU.pipelines.ring_buffer import RingBuffer
from GPU.pipelines.gpu_processor_detection_only import GPUBatchProcessorDetectionOnly
from GPU.pipelines.gpu_processor_fast_tracking import GPUBatchProcessorFastTracking

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleGUIDisplay:
    """Simple GUI display for detection speed testing"""

    def __init__(self, camera_ids: List[int]):
        self.camera_ids = camera_ids
        self.running = False
        self.display_thread = None
        self.latest_frames = {}
        self.latest_detections = {}
        self.stats = {}
        self.tracking_enabled = False  # Will be set by test function

        # Calculate grid layout
        self.num_cameras = len(camera_ids)
        if self.num_cameras <= 4:
            self.grid_cols = 2
            self.grid_rows = 2
        elif self.num_cameras <= 6:
            self.grid_cols = 3
            self.grid_rows = 2
        elif self.num_cameras <= 9:
            self.grid_cols = 3
            self.grid_rows = 3
        else:
            self.grid_cols = 4
            self.grid_rows = 3

        # Display settings
        self.cell_width = 320
        self.cell_height = 240
        self.display_width = self.grid_cols * self.cell_width
        self.display_height = self.grid_rows * self.cell_height + 100  # Extra space for stats

        logger.info(f"GUI Display initialized for {self.num_cameras} cameras ({self.grid_cols}x{self.grid_rows} grid)")

    def update_frame(self, camera_id: int, frame: np.ndarray, detections: List[Dict] = None):
        """Update frame and detections for a camera"""
        if frame is not None:
            self.latest_frames[camera_id] = frame.copy()
        if detections is not None:
            self.latest_detections[camera_id] = detections

    def update_stats(self, stats: Dict):
        """Update performance statistics"""
        self.stats = stats.copy()

    def get_tracking_color(self, detection: Dict) -> tuple:
        """Get color based on tracking state"""
        track_id = detection.get('track_id')
        track_age = detection.get('track_age', 0)

        if track_id is None:
            return (0, 255, 0)      # Green - Detection only
        elif track_age < 2:         # New track (< 2 frames)
            return (0, 255, 255)    # Yellow - New track
        else:
            return (0, 165, 255)    # Orange - Established track (2+ frames)

    def create_tracking_label(self, detection: Dict) -> str:
        """Create label with tracking information"""
        track_id = detection.get('track_id')
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'object')
        track_age = detection.get('track_age', 0)

        if track_id is not None:
            # Tracking mode: "ID:8001 pallet: 0.85 (15f)"
            return f"ID:{track_id} {class_name}: {confidence:.2f} ({track_age}f)"
        else:
            # Detection mode: "pallet: 0.85"
            return f"{class_name}: {confidence:.2f}"

    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict],
                                original_frame_size: tuple = None) -> np.ndarray:
        """Draw detection/tracking bounding boxes on frame with proper scaling and colors"""
        if not detections:
            return frame

        result_frame = frame.copy()
        current_height, current_width = frame.shape[:2]

        for detection in detections:
            # Get bounding box coordinates
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]

                # Scale bounding box if frame was resized
                if original_frame_size is not None:
                    orig_height, orig_width = original_frame_size
                    scale_x = current_width / orig_width
                    scale_y = current_height / orig_height

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                else:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, current_width - 1))
                y1 = max(0, min(y1, current_height - 1))
                x2 = max(0, min(x2, current_width - 1))
                y2 = max(0, min(y2, current_height - 1))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Get tracking color and label
                color = self.get_tracking_color(detection)
                label = self.create_tracking_label(detection)

                # Draw bounding box with tracking color
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_frame, (center_x, center_y), 3, color, -1)

                # Draw track ID badge for tracked objects
                track_id = detection.get('track_id')
                if track_id is not None:
                    # Large track ID in top-left corner of box
                    id_text = str(track_id)
                    cv2.putText(result_frame, id_text, (x1 - 20, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw enhanced label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

                # Position label above box, or below if too close to top
                label_y = y1 - 5 if y1 > label_size[1] + 10 else y2 + label_size[1] + 5
                label_x = x1

                # Draw label background
                cv2.rectangle(result_frame, (label_x, label_y - label_size[1] - 2),
                            (label_x + label_size[0], label_y + 2), color, -1)
                cv2.putText(result_frame, label, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return result_frame

    def create_camera_cell(self, camera_id: int, tracking_enabled: bool = False) -> np.ndarray:
        """Create display cell for a single camera with tracking status"""
        cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)

        # Camera header with tracking status
        if tracking_enabled:
            header_text = f"Camera {camera_id} - TRACKING"
            header_color = (0, 165, 255)  # Orange
        else:
            header_text = f"Camera {camera_id} - DETECTION"
            header_color = (0, 255, 0)    # Green

        cv2.putText(cell, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, header_color, 2)

        # Get frame and detections
        frame = self.latest_frames.get(camera_id)
        detections = self.latest_detections.get(camera_id, [])

        if frame is not None:
            # Store original frame size for bbox scaling
            original_frame_size = frame.shape[:2]  # (height, width)

            # Resize frame to fit cell (leave space for header)
            frame_height = self.cell_height - 30
            frame_resized = cv2.resize(frame, (self.cell_width, frame_height))

            # Draw detections with proper scaling
            frame_with_detections = self.draw_detections_on_frame(
                frame_resized, detections, original_frame_size)

            # Place frame in cell
            cell[30:, :] = frame_with_detections

            # Add tracking/detection count
            if tracking_enabled:
                tracked_count = sum(1 for d in detections if d.get('track_id') is not None)
                status_text = f"Tracks: {tracked_count}"
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = f"Detections: {len(detections)}"
                status_color = (0, 255, 255)  # Cyan

            cv2.putText(cell, status_text, (self.cell_width - 120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        else:
            # No frame available
            cv2.putText(cell, "NO SIGNAL", (self.cell_width//2 - 50, self.cell_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return cell

    def create_display_frame(self, tracking_enabled: bool = False) -> np.ndarray:
        """Create complete display frame with all cameras"""
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Place camera cells in grid
        for i, camera_id in enumerate(self.camera_ids):
            if i >= self.grid_cols * self.grid_rows:
                break  # Skip if too many cameras for grid

            row = i // self.grid_cols
            col = i % self.grid_cols

            x = col * self.cell_width
            y = row * self.cell_height

            cell = self.create_camera_cell(camera_id, tracking_enabled)
            display[y:y+self.cell_height, x:x+self.cell_width] = cell

        # Add statistics at bottom with tracking info
        stats_y = self.grid_rows * self.cell_height + 20
        if self.stats:
            if tracking_enabled:
                stats_text = [
                    f"Total Frames: {self.stats.get('total_frames', 0)}",
                    f"Active Tracks: {self.stats.get('active_tracks', 0)}",
                    f"Total Tracks Created: {self.stats.get('total_tracks_created', 0)}",
                    f"Avg FPS: {self.stats.get('avg_fps', 0.0):.1f}",
                    f"Avg Inference: {self.stats.get('avg_inference_time_ms', 0.0):.1f}ms"
                ]
            else:
                stats_text = [
                    f"Total Frames: {self.stats.get('total_frames', 0)}",
                    f"Total Detections: {self.stats.get('total_detections', 0)}",
                    f"Avg FPS: {self.stats.get('avg_fps', 0.0):.1f}",
                    f"Avg Inference: {self.stats.get('avg_inference_time_ms', 0.0):.1f}ms"
                ]

            # Display stats in two rows if tracking enabled
            for i, text in enumerate(stats_text):
                if tracking_enabled and i >= 3:
                    # Second row for tracking stats
                    x_pos = 20 + ((i - 3) * 200)
                    y_pos = stats_y + 25
                else:
                    # First row
                    x_pos = 20 + (i * 200)
                    y_pos = stats_y

                if x_pos < self.display_width - 150:
                    cv2.putText(display, text, (x_pos, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def display_loop(self):
        """Main display loop running in separate thread"""
        window_title = "Tracking Speed Test" if self.tracking_enabled else "Detection Speed Test"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, self.display_width, self.display_height)

        while self.running:
            try:
                display_frame = self.create_display_frame(self.tracking_enabled)
                cv2.imshow(window_title, display_frame)

                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    logger.info("GUI quit requested")
                    self.running = False
                    break
                elif key == ord('f'):  # 'f' for fullscreen
                    cv2.setWindowProperty("Detection Speed Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('w'):  # 'w' for windowed
                    cv2.setWindowProperty("Detection Speed Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(0.1)

        cv2.destroyAllWindows()
        logger.info("GUI display stopped")

    def start(self):
        """Start the display in a separate thread"""
        if not self.running:
            self.running = True
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()
            logger.info("GUI display started")

    def stop(self):
        """Stop the display"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        cv2.destroyAllWindows()

def test_detection_speed(camera_ids=[8, 9], duration=30, confidence=0.5, enable_gui=True, enable_tracking=False):
    """
    Test detection or tracking speed with PARALLEL camera reading

    Args:
        camera_ids: List of camera IDs to test
        duration: Test duration in seconds
        confidence: Detection confidence threshold
        enable_gui: Enable GUI display (default: True)
        enable_tracking: Enable tracking mode (default: False for detection only)
    """
    mode_name = "TRACKING" if enable_tracking else "DETECTION"
    logger.info(f"=== {mode_name} SPEED TEST (PARALLEL) ===")
    logger.info(f"Cameras: {camera_ids}")
    logger.info(f"Duration: {duration}s")
    logger.info(f"Confidence: {confidence}")
    logger.info(f"GUI Enabled: {enable_gui}")
    logger.info(f"Tracking Enabled: {enable_tracking}")

    # Initialize components
    ring_buffer = RingBuffer(num_cameras=11, buffer_size=30)

    # Initialize GPU processor based on mode
    if enable_tracking:
        gpu_processor = GPUBatchProcessorFastTracking(
            model_path='custom_yolo.pt',
            device='cuda:0',
            active_cameras=camera_ids,
            confidence=confidence,
            use_fp16=False
        )
        logger.info("? Fast tracking processor initialized")
        # Safe access to models attribute
        if hasattr(gpu_processor, 'models') and gpu_processor.models:
            logger.info(f"?? Using {len(gpu_processor.models)} separate model instances: custom_yolo.pt")
        else:
            logger.info("?? Using custom_yolo.pt model")
    else:
        gpu_processor = GPUBatchProcessorDetectionOnly(
            model_path='custom_yolo.pt',
            device='cuda:0',
            active_cameras=camera_ids,
            confidence=confidence,
            use_fp16=False
        )
        logger.info("? Detection-only processor initialized")

    # Initialize GUI display if enabled
    gui_display = None
    if enable_gui:
        gui_display = SimpleGUIDisplay(camera_ids)
        gui_display.tracking_enabled = enable_tracking  # Set tracking mode
        gui_display.start()
        logger.info(f"? GUI display started ({'tracking' if enable_tracking else 'detection'} mode)")
    
    # CHANGE: Create and start parallel camera workers
    camera_workers = []
    for cam_id in camera_ids:
        worker = ParallelCameraWorker(cam_id, ring_buffer, frame_skip=3)
        if worker.connect():
            worker.start()  # Start the thread!
            camera_workers.append(worker)
            logger.info(f"? Camera {cam_id} thread started")
    
    if not camera_workers:
        logger.error("No cameras connected")
        return
    
    # Wait for cameras to start producing frames
    logger.info("Waiting for cameras to stabilize...")
    time.sleep(3)
    
    # Warmup
    logger.info("Warming up GPU...")
    for _ in range(5):
        gpu_processor.process_batch(ring_buffer)
        time.sleep(0.1)
    
    # Reset stats
    gpu_processor.total_batches = 0
    gpu_processor.total_inference_time = 0
    gpu_processor.frames_processed = 0
    gpu_processor.detection_count = 0
    
    # Start actual test
    logger.info(f"\n?? Starting {duration} second speed test...")
    start_time = time.time()
    
    try:
        # CHANGE: Main loop just processes batches (no camera reading!)
        while time.time() - start_time < duration:
            # Check if GUI requested quit
            if gui_display and not gui_display.running:
                logger.info("GUI quit requested, stopping test")
                break

            # Just process batches - cameras feed themselves!
            # ?? FIX: Get synchronized frames and detections from GPU processor
            # This ensures the SAME frames used for processing are used for display
            processed_frames, detections_by_camera = gpu_processor.process_batch(ring_buffer)

            # Update GUI with synchronized frames and detections
            if gui_display:
                # ? Use the SAME frames that were processed for detections
                # This eliminates the flashing issue in tracking mode
                for cam_id in camera_ids:
                    frame = processed_frames.get(cam_id)
                    detections = detections_by_camera.get(cam_id, [])

                    # Minimal logging - focus on FPS performance
                    # Removed verbose detection logging to focus on performance metrics

                    if frame is not None:
                        gui_display.update_frame(cam_id, frame, detections)

                # Update performance stats
                stats = gpu_processor.get_stats()
                gui_display.update_stats(stats)

            # Small delay to prevent CPU spinning
            time.sleep(0.02)  # 20ms

    except KeyboardInterrupt:
        logger.info("Test interrupted")
    
    # Results
    elapsed = time.time() - start_time
    stats = gpu_processor.get_stats()

    logger.info("\n=== RESULTS ===")
    logger.info(f"Test duration: {elapsed:.1f}s")
    logger.info(f"Total frames processed: {stats['total_frames']}")

    if enable_tracking:
        logger.info(f"Total tracks created: {stats.get('total_tracks_created', 0)}")
        logger.info(f"Active tracks: {stats.get('active_tracks', 0)}")
    else:
        logger.info(f"Total detections: {stats.get('total_detections', 0)}")
        if stats.get('total_detections', 0) > 0:
            logger.info(f"Average detections per frame: {stats['total_detections']/max(stats['total_frames'],1):.1f}")

    logger.info(f"\n?? PERFORMANCE:")
    logger.info(f"  - Average inference time: {stats['avg_inference_time_ms']:.1f}ms per batch")
    logger.info(f"  - Average per frame: {stats['avg_ms_per_frame']:.1f}ms")
    logger.info(f"  - Average FPS: {stats['avg_fps']:.1f}")
    logger.info(f"  - Throughput: {stats['total_frames']/elapsed:.1f} frames/second")
    
    # Stop GUI display
    if gui_display:
        logger.info("Stopping GUI display...")
        gui_display.stop()

    # CHANGE: Stop parallel workers
    logger.info("\nStopping camera threads...")
    for worker in camera_workers:
        worker.stop()
    for worker in camera_workers:
        worker.join(timeout=1)

    # Cleanup
    gpu_processor.cleanup()

if __name__ == "__main__":
    """
    ?? FULL 11-CAMERA TRACKING PERFORMANCE TEST

    Testing individual camera processing fix:
    - All 11 warehouse cameras
    - Individual camera tracking (maintains persistence)
    - Focus on FPS performance measurement
    - Reduced logging for cleaner output

    Expected results:
    - Persistent track IDs (not resetting every frame)
    - Reasonable track creation (~100-200 total tracks)
    - Good FPS performance with 11 cameras
    """

    # ?? SEPARATE MODEL INSTANCES TEST - Test with 3 cameras first
    test_detection_speed(
        camera_ids=[1,2,3,4,5,6,7,8, 9, 10,11],  # Test with 3 cameras first to verify fix
        duration=30,            # Shorter test to verify separate model instances work
        confidence=0.5,
        enable_gui=True,
        enable_tracking=True    # Separate model instances per camera
    )