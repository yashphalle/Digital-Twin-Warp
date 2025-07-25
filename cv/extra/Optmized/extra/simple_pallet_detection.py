#!/usr/bin/env python3
"""
Simple Pallet Detection with Grounding DINO
Reuses existing camera and fisheye code, keeps it simple
"""

import cv2
import sys
import os
import time
import logging

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our existing camera display code
from single_camera_display import SingleCameraDisplay
from detector_tracker import DetectorTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePalletDetection(SingleCameraDisplay):
    """Simple pallet detection extending our existing camera display"""

    def __init__(self, camera_id: int = 1):
        # Initialize parent camera display
        super().__init__(camera_id)

        # FORCE override resolution to 1080p for better performance
        self.display_width = 1920   # 1080p instead of 4K
        self.display_height = 1080
        self.maintain_original_quality = False  # Override parent setting

        # Simple prompt options (based on existing config analysis)
        self.prompts = [
            "pallet",
            "wooden pallet",
            "shipping pallet",
            "warehouse pallet"
        ]

        self.current_prompt_index = 0
        self.current_prompt = self.prompts[0]

        # Use existing config values
        self.confidence_threshold = 0.20  # From Config.CONFIDENCE_THRESHOLD
        self.box_threshold = 0.20         # From Config.BOX_THRESHOLD
        self.text_threshold = 0.20        # From Config.TEXT_THRESHOLD
        
        # Detection components
        self.detector = None
        self.enable_detection = True
        self.detection_interval = 3  # Every 3rd frame
        self.detection_counter = 0
        self.last_detections = []
        
        # Initialize detector
        self._init_detector()

        # Update window name
        self.window_name = f"Simple Pallet Detection - {self.camera_name}"

        logger.info("Simple pallet detection initialized")
        logger.info(f"Available prompts: {self.prompts}")
        logger.info(f"Using 1080p resolution for better performance")

    def connect_camera(self) -> bool:
        """Override parent method to set 1080p resolution and fix display issues"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False

        logger.info(f"Connecting to {self.camera_name} at 1080p...")

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Set buffer size to 1 to reduce latency and avoid green screen
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set timeouts
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

            # FORCE 1080p resolution (this should work for RTSP)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False

            # Test frame capture with multiple attempts to avoid green screen
            for attempt in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Check if frame is valid (not all green)
                    if frame.mean() > 10:  # Valid frame should have some content
                        break
                logger.warning(f"Frame capture attempt {attempt + 1} failed or invalid")
                time.sleep(0.5)
            else:
                logger.error(f"Failed to capture valid frame from {self.camera_name}")
                self.cap.release()
                return False

            # Get actual stream properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"{self.camera_name} connected successfully")
            logger.info(f"Stream: {actual_width}x{actual_height} @ {fps:.1f}fps")

            # Update our display dimensions to match actual resolution
            self.display_width = actual_width
            self.display_height = actual_height

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def _init_detector(self):
        """Initialize Grounding DINO detector"""
        try:
            self.detector = DetectorTracker()
            logger.info("Grounding DINO detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            self.detector = None
    
    def next_prompt(self):
        """Switch to next prompt"""
        self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)
        self.current_prompt = self.prompts[self.current_prompt_index]
        logger.info(f"Switched to prompt: '{self.current_prompt}'")
    
    def previous_prompt(self):
        """Switch to previous prompt"""
        self.current_prompt_index = (self.current_prompt_index - 1) % len(self.prompts)
        self.current_prompt = self.prompts[self.current_prompt_index]
        logger.info(f"Switched to prompt: '{self.current_prompt}'")
    
    def adjust_threshold(self, increase=True):
        """Adjust confidence threshold"""
        step = 0.05
        if increase:
            self.confidence_threshold = min(0.95, self.confidence_threshold + step)
        else:
            self.confidence_threshold = max(0.05, self.confidence_threshold - step)
        logger.info(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def detect_pallets(self, frame):
        """Run pallet detection on frame"""
        if not self.detector or not self.enable_detection:
            return []

        try:
            # Update detector settings (using config values)
            self.detector.prompt = self.current_prompt
            self.detector.confidence_threshold = self.confidence_threshold

            # Also update box and text thresholds if available
            if hasattr(self.detector, 'box_threshold'):
                self.detector.box_threshold = self.box_threshold
            if hasattr(self.detector, 'text_threshold'):
                self.detector.text_threshold = self.text_threshold

            # Run detection
            tracked_objects, _ = self.detector.process_frame(frame)

            # Convert to simple format
            detections = []
            for obj in tracked_objects:
                if obj.get('confidence', 0) >= self.confidence_threshold:
                    detections.append({
                        'bbox': obj['bbox'],
                        'confidence': obj['confidence']
                    })

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _process_frame(self, frame):
        """Override parent method to add detection and fix display issues"""
        if frame is None:
            return None

        # Check for valid frame (avoid green screen)
        if frame.mean() < 10:
            logger.warning("Invalid frame detected (possible green screen)")
            return frame

        processed_frame = frame.copy()

        # Apply fisheye correction if enabled (simplified)
        if self.use_fisheye_correction:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")

        # Ensure frame is proper size for display
        if processed_frame.shape[:2] != (self.display_height, self.display_width):
            processed_frame = cv2.resize(processed_frame, (self.display_width, self.display_height))

        # Run detection every N frames
        if self.enable_detection:
            self.detection_counter += 1
            if self.detection_counter >= self.detection_interval:
                self.detection_counter = 0
                self.last_detections = self.detect_pallets(processed_frame)

        # Draw detections
        if self.last_detections:
            processed_frame = self._draw_detections(processed_frame)

        # Draw simple info overlay
        processed_frame = self._draw_simple_info(processed_frame)

        return processed_frame
    
    def _draw_detections(self, frame):
        """Draw detection bounding boxes"""
        for detection in self.last_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"Pallet: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _draw_simple_info(self, frame):
        """Draw simple info overlay"""
        height, width = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)
        
        y = 35
        cv2.putText(frame, f"Prompt: {self.current_prompt}", (20, y), font, 0.6, color, 2)
        
        y += 25
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (20, y), font, 0.6, color, 2)
        
        y += 25
        pallet_count = len(self.last_detections)
        cv2.putText(frame, f"Pallets: {pallet_count}", (20, y), font, 0.6, (0, 255, 0), 2)
        
        # Controls info
        cv2.putText(frame, "N/P-Prompt +/--Threshold D-Toggle Q-Quit", (20, height-20), font, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _display_loop(self):
        """Override parent display loop with simple controls"""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 800)
        
        logger.info("=== SIMPLE PALLET DETECTION ===")
        logger.info("Controls:")
        logger.info("  'n' - Next prompt")
        logger.info("  'p' - Previous prompt")
        logger.info("  '+' - Increase threshold")
        logger.info("  '-' - Decrease threshold")
        logger.info("  'd' - Toggle detection")
        logger.info("  'q' - Quit")
        logger.info("=" * 35)
        
        while self.running:
            try:
                # Capture frame with validation
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame, attempting reconnection...")
                    if not self._reconnect_camera():
                        break
                    continue

                # Validate frame quality (avoid green screen/corruption)
                if frame.mean() < 10:
                    logger.warning("Invalid frame detected, skipping...")
                    continue

                # Process frame (includes detection)
                processed_frame = self._process_frame(frame)

                if processed_frame is not None:
                    # Display frame
                    cv2.imshow(self.window_name, processed_frame)
                else:
                    logger.warning("Frame processing returned None")
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('n'):
                    self.next_prompt()
                elif key == ord('p'):
                    self.previous_prompt()
                elif key == ord('+') or key == ord('='):
                    self.adjust_threshold(increase=True)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_threshold(increase=False)
                elif key == ord('d'):
                    self.enable_detection = not self.enable_detection
                    logger.info(f"Detection: {'ON' if self.enable_detection else 'OFF'}")
                elif key == ord('c'):
                    self.use_fisheye_correction = not self.use_fisheye_correction
                    logger.info(f"Fisheye correction: {'ON' if self.use_fisheye_correction else 'OFF'}")
                
                # Update FPS
                self._update_fps()
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                break
        
        self.running = False


def main():
    """Main function"""
    print("SIMPLE PALLET DETECTION")
    print("=" * 30)
    print("4 simple prompts, easy controls")
    print("Reuses existing camera code")
    print("1080p resolution for better performance")
    print("=" * 30)
    print("Initializing...")
    
    detector = SimplePalletDetection(camera_id=1)
    
    try:
        detector.start_display()
        
        while detector.is_running():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_display()


if __name__ == "__main__":
    main()
