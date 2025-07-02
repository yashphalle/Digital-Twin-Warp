#!/usr/bin/env python3
"""
Area Analysis for Pallet Detection
Analyzes detection box areas to determine optimal size filtering limits
"""

import cv2
import numpy as np
import logging
import sys
import os
from typing import List, Dict
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AreaAnalysisDetector:
    """Analyze detection areas to find optimal size filtering limits"""
    
    def __init__(self, camera_id: int = 11):
        self.camera_id = camera_id
        self.warehouse_config = get_warehouse_config()
        
        # Get camera configuration
        if str(camera_id) in self.warehouse_config.camera_zones:
            self.camera_zone = self.warehouse_config.camera_zones[str(camera_id)]
            self.camera_name = self.camera_zone.camera_name
            self.rtsp_url = self.camera_zone.rtsp_url
        else:
            self.camera_name = f"Camera {camera_id}"
            self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id, "")
        
        # Camera connection
        self.cap = None
        self.connected = False
        self.running = False
        
        # Detection components
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        self.pallet_detector = SimplePalletDetector()
        
        # Set detection parameters
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]
        
        # Update DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.1
            self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
        
        # Area analysis data
        self.all_areas = []
        self.frame_count = 0
        self.detection_count = 0
        
        logger.info(f"Initialized area analysis for {self.camera_name}")
        logger.info(f"Prompts: {self.pallet_detector.sample_prompts}")
    
    def connect_camera(self) -> bool:
        """Connect to the camera"""
        if not self.rtsp_url:
            logger.error(f"No RTSP URL configured for camera {self.camera_id}")
            return False
        
        logger.info(f"Connecting to {self.camera_name}...")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.RTSP_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.rtsp_url}")
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to capture test frame from {self.camera_name}")
                self.cap.release()
                return False
            
            logger.info(f"{self.camera_name} connected successfully")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def analyze_areas(self, num_frames: int = 100):
        """Analyze detection areas over multiple frames"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Area Analysis - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        print("=" * 60)
        print("AREA ANALYSIS FOR PALLET DETECTION")
        print("=" * 60)
        print(f"Camera: {self.camera_name}")
        print(f"Prompts: {self.pallet_detector.sample_prompts}")
        print(f"Confidence: {self.pallet_detector.confidence_threshold}")
        print(f"Analyzing {num_frames} frames...")
        print("Press 'q' to stop early and see results")
        print("=" * 60)
        
        while self.running and self.frame_count < num_frames:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame with area info
                display_frame = self._draw_area_info(processed_frame)
                cv2.imshow(window_name, display_frame)
                
                # Print progress every 10 frames
                if self.frame_count % 10 == 0:
                    print(f"Processed {self.frame_count}/{num_frames} frames, "
                          f"Total detections: {self.detection_count}")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Analysis stopped by user")
                    break
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                break
        
        # Show final results
        self._print_area_analysis()
        self.stop_analysis()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and collect area data"""
        processed_frame = frame.copy()
        
        # Apply fisheye correction if enabled
        if Config.FISHEYE_CORRECTION_ENABLED:
            try:
                processed_frame = self.fisheye_corrector.correct(processed_frame)
            except Exception as e:
                logger.warning(f"Fisheye correction failed: {e}")
        
        # Resize for display
        height, width = processed_frame.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        # Run detection
        try:
            detections = self.pallet_detector.detect_pallets(processed_frame)
            
            # Collect area data
            for detection in detections:
                area = detection.get('area', 0)
                if area > 0:
                    self.all_areas.append(area)
                    self.detection_count += 1
                    
                    # Print individual detection info
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    print(f"Detection {self.detection_count}: Area={area:.0f}, "
                          f"Confidence={confidence:.3f}, "
                          f"BBox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        # Draw detections with area labels
        processed_frame = self._draw_detections_with_areas(processed_frame)
        
        return processed_frame
    
    def _draw_detections_with_areas(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections with area information"""
        result_frame = frame.copy()
        
        for detection in self.pallet_detector.last_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            area = detection.get('area', 0)
            
            x1, y1, x2, y2 = bbox
            
            # Color based on area size
            if area < 5000:
                color = (0, 0, 255)  # Red - very small
            elif area < 20000:
                color = (0, 165, 255)  # Orange - small
            elif area < 50000:
                color = (0, 255, 255)  # Yellow - medium
            elif area < 100000:
                color = (0, 255, 0)  # Green - large
            else:
                color = (255, 0, 255)  # Magenta - very large
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with area
            label = f"Area: {area:.0f}"
            label2 = f"Conf: {confidence:.2f}"
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1-40), (x1+max(label_size[0], 120), y1), color, -1)
            
            # Text
            cv2.putText(result_frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, label2, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def _draw_area_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw area analysis information"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)
        
        y_offset = 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), font, 0.6, color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Total Detections: {self.detection_count}", (20, y_offset), font, 0.6, color, 2)
        
        if self.all_areas:
            y_offset += 25
            current_min = min(self.all_areas)
            current_max = max(self.all_areas)
            current_avg = np.mean(self.all_areas)
            cv2.putText(frame, f"Min Area: {current_min:.0f}", (20, y_offset), font, 0.5, (0, 255, 0), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Max Area: {current_max:.0f}", (20, y_offset), font, 0.5, (0, 255, 0), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Avg Area: {current_avg:.0f}", (20, y_offset), font, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _print_area_analysis(self):
        """Print comprehensive area analysis"""
        print("\n" + "=" * 60)
        print("AREA ANALYSIS RESULTS")
        print("=" * 60)
        
        if not self.all_areas:
            print("No detections found!")
            return
        
        areas = np.array(self.all_areas)
        
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Detections: {self.detection_count}")
        print(f"Average Detections per Frame: {self.detection_count/self.frame_count:.2f}")
        print()
        
        print("AREA STATISTICS:")
        print(f"  Minimum Area: {np.min(areas):.0f} pixels")
        print(f"  Maximum Area: {np.max(areas):.0f} pixels")
        print(f"  Average Area: {np.mean(areas):.0f} pixels")
        print(f"  Median Area: {np.median(areas):.0f} pixels")
        print(f"  Standard Deviation: {np.std(areas):.0f} pixels")
        print()
        
        print("PERCENTILES:")
        print(f"  5th Percentile: {np.percentile(areas, 5):.0f} pixels")
        print(f"  25th Percentile: {np.percentile(areas, 25):.0f} pixels")
        print(f"  75th Percentile: {np.percentile(areas, 75):.0f} pixels")
        print(f"  95th Percentile: {np.percentile(areas, 95):.0f} pixels")
        print()
        
        print("RECOMMENDED FILTERING LIMITS:")
        # Conservative approach: exclude bottom 10% and top 10%
        min_recommended = np.percentile(areas, 10)
        max_recommended = np.percentile(areas, 90)
        print(f"  Conservative Min Area: {min_recommended:.0f} pixels")
        print(f"  Conservative Max Area: {max_recommended:.0f} pixels")
        print()
        
        # Aggressive approach: exclude bottom 5% and top 5%
        min_aggressive = np.percentile(areas, 5)
        max_aggressive = np.percentile(areas, 95)
        print(f"  Aggressive Min Area: {min_aggressive:.0f} pixels")
        print(f"  Aggressive Max Area: {max_aggressive:.0f} pixels")
        print()
        
        print("AREA DISTRIBUTION:")
        bins = [0, 1000, 5000, 10000, 20000, 50000, 100000, float('inf')]
        labels = ['<1K', '1K-5K', '5K-10K', '10K-20K', '20K-50K', '50K-100K', '>100K']
        
        for i in range(len(bins)-1):
            count = np.sum((areas >= bins[i]) & (areas < bins[i+1]))
            percentage = count / len(areas) * 100
            print(f"  {labels[i]}: {count} detections ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def stop_analysis(self):
        """Stop the analysis"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()
        
        logger.info(f"Stopped analysis for {self.camera_name}")


def main():
    """Main function"""
    print("AREA ANALYSIS FOR PALLET DETECTION")
    print("=" * 50)
    print("This will analyze detection box areas to find optimal size limits")
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("=" * 50)
    
    analyzer = AreaAnalysisDetector(camera_id=11)
    
    try:
        # Analyze 100 frames (adjust as needed)
        analyzer.analyze_areas(num_frames=100)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
    finally:
        analyzer.stop_analysis()


if __name__ == "__main__":
    main()
