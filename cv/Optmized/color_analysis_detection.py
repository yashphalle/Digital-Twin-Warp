#!/usr/bin/env python3
"""
Color Analysis for Pallet Detection
Analyzes colors in detected pallet regions to identify top colors
"""

import cv2
import numpy as np
import logging
import sys
import os
from typing import List, Dict, Tuple
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorAnalysisDetector:
    """Analyze colors in pallet detections to identify dominant colors"""
    
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
        
        # Set detection parameters - using your base prompts
        self.pallet_detector.confidence_threshold = 0.1
        self.pallet_detector.sample_prompts = ["pallet wrapped in plastic", "stack of goods on pallet"]
        self.pallet_detector.current_prompt_index = 0
        self.pallet_detector.current_prompt = self.pallet_detector.sample_prompts[0]
        
        # Update DetectorTracker settings
        if self.pallet_detector.grounding_dino:
            self.pallet_detector.grounding_dino.confidence_threshold = 0.1
            self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
        
        # Area filtering (from previous scripts)
        self.MIN_AREA = 10000
        self.MAX_AREA = 100000
        
        # Color analysis data
        self.color_samples = []
        self.dominant_colors = {}
        self.frame_count = 0
        self.detection_count = 0
        
        logger.info(f"Initialized color analysis for {self.camera_name}")
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
    
    def extract_dominant_colors(self, roi: np.ndarray, k: int = 5) -> List[Tuple]:
        """Extract dominant colors from region of interest using K-means clustering"""
        # Reshape image to be a list of pixels
        data = roi.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and count occurrences
        centers = np.uint8(centers)
        
        # Count pixels for each cluster
        label_counts = Counter(labels.flatten())
        
        # Sort by frequency
        dominant_colors = []
        for i in range(k):
            if i in label_counts:
                color = tuple(centers[i])
                percentage = label_counts[i] / len(labels) * 100
                dominant_colors.append((color, percentage))
        
        # Sort by percentage (most dominant first)
        dominant_colors.sort(key=lambda x: x[1], reverse=True)
        
        return dominant_colors
    
    def classify_color(self, bgr_color: Tuple[int, int, int]) -> str:
        """Classify BGR color into human-readable categories"""
        b, g, r = bgr_color
        
        # Convert to HSV for better color classification
        hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        
        # Color classification based on HSV
        if v < 50:
            return "Black/Dark"
        elif v > 200 and s < 30:
            return "White/Light"
        elif s < 50:
            return "Gray"
        elif h < 10 or h > 170:
            return "Red"
        elif h < 25:
            return "Orange"
        elif h < 35:
            return "Yellow"
        elif h < 85:
            return "Green"
        elif h < 125:
            return "Blue"
        elif h < 150:
            return "Purple"
        else:
            return "Pink/Magenta"
    
    def analyze_detection_colors(self, frame: np.ndarray, detections: List[Dict]):
        """Analyze colors in each detection region"""
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Get dominant colors
            try:
                dominant_colors = self.extract_dominant_colors(roi, k=3)
                
                # Store color information
                detection['dominant_colors'] = dominant_colors
                
                # Classify colors
                color_names = []
                for color, percentage in dominant_colors:
                    color_name = self.classify_color(color)
                    color_names.append((color_name, percentage))
                
                detection['color_names'] = color_names
                
                # Add to global color samples
                self.color_samples.extend(dominant_colors)
                
                # Print color analysis for this detection
                self.detection_count += 1
                print(f"Detection {self.detection_count}:")
                print(f"  Area: {detection.get('area', 0):.0f}")
                print(f"  Confidence: {detection['confidence']:.3f}")
                print(f"  Top Colors:")
                for j, ((b, g, r), percentage) in enumerate(dominant_colors):
                    color_name = self.classify_color((b, g, r))
                    print(f"    {j+1}. {color_name}: {percentage:.1f}% - RGB({r},{g},{b})")
                print()
                
            except Exception as e:
                logger.warning(f"Color analysis failed for detection {i}: {e}")
    
    def analyze_colors(self, num_frames: int = 50):
        """Analyze colors over multiple frames"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Color Analysis - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        print("=" * 60)
        print("COLOR ANALYSIS FOR PALLET DETECTION")
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
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Print progress every 10 frames
                if self.frame_count % 10 == 0:
                    print(f"Processed {self.frame_count}/{num_frames} frames, "
                          f"Total detections analyzed: {self.detection_count}")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Analysis stopped by user")
                    break
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                break
        
        # Show final results
        self._print_color_analysis()
        self.stop_analysis()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and analyze colors"""
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
            
            # Filter by area
            area_filtered = [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]
            
            # Analyze colors in filtered detections
            self.analyze_detection_colors(processed_frame, area_filtered)
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        # Draw detections with color info
        processed_frame = self._draw_detections_with_colors(processed_frame)
        
        return processed_frame
    
    def _draw_detections_with_colors(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections with color information"""
        result_frame = frame.copy()
        
        for detection in self.pallet_detector.last_detections:
            if not (self.MIN_AREA <= detection.get('area', 0) <= self.MAX_AREA):
                continue
                
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw color information if available
            if 'color_names' in detection:
                y_offset = y1 - 10
                for i, (color_name, percentage) in enumerate(detection['color_names'][:2]):  # Top 2 colors
                    label = f"{color_name}: {percentage:.1f}%"
                    cv2.putText(result_frame, label, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset -= 15
        
        return result_frame

    def _print_color_analysis(self):
        """Print comprehensive color analysis results"""
        print("\n" + "=" * 60)
        print("COLOR ANALYSIS RESULTS")
        print("=" * 60)

        if not self.color_samples:
            print("No color data collected!")
            return

        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Detections Analyzed: {self.detection_count}")
        print(f"Total Color Samples: {len(self.color_samples)}")
        print()

        # Aggregate all colors and classify them
        color_categories = {}

        for color, percentage in self.color_samples:
            color_name = self.classify_color(color)
            if color_name not in color_categories:
                color_categories[color_name] = []
            color_categories[color_name].append(percentage)

        # Calculate statistics for each color category
        print("COLOR DISTRIBUTION IN PALLET DETECTIONS:")
        print("-" * 50)

        total_samples = len(self.color_samples)
        color_stats = []

        for color_name, percentages in color_categories.items():
            count = len(percentages)
            avg_percentage = np.mean(percentages)
            occurrence_rate = count / total_samples * 100

            color_stats.append({
                'name': color_name,
                'count': count,
                'avg_percentage': avg_percentage,
                'occurrence_rate': occurrence_rate
            })

        # Sort by occurrence rate
        color_stats.sort(key=lambda x: x['occurrence_rate'], reverse=True)

        print(f"{'Color':<15} {'Occurrences':<12} {'Avg %':<8} {'Rate %':<8}")
        print("-" * 50)

        for stat in color_stats:
            print(f"{stat['name']:<15} {stat['count']:<12} {stat['avg_percentage']:<8.1f} {stat['occurrence_rate']:<8.1f}")

        print()
        print("TOP 3 COLORS FOR FILTERING:")
        print("-" * 30)

        top_colors = color_stats[:3]
        for i, color in enumerate(top_colors, 1):
            print(f"{i}. {color['name']}: {color['occurrence_rate']:.1f}% occurrence rate")

        print()
        print("RECOMMENDED COLOR FILTER SETTINGS:")
        print("-" * 40)

        if top_colors:
            primary_color = top_colors[0]['name']
            secondary_color = top_colors[1]['name'] if len(top_colors) > 1 else None

            print(f"Primary Color Filter: {primary_color}")
            if secondary_color:
                print(f"Secondary Color Filter: {secondary_color}")

            # Suggest HSV ranges based on top colors
            print("\nSuggested HSV Ranges:")
            for color in top_colors[:2]:
                hsv_range = self._get_hsv_range_for_color(color['name'])
                if hsv_range:
                    print(f"{color['name']}: {hsv_range}")

        print("=" * 60)

    def _get_hsv_range_for_color(self, color_name: str) -> str:
        """Get HSV range for color filtering"""
        hsv_ranges = {
            "Brown": "H: 10-20, S: 50-255, V: 20-200",
            "White/Light": "H: 0-180, S: 0-30, V: 200-255",
            "Gray": "H: 0-180, S: 0-50, V: 50-200",
            "Black/Dark": "H: 0-180, S: 0-255, V: 0-50",
            "Red": "H: 0-10 or 170-180, S: 50-255, V: 50-255",
            "Orange": "H: 10-25, S: 50-255, V: 50-255",
            "Yellow": "H: 25-35, S: 50-255, V: 50-255",
            "Green": "H: 35-85, S: 50-255, V: 50-255",
            "Blue": "H: 85-125, S: 50-255, V: 50-255"
        }
        return hsv_ranges.get(color_name, "Custom range needed")

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
    print("COLOR ANALYSIS FOR PALLET DETECTION")
    print("=" * 50)
    print("This will analyze colors in detected pallets to identify")
    print("the most common colors for filtering")
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("=" * 50)

    analyzer = ColorAnalysisDetector(camera_id=11)

    try:
        # Analyze 50 frames (adjust as needed)
        analyzer.analyze_colors(num_frames=50)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
    finally:
        analyzer.stop_analysis()


if __name__ == "__main__":
    main()
