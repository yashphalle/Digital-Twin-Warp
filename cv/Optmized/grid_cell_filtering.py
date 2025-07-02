#!/usr/bin/env python3
"""
Grid Cell Filtering for Pallet Detection
Uses 3x3 neighboring cell approach to avoid duplicate detections
Super efficient O(1) filtering method
"""

import cv2
import numpy as np
import logging
import sys
import os
from typing import List, Dict, Tuple, Set
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cv.configs.config import Config
from cv.configs.warehouse_config import get_warehouse_config
from fisheye_corrector import OptimizedFisheyeCorrector
from pallet_detector_simple import SimplePalletDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GridCellFilteringDetector:
    """Pallet detection with efficient grid cell filtering"""
    
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
        
        # AREA FILTERING SETTINGS
        self.MIN_AREA = 10000   # Exclude very small noise
        self.MAX_AREA = 100000  # Exclude very large background objects
        
        # GRID CELL FILTERING SETTINGS
        self.CELL_SIZE = 40     # Grid cell size in pixels (40x40 pixel cells - half of original)
        self.cell_step = 10     # Step size for adjusting cell size
        
        # Detection results
        self.raw_detections = []
        self.area_filtered_detections = []
        self.grid_filtered_detections = []
        self.filtered_out_by_grid = []
        
        # Display options
        self.show_grid = True
        self.show_centers = True
        
        logger.info(f"Initialized grid cell filtering for {self.camera_name}")
        logger.info(f"Area filter: {self.MIN_AREA} - {self.MAX_AREA} pixels")
        logger.info(f"Grid cell size: {self.CELL_SIZE}x{self.CELL_SIZE} pixels")
        logger.info(f"Minimum distance: ~{self.CELL_SIZE} pixels")
        logger.info(f"Exclusion zone: {self.CELL_SIZE*3}x{self.CELL_SIZE*3} pixels per detection")
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
    
    def calculate_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y
    
    def get_grid_cell(self, center: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid cell coordinates for a center point"""
        x, y = center
        cell_x = int(x // self.CELL_SIZE)
        cell_y = int(y // self.CELL_SIZE)
        return cell_x, cell_y
    
    def get_neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all 9 cells (current + 8 neighbors) for a given cell"""
        cell_x, cell_y = cell
        neighbors = []
        
        # Check 3x3 grid around the cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                neighbors.append(neighbor_cell)
        
        return neighbors
    
    def apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """Apply area-based filtering"""
        filtered = []
        for detection in detections:
            area = detection.get('area', 0)
            if self.MIN_AREA <= area <= self.MAX_AREA:
                filtered.append(detection)
        return filtered
    
    def apply_grid_cell_filter(self, detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply 3x3 neighboring cell filtering
        Super efficient O(1) lookup method to avoid duplicate detections
        """
        if len(detections) <= 1:
            return detections, []
        
        # Calculate centers for all detections
        for detection in detections:
            center = self.calculate_center(detection['bbox'])
            detection['center'] = center
            detection['grid_cell'] = self.get_grid_cell(center)
        
        # Sort by confidence (keep higher confidence detections first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        occupied_cells: Set[Tuple[int, int]] = set()
        filtered = []
        filtered_out = []
        
        for detection in sorted_detections:
            cell = detection['grid_cell']
            
            # Get all 9 neighboring cells (including current cell)
            neighbor_cells = self.get_neighbor_cells(cell)
            
            # Check if any of the 9 cells are already occupied
            conflict = False
            conflicting_cell = None
            
            for neighbor_cell in neighbor_cells:
                if neighbor_cell in occupied_cells:
                    conflict = True
                    conflicting_cell = neighbor_cell
                    break
            
            if conflict:
                # This detection is too close to an existing one
                detection['filter_reason'] = 'GRID_CONFLICT'
                detection['conflicting_cell'] = conflicting_cell
                filtered_out.append(detection)
            else:
                # This detection is good - mark its cell as occupied
                occupied_cells.add(cell)
                filtered.append(detection)
        
        return filtered, filtered_out
    
    def start_detection(self):
        """Start the detection with grid cell filtering"""
        if not self.connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        self.running = True
        
        # Create display window
        window_name = f"Grid Cell Filtering - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        logger.info("=== GRID CELL FILTERING ===")
        logger.info("Controls:")
        logger.info("  'q' or ESC - Quit")
        logger.info("  'n'/'p' - Next/Previous prompt")
        logger.info("  '+'/'-' - Increase/Decrease cell size")
        logger.info("  'g' - Toggle grid display")
        logger.info("  'c' - Toggle center points display")
        logger.info("=" * 50)
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n'):  # Next prompt
                    self.pallet_detector.next_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('p'):  # Previous prompt
                    self.pallet_detector.previous_prompt()
                    if self.pallet_detector.grounding_dino:
                        self.pallet_detector.grounding_dino.prompt = self.pallet_detector.current_prompt
                    logger.info(f"Switched to prompt: '{self.pallet_detector.current_prompt}'")
                elif key == ord('+') or key == ord('='):  # Increase cell size
                    self.CELL_SIZE += self.cell_step
                    logger.info(f"Cell size: {self.CELL_SIZE}x{self.CELL_SIZE} pixels")
                elif key == ord('-'):  # Decrease cell size
                    self.CELL_SIZE = max(20, self.CELL_SIZE - self.cell_step)
                    logger.info(f"Cell size: {self.CELL_SIZE}x{self.CELL_SIZE} pixels")
                elif key == ord('g'):  # Toggle grid display
                    self.show_grid = not self.show_grid
                    logger.info(f"Grid display: {'ON' if self.show_grid else 'OFF'}")
                elif key == ord('c'):  # Toggle center points
                    self.show_centers = not self.show_centers
                    logger.info(f"Center points: {'ON' if self.show_centers else 'OFF'}")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
        
        self.stop_detection()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and grid filtering"""
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
            self.raw_detections = self.pallet_detector.detect_pallets(processed_frame)
            self.area_filtered_detections = self.apply_area_filter(self.raw_detections)
            self.grid_filtered_detections, self.filtered_out_by_grid = self.apply_grid_cell_filter(self.area_filtered_detections)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.raw_detections = []
            self.area_filtered_detections = []
            self.grid_filtered_detections = []
            self.filtered_out_by_grid = []
        
        # Draw results
        processed_frame = self._draw_grid(processed_frame)
        processed_frame = self._draw_detections(processed_frame)
        processed_frame = self._draw_info_overlay(processed_frame)
        
        return processed_frame

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw grid lines to visualize cells"""
        if not self.show_grid:
            return frame

        height, width = frame.shape[:2]

        # Draw vertical lines
        for x in range(0, width, self.CELL_SIZE):
            cv2.line(frame, (x, 0), (x, height), (100, 100, 100), 1)

        # Draw horizontal lines
        for y in range(0, height, self.CELL_SIZE):
            cv2.line(frame, (0, y), (width, y), (100, 100, 100), 1)

        return frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results with grid filtering status"""
        result_frame = frame.copy()

        # Draw filtered out by grid (red)
        for detection in self.filtered_out_by_grid:
            bbox = detection['bbox']
            center = detection['center']

            x1, y1, x2, y2 = bbox

            # Red for grid conflicts
            color = (0, 0, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 1)

            # Draw center point
            if self.show_centers:
                cv2.circle(result_frame, center, 4, color, -1)

            # Label
            label = "GRID_CONFLICT"
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw accepted detections (green)
        for detection in self.grid_filtered_detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            area = detection.get('area', 0)
            grid_cell = detection['grid_cell']

            x1, y1, x2, y2 = bbox

            # Green for accepted
            color = (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            if self.show_centers:
                cv2.circle(result_frame, center, 8, color, -1)
                cv2.circle(result_frame, center, 8, (255, 255, 255), 2)

            # Label with grid cell info
            label = f"PALLET: {area:.0f}"
            conf_label = f"Conf: {confidence:.2f}"
            grid_label = f"Cell: ({grid_cell[0]},{grid_cell[1]})"

            cv2.putText(result_frame, label, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, conf_label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_frame, grid_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return result_frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw filtering information overlay"""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"Prompt: '{self.pallet_detector.current_prompt}'", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"Area Filter: {self.MIN_AREA} - {self.MAX_AREA} pixels", (20, y_offset), font, 0.5, color, 1)

        y_offset += 20
        cv2.putText(frame, f"Grid Cell Size: {self.CELL_SIZE}x{self.CELL_SIZE} pixels", (20, y_offset), font, font_scale, color, thickness)

        y_offset += 20
        cv2.putText(frame, f"Min Distance: ~{self.CELL_SIZE} pixels", (20, y_offset), font, 0.5, color, 1)

        y_offset += 25
        cv2.putText(frame, f"Raw Detections: {len(self.raw_detections)}", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"After Area Filter: {len(self.area_filtered_detections)}", (20, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"Final Accepted (Green): {len(self.grid_filtered_detections)}", (20, y_offset), font, 0.5, (0, 255, 0), 1)

        y_offset += 20
        cv2.putText(frame, f"Grid Conflicts (Red): {len(self.filtered_out_by_grid)}", (20, y_offset), font, 0.5, (0, 0, 255), 1)

        y_offset += 25
        cv2.putText(frame, "Controls: +/-=CellSize, g=Grid, c=Centers, n/p=Prompt", (20, y_offset), font, 0.4, (255, 255, 255), 1)

        return frame

    def stop_detection(self):
        """Stop the detection"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        cv2.destroyAllWindows()

        logger.info(f"Stopped detection for {self.camera_name}")


def main():
    """Main function"""
    print("GRID CELL FILTERING FOR PALLET DETECTION")
    print("=" * 50)
    print("Camera: 11")
    print("Prompts: ['pallet wrapped in plastic', 'stack of goods on pallet']")
    print("Confidence: 0.1")
    print("Area Filter: 10,000 - 100,000 pixels")
    print("Grid Cell Size: 40x40 pixels (reduced for better accuracy)")
    print("Minimum Distance: ~40 pixels between detections")
    print("Exclusion Zone: 120x120 pixels per accepted detection")
    print("=" * 50)
    print("\nHow it works:")
    print("1. Divides image into 80x80 pixel grid cells")
    print("2. Each detection center falls into one cell")
    print("3. Checks 3x3 neighboring cells (9 total)")
    print("4. If any neighbor cell is occupied, rejects detection")
    print("5. Super efficient O(1) lookup - no distance calculations!")
    print("=" * 50)

    detector = GridCellFilteringDetector(camera_id=11)

    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    except Exception as e:
        logger.error(f"Error running detector: {e}")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()
