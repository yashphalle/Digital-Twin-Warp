#!/usr/bin/env python3
"""
Detection Filtering Module
CPU-based detection filtering system for warehouse tracking
Extracted from main.py for modular architecture
"""

import logging
from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)

class DetectionFiltering:
    """CPU-based detection filtering system"""
    
    def __init__(self, camera_id: int = 1, min_area: int = 10000,
                 max_area: int = 100000, max_physical_size_ft: float = 50.0,
                 cell_size: int = 40):
        self.camera_id = camera_id
        self.MIN_AREA = min_area
        self.MAX_AREA = max_area
        self.MAX_PHYSICAL_SIZE_FT = max_physical_size_ft
        self.CELL_SIZE = cell_size
        
        logger.info(f"Detection filtering initialized for Camera {camera_id}")
        logger.info(f"Area filter: {min_area} - {max_area} pixels")
        logger.info(f"Physical size limit: {max_physical_size_ft}ft")
        logger.info(f"Grid cell size: {cell_size}px")

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

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                neighbors.append(neighbor_cell)

        return neighbors

    def apply_area_filter(self, detections: List[Dict]) -> List[Dict]:
        """CPU area filtering (same as combined filtering)"""
        if not detections:
            return []

        try:
            accepted = []
            for detection in detections:
                area = detection.get('area', 0)
                if self.MIN_AREA <= area <= self.MAX_AREA:
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"CPU area filtering failed: {e}")
            return [d for d in detections if self.MIN_AREA <= d.get('area', 0) <= self.MAX_AREA]

    def apply_grid_cell_filter(self, detections: List[Dict]) -> List[Dict]:
        """CPU grid cell filtering (same as combined filtering)"""
        if len(detections) <= 1:
            return detections

        try:
            # Calculate centers and grid cells for all detections
            for detection in detections:
                center = self.calculate_center(detection['bbox'])
                detection['center'] = center
                detection['grid_cell'] = self.get_grid_cell(center)

            # Sort by confidence (keep higher confidence detections first)
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            occupied_cells: Set[Tuple[int, int]] = set()
            accepted = []

            for detection in sorted_detections:
                cell = detection['grid_cell']
                neighbor_cells = self.get_neighbor_cells(cell)

                # Check if any of the 9 cells are already occupied
                conflict = False
                for neighbor_cell in neighbor_cells:
                    if neighbor_cell in occupied_cells:
                        conflict = True
                        break

                if not conflict:
                    # Accept this detection
                    occupied_cells.add(cell)
                    accepted.append(detection)

            return accepted

        except Exception as e:
            logger.error(f"CPU grid filtering failed: {e}")
            return detections

    def apply_physical_size_filter(self, detections: List[Dict]) -> List[Dict]:
        """Filter out objects that are too large in physical dimensions (>15ft length or width)"""
        if not detections:
            return []

        try:
            accepted = []
            for detection in detections:
                # Check if physical corners are available
                physical_corners = detection.get('physical_corners')
                if not physical_corners or len(physical_corners) != 4:
                    # If no physical corners, skip size filtering (keep the detection)
                    accepted.append(detection)
                    continue

                # Calculate physical dimensions from corners
                valid_corners = [c for c in physical_corners if c[0] is not None and c[1] is not None]
                if len(valid_corners) < 4:
                    # If not all corners are valid, skip size filtering (keep the detection)
                    accepted.append(detection)
                    continue

                # Extract x and y coordinates
                x_coords = [c[0] for c in valid_corners]
                y_coords = [c[1] for c in valid_corners]

                # Calculate physical width and height
                physical_width = max(x_coords) - min(x_coords)
                physical_height = max(y_coords) - min(y_coords)

                # Check if either dimension exceeds the maximum
                if physical_width <= self.MAX_PHYSICAL_SIZE_FT and physical_height <= self.MAX_PHYSICAL_SIZE_FT:
                    accepted.append(detection)
                    logger.debug(f"Camera {self.camera_id}: Object accepted - Size: {physical_width:.1f}ft x {physical_height:.1f}ft")
                else:
                    logger.info(f"Camera {self.camera_id}: Object filtered out - Size: {physical_width:.1f}ft x {physical_height:.1f}ft (exceeds {self.MAX_PHYSICAL_SIZE_FT}ft limit)")

            logger.debug(f"Physical size filtering: {len(accepted)} accepted from {len(detections)} detections")
            return accepted

        except Exception as e:
            logger.error(f"CPU physical size filtering failed: {e}")
            return detections

    def apply_all_filters(self, detections: List[Dict]) -> List[Dict]:
        """Apply all filtering methods in sequence"""
        if not detections:
            return []
        
        # Apply filters in order: area -> grid -> physical size
        filtered = self.apply_area_filter(detections)
        filtered = self.apply_grid_cell_filter(filtered)
        filtered = self.apply_physical_size_filter(filtered)
        
        logger.debug(f"Camera {self.camera_id}: Filtering complete - {len(filtered)} accepted from {len(detections)} detections")
        return filtered
