#!/usr/bin/env python3
"""
GUI Display Module
Visualization and overlay rendering for warehouse tracking system
Extracted from main.py for modular architecture
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class CPUDisplayManager:
    """CPU-based display manager for warehouse tracking visualization"""
    
    def __init__(self, camera_name: str = "Camera", camera_id: int = 1):
        self.camera_name = camera_name
        self.camera_id = camera_id
        
        # Display configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_height = 20
        
        # Color scheme for tracking status
        self.status_colors = {
            'new': (0, 255, 0),      # Green for new objects
            'existing': (0, 165, 255), # Orange for existing objects
            'failed': (0, 0, 255),    # Red for failed tracking
            'unknown': (128, 128, 128) # Gray for unknown status
        }
        
        # Text colors
        self.text_colors = {
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'orange': (255, 165, 0)
        }
        
        logger.info(f"✅ GUI Display Manager initialized for {camera_name}")

    def get_status_color_and_text(self, tracking_status: str) -> Tuple[Tuple[int, int, int], str]:
        """Get color and text for tracking status"""
        if tracking_status == 'new':
            return self.status_colors['new'], "NEW"
        elif tracking_status == 'existing':
            return self.status_colors['existing'], "TRACKED"
        elif tracking_status == 'failed':
            return self.status_colors['failed'], "FAILED"
        else:
            return self.status_colors['unknown'], "UNKNOWN"

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       filtering_module=None) -> np.ndarray:
        """Draw detection results with bounding boxes, labels, and status indicators"""
        result_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection.get('confidence', 0.0)
            area = detection.get('area', 0)
            
            # Calculate center if not available
            center = detection.get('center')
            if center is None and filtering_module:
                center = filtering_module.calculate_center(bbox)
            elif center is None:
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Get color and status text based on tracking status
            tracking_status = detection.get('tracking_status', 'unknown')
            color, status_text = self.get_status_color_and_text(tracking_status)

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(result_frame, center, 8, color, -1)

            # Text positioning
            y_offset = y1 - 10

            # Global ID and status
            global_id = detection.get('global_id', -1)
            if global_id != -1:
                id_label = f"CPU-ID:{global_id} ({status_text})"
                cv2.putText(result_frame, id_label, (x1, y_offset), self.font, 0.6, color, 2)
                y_offset -= self.line_height

            # Confidence and area
            conf_label = f"Conf:{confidence:.3f} Area:{area:.0f}"
            cv2.putText(result_frame, conf_label, (x1, y_offset), self.font, 0.5, self.text_colors['white'], 1)
            y_offset -= self.line_height

            # Pixel coordinates
            pixel_label = f"Pixel:({center[0]},{center[1]})"
            cv2.putText(result_frame, pixel_label, (x1, y_offset), self.font, 0.4, self.text_colors['white'], 1)
            y_offset -= self.line_height

            # Physical coordinates
            physical_x = detection.get('physical_x_ft')
            physical_y = detection.get('physical_y_ft')
            if physical_x is not None and physical_y is not None:
                coord_label = f"Physical:({physical_x:.1f},{physical_y:.1f})ft"
                coord_color = self.text_colors['cyan']  # Cyan for successful coordinates
            else:
                coord_label = "Physical: FAILED"
                coord_color = self.text_colors['red']  # Red for failed coordinates

            cv2.putText(result_frame, coord_label, (x1, y_offset), self.font, 0.4, coord_color, 1)

        return result_frame

    def draw_info_overlay(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """Draw information overlay with system statistics and pipeline status"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 350), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font_scale = 0.7
        color = self.text_colors['green']
        thickness = 2

        y_offset = 30
        cv2.putText(frame, f"🚀 CPU WAREHOUSE TRACKING", (20, y_offset), self.font, font_scale, color, thickness)

        y_offset += 25
        cv2.putText(frame, f"Camera: {self.camera_name}", (20, y_offset), self.font, 0.5, color, 1)

        # FPS display
        y_offset += 20
        fps = stats.get('fps', 0.0)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), self.font, 0.6, self.text_colors['green'], 2)

        # Frame statistics
        y_offset += 20
        frame_count = stats.get('frame_count', 0)
        frame_skip = stats.get('frame_skip', 20)
        processed_frames = (frame_count + frame_skip - 1) // frame_skip
        cv2.putText(frame, f"Frame: {frame_count} (Processed: {processed_frames}, Skip: {frame_skip})", (20, y_offset), self.font, 0.4, color, 1)

        # Detection pipeline statistics
        y_offset += 25
        cv2.putText(frame, f"CPU DETECTION PIPELINE:", (20, y_offset), self.font, 0.5, self.text_colors['white'], 1)

        pipeline_stats = [
            ("1. CPU Raw Detections", stats.get('raw_detections', 0)),
            ("2. CPU Area Filtered", stats.get('area_filtered_detections', 0)),
            ("3. CPU Grid Filtered", stats.get('grid_filtered_detections', 0)),
            ("4. CPU Size Filtered", stats.get('size_filtered_detections', 0)),
            ("5. CPU Final Tracked", stats.get('final_tracked_detections', 0))
        ]

        for i, (label, count) in enumerate(pipeline_stats):
            y_offset += 20 if i == 0 else 15
            text_color = self.text_colors['green'] if i == 4 else self.text_colors['white']
            cv2.putText(frame, f"{label}: {count}", (20, y_offset), self.font, 0.4, text_color, 1)

        # Tracking statistics
        y_offset += 25
        cv2.putText(frame, f"CPU TRACKING STATS:", (20, y_offset), self.font, 0.5, self.text_colors['white'], 1)

        tracking_stats = [
            ("New Objects", stats.get('new_objects', 0), self.text_colors['green']),
            ("CPU Tracked Objects", stats.get('existing_objects', 0), self.text_colors['orange']),
            ("Database Objects", stats.get('database_objects', 0), self.text_colors['white'])
        ]

        for label, count, text_color in tracking_stats:
            y_offset += 20 if label == "New Objects" else 15
            cv2.putText(frame, f"{label}: {count}", (20, y_offset), self.font, 0.4, text_color, 1)

        # System status
        y_offset += 25
        cv2.putText(frame, f"CPU OPTIMIZATIONS:", (20, y_offset), self.font, 0.5, self.text_colors['white'], 1)

        system_status = [
            ("CPU SIFT", True),
            ("CPU Matcher", True),
            ("CPU Coords", stats.get('coordinate_mapper_initialized', False)),
            ("Color Extract", True)
        ]

        for label, status in system_status:
            y_offset += 20 if label == "CPU SIFT" else 15
            status_symbol = "✅" if status else "❌"
            status_color = self.text_colors['green'] if status else self.text_colors['red']
            cv2.putText(frame, f"{label}: {status_symbol}", (20, y_offset), self.font, 0.4, status_color, 1)

        return frame

    def create_stats_dict(self, tracker) -> Dict:
        """Create statistics dictionary from tracker object"""
        # Get detection counts from frame processor if available
        if hasattr(tracker, 'frame_processor') and tracker.frame_processor:
            detection_counts = tracker.frame_processor.get_detection_counts()
        else:
            # Fallback to direct tracker attributes
            detection_counts = {
                'raw_detections': len(getattr(tracker, 'raw_detections', [])),
                'area_filtered_detections': len(getattr(tracker, 'area_filtered_detections', [])),
                'grid_filtered_detections': len(getattr(tracker, 'grid_filtered_detections', [])),
                'size_filtered_detections': len(getattr(tracker, 'size_filtered_detections', [])),
                'final_tracked_detections': len(getattr(tracker, 'final_tracked_detections', [])),
                'new_objects': getattr(tracker, 'new_objects', 0),
                'existing_objects': getattr(tracker, 'existing_objects', 0)
            }

        return {
            'frame_count': getattr(tracker, 'frame_count', 0),
            'frame_skip': getattr(tracker, 'FRAME_SKIP', 20),
            **detection_counts,
            'database_objects': len(getattr(tracker.global_db, 'features', {})) if hasattr(tracker, 'global_db') else 0,
            'coordinate_mapper_initialized': getattr(tracker, 'coordinate_mapper_initialized', False)
        }

    def render_frame(self, frame: np.ndarray, tracker, filtering_module=None) -> np.ndarray:
        """Complete frame rendering with detections and overlay"""
        # Get detections from frame processor if available
        if hasattr(tracker, 'frame_processor') and tracker.frame_processor:
            detections = tracker.frame_processor.final_tracked_detections
        else:
            detections = getattr(tracker, 'final_tracked_detections', [])

        # Draw detections
        result_frame = self.draw_detections(frame, detections, filtering_module)

        # Draw info overlay
        stats = self.create_stats_dict(tracker)
        result_frame = self.draw_info_overlay(result_frame, stats)

        return result_frame
