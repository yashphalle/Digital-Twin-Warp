#!/usr/bin/env python3
"""
YOLOv8 Tracking Database Module
Manages YOLOv8 track IDs and converts them to global warehouse IDs
Replaces SIFT-based feature matching with YOLOv8 built-in tracking
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class YOLOv8TrackingDatabase:
    """YOLOv8-based tracking database using built-in track IDs"""

    def __init__(self, database_file: str = "yolo_tracking_database.pkl", camera_id: int = 1):
        # Update database file path to use data/features directory
        if not database_file.startswith("data/features/"):
            database_file = f"data/features/{os.path.basename(database_file)}"
        
        self.database_file = database_file
        self.camera_id = camera_id
        self.track_mappings = {}  # track_id -> global_id mapping
        self.global_objects = {}  # global_id -> object data
        
        # Camera-prefixed Global IDs: Camera 8 → 8001, 8002, 8003...
        self.next_global_id = camera_id * 1000 + 1
        self.load_database()

        logger.info(f"🎯 YOLOv8 Tracking Database - Camera {camera_id} Global ID range: {camera_id}001 - {camera_id}999")
        logger.info(f"✅ YOLOv8 tracking database initialized with {len(self.global_objects)} objects")

    def load_database(self):
        """Load tracking database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.track_mappings = data.get('track_mappings', {})
                    self.global_objects = data.get('global_objects', {})
                    self.next_global_id = data.get('next_global_id', self.camera_id * 1000 + 1)
                logger.info(f"✅ Loaded YOLOv8 tracking database: {len(self.global_objects)} objects")
            else:
                logger.info("📁 No existing YOLOv8 tracking database found, starting fresh")
        except Exception as e:
            logger.error(f"❌ Error loading YOLOv8 tracking database: {e}")
            self.track_mappings = {}
            self.global_objects = {}

    def save_database(self):
        """Save tracking database to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
            
            data = {
                'track_mappings': self.track_mappings,
                'global_objects': self.global_objects,
                'next_global_id': self.next_global_id,
                'camera_id': self.camera_id,
                'last_updated': datetime.now().isoformat(),
                'tracking_method': 'yolov8_bytetrack'
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"💾 YOLOv8 tracking database saved")
        except Exception as e:
            logger.error(f"❌ Error saving YOLOv8 tracking database: {e}")

    def assign_global_id_from_track_id(self, detection: Dict) -> Tuple[int, str, float]:
        """
        Assign global ID based on YOLOv8 track ID
        
        Args:
            detection: Detection dictionary with track_id
            
        Returns:
            Tuple of (global_id, status, confidence)
        """
        track_id = detection.get('track_id')
        
        # If no track ID available, create new object
        if track_id is None:
            return self._create_new_global_object(detection)
        
        # Convert track_id to string for dictionary key
        track_key = str(track_id)
        
        # Check if we've seen this track ID before
        if track_key in self.track_mappings:
            # Existing tracked object
            global_id = self.track_mappings[track_key]
            self._update_existing_object(global_id, detection)
            return global_id, 'existing', 1.0
        else:
            # New track ID - create new global object
            global_id = self._create_new_global_object(detection)
            self.track_mappings[track_key] = global_id
            return global_id, 'new', 1.0

    def _create_new_global_object(self, detection: Dict) -> int:
        """Create new global object from detection"""
        global_id = self.next_global_id
        self.next_global_id += 1

        object_data = {
            'global_id': global_id,
            'track_id': detection.get('track_id'),
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'times_seen': 1,
            'camera_id': self.camera_id,
            
            # Detection properties
            'bbox': detection.get('bbox'),
            'confidence': detection.get('confidence'),
            'area': detection.get('area'),
            'class_id': detection.get('class_id'),
            'class_name': detection.get('class_name'),
            
            # Physical coordinates (if available)
            'physical_x_ft': detection.get('physical_x_ft'),
            'physical_y_ft': detection.get('physical_y_ft'),
            'coordinate_status': detection.get('coordinate_status'),
            
            # Color information (if available)
            'color_rgb': detection.get('color_rgb'),
            'color_hsv': detection.get('color_hsv'),
            'color_hex': detection.get('color_hex'),
            
            # Tracking metadata
            'tracking_method': 'yolov8_bytetrack',
            'detection_method': detection.get('detection_method', 'yolov8_tracking')
        }

        self.global_objects[global_id] = object_data
        logger.debug(f"🆕 Created new global object {global_id} for track_id {detection.get('track_id')}")
        return global_id

    def _update_existing_object(self, global_id: int, detection: Dict):
        """Update existing global object with new detection"""
        if global_id in self.global_objects:
            obj_data = self.global_objects[global_id]
            obj_data['last_seen'] = datetime.now().isoformat()
            obj_data['times_seen'] += 1
            
            # Update latest detection properties
            obj_data['bbox'] = detection.get('bbox')
            obj_data['confidence'] = detection.get('confidence')
            obj_data['area'] = detection.get('area')
            
            # Update physical coordinates if available
            if detection.get('physical_x_ft') is not None:
                obj_data['physical_x_ft'] = detection.get('physical_x_ft')
            if detection.get('physical_y_ft') is not None:
                obj_data['physical_y_ft'] = detection.get('physical_y_ft')
            if detection.get('coordinate_status') is not None:
                obj_data['coordinate_status'] = detection.get('coordinate_status')
            
            # Update color information if available
            if detection.get('color_rgb') is not None:
                obj_data['color_rgb'] = detection.get('color_rgb')
            if detection.get('color_hsv') is not None:
                obj_data['color_hsv'] = detection.get('color_hsv')
            if detection.get('color_hex') is not None:
                obj_data['color_hex'] = detection.get('color_hex')
            
            logger.debug(f"🔄 Updated global object {global_id} (seen {obj_data['times_seen']} times)")

    def cleanup_old_tracks(self, active_track_ids: List[int], max_age_minutes: int = 30):
        """
        Clean up old track mappings that haven't been seen recently
        
        Args:
            active_track_ids: List of currently active track IDs
            max_age_minutes: Maximum age in minutes before cleanup
        """
        current_time = datetime.now()
        tracks_to_remove = []
        
        for track_key, global_id in self.track_mappings.items():
            track_id = int(track_key)
            
            # If track is still active, skip
            if track_id in active_track_ids:
                continue
            
            # Check if object is too old
            if global_id in self.global_objects:
                obj_data = self.global_objects[global_id]
                last_seen = datetime.fromisoformat(obj_data['last_seen'])
                age_minutes = (current_time - last_seen).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    tracks_to_remove.append((track_key, global_id))
        
        # Remove old tracks
        for track_key, global_id in tracks_to_remove:
            del self.track_mappings[track_key]
            if global_id in self.global_objects:
                del self.global_objects[global_id]
            logger.info(f"🧹 Cleaned up old track {track_key} -> global_id {global_id}")

    def get_statistics(self) -> Dict:
        """Get tracking database statistics"""
        return {
            'total_global_objects': len(self.global_objects),
            'total_track_mappings': len(self.track_mappings),
            'next_global_id': self.next_global_id,
            'camera_id': self.camera_id,
            'tracking_method': 'yolov8_bytetrack'
        }

    def assign_global_id(self, image_region, detection_info):
        """
        Compatibility method for existing pipeline
        Redirects to YOLOv8 track ID-based assignment
        """
        return self.assign_global_id_from_track_id(detection_info)

    def mark_disappeared_objects(self, seen_ids):
        """
        Compatibility method for existing pipeline
        YOLOv8 tracking handles disappearance automatically via track IDs
        """
        # YOLOv8 ByteTrack handles object disappearance automatically
        # This method is kept for compatibility but doesn't need implementation
        logger.debug(f"YOLOv8 tracking: {len(seen_ids)} objects currently tracked")
        pass
