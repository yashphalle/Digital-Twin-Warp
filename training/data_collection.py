#!/usr/bin/env python3
"""
Automated Data Collection for Custom Training
Collects diverse images from warehouse cameras for training dataset
"""

import cv2
import os
import time
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cv.configs.warehouse_config import get_warehouse_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataCollector:
    """Collect training data from warehouse cameras"""
    
    def __init__(self, output_dir: str = "training/data/raw_images"):
        self.output_dir = output_dir
        self.warehouse_config = get_warehouse_config()
        self.collection_stats = {
            'total_images': 0,
            'images_per_camera': {},
            'collection_sessions': [],
            'quality_metrics': {}
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collection parameters
        self.capture_interval = 30  # seconds between captures
        self.min_image_quality = 0.7  # blur detection threshold
        self.max_images_per_camera = 500  # limit per camera
        
    def assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality using blur detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (higher = better quality)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0
    
    def capture_from_camera(self, camera_id: int, duration_minutes: int = 60) -> List[str]:
        """Capture images from a specific camera"""
        camera_config = self.warehouse_config.get(camera_id, {})
        rtsp_url = camera_config.get('rtsp_url_remote', 
                                   f'rtsp://admin:wearewarp!@104.181.138.5:556{camera_id}/Streaming/channels/1')
        
        logger.info(f"📹 Starting collection from Camera {camera_id}")
        logger.info(f"   URL: {rtsp_url}")
        logger.info(f"   Duration: {duration_minutes} minutes")
        
        # Connect to camera
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"❌ Failed to connect to Camera {camera_id}")
            return []
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        captured_files = []
        start_time = time.time()
        last_capture = 0
        image_count = 0
        
        camera_dir = os.path.join(self.output_dir, f"camera_{camera_id}")
        os.makedirs(camera_dir, exist_ok=True)
        
        try:
            while (time.time() - start_time) < (duration_minutes * 60):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from Camera {camera_id}")
                    time.sleep(1)
                    continue
                
                # Check if it's time to capture
                current_time = time.time()
                if current_time - last_capture >= self.capture_interval:
                    
                    # Assess image quality
                    quality = self.assess_image_quality(frame)
                    
                    if quality >= self.min_image_quality:
                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"camera_{camera_id}_{timestamp}_{image_count:04d}.jpg"
                        filepath = os.path.join(camera_dir, filename)
                        
                        # Save image
                        cv2.imwrite(filepath, frame)
                        captured_files.append(filepath)
                        image_count += 1
                        
                        logger.info(f"   📸 Captured: {filename} (quality: {quality:.3f})")
                        
                        # Update stats
                        if camera_id not in self.collection_stats['images_per_camera']:
                            self.collection_stats['images_per_camera'][camera_id] = 0
                        self.collection_stats['images_per_camera'][camera_id] += 1
                        self.collection_stats['total_images'] += 1
                        
                        last_capture = current_time
                        
                        # Check if we've reached the limit
                        if image_count >= self.max_images_per_camera:
                            logger.info(f"   ✅ Reached limit for Camera {camera_id}")
                            break
                    else:
                        logger.debug(f"   ⚠️ Low quality image skipped (quality: {quality:.3f})")
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info(f"   ⏹️ Collection stopped by user for Camera {camera_id}")
        except Exception as e:
            logger.error(f"   ❌ Error during collection: {e}")
        finally:
            cap.release()
        
        logger.info(f"   ✅ Camera {camera_id} collection complete: {len(captured_files)} images")
        return captured_files
    
    def collect_from_multiple_cameras(self, camera_ids: List[int], 
                                    duration_per_camera: int = 30) -> Dict:
        """Collect data from multiple cameras sequentially"""
        
        logger.info(f"🎯 Starting multi-camera collection")
        logger.info(f"   Cameras: {camera_ids}")
        logger.info(f"   Duration per camera: {duration_per_camera} minutes")
        
        session_start = datetime.now()
        all_captured_files = {}
        
        for camera_id in camera_ids:
            logger.info(f"\n📹 Processing Camera {camera_id}...")
            captured_files = self.capture_from_camera(camera_id, duration_per_camera)
            all_captured_files[camera_id] = captured_files
            
            # Brief pause between cameras
            time.sleep(5)
        
        # Record session
        session_info = {
            'start_time': session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'cameras': camera_ids,
            'duration_per_camera': duration_per_camera,
            'total_images': sum(len(files) for files in all_captured_files.values()),
            'files_per_camera': {cam: len(files) for cam, files in all_captured_files.items()}
        }
        
        self.collection_stats['collection_sessions'].append(session_info)
        
        return all_captured_files
    
    def save_collection_metadata(self):
        """Save collection statistics and metadata"""
        metadata_file = os.path.join(self.output_dir, "collection_metadata.json")
        
        with open(metadata_file, 'w') as f:
            json.dump(self.collection_stats, f, indent=2)
        
        logger.info(f"📄 Metadata saved to: {metadata_file}")
    
    def generate_collection_report(self):
        """Generate human-readable collection report"""
        report = []
        report.append("📊 DATA COLLECTION REPORT")
        report.append("=" * 50)
        report.append(f"Total Images Collected: {self.collection_stats['total_images']}")
        report.append(f"Number of Sessions: {len(self.collection_stats['collection_sessions'])}")
        report.append("")
        
        report.append("📹 Images per Camera:")
        for camera_id, count in self.collection_stats['images_per_camera'].items():
            report.append(f"   Camera {camera_id}: {count} images")
        
        report.append("")
        report.append("📅 Collection Sessions:")
        for i, session in enumerate(self.collection_stats['collection_sessions'], 1):
            report.append(f"   Session {i}: {session['start_time']}")
            report.append(f"      Cameras: {session['cameras']}")
            report.append(f"      Total Images: {session['total_images']}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = os.path.join(self.output_dir, "collection_report.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Print report
        print("\n" + report_text)
        
        return report_text

def main():
    """Main collection function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from warehouse cameras')
    parser.add_argument('--cameras', nargs='+', type=int, default=[8, 3, 4], 
                       help='Camera IDs to collect from (default: 8 3 4)')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration per camera in minutes (default: 30)')
    parser.add_argument('--output-dir', type=str, default='training/data/raw_images',
                       help='Output directory for images')
    parser.add_argument('--interval', type=int, default=30,
                       help='Capture interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Create collector
    collector = TrainingDataCollector(output_dir=args.output_dir)
    collector.capture_interval = args.interval
    
    # Start collection
    try:
        results = collector.collect_from_multiple_cameras(
            camera_ids=args.cameras,
            duration_per_camera=args.duration
        )
        
        # Save metadata and generate report
        collector.save_collection_metadata()
        collector.generate_collection_report()
        
        print(f"\n✅ Collection complete! Check {args.output_dir} for images")
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user")
        collector.save_collection_metadata()
        collector.generate_collection_report()

if __name__ == "__main__":
    main()
