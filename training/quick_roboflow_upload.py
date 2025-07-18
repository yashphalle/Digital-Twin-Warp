#!/usr/bin/env python3
"""
Quick Roboflow Upload Script - Get data to annotation team ASAP
Curates best images from collection and uploads to Roboflow
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from roboflow import Roboflow
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv.configs.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickRoboflowUploader:
    """Fast-track uploader for immediate annotation"""
    
    def __init__(self):
        # Load from environment
        self.api_key = os.getenv('ROBOFLOW_API_KEY')
        self.workspace = os.getenv('ROBOFLOW_WORKSPACE', 'Warp')
        self.project = os.getenv('ROBOFLOW_PROJECT', 'warp-computer-vision-system')
        
        if not self.api_key:
            raise ValueError("❌ ROBOFLOW_API_KEY not found in environment variables")
        
        # Initialize Roboflow
        self.rf = Roboflow(api_key=self.api_key)
        self.workspace_obj = self.rf.workspace(self.workspace)
        
        # Try to get existing project or create new one
        try:
            self.project_obj = self.workspace_obj.project(self.project)
            logger.info(f"✅ Connected to existing project: {self.project}")
        except:
            logger.info(f"🆕 Creating new project: {self.project}")
            self.project_obj = self.workspace_obj.create_project(
                project_name=self.project,
                project_type="object-detection"
            )
        
        self.stats = {'uploaded': 0, 'failed': 0, 'skipped': 0}
    
    def assess_image_quality(self, image_path: str) -> float:
        """Quick quality assessment - reuse existing logic"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {image_path}: {e}")
            return 0.0
    
    def curate_best_images(self, source_dir: str, target_count: int = 500) -> List[str]:
        """Quickly curate best images for annotation"""
        logger.info(f"🔍 Curating best {target_count} images from {source_dir}")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"❌ Source directory not found: {source_dir}")
            return []
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(source_path.rglob(f"*{ext}"))
            all_images.extend(source_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"📁 Found {len(all_images)} total images")
        
        # Quick quality assessment
        image_scores = []
        for i, img_path in enumerate(all_images):
            if i % 100 == 0:
                logger.info(f"📊 Assessing quality: {i}/{len(all_images)}")
            
            quality = self.assess_image_quality(str(img_path))
            image_scores.append((img_path, quality))
        
        # Sort by quality and take top images
        image_scores.sort(key=lambda x: x[1], reverse=True)
        best_images = [str(img_path) for img_path, score in image_scores[:target_count] if score > 0.3]
        
        logger.info(f"✅ Curated {len(best_images)} high-quality images")
        return best_images
    
    def upload_batch(self, image_paths: List[str], batch_size: int = 10) -> Dict:
        """Upload images in batches with progress tracking"""
        logger.info(f"🚀 Starting upload of {len(image_paths)} images")
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            logger.info(f"📤 Uploading batch {current_batch}/{total_batches} ({len(batch_paths)} images)")
            
            for img_path in batch_paths:
                try:
                    # Upload to Roboflow
                    self.project_obj.upload(
                        image_path=img_path,
                        split="train"  # Default to train split
                    )
                    
                    self.stats['uploaded'] += 1
                    logger.info(f"✅ Uploaded: {Path(img_path).name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to upload {img_path}: {e}")
                    self.stats['failed'] += 1
                
                # Rate limiting
                time.sleep(0.5)  # 0.5 seconds between uploads
            
            # Longer pause between batches
            if current_batch < total_batches:
                logger.info(f"⏸️ Batch {current_batch} complete. Pausing 5 seconds...")
                time.sleep(5)
        
        logger.info(f"📊 Upload complete: {self.stats['uploaded']} success, {self.stats['failed']} failed")
        return self.stats
    
    def setup_annotation_classes(self):
        """Setup annotation classes in Roboflow"""
        try:
            # Define your classes - SINGLE CLASS FOR PALLET DETECTION
            classes = ["pallet"]

            logger.info(f"🏷️ Setting up annotation classes: {classes}")
            logger.info("📝 Single class 'pallet' - no classification, just detection")

            # Note: Class setup is typically done through Roboflow UI
            # This is a placeholder for API-based class setup if available

            return True

        except Exception as e:
            logger.error(f"❌ Class setup error: {e}")
            return False

def main():
    """Main execution function"""
    try:
        # Initialize uploader
        uploader = QuickRoboflowUploader()
        
        # Setup annotation classes
        uploader.setup_annotation_classes()
        
        # Curate images from your collection
        collection_dir = Config.RAW_COLLECTION_DIR
        logger.info(f"📁 Looking for images in: {collection_dir}")
        
        # Curate best 500 images for immediate annotation
        best_images = uploader.curate_best_images(collection_dir, target_count=500)
        
        if not best_images:
            logger.error("❌ No suitable images found for upload")
            return
        
        # Upload to Roboflow
        results = uploader.upload_batch(best_images, batch_size=10)
        
        # Print results
        print(f"""
🎉 ROBOFLOW UPLOAD COMPLETE
==========================
✅ Successfully uploaded: {results['uploaded']} images
❌ Failed uploads: {results['failed']} images
📊 Success rate: {results['uploaded']/(results['uploaded']+results['failed'])*100:.1f}%

🔗 Next Steps:
1. Go to https://app.roboflow.com/{uploader.workspace}/{uploader.project}
2. Start annotating with your team
3. Use keyboard shortcuts for faster annotation
4. Export annotated data when ready

💡 Pro Tips:
- Use 'A' key for quick box drawing
- Use 'D' key to duplicate annotations
- Use batch operations for similar images
        """)
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
