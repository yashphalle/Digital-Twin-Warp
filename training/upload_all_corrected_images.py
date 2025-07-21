#!/usr/bin/env python3
"""
Upload ALL Corrected Images to Roboflow
Simple script that uploads every corrected image without any filtering
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List
from roboflow import Roboflow

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv.configs.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRoboflowUploader:
    """Simple uploader for ALL corrected images"""
    
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
    
    def find_all_corrected_images(self, source_dir: str) -> List[str]:
        """Find ALL corrected images (no filtering)"""
        logger.info(f"🔍 Finding ALL corrected images in: {source_dir}")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"❌ Source directory not found: {source_dir}")
            return []
        
        # Find all corrected images (look for "_corrected" in filename)
        corrected_images = []
        
        # Search for corrected images
        for pattern in ["*_corrected.jpg", "*_corrected.jpeg", "*_corrected.png"]:
            corrected_images.extend(source_path.rglob(pattern))
        
        # Convert to strings
        image_paths = [str(img_path) for img_path in corrected_images]
        
        logger.info(f"📁 Found {len(image_paths)} corrected images")
        
        # Show breakdown by camera if possible
        camera_counts = {}
        for img_path in image_paths:
            filename = Path(img_path).name
            # Extract camera ID from filename (assuming format: camera_X_timestamp_corrected.jpg)
            if 'camera_' in filename:
                try:
                    camera_part = filename.split('camera_')[1].split('_')[0]
                    camera_id = int(camera_part)
                    camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
                except:
                    pass
        
        if camera_counts:
            logger.info("📊 Images per camera:")
            for cam_id in sorted(camera_counts.keys()):
                logger.info(f"   Camera {cam_id}: {camera_counts[cam_id]} images")
        
        return image_paths
    
    def upload_all_images(self, image_paths: List[str], batch_size: int = 5) -> Dict:
        """Upload ALL images in batches"""
        logger.info(f"🚀 Starting upload of ALL {len(image_paths)} corrected images")
        logger.info("⚠️ NO QUALITY FILTERING - Uploading everything!")
        
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
                    filename = Path(img_path).name
                    logger.info(f"✅ Uploaded: {filename}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to upload {img_path}: {e}")
                    self.stats['failed'] += 1
                
                # Rate limiting - slower to avoid API limits
                time.sleep(1.0)  # 1 second between uploads
            
            # Longer pause between batches
            if current_batch < total_batches:
                logger.info(f"⏸️ Batch {current_batch} complete. Pausing 10 seconds...")
                time.sleep(10)
        
        logger.info(f"📊 Upload complete: {self.stats['uploaded']} success, {self.stats['failed']} failed")
        return self.stats

def main():
    """Main execution function"""
    try:
        # Initialize uploader
        uploader = SimpleRoboflowUploader()
        
        # Get corrected images directory
        collection_dir = Config.RAW_COLLECTION_DIR
        logger.info(f"📁 Looking for corrected images in: {collection_dir}")
        
        # Find ALL corrected images (no filtering)
        all_corrected_images = uploader.find_all_corrected_images(collection_dir)
        
        if not all_corrected_images:
            logger.error("❌ No corrected images found!")
            logger.info("💡 Make sure you have images with '_corrected' in the filename")
            return
        
        # Confirm upload
        print(f"\n🚨 UPLOAD CONFIRMATION")
        print(f"=" * 50)
        print(f"📁 Source: {collection_dir}")
        print(f"📊 Found: {len(all_corrected_images)} corrected images")
        print(f"🎯 Target: Roboflow project '{uploader.project}'")
        print(f"⚠️ This will upload ALL corrected images (no filtering)")
        print(f"=" * 50)
        
        response = input("Continue with upload? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Upload cancelled by user")
            return
        
        # Upload ALL corrected images
        results = uploader.upload_all_images(all_corrected_images, batch_size=5)
        
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
3. All corrected images are now available for annotation

💡 Note: All images uploaded were fisheye-corrected versions
        """)
        
    except KeyboardInterrupt:
        print("\n🛑 Upload interrupted by user")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
