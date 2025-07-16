#!/usr/bin/env python3
"""
Dataset Preparation for Custom Grounding DINO Training
Converts annotations to training format and creates train/val/test splits
"""

import os
import json
import shutil
import random
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    """Prepare annotated dataset for Grounding DINO training"""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.annotations_dir = self.raw_data_dir / "annotations"
        
        # Create output structure
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            (dir_path / "images").mkdir(parents=True, exist_ok=True)
            (dir_path / "annotations").mkdir(parents=True, exist_ok=True)
        
        # Dataset statistics
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'split_distribution': {'train': 0, 'val': 0, 'test': 0}
        }
    
    def convert_labelme_to_coco(self, labelme_file: str) -> Dict:
        """Convert LabelMe annotation to COCO format"""
        with open(labelme_file, 'r') as f:
            labelme_data = json.load(f)
        
        # Extract image info
        image_info = {
            'file_name': labelme_data['imagePath'],
            'height': labelme_data['imageHeight'],
            'width': labelme_data['imageWidth']
        }
        
        # Convert annotations
        annotations = []
        for shape in labelme_data['shapes']:
            if shape['shape_type'] == 'rectangle':
                # Convert polygon points to bbox
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                annotation = {
                    'category': shape['label'],
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],  # [x, y, width, height]
                    'area': (x_max - x_min) * (y_max - y_min),
                    'text_description': f"{shape['label']} on warehouse floor"
                }
                annotations.append(annotation)
        
        return {
            'image': image_info,
            'annotations': annotations
        }
    
    def create_grounding_dino_format(self, image_path: str, annotations: List[Dict]) -> Dict:
        """Create Grounding DINO training format"""
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Convert bounding boxes to normalized format
        normalized_boxes = []
        text_descriptions = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            
            # Normalize coordinates
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            norm_width = bbox[2] / width
            norm_height = bbox[3] / height
            
            normalized_boxes.append([x_center, y_center, norm_width, norm_height])
            text_descriptions.append(ann['text_description'])
        
        return {
            'image_path': image_path,
            'width': width,
            'height': height,
            'boxes': normalized_boxes,
            'texts': text_descriptions,
            'categories': [ann['category'] for ann in annotations]
        }
    
    def split_dataset(self, all_data: List[Dict], 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """Split dataset into train/val/test"""
        
        # Shuffle data
        random.shuffle(all_data)
        
        total_size = len(all_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]
        
        logger.info(f"📊 Dataset split:")
        logger.info(f"   Train: {len(train_data)} images ({len(train_data)/total_size*100:.1f}%)")
        logger.info(f"   Val: {len(val_data)} images ({len(val_data)/total_size*100:.1f}%)")
        logger.info(f"   Test: {len(test_data)} images ({len(test_data)/total_size*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def save_split_data(self, data: List[Dict], split_name: str):
        """Save data split to disk"""
        split_dir = getattr(self, f"{split_name}_dir")
        
        dataset_file = split_dir / "dataset.json"
        split_data = {
            'images': [],
            'annotations': []
        }
        
        for i, item in enumerate(data):
            # Copy image to split directory
            src_image = item['image_path']
            dst_image = split_dir / "images" / f"{split_name}_{i:06d}.jpg"
            shutil.copy2(src_image, dst_image)
            
            # Update image path in annotation
            item['image_path'] = str(dst_image.name)
            split_data['images'].append(item)
        
        # Save dataset file
        with open(dataset_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"✅ {split_name.capitalize()} split saved: {len(data)} images")
        self.stats['split_distribution'][split_name] = len(data)
    
    def process_annotations(self) -> List[Dict]:
        """Process all annotation files"""
        all_data = []
        
        # Find all annotation files
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        if not annotation_files:
            logger.error(f"❌ No annotation files found in {self.annotations_dir}")
            return []
        
        logger.info(f"📄 Processing {len(annotation_files)} annotation files...")
        
        for ann_file in annotation_files:
            try:
                # Convert LabelMe to COCO format
                coco_data = self.convert_labelme_to_coco(str(ann_file))
                
                # Find corresponding image
                image_name = coco_data['image']['file_name']
                image_path = self.raw_data_dir / "images" / image_name
                
                if not image_path.exists():
                    # Try to find image in camera subdirectories
                    for camera_dir in self.raw_data_dir.glob("camera_*"):
                        potential_path = camera_dir / image_name
                        if potential_path.exists():
                            image_path = potential_path
                            break
                
                if not image_path.exists():
                    logger.warning(f"⚠️ Image not found for annotation: {image_name}")
                    continue
                
                # Convert to Grounding DINO format
                grounding_data = self.create_grounding_dino_format(
                    str(image_path), 
                    coco_data['annotations']
                )
                
                all_data.append(grounding_data)
                
                # Update statistics
                self.stats['total_annotations'] += len(coco_data['annotations'])
                for ann in coco_data['annotations']:
                    category = ann['category']
                    self.stats['class_distribution'][category] = \
                        self.stats['class_distribution'].get(category, 0) + 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {ann_file}: {e}")
                continue
        
        self.stats['total_images'] = len(all_data)
        logger.info(f"✅ Processed {len(all_data)} images with annotations")
        
        return all_data
    
    def prepare_dataset(self):
        """Main dataset preparation function"""
        logger.info("🚀 Starting dataset preparation...")
        
        # Process annotations
        all_data = self.process_annotations()
        
        if not all_data:
            logger.error("❌ No valid data found. Check your annotations and images.")
            return False
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(all_data)
        
        # Save splits
        self.save_split_data(train_data, 'train')
        self.save_split_data(val_data, 'val')
        self.save_split_data(test_data, 'test')
        
        # Save statistics
        self.save_statistics()
        
        logger.info("✅ Dataset preparation complete!")
        return True
    
    def save_statistics(self):
        """Save dataset statistics"""
        stats_file = self.output_dir / "dataset_stats.json"
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Print summary
        print("\n📊 DATASET STATISTICS")
        print("=" * 40)
        print(f"Total Images: {self.stats['total_images']}")
        print(f"Total Annotations: {self.stats['total_annotations']}")
        print(f"Avg Annotations per Image: {self.stats['total_annotations']/self.stats['total_images']:.2f}")
        
        print("\n📋 Class Distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            print(f"   {class_name}: {count}")
        
        print("\n📂 Split Distribution:")
        for split, count in self.stats['split_distribution'].items():
            print(f"   {split}: {count}")
        
        logger.info(f"📄 Statistics saved to: {stats_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for Grounding DINO training')
    parser.add_argument('--raw-data', type=str, default='training/data/raw_images',
                       help='Directory containing raw images and annotations')
    parser.add_argument('--output', type=str, default='training/data/processed',
                       help='Output directory for processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logger.error("❌ Train/val/test ratios must sum to 1.0")
        return
    
    # Create preparator and run
    preparator = DatasetPreparator(args.raw_data, args.output)
    success = preparator.prepare_dataset()
    
    if success:
        print(f"\n✅ Dataset ready for training! Check {args.output}")
    else:
        print("\n❌ Dataset preparation failed")

if __name__ == "__main__":
    main()
