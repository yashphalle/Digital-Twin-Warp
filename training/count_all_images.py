#!/usr/bin/env python3
"""
Count All Images - Verify corrected vs raw image counts
Should show roughly 50/50 split between raw and corrected images
"""

import os
import sys
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv.configs.config import Config

def count_all_images():
    """Count all images by type and camera"""
    
    collection_dir = Path(Config.RAW_COLLECTION_DIR)
    print(f"🔍 Counting images in: {collection_dir}")
    
    if not collection_dir.exists():
        print(f"❌ Collection directory not found: {collection_dir}")
        return
    
    # Initialize counters
    total_raw = 0
    total_corrected = 0
    camera_breakdown = {}
    
    # Count by camera
    for camera_dir in sorted(collection_dir.glob("camera_*")):
        camera_id = camera_dir.name.replace("camera_", "")
        
        # Count raw images
        raw_dir = camera_dir / "raw_4k"
        raw_count = 0
        if raw_dir.exists():
            raw_count = len(list(raw_dir.glob("*.jpg"))) + len(list(raw_dir.glob("*.jpeg")))
        
        # Count corrected images  
        corrected_dir = camera_dir / "corrected_2k"
        corrected_count = 0
        if corrected_dir.exists():
            corrected_count = len(list(corrected_dir.glob("*.jpg"))) + len(list(corrected_dir.glob("*.jpeg")))
        
        # Store breakdown
        camera_breakdown[camera_id] = {
            'raw': raw_count,
            'corrected': corrected_count,
            'total': raw_count + corrected_count
        }
        
        total_raw += raw_count
        total_corrected += corrected_count
    
    # Print results
    print(f"\n📊 IMAGE COUNT BREAKDOWN:")
    print(f"{'Camera':<8} {'Raw 4K':<8} {'Corrected 2K':<12} {'Total':<8} {'Match?':<8}")
    print("-" * 50)
    
    for camera_id in sorted(camera_breakdown.keys(), key=int):
        data = camera_breakdown[camera_id]
        match_status = "✅" if data['raw'] == data['corrected'] else "❌"
        print(f"Camera {camera_id:<2} {data['raw']:<8} {data['corrected']:<12} {data['total']:<8} {match_status}")
    
    print("-" * 50)
    print(f"{'TOTAL':<8} {total_raw:<8} {total_corrected:<12} {total_raw + total_corrected:<8}")
    
    # Analysis
    print(f"\n🎯 ANALYSIS:")
    print(f"   Total Raw Images: {total_raw}")
    print(f"   Total Corrected Images: {total_corrected}")
    print(f"   Grand Total: {total_raw + total_corrected}")
    
    if total_raw == total_corrected:
        print(f"   ✅ PERFECT: Equal raw and corrected images (50/50 split)")
    elif abs(total_raw - total_corrected) <= 5:
        print(f"   ✅ GOOD: Nearly equal raw and corrected images")
    else:
        print(f"   ⚠️ WARNING: Unequal raw and corrected images")
        print(f"   Difference: {abs(total_raw - total_corrected)} images")
    
    # Upload verification
    print(f"\n📤 UPLOAD VERIFICATION:")
    print(f"   Script found: 1227 images")
    print(f"   Corrected count: {total_corrected}")
    
    if total_corrected == 1227:
        print(f"   ✅ CONFIRMED: Upload script is using corrected images")
    elif total_raw + total_corrected == 1227:
        print(f"   ⚠️ WARNING: Upload script might be using mixed images")
    else:
        print(f"   ❓ UNCLEAR: Upload count doesn't match any category")
    
    return total_raw, total_corrected, camera_breakdown

def main():
    """Main counting function"""
    print("📊 IMAGE COUNT VERIFICATION")
    print("=" * 60)
    
    total_raw, total_corrected, breakdown = count_all_images()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 CONCLUSION:")
    
    if total_corrected == 1227:
        print(f"✅ VERIFIED: Upload script is correctly using fisheye-corrected images")
        print(f"   - Found {total_corrected} corrected images")
        print(f"   - Upload script found 1227 images")
        print(f"   - Perfect match!")
    else:
        print(f"⚠️ NEEDS INVESTIGATION:")
        print(f"   - Corrected images: {total_corrected}")
        print(f"   - Upload script found: 1227")
        print(f"   - Difference: {abs(total_corrected - 1227)}")

if __name__ == "__main__":
    main()
