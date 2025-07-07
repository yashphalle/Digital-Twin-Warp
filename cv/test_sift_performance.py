#!/usr/bin/env python3
"""
SIFT Performance Test Script
Test how long SIFT processing takes with existing .pkl database
"""

import os
import sys
import time
import pickle
import cv2
import numpy as np
from datetime import datetime
import psutil
import threading

# Add the final modules to path
sys.path.append('final')
sys.path.append('final/modules')

def load_existing_sift_database():
    """Load the existing SIFT database"""
    pkl_file = "cpu_camera_4_global_features.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"❌ SIFT database not found: {pkl_file}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded SIFT database: {pkl_file}")
        print(f"📊 Database structure: {type(data)}")
        
        if isinstance(data, dict) and 'features' in data:
            features = data['features']
            print(f"🎯 Objects in database: {len(features)}")
            
            # Analyze feature counts
            feature_counts = []
            for obj_id, obj_data in features.items():
                if 'features' in obj_data and obj_data['features'] is not None:
                    feature_counts.append(len(obj_data['features']))
            
            if feature_counts:
                print(f"🔢 Features per object: {np.mean(feature_counts):.1f} avg, {sum(feature_counts)} total")
            
            return data
        else:
            print(f"⚠️ Unexpected database structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return data
            
    except Exception as e:
        print(f"❌ Error loading SIFT database: {e}")
        return None

def create_mock_detection():
    """Create a mock detection for testing"""
    return {
        'bbox': [100, 100, 200, 200],  # x1, y1, x2, y2
        'confidence': 0.85,
        'area': 10000,
        'center': [150, 150],
        'physical_x_ft': 10.5,
        'physical_y_ft': 15.2,
        'coordinate_status': 'valid'
    }

def create_test_frame():
    """Create a test frame similar to warehouse footage"""
    # Create a 1600x900 frame (resized from 4K)
    frame = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
    
    # Add some structured patterns (like warehouse shelving)
    for i in range(0, 1600, 100):
        cv2.line(frame, (i, 0), (i, 900), (128, 128, 128), 2)
    for i in range(0, 900, 100):
        cv2.line(frame, (0, i), (1600, i), (128, 128, 128), 2)
    
    # Add some rectangular regions (like pallets)
    for i in range(5):
        x = np.random.randint(50, 1400)
        y = np.random.randint(50, 700)
        w = np.random.randint(80, 200)
        h = np.random.randint(80, 150)
        color = tuple(np.random.randint(50, 200, 3).tolist())
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    return frame

def test_sift_feature_extraction(frame, detection, sift_detector):
    """Test SIFT feature extraction timing"""
    print(f"\n🔍 Testing SIFT Feature Extraction")
    print("=" * 50)
    
    # Extract region from detection
    x1, y1, x2, y2 = detection['bbox']
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        print("❌ Invalid ROI")
        return None, 0
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Time SIFT feature extraction
    start_time = time.time()
    
    try:
        keypoints, descriptors = sift_detector.detectAndCompute(gray_roi, None)
        extraction_time = time.time() - start_time
        
        feature_count = len(keypoints) if keypoints else 0
        descriptor_size = descriptors.shape if descriptors is not None else (0, 0)
        
        print(f"✅ Feature extraction completed")
        print(f"⏱️  Time: {extraction_time:.4f} seconds")
        print(f"🔢 Keypoints: {feature_count}")
        print(f"📊 Descriptor shape: {descriptor_size}")
        print(f"💾 Memory usage: {descriptor_size[0] * descriptor_size[1] * 4 / 1024:.1f} KB" if descriptors is not None else "💾 Memory usage: 0 KB")
        
        return descriptors, extraction_time
        
    except Exception as e:
        extraction_time = time.time() - start_time
        print(f"❌ Feature extraction failed after {extraction_time:.4f}s: {e}")
        return None, extraction_time

def test_sift_feature_matching(new_features, database_features, matcher):
    """Test SIFT feature matching timing"""
    print(f"\n🔗 Testing SIFT Feature Matching")
    print("=" * 50)
    
    if new_features is None or len(database_features) == 0:
        print("❌ No features to match")
        return 0
    
    total_matches = 0
    total_time = 0
    match_details = []
    
    for obj_id, obj_data in database_features.items():
        if 'features' not in obj_data or obj_data['features'] is None:
            continue
            
        existing_features = obj_data['features']
        
        start_time = time.time()
        try:
            # Perform feature matching
            matches = matcher.match(new_features, existing_features)
            match_time = time.time() - start_time
            
            # Filter good matches
            good_matches = [m for m in matches if m.distance < 0.7 * max([m2.distance for m2 in matches])] if matches else []
            
            match_details.append({
                'obj_id': obj_id,
                'total_matches': len(matches) if matches else 0,
                'good_matches': len(good_matches),
                'match_time': match_time,
                'existing_features': len(existing_features)
            })
            
            total_matches += len(good_matches)
            total_time += match_time
            
        except Exception as e:
            match_time = time.time() - start_time
            print(f"⚠️ Matching failed for object {obj_id}: {e}")
            total_time += match_time
    
    # Print results
    print(f"📊 Matching Results:")
    print(f"⏱️  Total matching time: {total_time:.4f} seconds")
    print(f"🎯 Objects tested: {len(match_details)}")
    print(f"🔗 Total good matches: {total_matches}")
    
    if match_details:
        avg_time = total_time / len(match_details)
        print(f"⏱️  Average time per object: {avg_time:.4f} seconds")
        
        # Show slowest matches
        sorted_matches = sorted(match_details, key=lambda x: x['match_time'], reverse=True)
        print(f"\n🐌 Slowest matches:")
        for i, match in enumerate(sorted_matches[:3]):
            print(f"   {i+1}. Object {match['obj_id']}: {match['match_time']:.4f}s ({match['existing_features']} features)")
    
    return total_time

def monitor_memory_usage():
    """Monitor memory usage during testing"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def memory_monitor():
        max_memory = initial_memory
        while True:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)
            except:
                break
        return max_memory
    
    return initial_memory, memory_monitor

def main():
    """Main performance test"""
    print("🚀 SIFT Performance Test")
    print("=" * 60)
    print(f"📅 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Monitor memory
    initial_memory, _ = monitor_memory_usage()
    print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    # Load existing SIFT database
    print(f"\n📂 Loading SIFT Database")
    print("=" * 30)
    sift_data = load_existing_sift_database()
    
    if sift_data is None:
        print("❌ Cannot proceed without SIFT database")
        return
    
    database_features = sift_data.get('features', {}) if isinstance(sift_data, dict) else {}
    
    # Initialize SIFT detector and matcher
    print(f"\n🔧 Initializing SIFT Components")
    print("=" * 30)
    
    try:
        sift_detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
        print("✅ SIFT detector and matcher initialized")
    except Exception as e:
        print(f"❌ Failed to initialize SIFT: {e}")
        return
    
    # Create test data
    print(f"\n🎬 Creating Test Data")
    print("=" * 30)
    
    test_frame = create_test_frame()
    test_detection = create_mock_detection()
    
    print(f"✅ Test frame created: {test_frame.shape}")
    print(f"✅ Test detection created: {test_detection['bbox']}")
    
    # Run performance tests
    print(f"\n🏃 Running Performance Tests")
    print("=" * 60)
    
    # Test 1: Feature extraction
    new_features, extraction_time = test_sift_feature_extraction(test_frame, test_detection, sift_detector)
    
    # Test 2: Feature matching
    if new_features is not None:
        matching_time = test_sift_feature_matching(new_features, database_features, matcher)
    else:
        matching_time = 0
        print("⚠️ Skipping feature matching due to extraction failure")
    
    # Calculate total time
    total_sift_time = extraction_time + matching_time
    
    # Final results
    print(f"\n📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"⏱️  Feature extraction: {extraction_time:.4f} seconds")
    print(f"⏱️  Feature matching: {matching_time:.4f} seconds")
    print(f"⏱️  Total SIFT time: {total_sift_time:.4f} seconds")
    print(f"🎯 Database objects: {len(database_features)}")
    
    # Memory usage
    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_increase = current_memory - initial_memory
    print(f"💾 Memory increase: {memory_increase:.1f} MB")
    
    # Performance assessment
    print(f"\n💡 PERFORMANCE ASSESSMENT")
    print("=" * 30)
    
    if total_sift_time < 1.0:
        print("✅ EXCELLENT: SIFT processing under 1 second")
    elif total_sift_time < 3.0:
        print("⚠️ ACCEPTABLE: SIFT processing under 3 seconds")
    elif total_sift_time < 5.0:
        print("🐌 SLOW: SIFT processing 3-5 seconds")
    else:
        print("🚨 CRITICAL: SIFT processing over 5 seconds")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS")
    print("=" * 30)
    
    if extraction_time > 2.0:
        print("• Feature extraction is slow - consider reducing ROI size")
    if matching_time > 3.0:
        print("• Feature matching is slow - consider database size limits")
    if len(database_features) > 20:
        print("• Database is large - consider periodic cleanup")
    if memory_increase > 100:
        print("• High memory usage - check for memory leaks")

if __name__ == "__main__":
    main()
