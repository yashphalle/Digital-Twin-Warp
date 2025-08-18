#!/usr/bin/env python3
"""
Simple BoT-SORT Test - Test if BoT-SORT tracking is working
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_botsort_tracking():
    """Test BoT-SORT tracking with a simple video"""
    print("🧪 Testing BoT-SORT tracking...")
    
    try:
        # Load YOLO model
        model_path = 'custom_yolo.pt'
        model = YOLO(model_path)
        print(f"✅ Loaded model: {model_path}")
        
        # Load BoT-SORT config
        botsort_config = '../configs/warehouse_botsort.yaml'
        print(f"✅ Using BoT-SORT config: {botsort_config}")
        
        # Try to use cam7.mp4 if available, otherwise create test frame
        try:
            cap = cv2.VideoCapture('cam7.mp4')
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Using frame from cam7.mp4")
            else:
                raise Exception("Could not read cam7.mp4")
        except:
            print("⚠️ cam7.mp4 not available, using test frame")
            # Create a simple test frame (black image with white rectangle)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        print("🎯 Running BoT-SORT tracking on test frame...")
        
        # Run tracking
        results = model.track(
            frame,
            conf=0.1,  # Low confidence for testing
            tracker=botsort_config,
            persist=True,
            verbose=True
        )
        
        print(f"📊 Results: {len(results)} result(s)")
        
        if results and len(results) > 0:
            result = results[0]
            print(f"📦 Result type: {type(result)}")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"📦 Boxes: {len(boxes)} detection(s)")
                
                # Check for track IDs
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                    print(f"🎯 Track IDs found: {track_ids}")
                    print("✅ BoT-SORT tracking is working!")
                    return True
                else:
                    print("❌ No track IDs found - BoT-SORT not working")
                    return False
            else:
                print("❌ No boxes found in results")
                return False
        else:
            print("❌ No results from model")
            return False
            
    except Exception as e:
        print(f"❌ BoT-SORT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_botsort_tracking()
    if success:
        print("🎉 BoT-SORT test PASSED!")
    else:
        print("💥 BoT-SORT test FAILED!")
