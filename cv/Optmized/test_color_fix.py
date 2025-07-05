#!/usr/bin/env python3
"""
Test color extraction field names fix
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))

def test_color_field_names():
    """Test that color extraction returns correct field names"""
    print("🎨 TESTING COLOR FIELD NAMES FIX")
    print("=" * 50)
    
    try:
        from cpu_11camera_configurable import ObjectColorExtractor
        
        color_extractor = ObjectColorExtractor()
        
        # Create a simple test image (gray)
        test_image = np.full((50, 50, 3), [100, 100, 100], dtype=np.uint8)
        
        # Extract color
        result = color_extractor.extract_dominant_color(test_image)
        
        print("🔍 Extracted color data:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Check required fields
        required_fields = ['color_rgb', 'color_hsv', 'color_hex', 'color_name', 'color_confidence']
        
        print(f"\n✅ FIELD VERIFICATION:")
        all_good = True
        for field in required_fields:
            if field in result and result[field] is not None:
                print(f"  ✅ {field}: PRESENT")
            else:
                print(f"  ❌ {field}: MISSING or NULL")
                all_good = False
        
        if all_good:
            print(f"\n🎉 SUCCESS! All color fields are present with correct names.")
            print(f"🔄 The CV system should now provide real RGB/HSV/hex values to the frontend.")
        else:
            print(f"\n❌ ISSUES FOUND! Some fields are missing.")
            
        return all_good
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_color_field_names()
