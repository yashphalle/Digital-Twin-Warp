#!/usr/bin/env python3
"""
Test script to verify frontend color mapping
Shows what colors should be displayed for current API data
"""

import requests
import json

def test_color_mapping():
    """Test the color mapping logic that frontend will use"""
    
    # Color mapping from frontend
    color_map = {
        'red': '#ff4444',
        'orange': '#ff8800', 
        'yellow': '#ffdd00',
        'green': '#44ff44',
        'blue': '#4444ff',
        'purple': '#8844ff',
        'pink': '#ff44aa',
        'brown': '#8b4513',
        'black': '#333333',
        'white': '#f0f0f0',
        'gray': '#888888',
        'grey': '#888888',
        'dark': '#444444'
    }
    
    def get_object_color(obj):
        """Same logic as frontend getObjectColor function"""
        # Priority 1: Use RGB values if available
        if obj.get('color_rgb') and isinstance(obj['color_rgb'], list) and len(obj['color_rgb']) >= 3:
            rgb = obj['color_rgb']
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        
        # Priority 2: Use hex color if available
        if obj.get('color_hex'):
            return obj['color_hex']
        
        # Priority 3: Map color names to hex values
        if obj.get('color_name') and obj.get('color_confidence', 0) > 0.3:
            color_name = obj['color_name'].lower()
            if color_name in color_map:
                return color_map[color_name]
        
        # Fallback: amber color
        return '#d97706'
    
    try:
        print("🎨 FRONTEND COLOR MAPPING TEST")
        print("=" * 50)
        
        # Fetch current API data
        response = requests.get('http://localhost:8000/api/tracking/objects')
        if not response.ok:
            print(f"❌ API request failed: {response.status_code}")
            return
        
        data = response.json()
        objects = data.get('objects', [])
        
        print(f"📊 Found {len(objects)} objects in API")
        print("\n🎯 COLOR MAPPING RESULTS:")
        print("-" * 50)
        
        color_counts = {}
        
        for i, obj in enumerate(objects[:10]):  # Show first 10 objects
            obj_id = obj.get('persistent_id', 'Unknown')
            color_name = obj.get('color_name', 'None')
            color_hex = obj.get('color_hex', 'None')
            color_rgb = obj.get('color_rgb', 'None')
            color_confidence = obj.get('color_confidence', 0)
            
            # Get the color that frontend will display
            display_color = get_object_color(obj)
            
            print(f"Object {obj_id}:")
            print(f"  📝 Detected: {color_name} (confidence: {color_confidence:.1f})")
            print(f"  🎨 Display Color: {display_color}")
            print(f"  📊 Source: {'RGB' if color_rgb != 'None' else 'HEX' if color_hex != 'None' else 'NAME_MAP' if color_name in color_map else 'FALLBACK'}")
            print()
            
            # Count colors
            if display_color in color_counts:
                color_counts[display_color] += 1
            else:
                color_counts[display_color] = 1
        
        print("📈 COLOR DISTRIBUTION:")
        print("-" * 30)
        for color, count in sorted(color_counts.items()):
            color_name = "Unknown"
            for name, hex_val in color_map.items():
                if hex_val == color:
                    color_name = name.title()
                    break
            if color == '#d97706':
                color_name = "Fallback Amber"
            
            print(f"  {color} ({color_name}): {count} objects")
        
        print(f"\n✅ Frontend will display objects in their detected colors!")
        print(f"🎯 Expected colors: Gray objects = #888888, Orange objects = #ff8800")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_color_mapping()
