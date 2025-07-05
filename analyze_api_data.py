import requests
import json

def analyze_api_data():
    try:
        r = requests.get('http://localhost:8000/api/tracking/objects')
        data = r.json()
        
        print('=== API DATA ANALYSIS ===')
        print(f'Total objects: {len(data["objects"])}')
        print()
        
        for i, obj in enumerate(data['objects'][:3]):
            print(f'Object {i+1}:')
            print(f'  ID: {obj.get("persistent_id")}')
            print(f'  Shape type: {obj.get("shape_type")}')
            print(f'  Physical corners: {obj.get("physical_corners")}')
            print(f'  Real center: {obj.get("real_center")}')
            print(f'  Color name: {obj.get("color_name")}')
            print(f'  Color hex: {obj.get("color_hex")}')
            print(f'  Color confidence: {obj.get("color_confidence")}')
            print()
            
        # Test coordinate conversion
        print('=== COORDINATE CONVERSION TEST ===')
        if data['objects']:
            obj = data['objects'][0]
            if obj.get('physical_corners'):
                warehouseWidthFt = 180
                warehouseLengthFt = 100
                
                print(f'Object {obj["persistent_id"]} coordinate conversion:')
                for i, corner in enumerate(obj['physical_corners']):
                    physX, physY = corner
                    x = ((warehouseWidthFt - physX) / warehouseWidthFt) * 100
                    y = (physY / warehouseLengthFt) * 100
                    print(f'  Corner {i+1}: ({physX}, {physY}) -> ({x:.1f}%, {y:.1f}%)')
                    
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    analyze_api_data()
