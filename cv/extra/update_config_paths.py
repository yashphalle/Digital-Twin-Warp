#!/usr/bin/env python3
"""
Update Config Paths Script
Automatically updates all import statements and file paths after moving configs to configs/ folder
"""

import os
import re
from pathlib import Path

def update_imports_in_file(filepath, updates):
    """Update import statements in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply each update
        for old_pattern, new_replacement in updates:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_replacement, content)
                changes_made.append(f"  ✓ {old_pattern} → {new_replacement}")
        
        # Write back if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {filepath}")
            for change in changes_made:
                print(change)
            return True
        else:
            print(f"⚪ No changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        return False

def main():
    """Main function to update all config paths"""
    print("🔧 UPDATING CONFIG PATHS FOR configs/ FOLDER")
    print("=" * 60)
    
    # Define the updates needed
    import_updates = [
        # Config imports
        (r'from config import', 'from configs.config import'),
        (r'import config', 'import configs.config as config'),
        
        # Warehouse config imports  
        (r'from warehouse_config import', 'from configs.warehouse_config import'),
        (r'import warehouse_config', 'import configs.warehouse_config as warehouse_config'),
    ]
    
    file_path_updates = [
        # Calibration file paths
        (r'"warehouse_calibration_camera_(\d+)\.json"', r'"configs/warehouse_calibration_camera_\1.json"'),
        (r"'warehouse_calibration_camera_(\d+)\.json'", r"'configs/warehouse_calibration_camera_\1.json'"),
        
        # General calibration paths
        (r'"warehouse_calibration\.json"', '"configs/warehouse_calibration.json"'),
        (r"'warehouse_calibration\.json'", "'configs/warehouse_calibration.json'"),
        
        # Warehouse configs directory
        (r'"warehouse_configs"', '"configs/warehouse_configs"'),
        (r"'warehouse_configs'", "'configs/warehouse_configs'"),
    ]
    
    # Files to update in cv/ directory
    cv_files_to_update = [
        'multi_camera_tracking_system.py',
        'detector_tracker.py', 
        'multi_camera_grid_display.py',
        'rtsp_camera_manager.py',
        'database_handler.py',
        'simplified_lorex_pipeline.py'
    ]
    
    # Files to update in backend/ directory  
    backend_files_to_update = [
        '../backend/live_server.py'
    ]
    
    updated_files = 0
    
    print("📝 Updating CV files...")
    print("-" * 30)
    
    # Update CV files
    for filename in cv_files_to_update:
        filepath = f"cv/{filename}"
        if os.path.exists(filepath):
            # Apply import updates
            if update_imports_in_file(filepath, import_updates):
                updated_files += 1
            # Apply file path updates  
            if update_imports_in_file(filepath, file_path_updates):
                updated_files += 1
        else:
            print(f"⚠️ File not found: {filepath}")
    
    print(f"\n📝 Updating backend files...")
    print("-" * 30)
    
    # Update backend files
    for filepath in backend_files_to_update:
        if os.path.exists(filepath):
            # Only apply file path updates for backend
            if update_imports_in_file(filepath, file_path_updates):
                updated_files += 1
        else:
            print(f"⚠️ File not found: {filepath}")
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"Files processed: {len(cv_files_to_update) + len(backend_files_to_update)}")
    print(f"Files updated: {updated_files}")
    
    print(f"\n🚀 NEXT STEPS")
    print("=" * 30)
    print("1. Create configs directory:")
    print("   mkdir cv/configs")
    print()
    print("2. Move config files:")
    print("   mv cv/config.py cv/configs/")
    print("   mv cv/warehouse_config.py cv/configs/")
    print("   mv cv/warehouse_calibration_camera_*.json cv/configs/")
    print("   mv cv/warehouse_configs cv/configs/")
    print()
    print("3. Test the system:")
    print("   python cv/multi_camera_tracking_system.py")

if __name__ == "__main__":
    main()
