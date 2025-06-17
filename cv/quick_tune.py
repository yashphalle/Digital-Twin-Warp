"""
Quick Tuning Script for Common Issues
Addresses intermittent detection and moving object tracking
"""

from config import Config
from config_manager import ConfigManager

def show_current_detection_settings():
    """Show current detection-related settings"""
    print("🎯 CURRENT DETECTION SETTINGS")
    print("=" * 40)
    print(f"CONFIDENCE_THRESHOLD: {Config.CONFIDENCE_THRESHOLD}")
    print(f"BOX_THRESHOLD: {Config.BOX_THRESHOLD}")
    print(f"TEXT_THRESHOLD: {Config.TEXT_THRESHOLD}")
    print(f"MIN_BOX_AREA: {Config.MIN_BOX_AREA}")
    print()

def show_current_tracking_settings():
    """Show current tracking-related settings"""
    print("🏃 CURRENT TRACKING SETTINGS")
    print("=" * 40)
    print(f"MIN_MATCH_COUNT: {Config.MIN_MATCH_COUNT}")
    print(f"GOOD_MATCH_RATIO: {Config.GOOD_MATCH_RATIO}")
    print(f"MATCH_SCORE_THRESHOLD: {Config.MATCH_SCORE_THRESHOLD}")
    print(f"MAX_DISAPPEARED_FRAMES: {Config.MAX_DISAPPEARED_FRAMES}")
    print(f"SIFT_N_FEATURES: {Config.SIFT_N_FEATURES}")
    print()

def fix_intermittent_detection():
    """Fix intermittent detection issues"""
    print("🔧 FIXING INTERMITTENT DETECTION")
    print("=" * 40)
    print("Lowering detection thresholds for more consistent detection...")
    
    # Lower detection thresholds
    Config.CONFIDENCE_THRESHOLD = 0.15
    Config.BOX_THRESHOLD = 0.15
    Config.TEXT_THRESHOLD = 0.15
    
    print(f"✅ CONFIDENCE_THRESHOLD: 0.20 → {Config.CONFIDENCE_THRESHOLD}")
    print(f"✅ BOX_THRESHOLD: 0.20 → {Config.BOX_THRESHOLD}")
    print(f"✅ TEXT_THRESHOLD: 0.20 → {Config.TEXT_THRESHOLD}")
    print()
    print("💡 This should detect objects more consistently across frames")
    print("⚠️ May increase false positives - monitor and adjust if needed")
    print()

def fix_moving_object_tracking():
    """Fix moving object tracking issues"""
    print("🏃 FIXING MOVING OBJECT TRACKING")
    print("=" * 40)
    print("Adjusting SIFT parameters for better moving object tracking...")
    
    # Make tracking more lenient for moving objects
    old_min_match = Config.MIN_MATCH_COUNT
    old_ratio = Config.GOOD_MATCH_RATIO
    old_threshold = Config.MATCH_SCORE_THRESHOLD
    old_features = Config.SIFT_N_FEATURES
    
    Config.MIN_MATCH_COUNT = 6
    Config.GOOD_MATCH_RATIO = 0.7
    Config.MATCH_SCORE_THRESHOLD = 0.05
    Config.SIFT_N_FEATURES = 600
    
    print(f"✅ MIN_MATCH_COUNT: {old_min_match} → {Config.MIN_MATCH_COUNT}")
    print(f"✅ GOOD_MATCH_RATIO: {old_ratio} → {Config.GOOD_MATCH_RATIO}")
    print(f"✅ MATCH_SCORE_THRESHOLD: {old_threshold} → {Config.MATCH_SCORE_THRESHOLD}")
    print(f"✅ SIFT_N_FEATURES: {old_features} → {Config.SIFT_N_FEATURES}")
    print()
    print("💡 This should maintain object IDs better when boxes move")
    print("⚠️ May accept some weaker matches - monitor match scores")
    print()

def apply_moving_objects_preset():
    """Apply the moving objects preset"""
    print("🎯 APPLYING MOVING OBJECTS PRESET")
    print("=" * 40)
    
    manager = ConfigManager()
    success = manager.apply_preset("moving_objects")
    
    if success:
        print("✅ Moving objects preset applied successfully!")
        print()
        print("📊 New settings optimized for:")
        print("  • Lower detection threshold (more consistent detection)")
        print("  • Lenient SIFT matching (better for moving objects)")
        print("  • More features (better tracking quality)")
        print()
    else:
        print("❌ Failed to apply preset")

def show_display_info():
    """Show what will be displayed"""
    print("📺 DISPLAY INFORMATION")
    print("=" * 40)
    print("With current settings, you'll see:")
    print()
    print("Object Label Format:")
    if Config.SHOW_OBJECT_IDS:
        label_parts = ["ID:5"]
        if Config.SHOW_MATCH_SCORES:
            label_parts.append("S:0.85")
        if Config.SHOW_DETECTION_CONFIDENCE:
            label_parts.append("C:0.23")
        print(f"  {' '.join(label_parts)} Duration:23s")
    
    print()
    print("Label Meaning:")
    print("  • ID:5 = Object identifier")
    print("  • S:0.85 = SIFT match score (tracking confidence)")
    print("  • C:0.23 = Detection confidence (Grounding DINO)")
    print("  • Duration:23s = How long object has been tracked")
    print()
    print("🎯 TUNING TIPS:")
    print("  • Low C values → Lower CONFIDENCE_THRESHOLD")
    print("  • Objects getting new IDs → Lower MIN_MATCH_COUNT")
    print("  • Low S values → Increase SIFT_N_FEATURES")
    print()

def interactive_quick_tune():
    """Interactive quick tuning"""
    print("🛠️ QUICK TUNING FOR YOUR ISSUES")
    print("=" * 50)
    
    while True:
        print("\nSelect issue to fix:")
        print("1. Intermittent detection (objects disappear/reappear)")
        print("2. Moving objects get new IDs")
        print("3. Apply moving objects preset (fixes both)")
        print("4. Show current settings")
        print("5. Show display information")
        print("6. Reset to defaults")
        print("7. Exit")
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                fix_intermittent_detection()
            elif choice == '2':
                fix_moving_object_tracking()
            elif choice == '3':
                apply_moving_objects_preset()
            elif choice == '4':
                show_current_detection_settings()
                show_current_tracking_settings()
            elif choice == '5':
                show_display_info()
            elif choice == '6':
                print("⚠️ Restart the application to reset to defaults")
            elif choice == '7':
                break
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Quick tuning complete!")

def main():
    """Main function"""
    print("🎯 WAREHOUSE TRACKING QUICK TUNE")
    print("=" * 50)
    print("Addresses your specific issues:")
    print("1. Objects detected intermittently")
    print("2. Moving boxes get new IDs")
    print("=" * 50)
    
    # Show current state
    show_current_detection_settings()
    show_current_tracking_settings()
    
    # Ask what to fix
    print("What would you like to fix?")
    print("a) Intermittent detection only")
    print("b) Moving object tracking only") 
    print("c) Both issues (recommended)")
    print("d) Interactive tuning")
    print("e) Just show display info")
    
    choice = input("\nEnter choice (a-e): ").strip().lower()
    
    if choice == 'a':
        fix_intermittent_detection()
    elif choice == 'b':
        fix_moving_object_tracking()
    elif choice == 'c':
        apply_moving_objects_preset()
    elif choice == 'd':
        interactive_quick_tune()
    elif choice == 'e':
        show_display_info()
    else:
        print("❌ Invalid choice")
        return
    
    print("🚀 Changes applied! Restart the tracking system to see effects.")
    print()
    print("📊 Monitor these values in the display:")
    print("  • C: values (detection confidence)")
    print("  • S: values (match scores)")
    print("  • Object ID stability when moving boxes")

if __name__ == "__main__":
    main()
