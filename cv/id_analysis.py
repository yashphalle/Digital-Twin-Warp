"""
ID Assignment Analysis Tool
Diagnoses why the system is creating too many object IDs
"""

from config import Config
import time

def analyze_id_assignment_issues():
    """Analyze common ID assignment issues"""
    print("🔍 ID ASSIGNMENT ANALYSIS")
    print("=" * 50)
    
    print("📊 CURRENT SETTINGS ANALYSIS:")
    print(f"CONFIDENCE_THRESHOLD: {Config.CONFIDENCE_THRESHOLD}")
    print(f"MIN_MATCH_COUNT: {Config.MIN_MATCH_COUNT}")
    print(f"GOOD_MATCH_RATIO: {Config.GOOD_MATCH_RATIO}")
    print(f"MATCH_SCORE_THRESHOLD: {Config.MATCH_SCORE_THRESHOLD}")
    print(f"MAX_DISAPPEARED_FRAMES: {Config.MAX_DISAPPEARED_FRAMES}")
    print()
    
    # Analyze potential issues
    issues = []
    recommendations = []
    
    # Check detection threshold
    if Config.CONFIDENCE_THRESHOLD > 0.18:
        issues.append("🚨 Detection threshold too high")
        recommendations.append("Lower CONFIDENCE_THRESHOLD to 0.15 or 0.12")
    
    # Check SIFT matching
    if Config.MIN_MATCH_COUNT > 8:
        issues.append("🚨 SIFT matching too strict")
        recommendations.append("Lower MIN_MATCH_COUNT to 6-8")
    
    if Config.GOOD_MATCH_RATIO < 0.6:
        issues.append("🚨 Match ratio too strict")
        recommendations.append("Increase GOOD_MATCH_RATIO to 0.7")
    
    if Config.MATCH_SCORE_THRESHOLD > 0.08:
        issues.append("🚨 Match score threshold too high")
        recommendations.append("Lower MATCH_SCORE_THRESHOLD to 0.05")
    
    # Check disappearance timeout
    if Config.MAX_DISAPPEARED_FRAMES < 20:
        issues.append("🚨 Objects removed too quickly")
        recommendations.append("Increase MAX_DISAPPEARED_FRAMES to 30-45")
    
    print("🚨 POTENTIAL ISSUES:")
    if issues:
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("  ✅ No obvious configuration issues detected")
    
    print()
    print("💡 RECOMMENDATIONS:")
    if recommendations:
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("  ✅ Current settings look reasonable")
    
    print()

def explain_id_assignment_process():
    """Explain how ID assignment works"""
    print("🎯 HOW ID ASSIGNMENT WORKS")
    print("=" * 50)
    
    print("""
📋 STEP-BY-STEP PROCESS:

1. 🔍 DETECTION PHASE:
   • Grounding DINO detects objects with confidence scores
   • Only objects above CONFIDENCE_THRESHOLD are considered
   • Each detection gets pixel coordinates and bounding box

2. 🎯 MATCHING PHASE:
   • Extract SIFT features from detected object region
   • Compare with SIFT features of all existing tracked objects
   • Calculate match scores using feature similarity

3. ✅ ASSIGNMENT DECISION:
   • If match score > MATCH_SCORE_THRESHOLD: Update existing object
   • If no good match found: Create NEW ID
   • If object disappears > MAX_DISAPPEARED_FRAMES: Remove from tracking

4. 🆕 NEW ID TRIGGERS:
   • No existing object has match score above threshold
   • Object reappears after being removed (timeout)
   • Visual features changed too much (movement, rotation, lighting)
   • Detection confidence fluctuated causing missed frames

🚨 WHY YOU GET HIGH ID NUMBERS (60-70 for 2-3 objects):

• INTERMITTENT DETECTION: Object detected → missed → detected → NEW ID
• POOR SIFT MATCHING: Movement changes features → no match → NEW ID  
• TIMEOUT REMOVAL: Object disappears briefly → removed → reappears → NEW ID
• LIGHTING CHANGES: Different appearance → no match → NEW ID
""")

def show_optimal_settings():
    """Show optimal settings for different scenarios"""
    print("⚙️ OPTIMAL SETTINGS FOR YOUR SCENARIO")
    print("=" * 50)
    
    print("🎯 FOR 2-3 STATIONARY OBJECTS:")
    print("CONFIDENCE_THRESHOLD = 0.18")
    print("MIN_MATCH_COUNT = 8")
    print("GOOD_MATCH_RATIO = 0.6")
    print("MATCH_SCORE_THRESHOLD = 0.08")
    print("MAX_DISAPPEARED_FRAMES = 45")
    print()
    
    print("🏃 FOR 2-3 MOVING OBJECTS:")
    print("CONFIDENCE_THRESHOLD = 0.15")
    print("MIN_MATCH_COUNT = 6")
    print("GOOD_MATCH_RATIO = 0.7")
    print("MATCH_SCORE_THRESHOLD = 0.05")
    print("MAX_DISAPPEARED_FRAMES = 30")
    print("SIFT_N_FEATURES = 600")
    print()
    
    print("🔍 FOR VERY SENSITIVE DETECTION:")
    print("CONFIDENCE_THRESHOLD = 0.12")
    print("BOX_THRESHOLD = 0.12")
    print("MIN_BOX_AREA = 50")
    print("SIFT_CONTRAST_THRESHOLD = 0.03")
    print()

def calculate_expected_ids():
    """Calculate expected vs actual ID usage"""
    print("📊 ID EFFICIENCY ANALYSIS")
    print("=" * 50)
    
    print("🎯 IDEAL SCENARIO (2-3 objects):")
    print("  • Expected IDs: 2-3")
    print("  • ID efficiency: 100%")
    print("  • No wasted IDs")
    print()
    
    print("🚨 YOUR CURRENT SCENARIO (60-70 IDs):")
    print("  • Actual IDs created: 60-70")
    print("  • Active objects: 2-3")
    print("  • ID efficiency: ~4% (96% wasted!)")
    print("  • Wasted IDs: 57-67")
    print()
    
    print("💡 TARGET IMPROVEMENT:")
    print("  • Target IDs: 5-8 (allowing some re-detection)")
    print("  • Target efficiency: 40-60%")
    print("  • Acceptable waste: 2-5 IDs")
    print()

def provide_quick_fixes():
    """Provide immediate fixes"""
    print("🛠️ IMMEDIATE FIXES TO TRY")
    print("=" * 50)
    
    print("1. 🎯 QUICK FIX - Lower Detection Threshold:")
    print("   Config.CONFIDENCE_THRESHOLD = 0.15")
    print("   → More consistent detection, fewer missed frames")
    print()
    
    print("2. 🏃 QUICK FIX - Lenient SIFT Matching:")
    print("   Config.MIN_MATCH_COUNT = 6")
    print("   Config.GOOD_MATCH_RATIO = 0.7")
    print("   Config.MATCH_SCORE_THRESHOLD = 0.05")
    print("   → Better matching for moving/changing objects")
    print()
    
    print("3. ⏱️ QUICK FIX - Longer Persistence:")
    print("   Config.MAX_DISAPPEARED_FRAMES = 45")
    print("   → Objects stay in memory longer")
    print()
    
    print("4. 🔧 APPLY PRESET:")
    print("   python quick_tune.py")
    print("   → Choose 'moving_objects' preset")
    print()

def monitor_id_assignment():
    """Show what to monitor"""
    print("👀 WHAT TO MONITOR")
    print("=" * 50)
    
    print("📺 IN THE GUI DISPLAY:")
    print("  • Watch for 'NEW ID:X' messages in console")
    print("  • Monitor C: values (detection confidence)")
    print("  • Monitor S: values (SIFT match scores)")
    print("  • Count how many unique IDs you see")
    print()
    
    print("⌨️ KEYBOARD SHORTCUTS:")
    print("  • Press 't' for tracking statistics")
    print("  • Press 'd' for database statistics")
    print("  • Press 's' for performance statistics")
    print()
    
    print("🎯 SUCCESS INDICATORS:")
    print("  • ID numbers stay low (under 10 for 2-3 objects)")
    print("  • Same objects keep same IDs when moving")
    print("  • C: values consistently above 0.15")
    print("  • S: values above 0.05 for matched objects")
    print()

def main():
    """Main analysis function"""
    print("🔍 WAREHOUSE TRACKING ID ASSIGNMENT ANALYSIS")
    print("=" * 60)
    print("Diagnoses why you're getting high ID numbers (60-70) for 2-3 objects")
    print("=" * 60)
    
    while True:
        print("\nSelect analysis:")
        print("1. Analyze current settings")
        print("2. Explain ID assignment process")
        print("3. Show optimal settings")
        print("4. Calculate ID efficiency")
        print("5. Get immediate fixes")
        print("6. Learn what to monitor")
        print("7. Exit")
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                analyze_id_assignment_issues()
            elif choice == '2':
                explain_id_assignment_process()
            elif choice == '3':
                show_optimal_settings()
            elif choice == '4':
                calculate_expected_ids()
            elif choice == '5':
                provide_quick_fixes()
            elif choice == '6':
                monitor_id_assignment()
            elif choice == '7':
                break
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Analysis complete!")

if __name__ == "__main__":
    main()
