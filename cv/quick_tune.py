#!/usr/bin/env python3
"""
Quick CV Algorithm Tuning Script
Use this to rapidly test different detection parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def set_high_sensitivity():
    """Set parameters for high sensitivity (more detections)"""
    print("🔍 Setting HIGH SENSITIVITY parameters...")
    Config.CONFIDENCE_THRESHOLD = 0.12
    Config.BOX_THRESHOLD = 0.12
    Config.TEXT_THRESHOLD = 0.12
    Config.DETECTION_PROMPT = "object. item. box. package. container."
    Config.MIN_MATCH_COUNT = 6
    Config.MATCH_SCORE_THRESHOLD = 0.05
    Config.MAX_DISAPPEARED_FRAMES = 60
    print(f"   • Confidence: {Config.CONFIDENCE_THRESHOLD}")
    print(f"   • Prompt: {Config.DETECTION_PROMPT}")
    print(f"   • Min matches: {Config.MIN_MATCH_COUNT}")

def set_balanced():
    """Set balanced parameters (default)"""
    print("⚖️ Setting BALANCED parameters...")
    Config.CONFIDENCE_THRESHOLD = 0.20
    Config.BOX_THRESHOLD = 0.20
    Config.TEXT_THRESHOLD = 0.20
    Config.DETECTION_PROMPT = "box. cardboard box. package."
    Config.MIN_MATCH_COUNT = 10
    Config.MATCH_SCORE_THRESHOLD = 0.2
    Config.MAX_DISAPPEARED_FRAMES = 30
    print(f"   • Confidence: {Config.CONFIDENCE_THRESHOLD}")
    print(f"   • Prompt: {Config.DETECTION_PROMPT}")
    print(f"   • Min matches: {Config.MIN_MATCH_COUNT}")

def set_high_precision():
    """Set parameters for high precision (fewer false positives)"""
    print("🎯 Setting HIGH PRECISION parameters...")
    Config.CONFIDENCE_THRESHOLD = 0.30
    Config.BOX_THRESHOLD = 0.30
    Config.TEXT_THRESHOLD = 0.30
    Config.DETECTION_PROMPT = "cardboard box. package."
    Config.MIN_MATCH_COUNT = 15
    Config.MATCH_SCORE_THRESHOLD = 0.3
    Config.MAX_DISAPPEARED_FRAMES = 20
    print(f"   • Confidence: {Config.CONFIDENCE_THRESHOLD}")
    print(f"   • Prompt: {Config.DETECTION_PROMPT}")
    print(f"   • Min matches: {Config.MIN_MATCH_COUNT}")

def enable_debug():
    """Enable debug visualization"""
    print("🔧 Enabling DEBUG mode...")
    Config.SHOW_ALL_DETECTIONS = True
    Config.SHOW_FILTERED_DETECTIONS = True
    Config.SHOW_ID_ASSIGNMENT_DEBUG = True
    Config.SHOW_DETECTION_CONFIDENCE = True
    print("   • All detections visible (green=good, red=filtered)")
    print("   • Console output enabled")

def test_with_settings(settings_name, settings_func):
    """Test the system with specific settings"""
    print(f"\n{'='*50}")
    print(f"🧪 TESTING: {settings_name}")
    print('='*50)
    
    settings_func()
    enable_debug()
    
    print("\n🚀 Starting test... Press 'q' to quit and try next setting")
    print("👀 Watch for:")
    print("   • Green boxes = detected and tracked")
    print("   • Red boxes = detected but filtered (low confidence)")
    print("   • Console messages about new IDs")
    
    # Import and run test
    from test_single_camera import test_single_camera
    test_single_camera()

def main():
    """Main tuning interface"""
    print("🎯 CV ALGORITHM TUNING TOOL")
    print("="*40)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "sensitive":
            test_with_settings("HIGH SENSITIVITY", set_high_sensitivity)
        elif mode == "balanced":
            test_with_settings("BALANCED", set_balanced)
        elif mode == "precise":
            test_with_settings("HIGH PRECISION", set_high_precision)
        else:
            print("❌ Unknown mode. Use: sensitive, balanced, or precise")
    else:
        print("📋 Available tuning modes:")
        print("   python quick_tune.py sensitive  # More detections")
        print("   python quick_tune.py balanced   # Default settings")
        print("   python quick_tune.py precise    # Fewer false positives")
        print("\n🔄 Or run interactive mode:")
        print("   python quick_tune.py interactive")
        
        if input("\n❓ Run interactive mode? (y/n): ").lower().startswith('y'):
            interactive_tuning()

def interactive_tuning():
    """Interactive tuning session"""
    print("\n🔄 INTERACTIVE TUNING SESSION")
    print("="*40)
    
    print("Testing each mode for 30 seconds...")
    
    modes = [
        ("HIGH SENSITIVITY", set_high_sensitivity),
        ("BALANCED", set_balanced), 
        ("HIGH PRECISION", set_high_precision)
    ]
    
    for name, func in modes:
        input(f"\n⏯️  Press ENTER to test {name}...")
        test_with_settings(name, func)
        print(f"\n✅ {name} test completed")
    
    print("\n🎉 Interactive tuning completed!")
    print("💡 Edit cv/config.py to make your preferred settings permanent")

if __name__ == "__main__":
    main() 