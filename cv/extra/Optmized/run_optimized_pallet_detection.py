#!/usr/bin/env python3
"""
Run Optimized Pallet Detection
Simple script to run pallet detection with optimal settings
"""

import sys
import os
import argparse
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from optimized_pallet_detection import OptimizedPalletDetector
from pallet_detection_config import PalletDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run optimized pallet detection')
    parser.add_argument('--camera', type=int, default=1, help='Camera ID (default: 1)')
    parser.add_argument('--preset', type=str, default='balanced', 
                       choices=['maximum_accuracy', 'balanced', 'performance', 'fast'],
                       help='Quality preset (default: balanced)')
    parser.add_argument('--prompt', type=str, help='Custom detection prompt')
    parser.add_argument('--threshold', type=float, help='Custom confidence threshold')
    parser.add_argument('--show-config', action='store_true', help='Show configuration and exit')
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        PalletDetectionConfig.print_configuration_summary()
        return
    
    print("OPTIMIZED PALLET DETECTION RUNNER")
    print("=" * 50)
    
    # Get preset settings
    try:
        preset_settings = PalletDetectionConfig.get_preset_settings(args.preset)
        print(f"Using preset: {args.preset.upper()}")
        print(f"Description: {preset_settings['description']}")
    except ValueError as e:
        logger.error(e)
        return
    
    # Create detector
    detector = OptimizedPalletDetector(camera_id=args.camera)
    
    # Apply custom settings if provided
    if args.prompt:
        detector.pallet_detector.set_custom_prompt(args.prompt)
        if detector.pallet_detector.grounding_dino:
            detector.pallet_detector.grounding_dino.prompt = args.prompt
        print(f"Using custom prompt: '{args.prompt}'")
    else:
        # Apply preset prompt
        detector.pallet_detector.set_custom_prompt(preset_settings['prompt'])
        if detector.pallet_detector.grounding_dino:
            detector.pallet_detector.grounding_dino.prompt = preset_settings['prompt']
        print(f"Using preset prompt: '{preset_settings['prompt']}'")
    
    if args.threshold:
        detector.pallet_detector.confidence_threshold = args.threshold
        if detector.pallet_detector.grounding_dino:
            detector.pallet_detector.grounding_dino.confidence_threshold = args.threshold
        print(f"Using custom threshold: {args.threshold}")
    else:
        # Apply preset threshold
        detector.pallet_detector.confidence_threshold = preset_settings['confidence_threshold']
        if detector.pallet_detector.grounding_dino:
            detector.pallet_detector.grounding_dino.confidence_threshold = preset_settings['confidence_threshold']
        print(f"Using preset threshold: {preset_settings['confidence_threshold']}")
    
    # Apply preset detection interval
    detector.detection_interval = preset_settings['detection_interval']
    print(f"Detection interval: every {preset_settings['detection_interval']} frame(s)")
    
    print("=" * 50)
    print("Starting detection...")
    print("Controls:")
    print("  'q' or ESC - Quit")
    print("  's' - Save detection results")
    print("  'i' - Toggle info display")
    print("=" * 50)
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    except Exception as e:
        logger.error(f"Error running detector: {e}")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()
