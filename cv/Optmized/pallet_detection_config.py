#!/usr/bin/env python3
"""
Pallet Detection Configuration
Stores optimal settings found through interactive tuning
"""

class PalletDetectionConfig:
    """Configuration class for optimized pallet detection"""
    
    # OPTIMAL SETTINGS FOR WOODEN SKIDS/PALLETS ON FLOOR
    # These settings were determined through interactive tuning
    
    # Best performing prompts (in order of effectiveness)
    OPTIMAL_PROMPTS = [
        "wooden skid on floor",           # Best for wooden skids
        "wooden pallet on floor",         # Good for standard pallets
        "pallet on concrete floor",       # Good for warehouse floors
        "wooden platform on ground",      # Alternative description
        "cube shaped pallet",             # For cube-shaped pallets
        "stacked wooden pallet",          # For stacked configurations
        "wooden shipping skid",           # Industrial terminology
        "rectangular wooden pallet",      # Shape-based description
    ]
    
    # Primary prompt (best performing)
    PRIMARY_PROMPT = OPTIMAL_PROMPTS[0]
    
    # Confidence thresholds for different scenarios
    CONFIDENCE_THRESHOLDS = {
        'very_sensitive': 0.10,    # Catches very weak detections (more false positives)
        'sensitive': 0.12,         # Good balance for 10-15% confidence detections
        'balanced': 0.15,          # Balanced approach
        'conservative': 0.20,      # Fewer false positives
        'strict': 0.25            # Only high-confidence detections
    }
    
    # Recommended threshold for wooden skids (based on your 10-15% observation)
    RECOMMENDED_THRESHOLD = CONFIDENCE_THRESHOLDS['sensitive']
    
    # Detection intervals (frames to skip between detections)
    DETECTION_INTERVALS = {
        'every_frame': 1,          # Process every frame (best accuracy, slower)
        'every_2nd': 2,           # Process every 2nd frame
        'every_3rd': 3,           # Process every 3rd frame (good balance)
        'every_5th': 5            # Process every 5th frame (faster, may miss objects)
    }
    
    # Recommended interval for pallet detection
    RECOMMENDED_INTERVAL = DETECTION_INTERVALS['every_frame']
    
    # Alternative prompts for different pallet types
    PALLET_TYPE_PROMPTS = {
        'wooden_skids': [
            "wooden skid on floor",
            "wooden skid",
            "wooden platform on ground"
        ],
        'standard_pallets': [
            "wooden pallet on floor",
            "shipping pallet on floor",
            "warehouse pallet"
        ],
        'industrial_pallets': [
            "industrial pallet",
            "freight pallet on floor",
            "storage pallet"
        ],
        'cube_pallets': [
            "cube shaped pallet",
            "rectangular wooden structure",
            "stacked wooden pallet"
        ]
    }
    
    # Detection quality settings
    QUALITY_PRESETS = {
        'maximum_accuracy': {
            'confidence_threshold': 0.10,
            'detection_interval': 1,
            'prompt': "wooden skid on floor",
            'description': "Highest accuracy, slowest performance"
        },
        'balanced': {
            'confidence_threshold': 0.12,
            'detection_interval': 2,
            'prompt': "wooden skid on floor",
            'description': "Good balance of accuracy and speed"
        },
        'performance': {
            'confidence_threshold': 0.15,
            'detection_interval': 3,
            'prompt': "wooden pallet on floor",
            'description': "Faster processing, good accuracy"
        },
        'fast': {
            'confidence_threshold': 0.20,
            'detection_interval': 5,
            'prompt': "wooden pallet",
            'description': "Fastest processing, conservative detection"
        }
    }
    
    # Recommended preset for wooden skids on floor
    RECOMMENDED_PRESET = 'balanced'
    
    @classmethod
    def get_optimal_settings(cls):
        """Get the optimal settings for wooden skids detection"""
        return {
            'prompt': cls.PRIMARY_PROMPT,
            'confidence_threshold': cls.RECOMMENDED_THRESHOLD,
            'detection_interval': cls.RECOMMENDED_INTERVAL,
            'description': 'Optimized for wooden skids/pallets on floor with 10-15% confidence'
        }
    
    @classmethod
    def get_preset_settings(cls, preset_name: str):
        """Get settings for a specific quality preset"""
        if preset_name in cls.QUALITY_PRESETS:
            return cls.QUALITY_PRESETS[preset_name]
        else:
            available = ', '.join(cls.QUALITY_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    @classmethod
    def get_prompts_for_pallet_type(cls, pallet_type: str):
        """Get prompts for a specific pallet type"""
        if pallet_type in cls.PALLET_TYPE_PROMPTS:
            return cls.PALLET_TYPE_PROMPTS[pallet_type]
        else:
            available = ', '.join(cls.PALLET_TYPE_PROMPTS.keys())
            raise ValueError(f"Unknown pallet type '{pallet_type}'. Available: {available}")
    
    @classmethod
    def print_configuration_summary(cls):
        """Print a summary of the optimal configuration"""
        print("OPTIMAL PALLET DETECTION CONFIGURATION")
        print("=" * 50)
        
        optimal = cls.get_optimal_settings()
        print(f"Primary Prompt: '{optimal['prompt']}'")
        print(f"Confidence Threshold: {optimal['confidence_threshold']}")
        print(f"Detection Interval: Every {optimal['detection_interval']} frame(s)")
        print(f"Description: {optimal['description']}")
        
        print("\nAVAILABLE QUALITY PRESETS:")
        print("-" * 30)
        for preset_name, settings in cls.QUALITY_PRESETS.items():
            marker = " (RECOMMENDED)" if preset_name == cls.RECOMMENDED_PRESET else ""
            print(f"{preset_name.upper()}{marker}:")
            print(f"  Prompt: '{settings['prompt']}'")
            print(f"  Confidence: {settings['confidence_threshold']}")
            print(f"  Interval: {settings['detection_interval']}")
            print(f"  Description: {settings['description']}")
            print()
        
        print("ALTERNATIVE PROMPTS BY PALLET TYPE:")
        print("-" * 40)
        for pallet_type, prompts in cls.PALLET_TYPE_PROMPTS.items():
            print(f"{pallet_type.upper()}:")
            for prompt in prompts:
                print(f"  - '{prompt}'")
            print()


def main():
    """Main function to display configuration"""
    PalletDetectionConfig.print_configuration_summary()


if __name__ == "__main__":
    main()
