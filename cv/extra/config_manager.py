"""
Configuration Manager for Warehouse Tracking System
Provides easy access to parameter tuning and preset configurations
"""

import json
from config import Config

class ConfigManager:
    """Manage and tune configuration parameters"""
    
    def __init__(self):
        self.current_config = {}
        self.load_current_config()
    
    def load_current_config(self):
        """Load current configuration values"""
        config_attrs = [attr for attr in dir(Config) if not attr.startswith('_') and not callable(getattr(Config, attr))]
        
        for attr in config_attrs:
            self.current_config[attr] = getattr(Config, attr)
    
    def show_tuning_guide(self):
        """Display parameter tuning guide"""
        print(Config.get_tuning_guide())
    
    def show_current_config(self, category=None):
        """Show current configuration values"""
        categories = {
            'detection': ['MODEL_ID', 'DETECTION_PROMPT', 'CONFIDENCE_THRESHOLD', 'BOX_THRESHOLD'],
            'sift': ['SIFT_N_FEATURES', 'MIN_MATCH_COUNT', 'GOOD_MATCH_RATIO', 'MATCH_SCORE_THRESHOLD'],
            'tracking': ['MAX_DISAPPEARED_FRAMES', 'MAX_MATCH_HISTORY', 'BOX_PADDING'],
            'display': ['SHOW_BOUNDING_BOXES', 'SHOW_OBJECT_IDS', 'COLOR_NEW_OBJECT', 'BBOX_THICKNESS'],
            'performance': ['MODEL_CACHE_FRAMES', 'FRAME_BUFFER_SIZE', 'PROCESSING_THREADS']
        }
        
        if category and category in categories:
            params = categories[category]
            print(f"\n📊 {category.upper()} PARAMETERS:")
            print("=" * 40)
            for param in params:
                if param in self.current_config:
                    print(f"{param}: {self.current_config[param]}")
        else:
            print("\n🎛️ AVAILABLE CATEGORIES:")
            for cat in categories.keys():
                print(f"  • {cat}")
            print("\nUsage: show_current_config('detection')")
    
    def apply_preset(self, preset_name):
        """Apply a preset configuration"""
        presets = Config.get_preset_configs()
        
        if preset_name not in presets:
            print(f"❌ Preset '{preset_name}' not found")
            print(f"Available presets: {list(presets.keys())}")
            return False
        
        preset = presets[preset_name]
        print(f"🎯 Applying preset: {preset_name}")
        
        for param, value in preset.items():
            if hasattr(Config, param):
                setattr(Config, param, value)
                self.current_config[param] = value
                print(f"  ✅ {param} = {value}")
            else:
                print(f"  ⚠️ Parameter {param} not found")
        
        return True
    
    def update_parameter(self, param_name, value):
        """Update a single parameter"""
        if hasattr(Config, param_name):
            old_value = getattr(Config, param_name)
            setattr(Config, param_name, value)
            self.current_config[param_name] = value
            print(f"✅ Updated {param_name}: {old_value} → {value}")
            return True
        else:
            print(f"❌ Parameter '{param_name}' not found")
            return False
    
    def save_config(self, filename="custom_config.json"):
        """Save current configuration to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_config, f, indent=2, default=str)
            print(f"✅ Configuration saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def load_config(self, filename="custom_config.json"):
        """Load configuration from file"""
        try:
            with open(filename, 'r') as f:
                loaded_config = json.load(f)
            
            for param, value in loaded_config.items():
                if hasattr(Config, param):
                    setattr(Config, param, value)
                    self.current_config[param] = value
            
            print(f"✅ Configuration loaded from: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        # This would require storing defaults - simplified for now
        print("⚠️ Reset to defaults not implemented - restart application to reset")
    
    def interactive_tuning(self):
        """Interactive parameter tuning session"""
        print("🎛️ INTERACTIVE PARAMETER TUNING")
        print("=" * 40)
        print("Commands:")
        print("  show <category>     - Show parameters for category")
        print("  set <param> <value> - Set parameter value")
        print("  preset <name>       - Apply preset configuration")
        print("  save <filename>     - Save current config")
        print("  guide               - Show tuning guide")
        print("  quit                - Exit tuning")
        print()
        
        while True:
            try:
                command = input("config> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                elif command[0] == 'show' and len(command) > 1:
                    self.show_current_config(command[1])
                elif command[0] == 'set' and len(command) > 2:
                    param = command[1]
                    value = command[2]
                    # Try to convert to appropriate type
                    try:
                        if '.' in value:
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                    except:
                        pass
                    self.update_parameter(param, value)
                elif command[0] == 'preset' and len(command) > 1:
                    self.apply_preset(command[1])
                elif command[0] == 'save':
                    filename = command[1] if len(command) > 1 else "custom_config.json"
                    self.save_config(filename)
                elif command[0] == 'guide':
                    self.show_tuning_guide()
                else:
                    print("❌ Invalid command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Exiting configuration tuning")

def main():
    """Main configuration management interface"""
    print("🎛️ WAREHOUSE TRACKING CONFIGURATION MANAGER")
    print("=" * 50)
    
    manager = ConfigManager()
    
    print("Available commands:")
    print("1. Show tuning guide")
    print("2. Show current config")
    print("3. Apply preset")
    print("4. Interactive tuning")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                manager.show_tuning_guide()
            elif choice == '2':
                category = input("Enter category (detection/sift/tracking/display/performance) or 'all': ").strip()
                if category == 'all':
                    manager.show_current_config()
                else:
                    manager.show_current_config(category)
            elif choice == '3':
                presets = list(Config.get_preset_configs().keys())
                print(f"Available presets: {presets}")
                preset = input("Enter preset name: ").strip()
                manager.apply_preset(preset)
            elif choice == '4':
                manager.interactive_tuning()
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Configuration manager closed")

if __name__ == "__main__":
    main()
