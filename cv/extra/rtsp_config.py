"""
RTSP Camera Configuration for Warehouse Tracking System
Configuration settings for Lorex RTSP cameras integration
"""

class RTSPConfig:
    """Configuration settings for RTSP cameras"""
    
    # ==================== RTSP CAMERA SETTINGS ====================
    
    # RTSP Camera URLs (Lorex cameras)
    RTSP_CAMERA_URLS = [
        "rtsp://admin:wearewarp!@192.168.0.79:554/Streaming/channels/1",  # Cam 8 Back
        "rtsp://admin:wearewarp!@192.168.0.80:554/Streaming/channels/1",  # Cam 9 Back 
        "rtsp://admin:wearewarp!@192.168.0.82:554/Streaming/channels/1"   # Cam 10 Back
    ]
    
    # Camera names for identification
    CAMERA_NAMES = [
        "Camera 8 - Back Left",
        "Camera 9 - Back Center", 
        "Camera 10 - Back Right"
    ]
    
    # ==================== CONNECTION SETTINGS ====================
    
    # RTSP connection parameters
    RTSP_TIMEOUT = 10  # seconds
    RTSP_BUFFER_SIZE = 5  # frames
    MAX_RECONNECTION_ATTEMPTS = 5
    RECONNECTION_DELAY = 5  # seconds
    
    # Network settings
    RTSP_USERNAME = "admin"
    RTSP_PASSWORD = "wearewarp!"
    RTSP_PORT = 554
    
    # ==================== CAMERA PROPERTIES ====================
    
    # Expected camera properties (Lorex cameras)
    EXPECTED_FRAME_WIDTH = 1920
    EXPECTED_FRAME_HEIGHT = 1080
    EXPECTED_FPS = 30
    
    # Lens properties
    LENS_MM = 2.8  # Fisheye lens focal length
    FISHEYE_CORRECTION_ENABLED = True
    
    # ==================== FRAME PROCESSING ====================
    
    # Frame combination settings
    COMBINE_MODE = "vertical"  # Options: "vertical", "horizontal", "grid"
    TARGET_FRAME_WIDTH = 1920
    TARGET_FRAME_HEIGHT = 1080
    
    # Performance settings
    PROCESSING_FPS = 30
    FRAME_BUFFER_SIZE = 10
    
    # ==================== FISHEYE CORRECTION ====================
    
    # Fisheye correction parameters
    FISHEYE_K1 = -0.1  # Radial distortion coefficient
    FISHEYE_K2 = 0.05  # Second radial distortion coefficient
    FISHEYE_P1 = 0.0   # Tangential distortion
    FISHEYE_P2 = 0.0   # Tangential distortion
    FISHEYE_K3 = 0.0   # Third radial distortion coefficient
    
    # ==================== INTEGRATION SETTINGS ====================
    
    # Integration with existing system
    USE_EXISTING_TRACKING = True
    USE_EXISTING_DATABASE = True
    USE_EXISTING_CALIBRATION = True
    
    # Camera mapping to existing zones
    CAMERA_ZONE_MAPPING = {
        1: [1, 2],  # Camera 8 covers zones 1 and 2
        2: [3, 4],  # Camera 9 covers zones 3 and 4  
        3: [5, 6]   # Camera 10 covers zones 5 and 6
    }
    
    # ==================== DISPLAY SETTINGS ====================
    
    # Display window settings
    DISPLAY_WINDOW_NAME = "RTSP Warehouse Tracking"
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080
    
    # Overlay settings
    SHOW_CAMERA_LABELS = True
    SHOW_TIMESTAMPS = True
    SHOW_FPS = True
    SHOW_CONNECTION_STATUS = True
    
    # ==================== LOGGING SETTINGS ====================
    
    # Log levels
    RTSP_LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_RTSP_CONNECTIONS = True
    LOG_FRAME_STATS = True
    
    # ==================== ERROR HANDLING ====================
    
    # Error handling settings
    CONTINUE_ON_CAMERA_FAILURE = True
    MAX_CONSECUTIVE_FAILURES = 10
    FAILURE_RETRY_DELAY = 2  # seconds
    
    # ==================== PERFORMANCE OPTIMIZATION ====================
    
    # Performance settings
    ENABLE_FRAME_SKIPPING = True
    MAX_FRAME_SKIP = 2
    ENABLE_MULTITHREADING = True
    THREAD_POOL_SIZE = 4
    
    # Memory management
    ENABLE_MEMORY_OPTIMIZATION = True
    MAX_FRAME_HISTORY = 30
    CLEANUP_INTERVAL = 1000  # frames
    
    # ==================== CALIBRATION SETTINGS ====================
    
    # Calibration file paths
    FISHEYE_CALIBRATION_FILE = "fisheye_calibration.npz"
    CAMERA_CALIBRATION_FILE = "rtsp_camera_calibration.json"
    
    # Calibration settings
    AUTO_CALIBRATE_FISHEYE = False
    CALIBRATION_CHECKERBOARD_SIZE = (9, 6)
    MIN_CALIBRATION_IMAGES = 10
    
    # ==================== BACKUP SETTINGS ====================
    
    # Fallback to USB cameras if RTSP fails
    ENABLE_USB_FALLBACK = True
    USB_CAMERA_IDS = [0, 1, 2]  # USB camera IDs to try as fallback
    
    # ==================== UTILITY METHODS ====================
    
    @classmethod
    def get_camera_url(cls, camera_index: int) -> str:
        """Get RTSP URL for specific camera"""
        if 0 <= camera_index < len(cls.RTSP_CAMERA_URLS):
            return cls.RTSP_CAMERA_URLS[camera_index]
        raise ValueError(f"Invalid camera index: {camera_index}")
    
    @classmethod
    def get_camera_name(cls, camera_index: int) -> str:
        """Get camera name for specific camera"""
        if 0 <= camera_index < len(cls.CAMERA_NAMES):
            return cls.CAMERA_NAMES[camera_index]
        return f"Camera {camera_index + 1}"
    
    @classmethod
    def get_camera_zones(cls, camera_index: int) -> list:
        """Get zones covered by specific camera"""
        return cls.CAMERA_ZONE_MAPPING.get(camera_index + 1, [])
    
    @classmethod
    def get_total_cameras(cls) -> int:
        """Get total number of configured cameras"""
        return len(cls.RTSP_CAMERA_URLS)
    
    @classmethod
    def get_camera_info(cls) -> dict:
        """Get information about all cameras"""
        info = {
            'total_cameras': cls.get_total_cameras(),
            'cameras': []
        }
        
        for i in range(cls.get_total_cameras()):
            camera_info = {
                'index': i,
                'name': cls.get_camera_name(i),
                'url': cls.get_camera_url(i),
                'zones': cls.get_camera_zones(i),
                'expected_resolution': f"{cls.EXPECTED_FRAME_WIDTH}x{cls.EXPECTED_FRAME_HEIGHT}",
                'expected_fps': cls.EXPECTED_FPS
            }
            info['cameras'].append(camera_info)
        
        return info
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate RTSP configuration"""
        try:
            # Check if URLs are properly formatted
            for i, url in enumerate(cls.RTSP_CAMERA_URLS):
                if not url.startswith("rtsp://"):
                    print(f"❌ Invalid RTSP URL format for camera {i+1}: {url}")
                    return False
                
                if cls.RTSP_USERNAME not in url or cls.RTSP_PASSWORD not in url:
                    print(f"❌ Missing credentials in RTSP URL for camera {i+1}")
                    return False
            
            # Check if camera names match
            if len(cls.CAMERA_NAMES) != len(cls.RTSP_CAMERA_URLS):
                print("❌ Number of camera names doesn't match number of URLs")
                return False
            
            # Check zone mapping
            for camera_id, zones in cls.CAMERA_ZONE_MAPPING.items():
                if not isinstance(zones, list):
                    print(f"❌ Invalid zone mapping for camera {camera_id}")
                    return False
            
            print("✅ RTSP configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ RTSP configuration validation failed: {e}")
            return False
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("\n📷 RTSP CAMERA CONFIGURATION SUMMARY")
        print("=" * 50)
        
        info = cls.get_camera_info()
        print(f"Total cameras: {info['total_cameras']}")
        print(f"Expected resolution: {cls.EXPECTED_FRAME_WIDTH}x{cls.EXPECTED_FRAME_HEIGHT}")
        print(f"Expected FPS: {cls.EXPECTED_FPS}")
        print(f"Fisheye correction: {'Enabled' if cls.FISHEYE_CORRECTION_ENABLED else 'Disabled'}")
        print(f"Lens focal length: {cls.LENS_MM}mm")
        
        print("\n📹 Camera Details:")
        for camera in info['cameras']:
            print(f"  • {camera['name']}: {camera['url']}")
            print(f"    Zones: {camera['zones']}")
        
        print(f"\n🔧 Integration Settings:")
        print(f"  • Use existing tracking: {cls.USE_EXISTING_TRACKING}")
        print(f"  • Use existing database: {cls.USE_EXISTING_DATABASE}")
        print(f"  • USB fallback enabled: {cls.ENABLE_USB_FALLBACK}")
        
        print(f"\n⚡ Performance Settings:")
        print(f"  • Processing FPS: {cls.PROCESSING_FPS}")
        print(f"  • Frame buffer size: {cls.FRAME_BUFFER_SIZE}")
        print(f"  • Multithreading: {cls.ENABLE_MULTITHREADING}")

# ==================== CONFIGURATION PRESETS ====================

class RTSPConfigPresets:
    """Preset configurations for different scenarios"""
    
    @staticmethod
    def get_development_config():
        """Configuration for development/testing"""
        config = RTSPConfig()
        config.PROCESSING_FPS = 15
        config.FRAME_BUFFER_SIZE = 3
        config.RTSP_LOG_LEVEL = "DEBUG"
        config.ENABLE_FRAME_SKIPPING = False
        return config
    
    @staticmethod
    def get_production_config():
        """Configuration for production use"""
        config = RTSPConfig()
        config.PROCESSING_FPS = 30
        config.FRAME_BUFFER_SIZE = 10
        config.RTSP_LOG_LEVEL = "INFO"
        config.ENABLE_FRAME_SKIPPING = True
        config.MAX_FRAME_SKIP = 1
        return config
    
    @staticmethod
    def get_high_performance_config():
        """Configuration for high-performance systems"""
        config = RTSPConfig()
        config.PROCESSING_FPS = 60
        config.FRAME_BUFFER_SIZE = 20
        config.THREAD_POOL_SIZE = 8
        config.ENABLE_MEMORY_OPTIMIZATION = True
        return config

def main():
    """Test and display RTSP configuration"""
    print("🔧 RTSP Camera Configuration Test")
    print("=" * 50)
    
    # Validate configuration
    if RTSPConfig.validate_config():
        RTSPConfig.print_config_summary()
    else:
        print("❌ Configuration validation failed")
    
    # Test camera info
    info = RTSPConfig.get_camera_info()
    print(f"\n📊 Camera Information:")
    for camera in info['cameras']:
        print(f"  Camera {camera['index']+1}: {camera['name']}")
        print(f"    URL: {camera['url']}")
        print(f"    Zones: {camera['zones']}")

if __name__ == "__main__":
    main() 