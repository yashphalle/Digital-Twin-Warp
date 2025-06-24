"""
Dynamic Camera Detection and Selection System
Automatically detects available cameras and allows selection for panoramic stitching
"""

import cv2
import numpy as np
import time
import threading
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class CameraInfo:
    """Information about a detected camera"""
    def __init__(self, camera_id: int, name: str = None):
        self.camera_id = camera_id
        self.name = name or f"Camera {camera_id}"
        self.resolution = None
        self.fps = None
        self.is_stereo = False
        self.backend = None
        self.working = False
        self.test_frame = None
        
    def __str__(self):
        status = "✅ Working" if self.working else "❌ Failed"
        stereo = " (Stereo)" if self.is_stereo else ""
        return f"ID {self.camera_id}: {self.name}{stereo} - {self.resolution} @ {self.fps}fps {status}"

class DynamicCameraDetector:
    """Detects and manages multiple cameras for panoramic stitching"""
    
    def __init__(self, max_cameras=10):
        self.max_cameras = max_cameras
        self.detected_cameras: List[CameraInfo] = []
        self.selected_cameras: List[int] = []
        self.camera_configs = {}
        
    def detect_cameras(self) -> List[CameraInfo]:
        """Detect all available cameras"""
        print("🔍 Scanning for available cameras...")
        detected = []
        
        for camera_id in range(self.max_cameras):
            print(f"Testing camera {camera_id}...", end=" ")
            
            camera_info = self._test_camera(camera_id)
            if camera_info and camera_info.working:
                detected.append(camera_info)
                print(f"✅ Found: {camera_info.name}")
            else:
                print("❌ Not available")
        
        self.detected_cameras = detected
        print(f"\n📷 Found {len(detected)} working cameras")
        
        return detected
    
    def _test_camera(self, camera_id: int) -> Optional[CameraInfo]:
        """Test if a camera is working and get its properties"""
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                return None
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return None
            
            # Get camera properties
            camera_info = CameraInfo(camera_id)
            camera_info.working = True
            camera_info.test_frame = frame.copy()
            
            # Get resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera_info.resolution = f"{width}x{height}"
            
            # Get FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            camera_info.fps = int(fps) if fps > 0 else "Unknown"
            
            # Get backend info
            backend = cap.get(cv2.CAP_PROP_BACKEND)
            camera_info.backend = int(backend)
            
            # Detect if it's a stereo camera (like ZED)
            if width >= 1280 and height >= 360:  # Common stereo resolutions
                # Check if it looks like side-by-side stereo
                left_half = frame[:, :width//2]
                right_half = frame[:, width//2:]
                
                # Simple check: if halves are different, likely stereo
                diff = cv2.absdiff(left_half, right_half)
                mean_diff = np.mean(diff)
                
                if mean_diff > 10:  # Threshold for stereo detection
                    camera_info.is_stereo = True
                    camera_info.name = f"Stereo Camera {camera_id}"
            
            cap.release()
            return camera_info
            
        except Exception as e:
            print(f"Error testing camera {camera_id}: {e}")
            return None
    
    def display_camera_selection(self) -> List[int]:
        """Interactive camera selection interface"""
        if not self.detected_cameras:
            print("❌ No cameras detected. Run detect_cameras() first.")
            return []
        
        print("\n📷 CAMERA SELECTION")
        print("=" * 50)
        
        # Show all detected cameras
        for i, camera in enumerate(self.detected_cameras):
            print(f"{i + 1}. {camera}")
        
        print("\n🎯 Camera Preview Mode")
        print("Controls:")
        print("  • Number keys (1-9): Select/deselect camera")
        print("  • 's': Save selection and continue")
        print("  • 'q': Quit without selection")
        print("  • 'a': Auto-select best cameras")
        
        # Create preview windows
        selected_set = set()
        
        while True:
            # Display preview for all cameras
            for i, camera in enumerate(self.detected_cameras):
                if camera.test_frame is not None:
                    preview_frame = camera.test_frame.copy()
                    
                    # Add selection indicator
                    color = (0, 255, 0) if i in selected_set else (0, 0, 255)
                    status = "SELECTED" if i in selected_set else "AVAILABLE"
                    
                    cv2.rectangle(preview_frame, (10, 10), (300, 60), color, -1)
                    cv2.putText(preview_frame, f"{camera.name}", (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(preview_frame, f"{status} - Press {i+1}", (15, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Resize for display
                    display_frame = cv2.resize(preview_frame, (400, 300))
                    cv2.imshow(f"Camera {camera.camera_id} Preview", display_frame)
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if len(selected_set) > 0:
                    self.selected_cameras = [self.detected_cameras[i].camera_id for i in selected_set]
                    print(f"\n✅ Selected cameras: {self.selected_cameras}")
                    break
                else:
                    print("⚠️ No cameras selected!")
            elif key == ord('a'):
                # Auto-select best cameras
                selected_set = self._auto_select_cameras()
                print(f"🤖 Auto-selected cameras: {[self.detected_cameras[i].camera_id for i in selected_set]}")
            elif ord('1') <= key <= ord('9'):
                camera_index = key - ord('1')
                if camera_index < len(self.detected_cameras):
                    if camera_index in selected_set:
                        selected_set.remove(camera_index)
                        print(f"➖ Deselected: {self.detected_cameras[camera_index].name}")
                    else:
                        selected_set.add(camera_index)
                        print(f"➕ Selected: {self.detected_cameras[camera_index].name}")
        
        # Close preview windows
        cv2.destroyAllWindows()
        
        return self.selected_cameras
    
    def _auto_select_cameras(self) -> set:
        """Automatically select best cameras for panoramic stitching"""
        selected = set()
        
        # Prioritize stereo cameras
        stereo_cameras = [i for i, cam in enumerate(self.detected_cameras) if cam.is_stereo]
        if stereo_cameras:
            selected.add(stereo_cameras[0])  # Select first stereo camera
            if len(stereo_cameras) > 1:
                selected.add(stereo_cameras[1])  # Select second if available
        
        # If no stereo cameras, select regular cameras
        if not selected:
            regular_cameras = [i for i, cam in enumerate(self.detected_cameras) if not cam.is_stereo]
            if regular_cameras:
                selected.add(regular_cameras[0])
                if len(regular_cameras) > 1:
                    selected.add(regular_cameras[1])
        
        return selected
    
    def save_camera_config(self, filename="camera_config.json"):
        """Save camera configuration to file"""
        if not self.selected_cameras:
            print("❌ No cameras selected to save")
            return False
        
        config = {
            "created_date": datetime.now().isoformat(),
            "selected_cameras": self.selected_cameras,
            "camera_details": {},
            "panoramic_config": {
                "stitch_mode": "panoramic" if len(self.selected_cameras) > 1 else "single",
                "overlap_percentage": 0.2,
                "blend_mode": "linear"
            }
        }
        
        # Add details for selected cameras
        for camera_id in self.selected_cameras:
            camera_info = next((cam for cam in self.detected_cameras if cam.camera_id == camera_id), None)
            if camera_info:
                config["camera_details"][str(camera_id)] = {
                    "name": camera_info.name,
                    "resolution": camera_info.resolution,
                    "fps": camera_info.fps,
                    "is_stereo": camera_info.is_stereo,
                    "backend": camera_info.backend
                }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Camera configuration saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving camera config: {e}")
            return False
    
    def load_camera_config(self, filename="camera_config.json") -> bool:
        """Load camera configuration from file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            self.selected_cameras = config.get("selected_cameras", [])
            self.camera_configs = config.get("camera_details", {})
            
            print(f"✅ Camera configuration loaded from: {filename}")
            print(f"📷 Selected cameras: {self.selected_cameras}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading camera config: {e}")
            return False
    
    def get_camera_capabilities(self) -> Dict:
        """Get capabilities summary of selected cameras"""
        if not self.selected_cameras:
            return {}
        
        capabilities = {
            "total_cameras": len(self.selected_cameras),
            "stereo_cameras": 0,
            "regular_cameras": 0,
            "max_resolution": "0x0",
            "panoramic_capable": len(self.selected_cameras) > 1
        }
        
        max_width, max_height = 0, 0
        
        for camera_id in self.selected_cameras:
            camera_info = next((cam for cam in self.detected_cameras if cam.camera_id == camera_id), None)
            if camera_info:
                if camera_info.is_stereo:
                    capabilities["stereo_cameras"] += 1
                else:
                    capabilities["regular_cameras"] += 1
                
                # Parse resolution
                if camera_info.resolution and 'x' in camera_info.resolution:
                    width, height = map(int, camera_info.resolution.split('x'))
                    if width > max_width:
                        max_width, max_height = width, height
        
        capabilities["max_resolution"] = f"{max_width}x{max_height}"
        
        return capabilities

def main():
    """Main camera detection and selection"""
    print("🎯 DYNAMIC CAMERA DETECTION SYSTEM")
    print("=" * 50)
    
    detector = DynamicCameraDetector()
    
    # Detect cameras
    cameras = detector.detect_cameras()
    
    if not cameras:
        print("❌ No cameras found!")
        return
    
    # Show camera selection interface
    selected = detector.display_camera_selection()
    
    if selected:
        # Save configuration
        detector.save_camera_config()
        
        # Show capabilities
        capabilities = detector.get_camera_capabilities()
        print("\n📊 CAMERA SYSTEM CAPABILITIES:")
        print(f"  • Total cameras: {capabilities['total_cameras']}")
        print(f"  • Stereo cameras: {capabilities['stereo_cameras']}")
        print(f"  • Regular cameras: {capabilities['regular_cameras']}")
        print(f"  • Max resolution: {capabilities['max_resolution']}")
        print(f"  • Panoramic capable: {'Yes' if capabilities['panoramic_capable'] else 'No'}")
        
        print("\n✅ Camera setup complete!")
        print("🚀 You can now run the tracking system with selected cameras")
    else:
        print("❌ No cameras selected")

if __name__ == "__main__":
    main()
