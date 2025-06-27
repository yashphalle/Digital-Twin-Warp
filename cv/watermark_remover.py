#!/usr/bin/env python3
"""
Camera Watermark Removal Utility
Removes timestamps, logos, and other watermarks from camera feeds
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class WatermarkRemover:
    """Removes watermarks from camera feeds using various techniques"""
    
    def __init__(self, camera_id: int = None):
        self.camera_id = camera_id
        self.mask_regions = {}
        self.crop_regions = {}
        self.setup_camera_specific_masks()
    
    def setup_camera_specific_masks(self):
        """Setup camera-specific watermark removal masks"""
        
        # Common watermark positions for most IP cameras
        # These are typical locations where cameras place timestamps and logos
        
        if self.camera_id == 8:
            # Camera 8 specific watermark positions
            # Adjust these coordinates based on observed watermarks
            self.mask_regions[8] = [
                # Top-left timestamp (common position)
                {"x": 0, "y": 0, "width": 300, "height": 60},
                
                # Top-right logo/brand (common position)  
                {"x": -300, "y": 0, "width": 300, "height": 60},
                
                # Bottom-left camera name (common position)
                {"x": 0, "y": -60, "width": 400, "height": 60},
                
                # Bottom-right timestamp (common position)
                {"x": -350, "y": -60, "width": 350, "height": 60}
            ]
        else:
            # Default mask regions for all cameras
            self.mask_regions[self.camera_id or 0] = [
                # Standard positions where most cameras place watermarks
                {"x": 0, "y": 0, "width": 250, "height": 50},        # Top-left
                {"x": -250, "y": 0, "width": 250, "height": 50},     # Top-right
                {"x": 0, "y": -50, "width": 300, "height": 50},      # Bottom-left
                {"x": -300, "y": -50, "width": 300, "height": 50}    # Bottom-right
            ]
    
    def create_mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create a mask to hide watermark regions"""
        height, width = frame_shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        regions = self.mask_regions.get(self.camera_id, self.mask_regions.get(0, []))
        
        for region in regions:
            # Handle negative coordinates (from right/bottom edge)
            x = region["x"] if region["x"] >= 0 else width + region["x"]
            y = region["y"] if region["y"] >= 0 else height + region["y"]
            
            # Ensure coordinates are within frame
            x = max(0, min(x, width - region["width"]))
            y = max(0, min(y, height - region["height"]))
            
            # Create mask region (0 = masked, 255 = keep)
            mask[y:y + region["height"], x:x + region["width"]] = 0
        
        return mask
    
    def remove_watermarks_by_masking(self, frame: np.ndarray) -> np.ndarray:
        """Remove watermarks by masking (blackout) specific regions"""
        if frame is None:
            return frame
        
        cleaned_frame = frame.copy()
        mask = self.create_mask(frame.shape)
        
        # Apply mask (set masked regions to black)
        cleaned_frame[mask == 0] = [0, 0, 0]
        
        return cleaned_frame
    
    def remove_watermarks_by_cropping(self, frame: np.ndarray, crop_pixels: int = 60) -> np.ndarray:
        """Remove watermarks by cropping edges where they typically appear"""
        if frame is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Crop from all edges to remove watermarks
        # Adjust crop_pixels based on your camera's watermark size
        cropped_frame = frame[
            crop_pixels:height-crop_pixels,  # Top and bottom
            crop_pixels:width-crop_pixels    # Left and right
        ]
        
        return cropped_frame
    
    def remove_watermarks_by_inpainting(self, frame: np.ndarray) -> np.ndarray:
        """Remove watermarks using OpenCV inpainting (advanced method)"""
        if frame is None:
            return frame
        
        # Create mask for watermark regions
        mask = self.create_mask(frame.shape)
        
        # Invert mask (inpainting needs 255 for areas to repair)
        inpaint_mask = cv2.bitwise_not(mask)
        
        # Apply inpainting to reconstruct the masked areas
        cleaned_frame = cv2.inpaint(frame, inpaint_mask, 3, cv2.INPAINT_TELEA)
        
        return cleaned_frame
    
    def remove_watermarks_smart(self, frame: np.ndarray, method: str = "crop") -> np.ndarray:
        """Smart watermark removal with multiple methods"""
        if frame is None:
            return frame
        
        if method == "mask":
            return self.remove_watermarks_by_masking(frame)
        elif method == "crop":
            return self.remove_watermarks_by_cropping(frame, crop_pixels=40)
        elif method == "inpaint":
            return self.remove_watermarks_by_inpainting(frame)
        elif method == "auto":
            # Try inpainting first, fallback to masking
            try:
                return self.remove_watermarks_by_inpainting(frame)
            except:
                return self.remove_watermarks_by_masking(frame)
        else:
            logger.warning(f"Unknown watermark removal method: {method}")
            return frame
    
    def detect_watermark_regions(self, frame: np.ndarray, threshold: int = 200) -> List[dict]:
        """Automatically detect potential watermark regions (experimental)"""
        if frame is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find bright text regions (typical of watermarks)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (typical watermark dimensions)
            if 50 < w < 400 and 20 < h < 80:
                # Check if it's in typical watermark positions
                height, width = frame.shape[:2]
                if (y < 100 or y > height - 100) and (x < 300 or x > width - 300):
                    detected_regions.append({
                        "x": x, "y": y, "width": w, "height": h,
                        "confidence": cv2.contourArea(contour)
                    })
        
        return detected_regions
    
    def preview_mask(self, frame: np.ndarray) -> np.ndarray:
        """Preview what regions will be masked (for debugging)"""
        if frame is None:
            return frame
        
        preview_frame = frame.copy()
        mask = self.create_mask(frame.shape)
        
        # Highlight masked regions in red
        preview_frame[mask == 0] = [0, 0, 255]  # Red overlay
        
        # Add transparency
        cv2.addWeighted(frame, 0.7, preview_frame, 0.3, 0, preview_frame)
        
        return preview_frame

def test_watermark_removal():
    """Test watermark removal on Camera 8"""
    print("🧹 Testing Watermark Removal")
    print("=" * 40)
    
    # Import here to avoid circular imports
    from configs.config import Config
    from lense_correct2 import OptimizedFisheyeCorrector
    
    # Setup
    camera_id = 8
    watermark_remover = WatermarkRemover(camera_id)
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    
    # Connect to camera
    camera_url = Config.RTSP_CAMERA_URLS[camera_id]
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print("❌ Failed to connect to camera")
        return
    
    print("✅ Connected to Camera 8")
    print("🎯 Testing watermark removal methods...")
    print("Press keys: '1'=Original, '2'=Mask, '3'=Crop, '4'=Inpaint, '5'=Preview, 'q'=Quit")
    
    mode = "original"
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        # Apply fisheye correction first
        corrected_frame = fisheye_corrector.correct(raw_frame)
        
        # Apply watermark removal based on mode
        if mode == "original":
            display_frame = corrected_frame
            window_title = "Original (Press 2-5 to test removal)"
        elif mode == "mask":
            display_frame = watermark_remover.remove_watermarks_smart(corrected_frame, "mask")
            window_title = "Watermark Masked (Black regions)"
        elif mode == "crop":
            display_frame = watermark_remover.remove_watermarks_smart(corrected_frame, "crop")
            window_title = "Watermark Cropped (Edges removed)"
        elif mode == "inpaint":
            display_frame = watermark_remover.remove_watermarks_smart(corrected_frame, "inpaint")
            window_title = "Watermark Inpainted (AI reconstructed)"
        elif mode == "preview":
            display_frame = watermark_remover.preview_mask(corrected_frame)
            window_title = "Mask Preview (Red = will be removed)"
        
        # Add mode indicator
        cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Resize for display
        height, width = display_frame.shape[:2]
        display_width = 1280
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(display_frame, (display_width, display_height))
        
        cv2.imshow(window_title, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = "original"
        elif key == ord('2'):
            mode = "mask"
        elif key == ord('3'):
            mode = "crop"
        elif key == ord('4'):
            mode = "inpaint"
        elif key == ord('5'):
            mode = "preview"
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Watermark removal test completed")

if __name__ == "__main__":
    test_watermark_removal() 