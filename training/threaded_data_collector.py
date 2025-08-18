#!/usr/bin/env python3
"""
Threaded Data Collection for Custom Training
Collects raw 4K + fisheye-corrected 2K images from all 11 warehouse cameras
"""

import cv2
import os
import time
import threading
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv.config.config import Config
from cv.config.warehouse_config import get_warehouse_config
from cv.modules.fisheye_corrector import OptimizedFisheyeCorrector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraCollectionThread(threading.Thread):
    """Individual camera collection thread"""
    
    def __init__(self, camera_id: int, output_dir: str, gui_callback=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.gui_callback = gui_callback
        self.running = False
        
        # Collection settings from config
        self.frame_skip_count = Config.COLLECTION_FRAME_SKIP  # 1 minute at 20fps
        self.raw_jpeg_quality = Config.RAW_IMAGE_QUALITY
        self.corrected_jpeg_quality = Config.CORRECTED_IMAGE_QUALITY
        
        # Camera connection
        self.cap = None
        self.warehouse_config = get_warehouse_config()
        self.rtsp_url = self._get_rtsp_url()
        
        # Fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'images_saved': 0,
            'last_save_time': None,
            'connection_status': 'Disconnected',
            'errors': 0,
            'last_error': None
        }
        
        # Create camera directories
        self.raw_dir = self.output_dir / f"camera_{camera_id}" / "raw_4k"
        self.corrected_dir = self.output_dir / f"camera_{camera_id}" / "corrected_2k"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.corrected_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Camera {camera_id} thread initialized")
    
    def _get_rtsp_url(self) -> str:
        """Get RTSP URL for camera - Use remote URLs for training data collection"""
        # Use remote URLs from config.py (updated URLs)
        from cv.config.config import Config
        remote_url = Config.REMOTE_RTSP_CAMERA_URLS.get(self.camera_id, "")
        if not remote_url:
            # Fallback to old format if camera not found
            remote_url = f'rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/{self.camera_id}01'
        logger.info(f"📡 Camera {self.camera_id}: Using remote RTSP URL: {remote_url}")
        return remote_url
    
    def _connect_camera(self) -> bool:
        """Connect to camera with retry logic"""
        max_retries = 3
        retry_delay = 5

        logger.info(f"🔧 Camera {self.camera_id}: Starting connection process...")
        logger.info(f"📡 Camera {self.camera_id}: RTSP URL: {self.rtsp_url}")

        for attempt in range(max_retries):
            try:
                logger.info(f"🔧 Camera {self.camera_id}: Connecting (attempt {attempt + 1}/{max_retries})")
                self.cap = cv2.VideoCapture(self.rtsp_url)
                logger.info(f"🔧 Camera {self.camera_id}: VideoCapture created")
                
                if self.cap.isOpened():
                    logger.info(f"✅ Camera {self.camera_id}: VideoCapture opened successfully")
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info(f"🔧 Camera {self.camera_id}: Buffer size set")

                    # Test read
                    logger.info(f"🔧 Camera {self.camera_id}: Testing frame read...")
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"✅ Camera {self.camera_id}: Test frame read successful - Frame shape: {frame.shape}")
                        self.stats['connection_status'] = 'Connected'
                        logger.info(f"🎉 Camera {self.camera_id}: Connected successfully")
                        return True
                    else:
                        logger.warning(f"❌ Camera {self.camera_id}: Test frame read failed - ret: {ret}, frame: {frame is not None}")
                        self.cap.release()
                        self.cap = None
                else:
                    logger.warning(f"❌ Camera {self.camera_id}: VideoCapture failed to open")
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Connection error: {e}")
                self.stats['last_error'] = str(e)
                self.stats['errors'] += 1
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        self.stats['connection_status'] = 'Failed'
        return False
    
    def _save_images(self, raw_frame: np.ndarray) -> bool:
        """Save raw and corrected images"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Save raw 4K image
            raw_filename = f"camera_{self.camera_id}_{timestamp}_raw.jpg"
            raw_path = self.raw_dir / raw_filename
            
            cv2.imwrite(str(raw_path), raw_frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.raw_jpeg_quality])
            
            # Apply fisheye correction
            corrected_frame = self.fisheye_corrector.correct(raw_frame)
            
            # Resize to 2K (2560x1440) if larger for better training data quality
            height, width = corrected_frame.shape[:2]
            if width > 2560 or height > 1440:
                # Calculate scale to fit 2560x1440 (2K resolution)
                scale_w = 2560 / width
                scale_h = 1440 / height
                scale = min(scale_w, scale_h)

                new_width = int(width * scale)
                new_height = int(height * scale)
                corrected_frame = cv2.resize(corrected_frame, (new_width, new_height))

            # Save corrected 2K image
            corrected_filename = f"camera_{self.camera_id}_{timestamp}_corrected.jpg"
            corrected_path = self.corrected_dir / corrected_filename
            
            cv2.imwrite(str(corrected_path), corrected_frame,
                       [cv2.IMWRITE_JPEG_QUALITY, self.corrected_jpeg_quality])
            
            # Update statistics
            self.stats['images_saved'] += 1
            self.stats['last_save_time'] = datetime.now()
            
            # Update GUI if callback provided
            if self.gui_callback:
                # Create thumbnail for GUI (resize corrected image to larger size)
                thumbnail = cv2.resize(corrected_frame, (320, 240))
                self.gui_callback(self.camera_id, thumbnail, self.stats.copy())
            
            logger.info(f"Camera {self.camera_id}: Saved images {self.stats['images_saved']}")
            return True
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Save error: {e}")
            self.stats['last_error'] = str(e)
            self.stats['errors'] += 1
            return False
    
    def run(self):
        """Main collection loop"""
        logger.info(f"🚀 Camera {self.camera_id}: Starting collection thread")
        self.running = True

        # Staggered startup (5 seconds per camera)
        startup_delay = (self.camera_id - 1) * 5
        if startup_delay > 0:
            logger.info(f"⏰ Camera {self.camera_id}: Waiting {startup_delay}s for staggered startup")
            time.sleep(startup_delay)

        logger.info(f"🔧 Camera {self.camera_id}: Initializing fisheye corrector...")
        # Initialize fisheye corrector
        self.fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
        logger.info(f"✅ Camera {self.camera_id}: Fisheye corrector initialized")

        frame_count = 0
        logger.info(f"🔧 Camera {self.camera_id}: Entering main collection loop...")
        
        while self.running:
            try:
                # Connect if not connected
                if self.cap is None or not self.cap.isOpened():
                    logger.info(f"🔧 Camera {self.camera_id}: Camera not connected, attempting connection...")
                    if not self._connect_camera():
                        logger.warning(f"❌ Camera {self.camera_id}: Connection failed, retrying in 30s")
                        time.sleep(30)
                        continue

                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"❌ Camera {self.camera_id}: Failed to read frame - ret: {ret}, frame: {frame is not None}")
                    self.cap.release()
                    self.cap = None
                    continue

                if frame_count == 0:  # Log first successful frame
                    logger.info(f"✅ Camera {self.camera_id}: Frame read successful - Shape: {frame.shape}")
                
                self.stats['frames_processed'] += 1
                frame_count += 1
                
                # Check if it's time to save (every 1200 frames = 1 minute at 20fps)
                if frame_count >= self.frame_skip_count:
                    logger.info(f"💾 Camera {self.camera_id}: Time to save image (frame {frame_count}/{self.frame_skip_count})")
                    success = self._save_images(frame)
                    if success:
                        logger.info(f"✅ Camera {self.camera_id}: Images saved successfully")
                    else:
                        logger.warning(f"❌ Camera {self.camera_id}: Failed to save images")
                    frame_count = 0  # Reset counter
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Runtime error: {e}")
                self.stats['last_error'] = str(e)
                self.stats['errors'] += 1
                
                # Reset connection on error
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                time.sleep(5)  # Wait before retry
        
        # Cleanup
        if self.cap:
            self.cap.release()
        
        logger.info(f"Camera {self.camera_id}: Collection thread stopped")
    
    def stop(self):
        """Stop collection thread"""
        self.running = False
        if self.cap:
            self.cap.release()

class CollectionGUI:
    """GUI for monitoring collection progress"""
    
    def __init__(self):
        logger.info("🔧 Initializing GUI...")
        self.root = tk.Tk()
        logger.info("✅ Tkinter root window created")

        self.root.title("Warehouse Camera Data Collection")
        self.root.geometry("1600x1200")

        # Force window to front and focus
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.focus_force()

        logger.info("✅ Window properties set and forced to front")

        # Camera thumbnails and info
        self.camera_frames = {}
        self.camera_labels = {}
        self.camera_info_labels = {}
        self.camera_images = {}  # Store PhotoImage objects
        logger.info("✅ Camera display dictionaries initialized")

        logger.info("🔧 Setting up GUI layout...")
        self.setup_gui()
        logger.info("✅ GUI layout complete")

        # Statistics
        self.total_stats = {
            'total_images': 0,
            'active_cameras': 0,
            'start_time': datetime.now()
        }
        logger.info("✅ GUI initialization complete")

        # Auto-start data collection
        self.root.after(1000, self.start_collection)  # Start after 1 second
        logger.info("🚀 Auto-start scheduled")

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Setup GUI layout"""
        logger.info("🔧 Creating main title...")
        # Main title
        title_label = tk.Label(self.root, text="Warehouse Camera Data Collection",
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        logger.info("✅ Main title created")

        logger.info("🔧 Creating statistics frame...")
        # Statistics frame
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(pady=5)

        self.stats_label = tk.Label(stats_frame, text="Initializing...",
                                   font=("Arial", 12))
        self.stats_label.pack()
        logger.info("✅ Statistics frame created")
        
        logger.info("🔧 Creating camera grid frame...")
        # Camera grid frame (4x3 for 11 cameras) - Don't expand to leave room for buttons
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=10, fill='both')
        logger.info("✅ Camera grid frame created")

        logger.info("🔧 Creating camera display grid for 11 cameras...")
        # Create camera display grid
        for camera_id in range(1, 12):  # Cameras 1-11
            logger.info(f"🔧 Creating camera {camera_id} display...")
            row = (camera_id - 1) // 4
            col = (camera_id - 1) % 4
            
            # Camera frame
            camera_frame = tk.Frame(grid_frame, relief='raised', borderwidth=2)
            camera_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Camera title
            title_label = tk.Label(camera_frame, text=f"Camera {camera_id}", 
                                  font=("Arial", 10, "bold"))
            title_label.pack()
            
            # Image display (larger size for better visibility)
            image_label = tk.Label(camera_frame, text="No Image",
                                  width=40, height=18, bg='gray')
            image_label.pack(pady=2)
            
            # Info display
            info_label = tk.Label(camera_frame, text="Status: Initializing", 
                                 font=("Arial", 8), justify='left')
            info_label.pack()
            
            # Store references
            self.camera_frames[camera_id] = camera_frame
            self.camera_labels[camera_id] = image_label
            self.camera_info_labels[camera_id] = info_label
            logger.info(f"✅ Camera {camera_id} display created")

        logger.info("🔧 Configuring grid weights...")
        # Configure grid weights for resizing
        for i in range(4):
            grid_frame.columnconfigure(i, weight=1)
        for i in range(3):
            grid_frame.rowconfigure(i, weight=1)
        logger.info("✅ Grid weights configured")
        
        logger.info("🔧 Creating status display...")
        # Status display instead of buttons
        status_frame = tk.Frame(self.root, bg='lightgreen', relief='raised', bd=2)
        status_frame.pack(pady=10, fill='x')

        self.status_label = tk.Label(status_frame, text="🚀 DATA COLLECTION RUNNING",
                                    bg='lightgreen', fg='darkgreen',
                                    font=("Arial", 14, "bold"))
        self.status_label.pack(pady=10)

        logger.info("✅ Status display created")
    
    def update_camera_display(self, camera_id: int, thumbnail: np.ndarray, stats: Dict):
        """Update camera display with new thumbnail and stats"""
        logger.info(f"🖼️ GUI: Updating display for Camera {camera_id}")
        try:
            # Convert OpenCV image to PIL
            logger.info(f"🖼️ GUI: Converting thumbnail for Camera {camera_id} - Shape: {thumbnail.shape}")
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(thumbnail_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            # Update image
            self.camera_labels[camera_id].configure(image=photo)
            self.camera_images[camera_id] = photo  # Keep reference
            logger.info(f"✅ GUI: Thumbnail updated for Camera {camera_id}")
            
            # Update info
            last_save = stats['last_save_time']
            last_save_str = last_save.strftime("%H:%M:%S") if last_save else "Never"
            
            info_text = f"Status: {stats['connection_status']}\n"
            info_text += f"Images: {stats['images_saved']}\n"
            info_text += f"Last Save: {last_save_str}\n"
            info_text += f"Errors: {stats['errors']}"
            
            self.camera_info_labels[camera_id].configure(text=info_text)
            
        except Exception as e:
            logger.error(f"GUI update error for camera {camera_id}: {e}")

    def on_closing(self):
        """Handle window close event"""
        logger.info("🛑 Window closing - stopping data collection...")
        self.stop_collection()
        self.root.destroy()

    def update_stats(self):
        """Update overall statistics"""
        if hasattr(self, 'collection_threads'):
            total_images = sum(thread.stats['images_saved'] for thread in self.collection_threads)
            active_cameras = sum(1 for thread in self.collection_threads 
                               if thread.stats['connection_status'] == 'Connected')
            
            runtime = datetime.now() - self.total_stats['start_time']
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            
            stats_text = f"Total Images: {total_images} | Active Cameras: {active_cameras}/11 | Runtime: {runtime_str}"
            self.stats_label.configure(text=stats_text)
        
        # Schedule next update
        self.root.after(5000, self.update_stats)  # Update every 5 seconds
    
    def start_collection(self):
        """Start data collection"""
        logger.info("🚀 Starting data collection...")
        output_dir = Config.RAW_COLLECTION_DIR
        logger.info(f"📁 Output directory: {output_dir}")

        # Create collection threads
        self.collection_threads = []
        logger.info("🔧 Creating camera collection threads...")

        for camera_id in range(1, 12):
            logger.info(f"🔧 Creating thread for Camera {camera_id}...")
            thread = CameraCollectionThread(
                camera_id=camera_id,
                output_dir=output_dir,
                gui_callback=self.update_camera_display
            )
            self.collection_threads.append(thread)
            logger.info(f"🚀 Starting thread for Camera {camera_id}...")
            thread.start()
            logger.info(f"✅ Camera {camera_id} thread started")

        logger.info(f"✅ All {len(self.collection_threads)} camera threads created and started")

        # Update status display
        self.status_label.config(text="🎉 DATA COLLECTION ACTIVE - All 11 Cameras Running",
                                bg='lightgreen', fg='darkgreen')
        logger.info("✅ Status display updated")

        # Start statistics updates
        self.total_stats['start_time'] = datetime.now()
        self.update_stats()
        logger.info("✅ Statistics updates started")

        logger.info("🎉 Data collection started for all cameras")
    
    def stop_collection(self):
        """Stop data collection"""
        if hasattr(self, 'collection_threads'):
            for thread in self.collection_threads:
                thread.stop()
            
            # Wait for threads to finish
            for thread in self.collection_threads:
                thread.join(timeout=5)
        
        # Update status display
        if hasattr(self, 'status_label'):
            self.status_label.config(text="🛑 DATA COLLECTION STOPPED",
                                    bg='lightcoral', fg='darkred')

        logger.info("Data collection stopped")
    
    def run(self):
        """Run GUI main loop"""
        logger.info("🚀 Starting GUI main loop...")
        try:
            logger.info("🔧 Calling root.mainloop()...")
            self.root.mainloop()
            logger.info("✅ GUI main loop ended normally")
        except KeyboardInterrupt:
            logger.info("⚠️ KeyboardInterrupt received")
            self.stop_collection()
        except Exception as e:
            logger.error(f"❌ GUI error: {e}")
            raise
        finally:
            logger.info("🔧 GUI cleanup...")
            if hasattr(self, 'collection_threads'):
                for thread in self.collection_threads:
                    thread.stop()
            logger.info("✅ GUI cleanup complete")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threaded warehouse camera data collection')
    parser.add_argument('--output-dir', type=str, default=Config.RAW_COLLECTION_DIR,
                       help='Output directory for collected images')
    parser.add_argument('--frame-skip', type=int, default=Config.COLLECTION_FRAME_SKIP,
                       help='Number of frames to skip between saves (default: 1200 = 1min at 20fps)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (headless mode)')
    
    args = parser.parse_args()
    
    if args.no_gui:
        # Headless mode - just run collection threads
        logger.info("Starting headless data collection...")
        
        output_dir = args.output_dir
        threads = []
        
        try:
            # Create and start threads
            for camera_id in range(1, 12):
                thread = CameraCollectionThread(camera_id=camera_id, output_dir=output_dir)
                thread.frame_skip_count = args.frame_skip
                threads.append(thread)
                thread.start()
            
            logger.info("All camera threads started. Press Ctrl+C to stop.")
            
            # Wait for keyboard interrupt
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping collection...")
            for thread in threads:
                thread.stop()
            
            for thread in threads:
                thread.join(timeout=5)
            
            logger.info("Collection stopped")
    
    else:
        # GUI mode
        logger.info("Starting GUI data collection...")
        logger.info("🔧 Creating CollectionGUI instance...")
        try:
            gui = CollectionGUI()
            logger.info("✅ CollectionGUI created successfully")
            logger.info("🚀 Starting GUI...")
            gui.run()
        except Exception as e:
            logger.error(f"❌ Failed to create or run GUI: {e}")
            raise

if __name__ == "__main__":
    main()
