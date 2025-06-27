#!/usr/bin/env python3
"""
Smart Queue Camera 8 Test - Advanced Frame Queue Management + Database Integration
Always processes the newest frame, discards old frames to prevent latency buildup
Implements circular buffer with automatic frame dropping for real-time processing
Stores detection results in MongoDB for frontend display
"""

import cv2
import time
import threading
import queue
from collections import deque
from detector_tracker import DetectorTracker
from config import Config
from lense_correct2 import OptimizedFisheyeCorrector
from database_handler import DatabaseHandler
from datetime import datetime

class SmartFrameQueue:
    """Smart frame queue that always provides the newest frame"""
    
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.frame_buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frames_dropped = 0
        self.frames_added = 0
        
    def put_frame(self, frame):
        """Add frame to queue, dropping old frames if necessary"""
        with self.lock:
            if len(self.frame_buffer) >= self.max_size:
                # Queue is full, drop oldest frame
                dropped_frame = self.frame_buffer.popleft()
                self.frames_dropped += 1
            
            self.frame_buffer.append({
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self.frames_added
            })
            self.frames_added += 1
    
    def get_newest_frame(self):
        """Get the newest frame, return None if empty"""
        with self.lock:
            if not self.frame_buffer:
                return None
            
            # Always return the newest (rightmost) frame
            newest = self.frame_buffer[-1]
            
            # Clear all old frames to prevent processing backlog
            self.frame_buffer.clear()
            
            return newest
    
    def get_stats(self):
        """Get queue statistics"""
        with self.lock:
            return {
                'queue_size': len(self.frame_buffer),
                'frames_added': self.frames_added,
                'frames_dropped': self.frames_dropped,
                'drop_rate': self.frames_dropped / max(self.frames_added, 1)
            }

class FrameCapture:
    """Threaded frame capture to prevent blocking"""
    
    def __init__(self, camera_url, frame_queue):
        self.camera_url = camera_url
        self.frame_queue = frame_queue
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.frames_captured = 0
        self.capture_errors = 0
        
    def start(self):
        """Start capture thread"""
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal camera buffer
        
        if not self.cap.isOpened():
            print("❌ Failed to connect to camera")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✅ Frame capture thread started")
        return True
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                self.capture_errors += 1
                
                if consecutive_failures >= max_failures:
                    print("🔄 Attempting to reconnect camera...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.camera_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                    
                    if self.cap.isOpened():
                        print("✅ Camera reconnected!")
                    else:
                        print("❌ Reconnection failed, retrying...")
                        time.sleep(3)
                continue
            
            consecutive_failures = 0
            self.frames_captured += 1
            
            # Add frame to smart queue
            self.frame_queue.put_frame(frame)
    
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("🛑 Frame capture stopped")
    
    def get_stats(self):
        """Get capture statistics"""
        return {
            'frames_captured': self.frames_captured,
            'capture_errors': self.capture_errors,
            'error_rate': self.capture_errors / max(self.frames_captured, 1)
        }

def main():
    print("🧠 SMART QUEUE CAMERA 8 TEST - REAL-TIME FRAME PROCESSING + DATABASE")
    print("=" * 70)
    print("Camera 8 Coverage: Column 1 Top (10-70ft, 0-22.5ft)")
    print("Origin: Top-right (0,0), Bottom-left (180,90)")
    print("Features: Smart queue + newest frame processing + auto-drop + MongoDB")
    print("Press 'q' to quit")
    print("=" * 70)
    
    # Initialize components
    tracker = DetectorTracker()
    tracker.set_camera_id(8)
    tracker.confidence_threshold = 0.15  # Lowered for better detection
    
    fisheye_corrector = OptimizedFisheyeCorrector(Config.FISHEYE_LENS_MM)
    print(f"✅ Fisheye corrector initialized for {Config.FISHEYE_LENS_MM}mm lens")
    
    # Initialize database handler
    try:
        db_handler = DatabaseHandler()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️  Continuing without database - frontend won't see objects")
        db_handler = None
    
    # Smart frame queue
    frame_queue = SmartFrameQueue(max_size=5)  # Keep max 5 frames
    
    # Threaded frame capture
    camera_url = Config.RTSP_CAMERA_URLS[8]
    frame_capture = FrameCapture(camera_url, frame_queue)
    
    if not frame_capture.start():
        print("❌ Failed to start camera capture")
        return
    
    print("✅ Smart queue camera system started")
    print("🎯 Processing newest frames only...")
    
    # Processing control
    frame_count = 0
    detection_count = 0
    skip_frames = 15  # Process every 16th detection cycle
    last_tracked_objects = []
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    processing_times = []
    frame_age_times = []
    db_store_times = []
    
    # Detection timing control
    last_detection_time = 0
    detection_interval = 1.0 / 3  # Target 3 detections per second
    
    # Database storage timing
    last_db_store_time = 0
    db_store_interval = 2.0  # Store to database every 2 seconds

    try:
        while True:
            # Get newest frame from queue
            frame_data = frame_queue.get_newest_frame()
            
            if frame_data is None:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            frame = frame_data['frame']
            frame_timestamp = frame_data['timestamp']
            frame_age = time.time() - frame_timestamp
            frame_age_times.append(frame_age)
            if len(frame_age_times) > 100:
                frame_age_times.pop(0)
            
            frame_count += 1
            fps_counter += 1
            
            # Apply fisheye correction
            corrected_frame = fisheye_corrector.correct(frame)
            
            # Smart detection timing - based on time rather than frame count
            current_time = time.time()
            should_detect = (current_time - last_detection_time) >= detection_interval
            
            if should_detect:
                process_start = time.time()
                
                # Run detection on newest frame
                tracked_objects, perf_stats = tracker.process_frame(corrected_frame)
                last_tracked_objects = tracked_objects
                detection_count += 1
                last_detection_time = current_time
                
                process_time = time.time() - process_start
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Store objects to database periodically
                should_store_db = (db_handler is not None and 
                                   tracked_objects and 
                                   (current_time - last_db_store_time) >= db_store_interval)
                
                if should_store_db:
                    db_start = time.time()
                    try:
                        # Prepare objects for database storage
                        db_objects = []
                        for obj in tracked_objects:
                            # Create database object structure
                            db_obj = {
                                'persistent_id': obj.get('persistent_id', 0),
                                'camera_id': 8,
                                'frame_number': frame_count,
                                'confidence': obj.get('confidence', 0.0),
                                'object_class': obj.get('class_name', 'person'),
                                'bbox': obj.get('bbox', [0, 0, 0, 0]),
                                'real_center': obj.get('real_center', [None, None]),
                                'real_center_x': obj.get('real_center', [None, None])[0],
                                'real_center_y': obj.get('real_center', [None, None])[1],
                                'pixel_center': obj.get('center', [0, 0]),
                                'first_seen': datetime.now(),
                                'last_seen': datetime.now(),
                                'times_seen': obj.get('times_seen', 1),
                                'age_seconds': obj.get('age_seconds', 0.0),
                                'source': 'smart_queue_camera8'
                            }
                            
                            # Only add objects with valid coordinates
                            if (db_obj['real_center_x'] is not None and 
                                db_obj['real_center_y'] is not None):
                                db_objects.append(db_obj)
                        
                        # Store objects in database
                        if db_objects:
                            # Use upsert to update existing objects
                            for db_obj in db_objects:
                                db_handler.upsert_object(db_obj)
                            
                            print(f"💾 Stored {len(db_objects)} objects to database")
                        
                        last_db_store_time = current_time
                        
                    except Exception as e:
                        print(f"❌ Database storage error: {e}")
                    
                    db_time = time.time() - db_start
                    db_store_times.append(db_time)
                    if len(db_store_times) > 10:
                        db_store_times.pop(0)
            else:
                # Use last detection results
                tracked_objects = last_tracked_objects
            
            # Draw detections
            display_frame = tracker.draw_tracked_objects(corrected_frame, tracked_objects)
            display_frame = tracker.draw_calibrated_zone_overlay(display_frame)
            
            # Calculate FPS
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            else:
                fps = fps_counter / max(current_time - last_fps_time, 0.001)
            
            # Get statistics
            queue_stats = frame_queue.get_stats()
            capture_stats = frame_capture.get_stats()
            avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
            avg_frame_age = sum(frame_age_times) / len(frame_age_times) if frame_age_times else 0
            avg_db_time = sum(db_store_times) / len(db_store_times) if db_store_times else 0
            
            # Status display
            status_text = f"Objects: {len(tracked_objects)} | FPS: {fps:.1f} | Age: {avg_frame_age*1000:.1f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Queue stats
            queue_text = f"Queue: {queue_stats['queue_size']}/5 | Dropped: {queue_stats['frames_dropped']} ({queue_stats['drop_rate']:.1%})"
            cv2.putText(display_frame, queue_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Processing stats with database info
            db_status = "ON" if db_handler else "OFF"
            process_text = f"Process: {avg_process_time*1000:.1f}ms | DB: {db_status} ({avg_db_time*1000:.1f}ms)"
            cv2.putText(display_frame, process_text, (10, display_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Detection rate
            det_text = f"Det Rate: {detection_count/max(frame_count, 1):.1%} | DB Stores: {len(db_store_times)}"
            cv2.putText(display_frame, det_text, (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print coordinates when detecting
            if should_detect and tracked_objects:
                print(f"\n🧠 Frame {frame_count}: {len(tracked_objects)} objects | Age: {frame_age*1000:.1f}ms | Process: {avg_process_time*1000:.1f}ms")
                for i, obj in enumerate(tracked_objects):
                    real_center = obj.get('real_center')
                    confidence = obj.get('confidence', 0)
                    if real_center and real_center[0] is not None:
                        x, y = real_center
                        print(f"  Object {i+1}: Global ({x:.1f}ft, {y:.1f}ft) | Conf: {confidence:.2f}")
            
            # Resize for display
            height, width = display_frame.shape[:2]
            display_width = 1280
            display_height = int(height * (display_width / width))
            display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Display
            cv2.imshow('Camera 8 - Smart Queue (Real-time + Database)', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                break
            elif key == ord('+') or key == ord('='):
                # Decrease detection rate (slower detection, faster display)
                detection_interval = min(detection_interval * 1.2, 2.0)
                print(f"🔧 Detection interval: {detection_interval:.2f}s ({1/detection_interval:.1f} Hz)")
            elif key == ord('-'):
                # Increase detection rate (faster detection)
                detection_interval = max(detection_interval / 1.2, 0.1)
                print(f"🔧 Detection interval: {detection_interval:.2f}s ({1/detection_interval:.1f} Hz)")
            elif key == ord('r'):
                # Reset detection rate
                detection_interval = 1.0 / 3
                print(f"🔧 Detection rate reset: {1/detection_interval:.1f} Hz")
            elif key == ord('d'):
                # Toggle database storage interval
                if db_store_interval == 2.0:
                    db_store_interval = 1.0
                    print("🔧 Database storage: FAST (1s)")
                elif db_store_interval == 1.0:
                    db_store_interval = 5.0
                    print("🔧 Database storage: SLOW (5s)")
                else:
                    db_store_interval = 2.0
                    print("🔧 Database storage: NORMAL (2s)")

    finally:
        # Cleanup
        frame_capture.stop()
        cv2.destroyAllWindows()
        tracker.cleanup()
        
        if db_handler:
            db_handler.close_connection()
            print("💾 Database connection closed")
        
        # Performance summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_frame_age = sum(frame_age_times) / len(frame_age_times) if frame_age_times else 0
        avg_db_time = sum(db_store_times) / len(db_store_times) if db_store_times else 0
        
        print(f"\n🧠 Smart queue test completed")
        print(f"📊 Performance Summary:")
        print(f"   Display FPS: {avg_fps:.1f}")
        print(f"   Detection Rate: {detection_count/max(total_time, 1):.1f} Hz")
        print(f"   Avg Process Time: {avg_process_time*1000:.1f}ms")
        print(f"   Avg Frame Age: {avg_frame_age*1000:.1f}ms")
        print(f"   Avg DB Store Time: {avg_db_time*1000:.1f}ms")
        print(f"   Frames Captured: {capture_stats['frames_captured']}")
        print(f"   Queue Drops: {queue_stats['frames_dropped']} ({queue_stats['drop_rate']:.1%})")
        print(f"   DB Stores: {len(db_store_times)}")
        print(f"   Total Runtime: {total_time:.1f}s")

if __name__ == "__main__":
    main() 