#!/usr/bin/env python3

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ultralytics import YOLO
import cv2
import time
import torch
import argparse
from pathlib import Path
from configs.config import Config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Custom YOLOv8 Model - Camera RTSP Processing')
    parser.add_argument('--local', action='store_true',
                       help='Use local camera URLs (192.168.x.x) instead of remote URLs')
    parser.add_argument('--remote', action='store_true',
                       help='Use remote camera URLs (104.181.138.5) instead of local URLs')
    parser.add_argument('--cameras', type=int, nargs='+', default=[7, 8],
                       help='Camera IDs to use (default: 7 8)')
    parser.add_argument('--camera', type=int,
                       help='Single camera ID (overrides --cameras if specified)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames to process (default: 1000, 0 for unlimited)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (save video only)')
    parser.add_argument('--no-save', action='store_true',
                       help='Run with GUI only (do not save video)')
    parser.add_argument('--resize-to', type=int, default=1080,
                       help='Resize frame height to this value (default: 1080, 0 for no resize)')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Process every Nth frame (default: 2, 1 for no skipping)')
    return parser.parse_args()

def load_custom_yolo_model(model_path: str, device: str = "cuda:0"):
    """Load custom YOLOv8 model"""
    try:
        print(f"📥 Loading custom YOLOv8 model: {model_path}")
        
        # Check if model file exists
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("💡 Available .pt files in current directory:")
            for pt_file in Path(".").glob("*.pt"):
                print(f"   - {pt_file}")
            return None, None
        
        # Load the model
        model = YOLO(model_path)
        
        # Set device
        if torch.cuda.is_available() and "cuda" in device:
            print(f"🚀 Using GPU device: {device}")
        else:
            device = "cpu"
            print(f"⚠️ Using CPU device")
        
        print(f"✅ Custom YOLOv8 model loaded successfully!")
        print(f"   Model: {model_path}")
        print(f"   Device: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def draw_detections(frame, results, confidence_threshold=0.25):
    """Draw bounding boxes and labels on frame"""
    annotated_frame = frame.copy()
    
    if results and len(results) > 0:
        # Get the first result (single image inference)
        result = results[0]
        
        # Get boxes, confidences, and class IDs
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = result.names
            
            detection_count = 0
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf >= confidence_threshold:
                    detection_count += 1
                    
                    # Extract coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name
                    class_name = class_names[class_id] if class_id in class_names else f"Class_{class_id}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return annotated_frame, detection_count
    
    return annotated_frame, 0

def process_single_camera(camera_id, model, device, confidence_threshold, max_frames, show_gui, save_video):
    """Process a single camera stream"""
    # Get Camera RTSP URL
    camera_info = Config.get_camera_info(camera_id)
    rtsp_url = camera_info['current_url']
    output_video = f"cam{camera_id}_rtsp_custom_yolo_detected.mp4"

    print(f"\n📡 Camera {camera_id} Configuration:")
    print(f"   • RTSP URL: {rtsp_url}")
    print(f"   • Output: {output_video if save_video else 'No video saving'}")

    try:
        # Open RTSP stream with aggressive optimization
        print(f"📡 Connecting to Camera {camera_id} RTSP stream...")

        # Add RTSP transport options to reduce network issues
        rtsp_url_optimized = rtsp_url + "?tcp"  # Force TCP instead of UDP
        cap = cv2.VideoCapture(rtsp_url_optimized, cv2.CAP_FFMPEG)

        # Aggressive buffer and timeout settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # Minimal buffer
        cap.set(cv2.CAP_PROP_FPS, 15)                # Limit FPS to reduce load
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)      # Request lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)     # Request lower resolution

        # Set read timeout to prevent hanging
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not cap.isOpened():
            print(f"❌ Could not connect to Camera {camera_id} RTSP stream: {rtsp_url}")
            return None

        # Get stream properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Camera {camera_id} Stream Properties:")
        print(f"   • Resolution: {width}x{height}")
        print(f"   • FPS: {fps}")

        # Setup video writer (only if saving video)
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Setup GUI window (only if showing GUI)
        window_name = None
        if show_gui:
            window_name = f"Custom YOLOv8 - Camera {camera_id} Live Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)  # Smaller for multiple cameras

        return {
            'camera_id': camera_id,
            'cap': cap,
            'out': out,
            'window_name': window_name,
            'width': width,
            'height': height,
            'fps': fps,
            'rtsp_url': rtsp_url,
            'output_video': output_video,
            'frame_count': 0,
            'total_detections': 0,
            'processing_times': [],
            'start_time': time.time(),
            'frame_skip_counter': 0
        }

    except Exception as e:
        print(f"❌ Error setting up Camera {camera_id}: {e}")
        return None

def main():
    args = parse_arguments()

    print("🎥 Custom YOLOv8 Model - Multi-Camera RTSP Processing")
    print("=" * 60)

    # Configure camera URLs based on arguments
    if args.local:
        Config.switch_to_local_cameras()
    elif args.remote:
        Config.switch_to_remote_cameras()

    # Determine which cameras to use
    if args.camera:
        camera_ids = [args.camera]
    else:
        camera_ids = args.cameras

    # Configuration
    MODEL_PATH = "custom_yolo.pt"
    CONFIDENCE_THRESHOLD = args.confidence
    MAX_FRAMES = args.max_frames
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    SHOW_GUI = not args.no_gui
    SAVE_VIDEO = not args.no_save
    RESIZE_HEIGHT = args.resize_to
    FRAME_SKIP = args.frame_skip

    print(f"📡 Multi-Camera Configuration:")
    print(f"   • Camera IDs: {camera_ids}")
    print(f"   • URL Type: {'LOCAL' if Config.USE_LOCAL_CAMERAS else 'REMOTE'}")
    print(f"   • Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   • Max Frames: {MAX_FRAMES if MAX_FRAMES > 0 else 'Unlimited'}")
    print(f"   • GUI Display: {'ENABLED' if SHOW_GUI else 'DISABLED'}")
    print(f"   • Save Video: {'ENABLED' if SAVE_VIDEO else 'DISABLED'}")

    try:
        # Load custom YOLOv8 model
        model, device = load_custom_yolo_model(MODEL_PATH, DEVICE)
        if model is None:
            exit(1)

        # Setup all cameras
        cameras = []
        for camera_id in camera_ids:
            camera_setup = process_single_camera(
                camera_id, model, device, CONFIDENCE_THRESHOLD,
                MAX_FRAMES, SHOW_GUI, SAVE_VIDEO
            )
            if camera_setup:
                cameras.append(camera_setup)
            else:
                print(f"⚠️ Skipping Camera {camera_id} due to connection issues")

        if not cameras:
            print("❌ No cameras could be connected. Exiting.")
            exit(1)

        print(f"\n✅ Successfully connected to {len(cameras)} camera(s)")

        # Global processing variables
        global_start_time = time.time()
        total_frames_processed = 0
        total_frames_read = 0
        debug_counter = 0

        print(f"\n🔄 Processing {len(cameras)} RTSP streams with custom YOLOv8 model...")
        print(f"🔧 Debug: Frame skip = {FRAME_SKIP}, Resize = {RESIZE_HEIGHT}")
        if SHOW_GUI:
            print("Press 'q' in any GUI window or Ctrl+C to stop processing")
        else:
            print("Press Ctrl+C to stop processing")
        
        loop_count = 0
        max_loops = 10000  # Prevent infinite loops

        while loop_count < max_loops:
            loop_count += 1
            quit_requested = False
            active_cameras = 0

            # Process each camera
            for camera in cameras:
                ret, frame = camera['cap'].read()
                total_frames_read += 1

                if not ret or frame is None:
                    # Skip bad frames but don't be too aggressive
                    if debug_counter % 30 == 0:  # Print every 30 failed reads
                        print(f"⚠️ Camera {camera['camera_id']}: Frame read failed, skipping...")
                    continue

                # Basic frame validation (less strict)
                if frame.size == 0:
                    if debug_counter % 30 == 0:
                        print(f"⚠️ Camera {camera['camera_id']}: Empty frame, skipping...")
                    continue

                # Frame skipping to reduce network load
                camera['frame_skip_counter'] += 1
                if camera['frame_skip_counter'] % FRAME_SKIP != 0:
                    continue  # Skip this frame

                debug_counter += 1

                active_cameras += 1
                camera['frame_count'] += 1
                total_frames_processed += 1

                # Check max frames limit (per camera)
                if MAX_FRAMES > 0 and camera['frame_count'] > MAX_FRAMES:
                    print(f"📊 Camera {camera['camera_id']}: Reached maximum frame limit: {MAX_FRAMES}")
                    continue

                # Resize frame for better performance (if enabled)
                if RESIZE_HEIGHT > 0 and frame.shape[0] > RESIZE_HEIGHT:
                    scale_factor = RESIZE_HEIGHT / frame.shape[0]
                    new_width = int(frame.shape[1] * scale_factor)
                    frame = cv2.resize(frame, (new_width, RESIZE_HEIGHT))

                # Run YOLOv8 inference
                frame_start = time.time()
                results = model(frame, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False)
                frame_end = time.time()

                inference_time = frame_end - frame_start
                camera['processing_times'].append(inference_time)

                # Draw detections on frame
                annotated_frame, frame_detections = draw_detections(frame, results, CONFIDENCE_THRESHOLD)
                camera['total_detections'] += frame_detections

                # Add frame info overlay
                info_text = f"Cam {camera['camera_id']} | Frame: {camera['frame_count']} | Det: {frame_detections} | FPS: {1/inference_time:.1f}"
                cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add model info overlay
                model_text = f"Model: {MODEL_PATH} | Device: {device}"
                cv2.putText(annotated_frame, model_text, (10, camera['height'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Show live GUI
                if SHOW_GUI and camera['window_name']:
                    cv2.imshow(camera['window_name'], annotated_frame)

                # Write frame to output video (if saving)
                if SAVE_VIDEO and camera['out'] is not None:
                    camera['out'].write(annotated_frame)

            # Check for 'q' key press to quit (only if GUI is enabled)
            if SHOW_GUI:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 GUI quit requested by user")
                    quit_requested = True

            # Break if no active cameras or quit requested
            if active_cameras == 0 or quit_requested:
                if active_cameras == 0:
                    print(f"⚠️ No active cameras, terminating after {loop_count} loops")
                break

            # Progress update every 60 total frames (across all cameras)
            if total_frames_processed % 60 == 0:
                elapsed = time.time() - global_start_time
                total_detections = sum(cam['total_detections'] for cam in cameras)

                print(f"   📈 Total Frames: {total_frames_processed} | Elapsed: {elapsed:.0f}s | Total Detections: {total_detections}")

                # Show per-camera stats
                for camera in cameras:
                    if camera['processing_times']:
                        if len(camera['processing_times']) >= 30:
                            avg_fps = 1 / (sum(camera['processing_times'][-30:]) / len(camera['processing_times'][-30:]))
                        else:
                            avg_fps = 1 / (sum(camera['processing_times']) / len(camera['processing_times']))
                        print(f"      Cam {camera['camera_id']}: {camera['frame_count']} frames | {avg_fps:.1f} FPS | {camera['total_detections']} detections")
        
        # Cleanup all cameras
        for camera in cameras:
            camera['cap'].release()
            if camera['out'] is not None:
                camera['out'].release()

        if SHOW_GUI:
            cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - global_start_time
        total_frames = sum(cam['frame_count'] for cam in cameras)
        total_detections = sum(cam['total_detections'] for cam in cameras)

        print(f"\n🎉 Multi-Camera RTSP Stream Processing Complete!")
        print("=" * 60)
        print(f"📊 Processing Summary:")
        print(f"   • Custom Model: {MODEL_PATH}")
        print(f"   • Cameras Processed: {len(cameras)} ({[cam['camera_id'] for cam in cameras]})")
        print(f"   • Device Used: {device}")
        print(f"   • Total Frames Processed: {total_frames}")
        print(f"   • Total Processing Time: {total_time:.1f} seconds")
        print(f"   • Total Detections: {total_detections}")
        if total_frames > 0:
            print(f"   • Average Detections per Frame: {total_detections/total_frames:.1f}")
        else:
            print(f"   • Average Detections per Frame: N/A (no frames processed)")
        print(f"   • Confidence Threshold: {CONFIDENCE_THRESHOLD}")

        # Per-camera statistics
        print(f"\n📊 Per-Camera Statistics:")
        for camera in cameras:
            print(f"   Camera {camera['camera_id']}:")
            print(f"      • Frames: {camera['frame_count']}")
            print(f"      • Detections: {camera['total_detections']}")

            if camera['processing_times'] and len(camera['processing_times']) > 0:
                avg_inference_time = sum(camera['processing_times']) / len(camera['processing_times'])
                avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
                print(f"      • Inference FPS: {avg_fps:.1f}")
            else:
                print(f"      • Inference FPS: N/A")

            if total_time > 0:
                actual_fps = camera['frame_count'] / total_time
                print(f"      • Actual Camera FPS: {actual_fps:.1f}")
            else:
                print(f"      • Actual Camera FPS: N/A")

            if camera['frame_count'] > 0:
                print(f"      • Detections/Frame: {camera['total_detections']/camera['frame_count']:.1f}")
            else:
                print(f"      • Detections/Frame: N/A")

            if SAVE_VIDEO:
                print(f"      • Output Video: {camera['output_video']}")

        if SAVE_VIDEO:
            print(f"\n✅ Annotated videos saved for all cameras!")
            print("🎬 You can now open the output videos to see custom YOLOv8 detections!")
        else:
            print(f"\n🖥️ Live GUI session completed - no videos saved")

    except KeyboardInterrupt:
        total_frames = sum(cam['frame_count'] for cam in cameras) if 'cameras' in locals() else 0
        print(f"\n⏹️ Processing stopped by user at {total_frames} total frames")

        # Cleanup all cameras
        if 'cameras' in locals():
            for camera in cameras:
                if camera['cap']:
                    camera['cap'].release()
                if camera['out'] is not None:
                    camera['out'].release()

        if SHOW_GUI:
            cv2.destroyAllWindows()

        if SAVE_VIDEO and 'cameras' in locals():
            print(f"✅ Partial videos saved for {len(cameras)} cameras")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        if 'cameras' in locals():
            for camera in cameras:
                if camera['cap']:
                    camera['cap'].release()
                if camera['out'] is not None:
                    camera['out'].release()

        if SHOW_GUI:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
