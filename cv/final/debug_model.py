from inference import get_model
import supervision as sv
import cv2
import time

print("🎥 WARP Computer Vision System - Video Processing")
print("=" * 55)

# Configuration
API_KEY = "jpGF414JaerhGzPz651h"

MODEL_ID = "warp-computer-vision-system/5"
VIDEO_PATH = "cam7.mp4"
OUTPUT_VIDEO = "cam7_warp_detected.mp4"
CONFIDENCE_THRESHOLD = 0.25

def get_model_instance():
    """Get a model instance for testing"""
    return get_model(model_id=MODEL_ID, api_key=API_KEY)

def load_test_frames(video_path: str, max_frames: int = 100):
    """Load frames from video for testing"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

try:
    print("🔑 Setting up API authentication...")
    
    # Load your WARP model
    print(f"📥 Loading model: {MODEL_ID}")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    print("✅ Model loaded successfully!")
    
    # Open video
    print(f"🎬 Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        exit()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video Properties:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • FPS: {fps}")
    print(f"   • Total Frames: {total_frames}")
    print(f"   • Duration: {duration:.1f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Setup annotators
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
    
    # Processing variables
    frame_count = 0
    total_detections = 0
    processing_times = []
    start_time = time.time()
    
    print(f"\n🔄 Processing video...")
    print("Press Ctrl+C to stop processing early")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        frame_start = time.time()
        results = model.infer(frame)[0]
        frame_end = time.time()
        
        inference_time = frame_end - frame_start
        processing_times.append(inference_time)
        
        # Filter detections by confidence
        filtered_predictions = [
            pred for pred in results.predictions 
            if pred.confidence >= CONFIDENCE_THRESHOLD
        ]
        
        frame_detections = len(filtered_predictions)
        total_detections += frame_detections
        
        # Create supervision detections object
        detections = sv.Detections.from_inference(results)
        
        # Filter detections by confidence in supervision format
        detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
        
        # Annotate frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Add frame info overlay
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections} | FPS: {1/inference_time:.1f}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            avg_fps = 1 / (sum(processing_times[-30:]) / len(processing_times[-30:]))
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            
            print(f"   📈 Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | ETA: {eta:.0f}s")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_inference_time = sum(processing_times) / len(processing_times)
    avg_fps = 1 / avg_inference_time
    
    print(f"\n🎉 Video Processing Complete!")
    print("=" * 55)
    print(f"📊 Processing Summary:")
    print(f"   • Input Video: {VIDEO_PATH}")
    print(f"   • Output Video: {OUTPUT_VIDEO}")
    print(f"   • Frames Processed: {frame_count}")
    print(f"   • Total Processing Time: {total_time:.1f} seconds")
    print(f"   • Average Inference FPS: {avg_fps:.1f}")
    print(f"   • Total Detections: {total_detections}")
    print(f"   • Average Detections per Frame: {total_detections/frame_count:.1f}")
    print(f"   • Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    # Detection breakdown
    print(f"\n🎯 Detection Statistics:")
    
    # Quick sample to show what was detected
    print("   Sample detections from last frame:")
    if filtered_predictions:
        class_counts = {}
        for pred in filtered_predictions:
            class_name = pred.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"     • {class_name}: {count}")
    else:
        print("     • No detections in final frame")
    
    print(f"\n✅ Annotated video saved as: {OUTPUT_VIDEO}")
    print("🎬 You can now open the output video to see detections!")

except KeyboardInterrupt:
    print(f"\n⏹️ Processing stopped by user at frame {frame_count}")
    cap.release()
    out.release()
    print(f"✅ Partial video saved as: {OUTPUT_VIDEO}")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    
    # Cleanup on error
    if 'cap' in locals():
        cap.release()
    if 'out' in locals():
        out.release()