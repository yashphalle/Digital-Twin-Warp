
import os
import cv2
import threading
import numpy as np
import time

# --- IMPORTANT: Set environment variable to force TCP transport ---
# This MUST be done before importing cv2 or initializing VideoCapture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Dictionary of RTSP stream URLs provided
REMOTE_RTSP_CAMERA_URLS = {
    # Row 1
    1: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/101",
    2: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/201",
    3: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/301",
    4: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/401",
    # Row 2
    5: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/501",
    6: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/601",
    7: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/701",
    # Row 3
    8: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/801",
    9: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/901",
    10: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/1001",
    11: "rtsp://admin:wearewarp!@104.13.230.137:6554/Streaming/Channels/1101"
}

# --- Threading Class for Camera Stream ---
class CameraThread(threading.Thread):
    def __init__(self, thread_id, url):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.url = url
        self.latest_frame = None
        self.is_running = True
        self.reconnect_delay = 5  # seconds
        self.capture = cv2.VideoCapture(self.url)

    def run(self):
        print(f"Starting camera thread {self.thread_id} for URL: {self.url}")
        while self.is_running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    self.latest_frame = frame
                else:
                    print(f"Error reading frame from camera {self.thread_id}. Attempting to reconnect...")
                    self.capture.release()
                    time.sleep(self.reconnect_delay)
                    self.capture = cv2.VideoCapture(self.url)
            else:
                print(f"Camera {self.thread_id} disconnected. Attempting to reconnect...")
                time.sleep(self.reconnect_delay)
                self.capture = cv2.VideoCapture(self.url)

        self.capture.release()
        print(f"Camera thread {self.thread_id} stopped.")

    def stop(self):
        self.is_running = False

# --- Main Application ---
def main():
    # --- Configuration ---
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    GRID_COLS = 4
    GRID_ROWS = 3

    blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank_frame, 'NO SIGNAL', (FRAME_WIDTH // 3, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    threads = []
    for cam_id, url in REMOTE_RTSP_CAMERA_URLS.items():
        thread = CameraThread(cam_id, url)
        thread.start()
        threads.append(thread)

    try:
        while True:
            grid_rows_list = []
            for r in range(GRID_ROWS):
                row_frames = []
                for c in range(GRID_COLS):
                    cam_index = r * GRID_COLS + c + 1
                    frame = None
                    for t in threads:
                        if t.thread_id == cam_index:
                            frame = t.latest_frame
                            break
                    
                    if frame is None:
                        display_frame = blank_frame.copy()
                        # Add cam index text to blank frame for identification
                        cv2.putText(display_frame, f'CAM {cam_index}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                    
                    row_frames.append(display_frame)
                
                grid_rows_list.append(np.hstack(row_frames))

            final_grid = np.vstack(grid_rows_list)

            cv2.imshow('Multi Camera RTSP Viewer (TCP Mode)', final_grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping all camera threads...")
        for thread in threads:
            thread.stop()
        for thread in threads:
            thread.join()
        
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()
