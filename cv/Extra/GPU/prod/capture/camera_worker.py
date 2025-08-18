import threading
import logging
import time
from typing import Tuple
import cv2

from GPU.modules.camera_manager import CPUCameraManager
from GPU.modules.fisheye_corrector import OptimizedFisheyeCorrector
from GPU.configs.config import Config
from GPU.configs.warehouse_config import get_warehouse_config


class CameraWorker(threading.Thread):
    """Per-camera capture thread: frame skip -> fisheye -> resize -> publish latest."""

    def __init__(self, camera_id: int, latest_store, frame_skip: int = 4, resize: Tuple[int, int] = (1280, 720),
                 debug: bool = False):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.latest_store = latest_store
        self.frame_skip = max(1, frame_skip)
        self.resize = resize
        self.debug = debug
        self._running = False

        # Remote URLs preferred
        Config.switch_to_remote_cameras()
        self.rtsp_url = Config.RTSP_CAMERA_URLS.get(camera_id)
        if not self.rtsp_url:
            raise ValueError(f"No RTSP URL for camera {camera_id}")

        # Camera display name
        self.warehouse_config = get_warehouse_config()
        self.camera_zone = self.warehouse_config.camera_zones.get(camera_id)
        self.camera_name = self.camera_zone.camera_name if self.camera_zone else f"Camera {camera_id}"

        self.cam_mgr = CPUCameraManager(camera_id=camera_id, rtsp_url=self.rtsp_url, camera_name=self.camera_name)
        self.fisheye = OptimizedFisheyeCorrector(lens_mm=2.8)

        self._frame_counter = 0
        self._processed = 0
        self._last_log_t = 0.0
        self._last_proc = 0
        self.logger = logging.getLogger(f"CameraWorker[{camera_id}]")

    def connect(self) -> bool:
        ok = self.cam_mgr.connect_camera()
        if ok:
            self.logger.info("Connected")
        else:
            self.logger.error("Failed to connect")
        return ok

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self.cam_mgr.read_frame()
            if not ret:
                time.sleep(0.05)
                continue

            self._frame_counter += 1
            # Frame skip first to save compute
            if self._frame_counter % self.frame_skip != 1:
                continue

            # Fisheye then resize on kept frames
            try:
                frame = self.fisheye.correct(frame)
            except Exception:
                pass
            if self.resize and (frame.shape[1], frame.shape[0]) != self.resize:
                frame = cv2.resize(frame, self.resize)

            # Use capture timestamp for precise sync across pipeline components
            capture_ts = time.time()
            self.latest_store.put(self.camera_id, frame, ts=capture_ts)
            self._processed += 1

            # periodic ingest FPS (unique post-skip, post-fisheye, post-resize)
            now = time.time()
            if now - self._last_log_t >= 2.0:
                proc_delta = self._processed - self._last_proc
                fps = proc_delta / max(1e-6, (now - self._last_log_t))
                self.logger.info(f"ingest_fps={fps:.1f}")
                self._last_log_t = now
                self._last_proc = self._processed

            if self.debug and self._processed % 30 == 0:
                self.logger.debug(f"Processed={self._processed}, skipped={self._frame_counter - self._processed}")

        self.cam_mgr.cleanup_camera()
        self.logger.info("Stopped")

