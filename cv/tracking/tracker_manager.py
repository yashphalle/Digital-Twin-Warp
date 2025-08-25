import threading
import time
import logging
from ultralytics import YOLO

# Reuse FastGlobalIDManager for camera-prefixed IDs
from pipelines.gpu_processor_fast_tracking import FastGlobalIDManager
from modules.movement_monitor import MovementMonitor
from pipelines.crop_writer import CropWriter


class CameraTrackerThread(threading.Thread):
    """Per-camera tracker using Ultralytics .track with BoT-SORT."""

    def __init__(self, camera_id: int, latest_store, model_path: str, device: str,
                 confidence: float, botsort_yaml_path: str, output_queue, log_interval_s: float = 2.0,
                 debug: bool = False):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.latest_store = latest_store
        self.output_queue = output_queue
        self.log_interval_s = log_interval_s
        self.debug = debug
        self.logger = logging.getLogger(f"CameraTracker[{camera_id}]")

        # Model per camera (stateful tracker)
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf = confidence
        self.tracker_cfg = botsort_yaml_path

        self.id_manager = FastGlobalIDManager(camera_id)
        # Movement-based crop components
        self.movement_monitor = MovementMonitor(camera_id, threshold_ft=5.0, cooldown_sec=120.0, max_crops_per_track=5, padding_px=32, force_test_mode=False, established_age_frames=5)
        # In GCP-first mode, CropWriter will upload directly to GCS (controlled by USE_GCS_CROPS env)
        self.crop_writer = CropWriter(camera_id, base_path='images', queue_size=20, jpg_quality=85)
        self.crop_writer.start()

        self._last_ts: float = 0.0
        self._running = False

        self._frames = 0
        self._frames_total = 0
        self._last_log_t = 0.0
        self._last_frame_count = 0

    def stop(self):
        self._running = False
        try:
            self.crop_writer.stop(wait=True)
        except Exception:
            pass

    def _parse_results(self, results) -> list:
        tracks = []
        if not results:
            return tracks
        r0 = results[0]
        boxes = getattr(r0, 'boxes', None)
        if boxes is None or getattr(boxes, 'id', None) is None:
            return tracks
        try:
            for i in range(len(boxes)):
                y_id = int(boxes.id[i].cpu().item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                global_id = self.id_manager.get_global_id(y_id)
                age = self.id_manager.get_track_age(global_id)
                tracks.append({
                    'camera_id': self.camera_id,
                    'track_id': y_id,
                    'global_id': global_id,
                    'age': age,
                    'bbox': bbox,
                    'confidence': conf,
                    'class': self.model.names.get(cls, f'class_{cls}')
                })
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
        return tracks

    def run(self):
        self._running = True
        while self._running:
            item = self.latest_store.latest(self.camera_id)
            if item is None:
                time.sleep(0.005)
                continue
            frame, ts = item
            if ts <= self._last_ts:
                time.sleep(0.002)
                continue
            try:
                results = self.model.track(
                    frame,
                    device=self.device,
                    conf=self.conf,
                    tracker=self.tracker_cfg,
                    persist=True,
                    verbose=False,
                )
            except Exception as e:
                self.logger.error(f"track() failed: {e}")
                time.sleep(0.01)
                continue
            tracks = self._parse_results(results)
            now = time.time()
            self._last_ts = ts

            # Non-blocking crop trigger per track
            try:
                # Movement check uses bbox and age; we need frame shape which is known here
                fshape = frame.shape
                for t in tracks:
                    # Normalize fields for monitor
                    det = {
                        'persistent_id': int(t.get('global_id')),
                        'bbox': t.get('bbox'),
                        'track_age': int(t.get('age', 0)),
                        'camera_id': self.camera_id,
                    }
                    job = self.movement_monitor.check_and_build_job(det, fshape, now_ts_ms=int(ts*1000))
                    if job is not None:
                        x1, y1, x2, y2 = det['bbox']
                        pad = 32
                        h, w = fshape[:2]
                        px1 = max(0, int(x1 - pad))
                        py1 = max(0, int(y1 - pad))
                        px2 = min(w, int(x2 + pad))
                        py2 = min(h, int(y2 + pad))
                        if px2 > px1 and py2 > py1:
                            roi = frame[py1:py2, px1:px2].copy()
                            job['roi'] = roi
                            job['camera_id'] = self.camera_id
                            ok = self.crop_writer.enqueue(job)
                            if not ok:
                                self.logger.warning(f"Crop enqueue failed for cam={self.camera_id} pid={det['persistent_id']}")
            except Exception as e:
                self.logger.warning(f"Crop path error cam={self.camera_id}: {e}")

            # push tracks
            try:
                if self.output_queue.full():
                    _ = self.output_queue.get_nowait()
                self.output_queue.put_nowait({'camera_id': self.camera_id, 'tracks': tracks, 'ts': ts})
            except Exception:
                pass

            self._frames += 1
            if now - self._last_log_t >= max(0.5, self.log_interval_s):
                # true tracking FPS over the last interval
                dt = now - (self._last_log_t or now)
                trk_fps = self._frames / max(1e-6, dt)
                self._frames_total += self._frames
                self.logger.info(f"trk_fps={trk_fps:.1f} active_tracks={len(tracks)}")
                self._frames = 0
                self._last_log_t = now


class TrackerManager:
    """Manager for per-camera tracker threads."""

    def __init__(self, camera_ids: list, latest_store, output_queue, model_path: str,
                 device: str, confidence: float, botsort_yaml_path: str, log_interval_s: float = 2.0,
                 debug: bool = False):
        self.threads = [
            CameraTrackerThread(cid, latest_store, model_path, device, confidence, botsort_yaml_path,
                                 output_queue, log_interval_s, debug)
            for cid in camera_ids
        ]

    def start(self):
        for t in self.threads:
            t.start()

    def stop(self):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join(timeout=2)

