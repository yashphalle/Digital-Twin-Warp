import threading
import time
import logging
from typing import Dict, List, Tuple
import torch
import numpy as np
from ultralytics import YOLO

from config.runtime import RuntimeConfig


class DetectorWorker(threading.Thread):
    """Cross-camera dynamic batching detector (FP32)."""

    def __init__(self, cfg: RuntimeConfig, latest_store, output_queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.latest_store = latest_store
        self.output_queue = output_queue
        self.logger = logging.getLogger("DetectorWorker")
        self._running = False

        # Load model (FP32)
        self.model = YOLO(cfg.model_path)
        self.model.to(cfg.device)
        # No .half(); keep FP32 as per preference

        # Track last processed timestamp per camera to avoid reprocessing the same frame
        self._last_ts: Dict[int, float] = {cid: 0.0 for cid in (cfg.active_cameras or [])}

        self._last_log_t = 0.0
        self._frames_proc = 0
        self._batches = 0
        self._inf_time = 0.0

    def stop(self):
        self._running = False

    def _collect_batch(self) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """Build a batch using the latest available frame per camera.
        Modes:
          - latest_full: include latest frame for each camera immediately (may include duplicates vs last_ts)
          - new_only: include only frames with ts > last_ts within a small window
        """
        latest_map = self.latest_store.latest_many(self.cfg.active_cameras)
        frames: List[np.ndarray] = []
        cam_ids: List[int] = []
        ts_list: List[float] = []

        if getattr(self.cfg, 'detection_batch_mode', 'latest_full') == 'latest_full':
            # Take the latest available frame for each camera (no waiting)
            for cid in self.cfg.active_cameras:
                item = latest_map.get(cid)
                if not item:
                    continue  # skip cams with no frame yet
                frame, ts = item
                frames.append(frame)
                cam_ids.append(cid)
                ts_list.append(ts)
                if len(frames) >= self.cfg.max_batch:
                    break
            return frames, cam_ids, ts_list

        # Fallback: new_only with small window (for completeness)
        end_t = time.time() + max(0.0, self.cfg.batch_window_ms) / 1000.0
        seen = set()
        while time.time() < end_t and len(frames) < self.cfg.max_batch and len(seen) < len(self.cfg.active_cameras):
            latest_map = self.latest_store.latest_many(self.cfg.active_cameras)
            for cid in self.cfg.active_cameras:
                if cid in seen:
                    continue
                item = latest_map.get(cid)
                if not item:
                    continue
                frame, ts = item
                if ts <= self._last_ts.get(cid, 0.0):
                    continue
                frames.append(frame)
                cam_ids.append(cid)
                ts_list.append(ts)
                seen.add(cid)
                if len(frames) >= self.cfg.max_batch:
                    break
            time.sleep(0.003)
        return frames, cam_ids, ts_list

    def run(self):
        self._running = True
        while self._running:
            frames, cam_ids, ts_list = self._collect_batch()
            if not frames:
                time.sleep(0.005)
                continue

            # If in latest_full mode, we may have included cameras whose ts did not advance
            latest_full = getattr(self.cfg, 'detection_batch_mode', 'latest_full') == 'latest_full'

            t0 = time.time()
            try:
                results = self.model(
                    frames,
                    conf=self.cfg.confidence,
                    verbose=False,
                    stream=False,
                )
            except Exception as e:
                self.logger.error(f"Detection failed: {e}")
                time.sleep(0.01)
                continue

            inf_dt = time.time() - t0
            self._inf_time += inf_dt
            self._batches += 1

            # Demux per camera and push
            now = time.time()
            for idx, cid in enumerate(cam_ids):
                dets = []
                try:
                    r = results[idx]
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes = r.boxes
                        for i in range(len(boxes)):
                            bbox = boxes.xyxy[i].cpu().numpy().tolist()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls = int(boxes.cls[i].cpu().numpy())
                            if conf >= float(self.cfg.confidence):  # explicit post-filter
                                dets.append({
                                    'camera_id': cid,
                                    'bbox': bbox,
                                    'confidence': conf,
                                    'class': self.model.names.get(cls, f'class_{cls}')
                                })
                except Exception as e:
                    self.logger.error(f"Result parse failed for cam {cid}: {e}")

                # Only forward unique frames (drop duplicates in latest_full mode)
                emit = True
                if latest_full:
                    ts = ts_list[idx]
                    if ts <= self._last_ts.get(cid, 0.0):
                        emit = False
                if emit:
                    try:
                        if self.output_queue.full():
                            _ = self.output_queue.get_nowait()
                        self.output_queue.put_nowait({'camera_id': cid, 'detections': dets, 'ts': ts_list[idx]})
                        # mark last processed ts for uniqueness gating
                        self._last_ts[cid] = max(self._last_ts.get(cid, 0.0), ts_list[idx])
                    except Exception:
                        pass
                    self._frames_proc += 1

            now = time.time()
            if now - self._last_log_t >= max(0.5, self.cfg.log_interval_s):
                avg_inf_ms = (self._inf_time / max(1, self._batches)) * 1000.0
                # frames_proc here equals number of camera outputs emitted
                fps_total = self._frames_proc / max(1e-6, now - (self._last_log_t or now))
                self.logger.info(f"batch={len(frames)} avg_inf={avg_inf_ms:.1f}ms total_det_msgs_per_s={fps_total:.1f}")
                self._last_log_t = now
                self._frames_proc = 0
                self._batches = 0
                self._inf_time = 0.0

