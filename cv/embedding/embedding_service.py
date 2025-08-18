import threading
import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import queue

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class EmbeddingService(threading.Thread):
    """Batched appearance embedding extractor.

    - Consumes detection messages: { 'camera_id', 'ts', 'detections': [ {bbox, confidence, class}, ... ] }
    - Produces enriched messages with aligned embeddings: same dict plus 'embeddings': [np.ndarray|None, ...]
    - Crops from LatestFrameStore; batches all crops across cameras; runs yolov8n-cls once per batch.
    - Non-blocking output: if out_q is full, drop oldest before enqueueing newest (preserve latest behavior).
    """

    def __init__(self,
                 cfg,
                 latest_store,
                 in_q: queue.Queue,
                 out_q: queue.Queue,
                 name: str = 'EmbeddingService'):
        super().__init__(daemon=True, name=name)
        self.cfg = cfg
        self.latest_store = latest_store
        self.in_q = in_q
        self.out_q = out_q
        self._running = False
        self.logger = logging.getLogger(name)
        # Load classifier once, FP32
        self.model = YOLO(getattr(cfg, 'yolo_cls_model_path', 'yolov8n-cls.pt'))
        self.model.to(getattr(cfg, 'device', 'cuda:0'))
        self.model.fuse() if hasattr(self.model, 'fuse') else None

        # Stats
        self._last_log = 0.0
        self._batches = 0
        self._inf_time = 0.0
        self._msgs_out = 0
        self._crops = 0

    def stop(self):
        self._running = False

    @staticmethod
    def _area(b: List[float]) -> float:
        return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

    @staticmethod
    def _crop_resize(frame: np.ndarray, bbox: List[float], tw: int, th: int) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        img = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        return img

    def _drain_window(self, max_ms: int, max_msgs: int = 64) -> List[Dict]:
        items: List[Dict] = []
        end_t = time.time() + max(0.0, max_ms) / 1000.0
        while len(items) < max_msgs and time.time() < end_t:
            try:
                m = self.in_q.get(timeout=0.002)
                items.append(m)
            except Exception:
                pass
        return items

    def run(self):
        self._running = True
        while self._running:
            try:
                # Always take at least one item (blocks briefly), then drain a tiny window for batching
                try:
                    first = self.in_q.get(timeout=0.05)
                except Exception:
                    continue
                batch = [first]
                if getattr(self.cfg, 'emb_batch_window_ms', 4) > 0:
                    more = self._drain_window(getattr(self.cfg, 'emb_batch_window_ms', 4))
                    if more:
                        batch.extend(more)

                # Prepare crops
                crops: List[np.ndarray] = []
                map_indices: List[Tuple[int, int]] = []  # (msg_idx, det_idx)
                tw = int(getattr(self.cfg, 'emb_target_w', 224))
                th = int(getattr(self.cfg, 'emb_target_h', 224))
                a_min = float(getattr(self.cfg, 'emb_filter_area_min', 0))
                a_max = float(getattr(self.cfg, 'emb_filter_area_max', 1e12))
                class_whitelist = getattr(self.cfg, 'emb_filter_classes', None)

                for mi, msg in enumerate(batch):
                    cid = msg.get('camera_id')
                    dets = msg.get('detections') or []
                    latest = self.latest_store.latest(cid)
                    frame = latest[0] if latest else None
                    emb_list = [None] * len(dets)  # placeholder
                    msg['embeddings'] = emb_list  # will be filled after inference
                    if frame is None:
                        continue
                    for di, d in enumerate(dets):
                        bbox = d.get('bbox'); cls = d.get('class');
                        if not bbox or len(bbox) != 4:
                            continue
                        area = self._area(bbox)
                        if area < a_min or area > a_max:
                            continue
                        if class_whitelist is not None and cls not in class_whitelist:
                            continue
                        img = self._crop_resize(frame, bbox, tw, th)
                        if img is None:
                            continue
                        crops.append(img)
                        map_indices.append((mi, di))
                        if len(crops) >= int(getattr(self.cfg, 'emb_max_batch_crops', 128)):
                            break

                # If no crops, forward messages as-is
                if not crops:
                    for m in batch:
                        try:
                            if self.out_q.full():
                                _ = self.out_q.get_nowait()
                            self.out_q.put_nowait(m)
                            self._msgs_out += 1
                        except Exception:
                            pass
                    # Periodic log
                    now = time.time()
                    if now - self._last_log >= max(0.5, getattr(self.cfg, 'log_interval_s', 2.0)):
                        self.logger.info(f"emb: batches={self._batches} msgs_out={self._msgs_out} crops={self._crops} avg_inf_ms={(self._inf_time/max(1,self._batches))*1000.0:.1f}")
                        self._last_log = now
                    continue

                # Run classifier once on batch
                t0 = time.time()
                try:
                    res = self.model.predict(crops, device=getattr(self.cfg, 'device', 'cuda:0'), verbose=False)
                except Exception as e:
                    self.logger.error(f"Embedding inference failed: {e}")
                    res = None
                dt = time.time() - t0
                self._inf_time += dt
                self._batches += 1
                self._crops += len(crops)

                # Fill embeddings back
                if res is not None:
                    for k, (mi, di) in enumerate(map_indices):
                        try:
                            r = res[k]
                            probs = getattr(r, 'probs', None)
                            if probs is None:
                                vec = None
                            else:
                                vec = probs.data
                                vec = vec.cpu().numpy() if hasattr(vec, 'cpu') else np.asarray(vec)
                                n = np.linalg.norm(vec) + 1e-8
                                vec = (vec / n).astype(np.float32)
                            m = batch[mi]
                            m['embeddings'][di] = vec
                        except Exception:
                            pass

                # Forward messages
                for m in batch:
                    try:
                        if self.out_q.full():
                            _ = self.out_q.get_nowait()
                        self.out_q.put_nowait(m)
                        self._msgs_out += 1
                    except Exception:
                        pass

                # Periodic stats
                now = time.time()
                if now - self._last_log >= max(0.5, getattr(self.cfg, 'log_interval_s', 2.0)):
                    avg_ms = (self._inf_time / max(1, self._batches)) * 1000.0
                    self.logger.info(f"EmbeddingService: batch_msgs={len(batch)} crops={len(crops)} avg_inf={avg_ms:.1f}ms msgs_out_per_s={self._msgs_out/max(1e-6, now-(self._last_log or now)):.1f}")
                    self._last_log = now
                    self._msgs_out = 0
                    self._batches = 0
                    self._inf_time = 0.0
                    self._crops = 0

            except Exception as e:
                self.logger.error(f"EmbeddingService loop error: {e}")
                time.sleep(0.01)

