import threading
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FeatureWorker(threading.Thread):
    """Extract Yolov8n-cls embeddings from ROI crops in a non-blocking worker.

    Input queue entries: { 'camera_id', 'track_id', 'global_id', 'bbox' }
    Output queue entries: { 'camera_id', 'track_id', 'global_id', 'bbox', 'embedding': np.ndarray, 'ts': float }
    """

    def __init__(self, latest_store, input_q, output_q, cls_model_path: str, device: str = 'cuda:0', name: str = 'FeatureWorker'):
        super().__init__(daemon=True, name=name)
        self.latest_store = latest_store
        self.in_q = input_q
        self.out_q = output_q
        self._running = False
        self.logger = logging.getLogger(name)
        self.model = YOLO(cls_model_path)
        self.model.to(device)
        self.device = device

    def stop(self):
        self._running = False

    @staticmethod
    def _prep_crop(frame: np.ndarray, bbox: List[float], pad: float = 0.05) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        dx = (x2 - x1) * pad
        dy = (y2 - y1) * pad
        x1 = int(max(0, min(w - 1, x1 - dx)))
        x2 = int(max(0, min(w - 1, x2 + dx)))
        y1 = int(max(0, min(h - 1, y1 - dy)))
        y2 = int(max(0, min(h - 1, y2 + dy)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        return crop

    @staticmethod
    def _prep_image_for_model(image: np.ndarray) -> np.ndarray:
        # Return HWC uint8 image; Ultralytics will handle color internally
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(vec) + eps
        return vec / n

    def run(self):
        self._running = True
        while self._running:
            try:
                item = self.in_q.get(timeout=0.05)
            except Exception:
                continue
            try:
                cid = item['camera_id']
                bbox = item['bbox']
                latest = self.latest_store.latest(cid)
                if not latest:
                    continue
                frame, _ = latest
                crop = self._prep_crop(frame, bbox)
                if crop is None:
                    continue
                img = self._prep_image_for_model(crop)
                res = self.model.predict(img, device=self.device, verbose=False)
                if not res:
                    continue
                r0 = res[0]
                # Ultralytics cls provides probabilities in r0.probs
                probs = getattr(r0, 'probs', None)
                if probs is None:
                    continue
                try:
                    vec = probs.data
                    if hasattr(vec, 'cpu'):
                        vec = vec.cpu().numpy()
                    else:
                        vec = np.asarray(vec)
                except Exception:
                    continue
                emb = self._l2_normalize(vec.astype(np.float32).squeeze())
                out = {
                    'camera_id': item['camera_id'],
                    'track_id': item['track_id'],
                    'global_id': item['global_id'],
                    'bbox': item['bbox'],
                    'embedding': emb,
                    'ts': time.time(),
                }
                try:
                    if self.out_q.full():
                        _ = self.out_q.get_nowait()
                    self.out_q.put_nowait(out)
                except Exception:
                    pass
            except Exception as e:
                self.logger.error(f"feature failed: {e}")

