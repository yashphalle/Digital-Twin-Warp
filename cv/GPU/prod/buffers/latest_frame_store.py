from collections import deque
from typing import Dict, Optional, Tuple, List
import threading
import time
import numpy as np


class LatestFrameStore:
    """Tiny per-camera ring buffer storing only the newest N frames (replace-old)."""

    def __init__(self, camera_ids: List[int], depth: int = 2):
        self.depth = max(1, depth)
        self._buffers: Dict[int, deque] = {cid: deque(maxlen=self.depth) for cid in camera_ids}
        self._locks: Dict[int, threading.Lock] = {cid: threading.Lock() for cid in camera_ids}

    def put(self, camera_id: int, frame: np.ndarray) -> None:
        lock = self._locks.get(camera_id)
        if lock is None:
            return
        with lock:
            self._buffers[camera_id].append((frame, time.time()))

    def latest(self, camera_id: int) -> Optional[Tuple[np.ndarray, float]]:
        lock = self._locks.get(camera_id)
        if lock is None:
            return None
        with lock:
            if not self._buffers[camera_id]:
                return None
            return self._buffers[camera_id][-1]

    def pop_all(self, camera_id: int) -> List[Tuple[np.ndarray, float]]:
        lock = self._locks.get(camera_id)
        if lock is None:
            return []
        with lock:
            items = list(self._buffers[camera_id])
            self._buffers[camera_id].clear()
            return items

    def latest_many(self, camera_ids: List[int]) -> Dict[int, Tuple[np.ndarray, float]]:
        out: Dict[int, Tuple[np.ndarray, float]] = {}
        for cid in camera_ids:
            item = self.latest(cid)
            if item is not None:
                out[cid] = item
        return out

