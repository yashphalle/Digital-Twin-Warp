from collections import deque
from typing import Dict, Optional, Tuple, List
import threading
import time
import numpy as np


class LatestFrameStore:
    """Tiny per-camera ring buffer storing only the newest N frames (replace-old).

    Notes on sync:
    - We store (frame, ts) pairs where ts is time.time() at put().
    - For GUI sync with pipeline messages carrying 'ts', use get_nearest_by_ts() to fetch
      the frame closest to that timestamp (within a small tolerance) instead of latest().
    """

    def __init__(self, camera_ids: List[int], depth: int = 2):
        self.depth = max(1, depth)
        self._buffers: Dict[int, deque] = {cid: deque(maxlen=self.depth) for cid in camera_ids}
        self._locks: Dict[int, threading.Lock] = {cid: threading.Lock() for cid in camera_ids}

    def put(self, camera_id: int, frame: np.ndarray, ts: float | None = None) -> None:
        """Insert a frame with an explicit capture timestamp.
        If ts is None, time.time() is used.
        """
        lock = self._locks.get(camera_id)
        if lock is None:
            return
        if ts is None:
            ts = time.time()
        with lock:
            self._buffers[camera_id].append((frame, ts))

    def latest(self, camera_id: int) -> Optional[Tuple[np.ndarray, float]]:
        lock = self._locks.get(camera_id)
        if lock is None:
            return None
        with lock:
            if not self._buffers[camera_id]:
                return None
            return self._buffers[camera_id][-1]

    def get_nearest_by_ts(self, camera_id: int, target_ts: float, tolerance_s: float = 0.2) -> Optional[Tuple[np.ndarray, float]]:
        """Return the frame whose timestamp is nearest to target_ts within tolerance.
        Preference order: latest <= target_ts; else nearest absolute difference <= tolerance; else None.
        """
        lock = self._locks.get(camera_id)
        if lock is None:
            return None
        with lock:
            buf = self._buffers.get(camera_id)
            if not buf:
                return None
            best = None
            best_dt = None
            # iterate from newest to oldest
            for (f, ts) in reversed(buf):
                dt = abs(ts - target_ts)
                if ts <= target_ts and dt <= tolerance_s:
                    return (f, ts)
                if dt <= tolerance_s and (best_dt is None or dt < best_dt):
                    best = (f, ts)
                    best_dt = dt
            return best

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

