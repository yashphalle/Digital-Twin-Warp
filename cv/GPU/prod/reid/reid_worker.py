import threading
import time
import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


class ReIDWorker(threading.Thread):
    """Matches new embeddings against Redis candidates (same cam then neighbors)."""

    def __init__(self, input_q, output_q, redis_client, neighbor_map: Dict[int, List[int]],
                 same_cam_window_s: int, neighbor_window_s: int, topk: int, sim_threshold: float,
                 name: str = 'ReIDWorker'):
        super().__init__(daemon=True, name=name)
        self.in_q = input_q
        self.out_q = output_q
        self.redis = redis_client
        self.neighbors = neighbor_map
        self.tw_same = same_cam_window_s
        self.tw_nei = neighbor_window_s
        self.topk = topk
        self.th = sim_threshold
        self._running = False
        self.logger = logging.getLogger(name)

    def stop(self):
        self._running = False

    def _fetch_candidates(self, cid: int, now_ts: float, window_s: int) -> List[int]:
        if not self.redis or not self.redis.connected():
            return []
        min_ts = now_ts - window_s
        return self.redis.zrevrangebyscore(cid, min_ts, now_ts, self.topk)

    def _get_feature(self, gid: int) -> Optional[np.ndarray]:
        vals = self.redis.hmget_entity(gid, [b'feature']) if self.redis else None
        if not vals:
            return None
        f = vals[0]
        if f is None:
            return None
        try:
            arr = np.frombuffer(f, dtype=np.float16).astype(np.float32)
            # Assume already L2-normalized when stored
            return arr
        except Exception:
            return None

    def run(self):
        self._running = True
        while self._running:
            try:
                item = self.in_q.get(timeout=0.05)
            except Exception:
                continue
            try:
                cid = item['camera_id']
                now_ts = item['ts']
                emb = item['embedding']

                # Same camera candidates first
                best_gid = None
                best_sim = -1.0
                cands = self._fetch_candidates(cid, now_ts, self.tw_same)
                for gid in cands:
                    f = self._get_feature(gid)
                    if f is None:
                        continue
                    s = cosine_sim(emb, f)
                    if s > best_sim:
                        best_sim = s
                        best_gid = gid

                # Neighbors if needed
                if (best_gid is None or best_sim < self.th) and cid in self.neighbors:
                    for nid in self.neighbors[cid]:
                        cands = self._fetch_candidates(nid, now_ts, self.tw_nei)
                        for gid in cands:
                            f = self._get_feature(gid)
                            if f is None:
                                continue
                            s = cosine_sim(emb, f)
                            if s > best_sim:
                                best_sim = s
                                best_gid = gid

                resolved_gid = None
                if best_gid is not None and best_sim >= self.th:
                    resolved_gid = best_gid

                out = dict(item)
                out['resolved_gid'] = resolved_gid
                out['similarity'] = best_sim

                try:
                    if self.out_q.full():
                        _ = self.out_q.get_nowait()
                    self.out_q.put_nowait(out)
                except Exception:
                    pass

            except Exception as e:
                self.logger.error(f"reid failed: {e}")

