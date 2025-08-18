import threading
import time
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class RedisWriter(threading.Thread):
    """Batched Redis upserts for entity:{gid} and idx:cam:{cid}."""

    def __init__(self, input_q, redis_client, batch_interval_s: float = 0.5, name: str = 'RedisWriter'):
        super().__init__(daemon=True, name=name)
        self.in_q = input_q
        self.redis = redis_client
        self.batch_interval_s = batch_interval_s
        self._running = False
        self.logger = logging.getLogger(name)
        self._pending: List[Dict] = []

    def stop(self):
        self._running = False

    def _flush(self):
        if not self.redis or not self.redis.connected():
            self._pending.clear()
            return
        n = 0
        for item in self._pending:
            try:
                gid = int(item['global_id'])
                cid = int(item['camera_id'])
                ts = float(item['ts'])
                bbox = item.get('bbox')
                emb = item.get('embedding')
                fields = {}
                if emb is not None:
                    # Store as float16 bytes
                    f16 = emb.astype(np.float16).tobytes()
                    fields[b'feature'] = f16
                fields[b'last_seen_ts'] = str(ts).encode()
                fields[b'last_camera'] = str(cid).encode()
                if bbox is not None:
                    fields[b'last_bbox'] = str(bbox).encode()
                self.redis.hmset_entity(gid, fields)
                self.redis.zadd_cam_index(cid, gid, ts)
                n += 1
            except Exception:
                pass
        if n:
            self.logger.info(f"RedisWriter flushed {n} entities")
        self._pending.clear()

    def run(self):
        self._running = True
        last = time.time()
        while self._running:
            try:
                item = self.in_q.get(timeout=0.05)
                self._pending.append(item)
            except Exception:
                pass
            now = time.time()
            if now - last >= self.batch_interval_s:
                try:
                    self._flush()
                except Exception:
                    pass
                last = now
        try:
            self._flush()
        except Exception:
            pass

