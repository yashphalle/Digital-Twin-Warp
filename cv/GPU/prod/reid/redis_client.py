import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


class RedisClient:
    """Thin wrapper around redis-py with safe operations.

    All methods are non-throwing; failures are logged and return None/False.
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._client = None

    def connect(self) -> bool:
        if redis is None:
            logger.warning("Redis library not available; continuing without Redis")
            return False
        try:
            self._client = redis.from_url(self._uri, decode_responses=False, socket_keepalive=True, health_check_interval=30)
            self._client.ping()
            logger.info(f"Redis connected: {self._uri}")
            return True
        except Exception as e:
            logger.error(f"Redis connect failed: {e}")
            self._client = None
            return False

    def connected(self) -> bool:
        return self._client is not None

    # Entity ops
    def hmset_entity(self, gid: int, fields: Dict[bytes, bytes]) -> bool:
        c = self._client
        if c is None:
            return False
        try:
            key = f"entity:{gid}".encode()
            # Use per-field HSET to avoid server arg parsing issues across versions
            # and ensure we never send an empty mapping.
            wrote = False
            for k, v in fields.items():
                if v is None:
                    continue
                c.hset(key, k, v)
                wrote = True
            return wrote
        except Exception as e:
            logger.error(f"Redis HMSET entity failed: {e}")
            return False

    def hmget_entity(self, gid: int, field_names: List[bytes]) -> Optional[List[Optional[bytes]]]:
        c = self._client
        if c is None:
            return None
        try:
            key = f"entity:{gid}".encode()
            return c.hmget(key, field_names)
        except Exception as e:
            logger.error(f"Redis HMGET entity failed: {e}")
            return None

    # Index ops
    def zadd_cam_index(self, cid: int, gid: int, ts: float) -> bool:
        c = self._client
        if c is None:
            return False
        try:
            key = f"idx:cam:{cid}".encode()
            c.zadd(key, {str(gid).encode(): ts})
            return True
        except Exception as e:
            logger.error(f"Redis ZADD failed: {e}")
            return False

    def zrevrangebyscore(self, cid: int, min_ts: float, max_ts: float, topk: int) -> List[int]:
        c = self._client
        if c is None:
            return []
        try:
            key = f"idx:cam:{cid}".encode()
            vals = c.zrevrangebyscore(key, max_ts, min_ts, start=0, num=topk)
            out: List[int] = []
            for v in vals:
                try:
                    out.append(int(v.decode()))
                except Exception:
                    pass
            return out
        except Exception as e:
            logger.error(f"Redis ZREVRANGEBYSCORE failed: {e}")
            return []

