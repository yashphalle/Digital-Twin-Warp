import threading
import time
import logging
import queue
from typing import Dict, List

from GPU.pipelines.gpu_processor_fast_tracking import FastGlobalIDManager

logger = logging.getLogger(__name__)


class AssociationRouter(threading.Thread):
    """Routes tracker outputs through ReID (async) and forwards resolved tracks to DB.

    Non-blocking:
    - Continued tracks are forwarded immediately (RAM continuity) and a Redis update is enqueued in background
    - New tracks are enqueued to feature/reid workers; when resolved, we rebind and forward
    """

    def __init__(self, cfg, latest_store, neighbor_map: Dict[int, List[int]], db_writer, redis_client,
                 tracker_manager,
                 feature_workers: List[threading.Thread], reid_workers: List[threading.Thread],
                 feature_in_q: queue.Queue, feature_out_q: queue.Queue,
                 reid_out_q: queue.Queue, redis_write_q: queue.Queue):
        super().__init__(daemon=True, name='AssociationRouter')
        self.cfg = cfg
        self.latest_store = latest_store
        self.neighbors = neighbor_map
        self.db_writer = db_writer
        self.redis = redis_client
        self.tracker_manager = tracker_manager
        self.feature_workers = feature_workers
        self.reid_workers = reid_workers
        self.feature_in_q = feature_in_q
        self.feature_out_q = feature_out_q
        self.reid_out_q = reid_out_q
        self.redis_write_q = redis_write_q
        self._running = False
        self.logger = logging.getLogger('AssociationRouter')

        # RAM mapping: (cam, yolo_track_id) -> global_id
        self.ram_map: Dict[tuple, int] = {}

    def stop(self):
        self._running = False

    def enqueue(self, msg: Dict):
        """Entry point: called by app main loop instead of db_writer.enqueue."""
        cid = msg.get('camera_id')
        tracks = msg.get('tracks') or []
        ts = msg.get('ts', time.time())

        # Forward continued tracks immediately; enqueue Redis background update
        for t in tracks:
            gid = int(t.get('global_id'))
            yid = int(t.get('track_id')) if 'track_id' in t else None
            age = int(t.get('age', 0))
            bbox = t.get('bbox')

            key = (cid, yid)
            if age > 1 and self.ram_map.get(key) == gid:
                # Fast path: forward to DB and update Redis async
                self._forward_db(cid, [t], ts)
                self._enqueue_redis_update(cid, gid, bbox, ts)
                # One-time feature fill: if Redis lacks feature for this gid and track is mature (age>=5), enqueue extraction
                try:
                    needs_feat = False
                    if self.redis and self.redis.connected():
                        vals = self.redis.hmget_entity(gid, [b'feature'])
                        needs_feat = not (vals and vals[0])
                    if needs_feat and age >= 5:
                        item = {'camera_id': cid, 'track_id': yid, 'global_id': gid, 'bbox': bbox}
                        if self.feature_in_q.full():
                            _ = self.feature_in_q.get_nowait()
                        self.feature_in_q.put_nowait(item)
                except Exception:
                    pass
            else:
                # New or unknown mapping: push to feature extraction
                try:
                    item = {'camera_id': cid, 'track_id': yid, 'global_id': gid, 'bbox': bbox}
                    if self.feature_in_q.full():
                        _ = self.feature_in_q.get_nowait()
                    self.feature_in_q.put_nowait(item)
                except Exception:
                    pass

    def _forward_db(self, cid: int, tracks: List[Dict], ts: float):
        try:
            self.db_writer.enqueue({'camera_id': cid, 'tracks': tracks, 'ts': ts})
        except Exception:
            pass

    def _enqueue_redis_update(self, cid: int, gid: int, bbox, ts: float):
        try:
            item = {'camera_id': cid, 'global_id': gid, 'bbox': bbox, 'ts': ts}
            if self.redis_write_q.full():
                _ = self.redis_write_q.get_nowait()
            self.redis_write_q.put_nowait(item)
        except Exception:
            pass

    def run(self):
        self._running = True
        last_log = time.time()
        forwarded = 0
        resolved = 0
        while self._running:
            # Drain feature_out -> feed reid
            try:
                f = self.feature_out_q.get(timeout=0.01)
                try:
                    if self.reid_out_q.full():
                        _ = self.reid_out_q.get_nowait()
                    self.reid_out_q.put_nowait(f)
                except Exception:
                    pass
            except Exception:
                pass

            # Drain reid_out -> bind id, forward db, schedule redis update
            try:
                r = self.reid_out_q.get(timeout=0.01)
                cid = r['camera_id']
                yid = r['track_id']
                gid = r['global_id']
                bbox = r['bbox']
                ts = r['ts']
                resolved_gid = r.get('resolved_gid')
                if resolved_gid is not None and resolved_gid != gid:
                    # External bind to matched gid (e.g., 8005 across cams)
                    try:
                        mgr = getattr(self.tracker_manager, 'threads', None)
                        if mgr:
                            for t in mgr:
                                if getattr(t, 'camera_id', None) == cid:
                                    # Bind YOLO track id to resolved global id
                                    idm = getattr(t, 'id_manager', None)
                                    if idm:
                                        idm.assign_external_global_id(yid, int(resolved_gid))
                                    break
                    except Exception:
                        pass
                    # Update RAM map to resolved id
                    gid = resolved_gid

                # Update RAM continuity
                self.ram_map[(cid, yid)] = gid

                # Bind resolved global id back into orchestrator if available
                try:
                    bind = getattr(self.tracker_manager, 'bind_global_id', None)
                    if callable(bind) and yid is not None:
                        bind(cid, yid, int(gid))
                except Exception:
                    pass

                # Forward to DB, mapping similarity -> similarity_score if present
                db_msg = {**r, 'global_id': gid}
                if 'similarity' in db_msg and 'similarity_score' not in db_msg:
                    db_msg['similarity_score'] = float(db_msg.pop('similarity'))
                self._forward_db(cid, [db_msg], ts)

                # Enqueue Redis upsert with embedding
                try:
                    item = {'camera_id': cid, 'global_id': gid, 'bbox': bbox, 'ts': ts, 'embedding': r.get('embedding')}
                    if self.redis_write_q.full():
                        _ = self.redis_write_q.get_nowait()
                    self.redis_write_q.put_nowait(item)
                except Exception:
                    pass

                resolved += 1

            except Exception:
                pass

            # lightweight idle
            time.sleep(0.002)

            if time.time() - last_log >= 2.0:
                self.logger.info(f"Router stats: forwarded={forwarded} resolved={resolved} queued_feat={self.feature_in_q.qsize()} queued_reid={self.reid_out_q.qsize()}")
                last_log = time.time()

