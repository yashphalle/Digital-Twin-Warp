import threading
import time
import logging
import queue
from typing import Dict, List
import numpy as np

from tracking.tracker_core import TrackerState, associate
from pipelines.gpu_processor_fast_tracking import FastGlobalIDManager

logger = logging.getLogger(__name__)

class TrackingOrchestrator(threading.Thread):
    """Consumes batched detections and emits per-camera tracks.

    Input messages from DetectorWorker have schema:
      { 'camera_id': int, 'detections': [ {bbox, confidence, class}, ... ], 'ts': float }
    Output messages to app/db/router:
      { 'camera_id': int, 'tracks': [ {track_id, global_id, age, bbox, confidence, class}, ... ], 'ts': float }
    """
    def __init__(self, cfg, det_queue: queue.Queue, out_queue: queue.Queue, camera_ids: List[int], latest_store=None):
        super().__init__(daemon=True, name='TrackingOrchestrator')
        self.cfg = cfg
        self.det_queue = det_queue
        self.out_queue = out_queue
        self.latest_store = latest_store
        self._running = False
        self.logger = logging.getLogger('TrackingOrchestrator')

        # Per-camera tracker state and ID managers
        self.states: Dict[int, TrackerState] = {cid: TrackerState() for cid in camera_ids}
        self.idm: Dict[int, FastGlobalIDManager] = {cid: FastGlobalIDManager(cid) for cid in camera_ids}

        # Per-camera track embedding cache: {cid: {track_id: {'emb': np.ndarray, 'age': int}}}
        self.trk_cache: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {cid: {} for cid in camera_ids}

        # Disable internal yolov8n-cls; embeddings are provided by EmbeddingService
        self.app_model = None

        self._frames = 0
        self._last_log = 0.0

    def stop(self):
        self._running = False

    # Note: Internal embedding disabled; embeddings are provided by EmbeddingService

    def bind_global_id(self, camera_id: int, track_id: int, new_gid: int):
        # Bind resolved global id in ID manager
        try:
            idm = self.idm.get(camera_id)
            if idm:
                idm.assign_external_global_id(track_id, new_gid)
        except Exception:
            pass

    def run(self):
        self._running = True
        while self._running:
            try:
                msg = self.det_queue.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.002)
                continue

            cid = msg.get('camera_id')
            detections = msg.get('detections') or []
            ts = msg.get('ts', time.time())

            # Extract xyxy boxes
            det_xyxy = []
            det_confs = []
            det_classes = []
            for d in detections:
                bb = d.get('bbox')
                if not bb or len(bb) != 4:
                    continue
                det_xyxy.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                det_confs.append(float(d.get('confidence', 0.0)))
                det_classes.append(d.get('class'))

            # On-demand appearance: only for ambiguous matches; use provided embeddings
            app_sim = None
            state = self.states[cid]
            T = len(state.tracks)
            D = len(det_xyxy)
            # Build IoU matrix for ambiguity analysis
            if det_xyxy and T > 0 and D > 0:
                try:
                    dets_np = np.array(det_xyxy, dtype=np.float32)
                    iou_mat = np.zeros((T, D), dtype=np.float32)
                    for i, tr in enumerate(state.tracks):
                        x1t,y1t,x2t,y2t = tr.bbox.tolist()
                        for j in range(D):
                            x1d,y1d,x2d,y2d = dets_np[j]
                            xi1 = max(x1t, x1d); yi1 = max(y1t, y1d)
                            xi2 = min(x2t, x2d); yi2 = min(y2t, y2d)
                            w = max(0.0, xi2 - xi1); h = max(0.0, yi2 - yi1)
                            inter = w*h
                            a_tr = max(0.0, x2t-x1t) * max(0.0, y2t-y1t)
                            a_det = max(0.0, x2d-x1d) * max(0.0, y2d-y1d)
                            union = a_tr + a_det - inter + 1e-6
                            iou_mat[i, j] = inter/union
                    # Find ambiguous rows/cols
                    iou_strong = float(getattr(self.cfg, 'iou_strong', 0.70))
                    iou_conflict_delta = float(getattr(self.cfg, 'iou_conflict_delta', 0.05))
                    amb_rows = []
                    cand_cols = set()
                    topk = int(getattr(self.cfg, 'amb_topk', 3))
                    for i in range(T):
                        row = iou_mat[i]
                        order = np.argsort(row)[::-1]
                        best = row[order[0]] if D>0 else 0.0
                        second = row[order[1]] if D>1 else 0.0
                        if best < iou_strong or (best - second) < iou_conflict_delta:
                            amb_rows.append(i)
                            for k in range(min(topk, D)):
                                cand_cols.add(int(order[k]))
                    if amb_rows and cand_cols:
                        # Use provided embeddings to build app_sim
                        emb_list = msg.get('embeddings') or []
                        # Track embedding cache
                        trk_embs: Dict[int, np.ndarray] = {}
                        for i in amb_rows:
                            tr = state.tracks[i]
                            cache = self.trk_cache[cid].get(tr.track_id)
                            if cache is not None and 'emb' in cache:
                                trk_embs[i] = cache['emb']
                        det_emb_map: Dict[int, np.ndarray] = {}
                        for j in cand_cols:
                            if 0 <= j < len(emb_list):
                                b = emb_list[j]
                                if b is not None:
                                    det_emb_map[j] = np.asarray(b, dtype=np.float32)
                        if trk_embs and det_emb_map:
                            app_sim = np.ones((T, D), dtype=np.float32)
                            for i in amb_rows:
                                a = trk_embs.get(i)
                                if a is None: continue
                                for j in cand_cols:
                                    b = det_emb_map.get(j)
                                    if b is None: continue
                                    app_sim[i, j] = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))
                except Exception:
                    app_sim = None

            # Association with optional appearance similarity
            state = self.states[cid]
            tracks, _ = associate(det_xyxy, state,
                                  match_thresh=getattr(self.cfg, 'match_thresh', 0.5),
                                  min_hits=getattr(self.cfg, 'min_hits', 3),
                                  max_age=getattr(self.cfg, 'max_age', 30),
                                  fuse_score=getattr(self.cfg, 'fuse_score', True),
                                  app_sim=app_sim,
                                  wiou=getattr(self.cfg, 'wiou', 0.7),
                                  wapp=getattr(self.cfg, 'wapp', 0.3))

            # Build output with camera-prefixed global IDs and ages
            # Suppress stale tracks: do not emit if missed beyond display_max_missed (default 0)
            display_max_missed = int(getattr(self.cfg, 'display_max_missed', 0))
            out_tracks = []
            for tr in tracks:
                if tr.time_since_update > display_max_missed:
                    continue
                gid = self.idm[cid].get_global_id(tr.track_id)
                age = self.idm[cid].get_track_age(gid)
                # Include track embedding if cached
                emb_cached = None
                try:
                    cache = self.trk_cache[cid].get(tr.track_id)
                    emb_cached = cache.get('emb') if cache else None
                except Exception:
                    emb_cached = None
                out_tracks.append({
                    'camera_id': cid,
                    'track_id': tr.track_id,
                    'global_id': gid,
                    'age': age,
                    'bbox': [float(x) for x in tr.bbox.tolist()],
                    'class': det_classes[0] if det_classes else 'object',
                    'embedding': emb_cached,
                })

            # Emit
            try:
                if self.out_queue.full():
                    _ = self.out_queue.get_nowait()
                self.out_queue.put_nowait({'camera_id': cid, 'tracks': out_tracks, 'ts': ts})
            except Exception:
                pass

            # Update embedding cache for matched tracks using provided embeddings
            try:
                emb_list = msg.get('embeddings') or []
                for tr in tracks:
                    # if a detection matched this track this frame, we can infer via bbox equality to update cache
                    # Simpler: copy the first valid embedding if available and cache under this track_id
                    # More robust mapping requires match indices; for now, only set if track has no cache
                    if self.trk_cache[cid].get(tr.track_id) is None:
                        for emb in emb_list:
                            if emb is not None:
                                self.trk_cache[cid][tr.track_id] = {'emb': np.asarray(emb, dtype=np.float32)}
                                break
            except Exception:
                pass

            # Stats
            self._frames += 1
            now = time.time()
            if now - self._last_log >= max(0.5, self.cfg.log_interval_s):
                fps = self._frames / max(1e-6, now - (self._last_log or now))
                self.logger.info(f"orchestrator_cam={cid} trk_fps={fps:.1f} active_tracks={len(out_tracks)}")
                self._frames = 0
                self._last_log = now

