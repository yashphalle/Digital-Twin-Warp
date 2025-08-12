import threading
import time
import logging
import queue
from typing import Dict, List
import numpy as np
import cv2

from GPU.prod.tracking.tracker_core import TrackerState, associate
from GPU.pipelines.gpu_processor_fast_tracking import FastGlobalIDManager

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

        # Lazy-load yolov8n-cls for embeddings if enabled
        from ultralytics import YOLO
        try:
            self.app_model = YOLO(getattr(cfg, 'yolo_cls_model_path', 'yolov8n-cls.pt'))
            self.app_model.to(getattr(cfg, 'device', 'cuda:0'))
        except Exception:
            self.app_model = None

        self._frames = 0
        self._last_log = 0.0

    def stop(self):
        self._running = False

    def _embed_detections(self, cid: int, det_xyxy: List[List[float]]) -> np.ndarray | None:
        if self.app_model is None or not det_xyxy:
            return None
        # Fetch latest frame for this camera for cropping
        try:
            from GPU.prod.buffers.latest_frame_store import LatestFrameStore  # type: ignore
        except Exception:
            return None
        # Attempt to get a frame via a back-reference path (we don't have latest_store here; rely on Redis/DB path if needed)
        # For now, skip embedding if frame cannot be fetched; we keep IoU-only in that case.
        # In a future change, we can pass latest_store into orchestrator constructor.
        return None

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

            # On-demand appearance: only for ambiguous matches
            app_sim = None
            state = self.states[cid]
            T = len(state.tracks)
            D = len(det_xyxy)
            if self.app_model is not None and det_xyxy and self.latest_store is not None and T > 0 and D > 0:
                item = self.latest_store.latest(cid)
                try:
                    if item is not None:
                        frame = item[0]
                        # Compute IoU matrix for ambiguity analysis
                        dets_np = np.array(det_xyxy, dtype=np.float32)
                        iou_mat = np.zeros((T, D), dtype=np.float32)
                        for i, tr in enumerate(state.tracks):
                            x1t,y1t,x2t,y2t = tr.bbox.tolist()
                            a_tr = max(0.0, x2t-x1t) * max(0.0, y2t-y1t)
                            for j in range(D):
                                x1d,y1d,x2d,y2d = dets_np[j]
                                xi1 = max(x1t, x1d); yi1 = max(y1t, y1d)
                                xi2 = min(x2t, x2d); yi2 = min(y2t, y2d)
                                w = max(0.0, xi2 - xi1); h = max(0.0, yi2 - yi1)
                                inter = w*h
                                a_det = max(0.0, x2d-x1d) * max(0.0, y2d-y1d)
                                union = a_tr + a_det - inter + 1e-6
                                iou_mat[i, j] = inter/union
                        # Find ambiguous tracks
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
                            # Build detection crops for candidate columns only
                            cand_cols = sorted(list(cand_cols))
                            det_crops = []
                            det_col_to_idx = {}
                            for idx_j, j in enumerate(cand_cols):
                                x1,y1,x2,y2 = [int(max(0, v)) for v in det_xyxy[j]]
                                x2 = min(x2, frame.shape[1]-1)
                                y2 = min(y2, frame.shape[0]-1)
                                if x2 <= x1 or y2 <= y1:
                                    det_col_to_idx[j] = None
                                    continue
                                crop = cv2.resize(frame[y1:y2, x1:x2], (224,224), interpolation=cv2.INTER_AREA)
                                det_crops.append(crop)
                                det_col_to_idx[j] = len(det_crops)-1
                            det_emb_map: Dict[int, np.ndarray] = {}
                            if det_crops:
                                res = self.app_model.predict(det_crops, device=getattr(self.cfg,'device','cuda:0'), verbose=False)
                                for j in cand_cols:
                                    ridx = det_col_to_idx.get(j)
                                    if ridx is None: continue
                                    probs = getattr(res[ridx], 'probs', None)
                                    if probs is None: continue
                                    vec = probs.data
                                    vec = vec.cpu().numpy() if hasattr(vec,'cpu') else np.asarray(vec)
                                    vec = (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)
                                    det_emb_map[j] = vec
                            # Prepare/refresh track embeddings for ambiguous rows
                            refresh_period = int(getattr(self.cfg, 'embed_refresh_period', 10))
                            trk_embs: Dict[int, np.ndarray] = {}
                            trk_crops = []
                            trk_row_to_idx = {}
                            for i in amb_rows:
                                tr = state.tracks[i]
                                cache = self.trk_cache[cid].get(tr.track_id)
                                need_refresh = cache is None or (tr.age % max(1, refresh_period) == 0)
                                if need_refresh:
                                    x1,y1,x2,y2 = [int(max(0, v)) for v in tr.bbox.tolist()]
                                    x2 = min(x2, frame.shape[1]-1)
                                    y2 = min(y2, frame.shape[0]-1)
                                    if x2 > x1 and y2 > y1:
                                        crop = cv2.resize(frame[y1:y2, x1:x2], (224,224), interpolation=cv2.INTER_AREA)
                                        trk_row_to_idx[i] = len(trk_crops)
                                        trk_crops.append(crop)
                                else:
                                    trk_emb = cache.get('emb') if cache is not None else None
                                    if trk_emb is not None:
                                        trk_embs[i] = trk_emb
                            if trk_crops:
                                res_t = self.app_model.predict(trk_crops, device=getattr(self.cfg,'device','cuda:0'), verbose=False)
                                for i in amb_rows:
                                    ridx = trk_row_to_idx.get(i)
                                    if ridx is None: continue
                                    probs = getattr(res_t[ridx], 'probs', None)
                                    if probs is None: continue
                                    vec = probs.data
                                    vec = vec.cpu().numpy() if hasattr(vec,'cpu') else np.asarray(vec)
                                    vec = (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)
                                    tr = state.tracks[i]
                                    self.trk_cache[cid][tr.track_id] = {'emb': vec}
                                    trk_embs[i] = vec
                            # Build app_sim matrix initialized to 1.0 (neutral)
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
            out_tracks = []
            for tr in tracks:
                gid = self.idm[cid].get_global_id(tr.track_id)
                age = self.idm[cid].get_track_age(gid)
                out_tracks.append({
                    'camera_id': cid,
                    'track_id': tr.track_id,
                    'global_id': gid,
                    'age': age,
                    'bbox': [float(x) for x in tr.bbox.tolist()],
                    'class': det_classes[0] if det_classes else 'object'
                })

            # Emit
            try:
                if self.out_queue.full():
                    _ = self.out_queue.get_nowait()
                self.out_queue.put_nowait({'camera_id': cid, 'tracks': out_tracks, 'ts': ts})
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

