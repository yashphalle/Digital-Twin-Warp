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

            # Appearance-by-default: embed all detections for this camera (batched)
            app_sim = None
            if self.app_model is not None and det_xyxy and self.latest_store is not None:
                item = self.latest_store.latest(cid)
                try:
                    if item is not None:
                        frame = item[0]
                        crops = []
                        for bb in det_xyxy:
                            x1,y1,x2,y2 = [int(max(0, v)) for v in bb]
                            x2 = min(x2, frame.shape[1]-1)
                            y2 = min(y2, frame.shape[0]-1)
                            if x2 <= x1 or y2 <= y1:
                                crops.append(None)
                                continue
                            crop = cv2.resize(frame[y1:y2, x1:x2], (224,224), interpolation=cv2.INTER_AREA)
                            crops.append(crop)
                        # Batch predict only valid crops
                        valid_idxs = [i for i,c in enumerate(crops) if c is not None]
                        if valid_idxs:
                            imgs = [crops[i] for i in valid_idxs]
                            res = self.app_model.predict(imgs, device=getattr(self.cfg,'device','cuda:0'), verbose=False)
                            det_embs = []
                            for r in res:
                                probs = getattr(r, 'probs', None)
                                if probs is None: det_embs.append(None); continue
                                vec = probs.data
                                vec = vec.cpu().numpy() if hasattr(vec,'cpu') else np.asarray(vec)
                                # L2 normalize
                                norm = np.linalg.norm(vec) + 1e-8
                                det_embs.append((vec / norm).astype(np.float32))
                            # Build similarity matrix vs tracks with cached embeddings (refresh every N frames)
                            state = self.states[cid]
                            T = len(state.tracks)
                            D = len(det_xyxy)
                            app_sim = np.zeros((T,D), dtype=np.float32)

                            # Prepare per-track embeddings: refresh if missing or every embed_refresh_period frames
                            refresh_period = int(getattr(self.cfg, 'embed_refresh_period', 10))
                            trk_embs: List[np.ndarray | None] = [None]*T
                            for i, tr in enumerate(state.tracks):
                                cache = self.trk_cache[cid].get(tr.track_id)
                                need_refresh = cache is None or (tr.age % max(1, refresh_period) == 0)
                                if need_refresh:
                                    # crop from frame using current bbox
                                    x1,y1,x2,y2 = [int(max(0, v)) for v in tr.bbox.tolist()]
                                    x2 = min(x2, frame.shape[1]-1)
                                    y2 = min(y2, frame.shape[0]-1)
                                    if x2 > x1 and y2 > y1:
                                        crop = cv2.resize(frame[y1:y2, x1:x2], (224,224), interpolation=cv2.INTER_AREA)
                                        # Single predict; could batch later if needed
                                        r = self.app_model.predict(crop, device=getattr(self.cfg,'device','cuda:0'), verbose=False)
                                        if r:
                                            probs = getattr(r[0], 'probs', None)
                                            if probs is not None:
                                                vec = probs.data
                                                vec = vec.cpu().numpy() if hasattr(vec,'cpu') else np.asarray(vec)
                                                vec = (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)
                                                self.trk_cache[cid][tr.track_id] = {'emb': vec}
                                                trk_embs[i] = vec
                                else:
                                    if cache is not None:
                                        trk_embs[i] = cache.get('emb')

                            # Compute cosine sims
                            # Map det_embs back to full D list indices
                            det_map = {full_j: det_embs[vi] for full_j, vi in zip(valid_idxs, range(len(valid_idxs)))}
                            for i in range(T):
                                a = trk_embs[i]
                                if a is None: continue
                                for j in range(D):
                                    b = det_map.get(j)
                                    if b is None: continue
                                    app_sim[i,j] = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))
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

