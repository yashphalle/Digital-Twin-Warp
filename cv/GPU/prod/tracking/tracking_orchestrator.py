import threading
import time
import logging
import queue
from typing import Dict, List
import numpy as np

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
    def __init__(self, cfg, det_queue: queue.Queue, out_queue: queue.Queue, camera_ids: List[int]):
        super().__init__(daemon=True, name='TrackingOrchestrator')
        self.cfg = cfg
        self.det_queue = det_queue
        self.out_queue = out_queue
        self._running = False
        self.logger = logging.getLogger('TrackingOrchestrator')
        
        # Per-camera tracker state and ID managers
        self.states: Dict[int, TrackerState] = {cid: TrackerState() for cid in camera_ids}
        self.idm: Dict[int, FastGlobalIDManager] = {cid: FastGlobalIDManager(cid) for cid in camera_ids}

        self._frames = 0
        self._last_log = 0.0

    def stop(self):
        self._running = False

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

            # Call lightweight association
            state = self.states[cid]
            tracks, _ = associate(det_xyxy, state,
                                  match_thresh=getattr(self.cfg, 'match_thresh', 0.5),
                                  min_hits=getattr(self.cfg, 'min_hits', 3),
                                  max_age=getattr(self.cfg, 'max_age', 30),
                                  fuse_score=getattr(self.cfg, 'fuse_score', True))

            # Build output with camera-prefixed global IDs and ages
            out_tracks = []
            for tr in tracks:
                # Map internal track_id to camera-prefixed global via FastGlobalIDManager using YOLO-style IDs
                # Here we reuse internal track_id directly as yolo_track_id placeholder
                gid = self.idm[cid].get_global_id(tr.track_id)
                age = self.idm[cid].get_track_age(gid)
                out_tracks.append({
                    'camera_id': cid,
                    'track_id': tr.track_id,
                    'global_id': gid,
                    'age': age,
                    'bbox': [float(x) for x in tr.bbox.tolist()],
                    'confidence': 1.0,  # use detection conf if needed (e.g., average)
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

