import threading
import time
import logging
from typing import Dict, Tuple, List, Optional
import os
from collections import deque
import numpy as np
import cv2


# pymongo
try:
    from pymongo import MongoClient, UpdateOne
except Exception as e:
    MongoClient = None
    UpdateOne = None

from final.modules.coordinate_mapper import CoordinateMapper  # reuse tested mapper
from final.modules.color_extractor import ObjectColorExtractor  # enabled for insert-only


class MongoWriterThread(threading.Thread):
    """Async MongoDB writer for tracking outputs.

    - Consumes per-camera tracking messages: {'camera_id', 'tracks': [...], 'ts'}
    - Only writes when age >= age_threshold (default 5)
    - Batches writes every batch_interval_s seconds
    - Schema preserved to match existing frontend/backend
    - Timestamps stored in PST (naive, per user preference)
    """

    # Calibration expects fisheye-corrected 3840x2160 pixels; frames are resized to 1280x720
    CALIB_W = 3840
    CALIB_H = 2160

    def __init__(
        self,
        latest_store,
        active_cameras: List[int],
        atlas_uri: str,
        database: str = "warehouse_tracking",
        collection: str = "detections",
        age_threshold: int = 5,
        batch_interval_s: float = 2.0,
        logger_name: str = "MongoWriter",
    ):
        super().__init__(daemon=True)
        self.latest_store = latest_store
        self.active_cameras = list(active_cameras)
        self.atlas_uri = atlas_uri
        self.database_name = database
        self.collection_name = collection
        self.age_threshold = max(1, age_threshold)
        self.batch_interval_s = max(0.5, batch_interval_s)
        self.logger = logging.getLogger(logger_name)
        self._running = False

        # State
        self._lock = threading.Lock()
        self._pending_inserts: List[Dict] = []
        self._pending_updates: List[Tuple[Dict, Dict]] = []  # (filter, update)
        # Use plain set() for Python 3.10/3.11 compatibility
        self._seen_keys = set()  # type: ignore[var-annotated]  # (camera_id, global_id)
        # Internal non-blocking queue of raw tracking messages (fast enqueue)
        self._raw_msgs = deque(maxlen=2000)

        # Per-camera coordinate mappers
        self._mappers: Dict[int, CoordinateMapper] = {}
        for cid in self.active_cameras:
            try:
                mapper = CoordinateMapper(camera_id=cid)
                # Load per-camera calibration from cv/GPU/configs
                calib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', f'warehouse_calibration_camera_{cid}.json'))
                mapper.load_calibration(calib_path)
                self._mappers[cid] = mapper
            except Exception:
                self._mappers[cid] = None

        # Color extraction: enabled for insert-only to minimize CPU
        try:
            self._color = ObjectColorExtractor()
        except Exception as e:
            self._color = None
            self.logger.warning(f"Color extractor init failed; disabling color extraction: {e}")

        # DB
        self.client = None
        self.collection = None
        self._connect_db()

    def _connect_db(self):
        if MongoClient is None:
            self.logger.error("pymongo not available; DB writer disabled")
            return
        try:
            self.client = MongoClient(self.atlas_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            db = self.client[self.database_name]
            self.collection = db[self.collection_name]
            # Indexes (non-unique to avoid insert failures)
            try:
                self.collection.create_index([("global_id", 1), ("camera_id", 1)])
                self.collection.create_index([("camera_id", 1), ("last_seen", -1)])
                self.collection.create_index([("first_seen", -1)])
                self.collection.create_index([("warp_id", 1)], sparse=True)
            except Exception:
                pass
            self.logger.info(f"Mongo connected: db={self.database_name} coll={self.collection_name}")
        except Exception as e:
            self.logger.error(f"Mongo connect failed: {e}")
            self.client = None
            self.collection = None

    @staticmethod
    def _now_pst_naive():
        # PST timezone without DST handling (simple offset). For robust TZ use zoneinfo.
        # User requested simple PST only to match existing.
        from datetime import datetime, timezone, timedelta
        now_utc = datetime.now(timezone.utc)
        pst = now_utc.astimezone(timezone(timedelta(hours=-8)))
        return pst.replace(tzinfo=None)

    @staticmethod
    def _bbox_to_corners(b: List[float]) -> List[List[int]]:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @staticmethod
    def _center_area(b: List[float]) -> Tuple[List[int], int]:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        return [cx, cy], int(area)

    def _physical_from_pixels(self, cid: int, corners_px: List[List[int]]) -> Tuple[Optional[List[List[float]]], Optional[List[float]], Optional[float], Optional[float], str]:
        mapper = self._mappers.get(cid)
        if not mapper:
            return None, None, None, None, 'missing_mapper'
        try:
            # Determine actual frame size for this camera from latest_store
            item = self.latest_store.latest(cid)
            if item and item[0] is not None:
                frame = item[0]
                fw, fh = frame.shape[1], frame.shape[0]
            else:
                # Fallback to detection resize default
                fw, fh = 1280, 720

            # Scale pixel coords to calibration resolution (4K) for homography
            scale_x = self.CALIB_W / float(fw)
            scale_y = self.CALIB_H / float(fh)

            scaled: List[Tuple[float, float]] = []
            for (x, y) in corners_px:
                sx = float(x) * scale_x
                sy = float(y) * scale_y
                scaled.append((sx, sy))

            phys_corners: List[List[float]] = []
            for (sx, sy) in scaled:
                rx, ry = mapper.pixel_to_real(sx, sy)
                if rx is None or ry is None:
                    return None, None, None, None, 'transform_failed'
                phys_corners.append([float(rx), float(ry)])

            # Compute center in pixel space and map center directly for accuracy
            cx = sum([p[0] for p in corners_px]) / 4.0
            cy = sum([p[1] for p in corners_px]) / 4.0
            cx_s = cx * scale_x
            cy_s = cy * scale_y
            rcx, rcy = mapper.pixel_to_real(cx_s, cy_s)
            if rcx is None or rcy is None:
                # fallback to average of corner mappings
                xs = [p[0] for p in phys_corners]
                ys = [p[1] for p in phys_corners]
                phys_center = [float(sum(xs) / 4.0), float(sum(ys) / 4.0)]
            else:
                phys_center = [float(rcx), float(rcy)]

            return phys_corners, phys_center, phys_center[0], phys_center[1], 'ok'
        except Exception:
            return None, None, None, None, 'transform_failed'

    def _color_from_roi(self, cid: int, bbox: List[float]) -> Dict:
        # Fast quantized RGB sampling (insert-time only)
        item = self.latest_store.latest(cid)
        if not item:
            return {}
        frame, _ = item
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return {}
        roi = frame[y1:y2, x1:x2]
        try:
            # Sample up to N pixels uniformly without replacement (or all if small)
            N = 512
            H, W = roi.shape[:2]
            total = H * W
            if total == 0:
                return {}
            if total <= N:
                # Use all pixels
                samp = roi.reshape(-1, 3)
            else:
                # Random uniform sample of indices
                idx = np.random.choice(total, size=N, replace=False)
                samp = roi.reshape(-1, 3)[idx]
            # Convert BGR->RGB and quantize to 4 bits per channel
            rgb = samp[:, ::-1].astype(np.uint8)
            q = (rgb >> 4).astype(np.uint16)
            bins = (q[:,0] << 8) | (q[:,1] << 4) | q[:,2]
            # Histogram and dominant bin
            hist = np.bincount(bins, minlength=4096)
            dom = int(hist.argmax())
            r4 = (dom >> 8) & 0xF; g4 = (dom >> 4) & 0xF; b4 = dom & 0xF
            # Bin center back to 8-bit
            r = int(r4 * 17); g = int(g4 * 17); b = int(b4 * 17)
            # Convert representative RGB to HSV
            rgb1 = np.array([[[r, g, b]]], dtype=np.uint8)
            hsv1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2HSV)[0,0]
            conf = float(hist[dom] / max(1, len(samp)))
            return {
                'color_rgb': [r, g, b],
                'color_hsv': [int(hsv1[0]), int(hsv1[1]), int(hsv1[2])],
                'color_hex': f"#{r:02x}{g:02x}{b:02x}",
                'color_name': 'unknown',
                'color_confidence': conf,
                'extraction_method': 'quantized_rgb_4bit',
            }
        except Exception:
            return {}

    def enqueue(self, msg: Dict):
        """Non-blocking: push raw message to internal queue; heavy work happens in run()."""
        if self.collection is None:
            return
        try:
            self._raw_msgs.append(msg)
        except Exception:
            # If deque is full, drop oldest by popping left then append
            try:
                _ = self._raw_msgs.popleft()
                self._raw_msgs.append(msg)
            except Exception:
                pass
    def _drain_raw_msgs(self, max_items: int = 200) -> List[Dict]:
        out = []
        for _ in range(max_items):
            try:
                out.append(self._raw_msgs.popleft())
            except Exception:
                break
        return out


    def stop(self):
        self._running = False

    def _flush_once(self):
        if self.collection is None:
            return
        inserts: List[Dict]
        updates: List[Tuple[Dict, Dict]]
        with self._lock:
            inserts = self._pending_inserts
            updates = self._pending_updates
            self._pending_inserts = []
            self._pending_updates = []
        if not inserts and not updates:
            return
        try:
            n_ins = 0
            n_upd = 0
            if inserts:
                try:
                    self.collection.insert_many(inserts, ordered=False)
                    n_ins = len(inserts)
                except Exception as e:
                    # Fallback: individual inserts
                    for d in inserts:
                        try:
                            self.collection.insert_one(d)
                            n_ins += 1
                        except Exception:
                            pass
            if updates:
                if UpdateOne is not None:
                    ops = [UpdateOne(f, u, upsert=False) for (f, u) in updates]
                    res = self.collection.bulk_write(ops, ordered=False)
                    n_upd = res.modified_count
                else:
                    for f, u in updates:
                        try:
                            r = self.collection.update_one(f, u)
                            if r.modified_count:
                                n_upd += 1
                        except Exception:
                            pass
            self.logger.info(f"Flushed inserts={n_ins} updates={n_upd}")
        except Exception as e:
            self.logger.error(f"flush failed: {e}")

    def run(self):
        self._running = True
        last = 0.0
        try:
            while self._running:
                # Drain raw messages and build pending DB ops off the main app thread
                msgs = self._drain_raw_msgs(max_items=400)
                if msgs:
                    now_pst = self._now_pst_naive()
                    for msg in msgs:
                        cid = msg.get('camera_id')
                        tracks = msg.get('tracks') or []
                        for t in tracks:
                            try:
                                age = int(t.get('age', 0))
                                if age < self.age_threshold:
                                    continue
                                gid = int(t.get('global_id'))
                                bbox = t.get('bbox')
                                if not bbox or len(bbox) != 4:
                                    continue
                                corners_px = self._bbox_to_corners(bbox)
                                center_px, area = self._center_area(bbox)
                                phys_corners, phys_center, px, py, coord_status = self._physical_from_pixels(cid, corners_px)

                                key = (cid, gid)
                                is_new = key not in self._seen_keys

                                common = {
                                    'camera_id': cid,
                                    'global_id': gid,
                                    'warp_id': t.get('warp_id'),
                                    'bbox': [int(round(v)) for v in bbox],
                                    'corners': corners_px,
                                    'shape_type': 'quadrangle',
                                    'center': center_px,
                                    'area': area,
                                    'class': t.get('class'),
                                    'confidence': float(t.get('confidence', 0.0)),
                                    'similarity_score': float(t.get('similarity_score', 0.0)),
                                    'physical_corners': phys_corners,
                                    'real_center': phys_center,
                                    'physical_x_ft': px,
                                    'physical_y_ft': py,
                                    'coordinate_status': coord_status,
                                }

                                with self._lock:
                                    if is_new:
                                        doc = dict(common)
                                        # Insert-time only: extract color once
                                        try:
                                            color_fields = self._color_from_roi(cid, bbox)
                                            if color_fields:
                                                doc.update(color_fields)
                                        except Exception:
                                            pass
                                        doc.update({
                                            'persistent_id': gid,
                                            'timestamp': now_pst,
                                            'first_seen': now_pst,
                                            'last_seen': now_pst,
                                            'times_seen': 1,
                                            'status': 'new',
                                            'is_new': True,
                                        })
                                        self._pending_inserts.append(doc)
                                        self._seen_keys.add(key)
                                    else:
                                        filt = {'camera_id': cid, 'global_id': gid}
                                        update = {
                                            '$set': {
                                                'last_seen': now_pst,
                                                'bbox': common['bbox'],
                                                'corners': common['corners'],
                                                'physical_corners': common['physical_corners'],
                                                'shape_type': common['shape_type'],
                                                'real_center': common['real_center'],
                                                'confidence': common['confidence'],
                                                'area': common['area'],
                                                'center': common['center'],
                                                'physical_x_ft': common['physical_x_ft'],
                                                'physical_y_ft': common['physical_y_ft'],
                                                'coordinate_status': common['coordinate_status'],
                                                'similarity_score': common['similarity_score'],
                                                'status': 'tracked',
                                            },
                                            '$unset': {'inactive_since': ""},
                                            '$inc': {'times_seen': 1}
                                        }
                                        self._pending_updates.append((filt, update))
                            except Exception:
                                pass

                now = time.time()
                if now - last >= self.batch_interval_s:
                    self._flush_once()
                    last = now
                time.sleep(0.02)
        except Exception as e:
            self.logger.error(f"writer loop failed: {e}")
        # Final flush
        try:
            self._flush_once()
        except Exception:
            pass

            pass
