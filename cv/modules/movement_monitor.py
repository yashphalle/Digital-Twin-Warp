import time
from typing import Dict, Tuple, Optional
import numpy as np
import logging
import os

from .coordinate_mapper import CoordinateMapper


class MovementMonitor:
    """
    Lightweight snapshot + movement-based cropping control per persistent ID.
    Behavior:
      - Establishment snapshot: once when a track becomes established (age >= established_age_frames), regardless of calibration
      - Movement snapshots: require world distance >= threshold_ft and cooldown
      - Max crops per track applies; when at cap, we replace the most recent saved image (do not increase count)
    Designed to be called in the hot path with minimal overhead.
    """

    def __init__(
        self,
        camera_id: int,
        threshold_ft: float = 5.0,
        cooldown_sec: float = 60.0,
        max_crops_per_track: int = 5,
        padding_px: int = 32,
        force_test_mode: bool = False,
        established_age_frames: int = 5,
    ):
        self._log = logging.getLogger(f"MovementMonitor[{camera_id}]")
        # Allow test-mode via env to force early snapshot: MOVEMENT_TEST_MODE=true
        try:
            if os.getenv('MOVEMENT_TEST_MODE', 'false').lower() == 'true':
                force_test_mode = True
                established_age_frames = 0
                cooldown_sec = 0.0
                threshold_ft = 0.0
                self._log.warning("MOVEMENT_TEST_MODE enabled: disabling age gate, cooldown, and movement threshold")
        except Exception:
            pass
        self.camera_id = camera_id
        self.threshold_ft = float(threshold_ft)
        self.cooldown_sec = float(cooldown_sec)
        self.max_crops_per_track = int(max_crops_per_track)
        self.padding_px = int(padding_px)
        self.force_test_mode = bool(force_test_mode)
        self.established_age_frames = int(established_age_frames)

        # Per persistent ID state: {pid: (x_ft, y_ft)} and last ts and count
        self._last_xy: Dict[int, Tuple[float, float]] = {}
        self._last_ts: Dict[int, float] = {}
        self._count: Dict[int, int] = {}
        # Track a replacement policy hint: when count >= max, future saves should replace most recent
        self._replace_latest: Dict[int, bool] = {}

        # Mapper per camera; reuse tested module
        self._mapper = CoordinateMapper(camera_id=camera_id)
        # Load calibration file based on standard naming in cv/config
        try:
            calib_path = f"cv/config/warehouse_calibration_camera_{camera_id}.json"
            self._mapper.load_calibration(calib_path)
        except Exception:
            pass

    @staticmethod
    def _centroid_from_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _pixel_to_world_feet(self, center_px: Tuple[float, float], frame_shape) -> Optional[Tuple[float, float]]:
        if not getattr(self._mapper, 'is_calibrated', False):
            return None
        h, w = frame_shape[:2]
        # Calibration built for 3840x2160; scale up to that space
        scale_x = 3840.0 / float(w)
        scale_y = 2160.0 / float(h)
        sx = center_px[0] * scale_x
        sy = center_px[1] * scale_y
        x_ft, y_ft = self._mapper.pixel_to_real(sx, sy)
        if x_ft is None or y_ft is None:
            return None
        return x_ft, y_ft

    @staticmethod
    def _l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def check_and_build_job(
        self,
        detection: Dict,
        frame_shape,
        now_ts_ms: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Return a crop job dict if trigger conditions are met; else None.
        detection must contain keys: 'persistent_id', 'bbox', optionally 'track_age'.
        """
        pid = detection.get('persistent_id')
        bbox = detection.get('bbox')
        if pid is None or bbox is None:
            return None

        # Age gate (>= established_age_frames)
        age = detection.get('track_age', 0)
        if age < self.established_age_frames and not self.force_test_mode:
            if age % 10 == 0:
                self._log.debug(f"age gate: age={age} < established={self.established_age_frames} pid={pid}")
            return None

        # Compute world center now (may be None if not calibrated)
        cx, cy = self._centroid_from_bbox(tuple(bbox))
        world = self._pixel_to_world_feet((cx, cy), frame_shape)

        # Bookkeeping
        last_xy = self._last_xy.get(pid)
        last_ts = self._last_ts.get(pid, 0.0)
        count = self._count.get(pid, 0)
        now_ms = int(now_ts_ms if now_ts_ms is not None else time.time() * 1000)

        replace_latest = False

        if count == 0:
            # Establishment snapshot (no calibration requirement)
            pass
        else:
            # Movement snapshot requires calibration and distance/cooldown checks
            if not self.force_test_mode:
                if world is None:
                    # No world mapping available yet, set a baseline and skip
                    if last_xy is None:
                        if world is not None:
                            self._last_xy[pid] = world
                    if (count % 20) == 0:
                        self._log.debug(f"no world mapping yet pid={pid} cam={self.camera_id} count={count}")
                    return None
                # Cooldown
                if now_ms - (last_ts or 0) < self.cooldown_sec * 1000.0:
                    return None
                # Distance
                if last_xy is None:
                    # Set baseline and wait for next movement
                    self._last_xy[pid] = world
                    return None
                dist = self._l2(world, last_xy)
                if dist < self.threshold_ft:
                    return None
            # At cap: replace most recent instead of skipping
            if count >= self.max_crops_per_track:
                replace_latest = True

        # Build job skeleton; caller will fill roi and enqueue
        job = {
            'persistent_id': pid,
            'camera_id': detection.get('camera_id', self.camera_id),
            'ts_ms': now_ms,
            'replace_latest': replace_latest,
        }

        # Update state immediately to prevent double-enqueue on same frame
        if world is not None:
            self._last_xy[pid] = world
        self._last_ts[pid] = now_ms
        if not replace_latest:
            self._count[pid] = count + 1

        return job

    def get_stats(self) -> Dict:
        return {
            'tracked_ids': len(self._last_xy),
            'total_saved_counts': sum(self._count.values()) if self._count else 0,
        }

