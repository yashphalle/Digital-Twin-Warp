import threading
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


class GridGUIThread(threading.Thread):
    """Non-blocking GUI that renders a grid of latest frames with optional overlays.

    - Uses LatestFrameStore to fetch latest frames per camera.
    - Receives detection/track messages via push_msg() and overlays them.
    - Runs at ~5-10 Hz; does not block the main pipeline.
    """

    def __init__(self, window_name: str, latest_store, camera_ids: List[int], tracking_mode: bool = False,
                 target_hz: float = 8.0):
        super().__init__(daemon=True)
        self.window_name = window_name
        self.latest_store = latest_store
        self.camera_ids = list(camera_ids)
        self.tracking_mode = tracking_mode
        self.period_s = 1.0 / max(1e-3, target_hz)
        self._running = False

        # Overlay/display settings (tunable via hotkeys)
        self.font_scale = 1.0
        self.font_thickness = 2
        self.box_thickness = 3
        self.low_conf_thresh = 0.5
        self.show_conf = True
        self.show_fps = True

        # View state
        self.single_view = False
        self.current_cam_index = 0  # index into self.camera_ids for single view
        self.fullscreen = False

        # Last per-camera outputs we received from the app's monitor loop
        self._last_dets: Dict[int, List[Dict]] = {}
        self._last_tracks: Dict[int, List[Dict]] = {}
        self._lock = threading.Lock()

        # Signal to repaint on new pipeline message (sync GUI to pipeline cadence)
        self._tick_event = threading.Event()

        # Simple per-camera FPS estimation (EMA on capture timestamps)
        self._last_ts_map: Dict[int, float] = {}
        self._fps_map: Dict[int, float] = {}
        # System (pipeline) FPS injected by app monitor loop
        self._sys_fps_map: Dict[int, float] = {}

    def stop(self):
        self._running = False

    def push_msg(self, msg: Dict):
        """Receive per-camera output message from app monitor loop.
        msg: {'camera_id': int, 'detections'| 'tracks': list, 'ts': float, 'sys_fps': float?}
        """
        cid = msg.get('camera_id')
        if cid is None:
            return
        with self._lock:
            if 'detections' in msg:
                self._last_dets[cid] = msg.get('detections', [])
            if 'tracks' in msg:
                self._last_tracks[cid] = msg.get('tracks', [])
            if 'sys_fps' in msg:
                self._sys_fps_map[cid] = float(msg.get('sys_fps'))
        # Signal a new pipeline tick so GUI repaints immediately
        try:
            self._tick_event.set()
        except Exception:
            pass

    def _draw_text(self, img: np.ndarray, text: str, org: Tuple[int, int], color=(0, 255, 0)):
        # outlined text for readability
        cv2.putText(img, text, (org[0]+1, org[1]+1), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0,0,0), self.font_thickness+1, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

    def _draw_overlays(self, img: np.ndarray, cid: int) -> np.ndarray:
        out = img.copy()
        with self._lock:
            dets = list(self._last_dets.get(cid, []))
            tracks = list(self._last_tracks.get(cid, []))
        if self.tracking_mode:
            # Tracks: orange boxes, label: T:track_id G:global_id A:age [class conf]
            for t in tracks:
                bbox = t.get('bbox') or t.get('xyxy')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), self.box_thickness)
                    gid = t.get('global_id', '')
                    age = t.get('age', '')
                    cls = t.get('class', '')
                    # Tracking label: only persisted (global) ID + age (+ optional class). No track id, no conf.
                    label = f"G:{gid} A:{age}"
                    if cls:
                        label += f" {cls}"
                    self._draw_text(out, label, (x1, max(20, y1 - 8)), (0, 200, 255))
        else:
            # Detections: green boxes, red if low confidence; label: class conf
            for d in dets:
                bbox = d.get('bbox')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    conf = float(d.get('confidence', 0.0))
                    color = (0, 255, 0) if conf >= self.low_conf_thresh else (0, 0, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, self.box_thickness)
                    if self.show_conf:
                        cls = d.get('class', '')
                        self._draw_text(out, f"{cls} {conf:.2f}", (x1, max(20, y1 - 8)), (0, 255, 0))
        # Camera label + optional FPS (GUI render FPS + System FPS when available)
        label = f"Cam {cid}"
        gui_fps = self._fps_map.get(cid)
        sys_fps = self._sys_fps_map.get(cid)
        if self.show_fps:
            parts = []
            if sys_fps is not None:
                parts.append(f"Sys:{sys_fps:.1f}")
            if gui_fps is not None:
                parts.append(f"GUI:{gui_fps:.1f}")
            if parts:
                label += "  " + " ".join(parts)
        self._draw_text(out, label, (10, 25), (255, 255, 0))
        return out

    @staticmethod
    def _tile_frames(frames: List[Tuple[int, np.ndarray]], grid_cols: int = 4, tile_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # Determine tile size
        if tile_size is None:
            h, w = frames[0][1].shape[:2]
            tile_w, tile_h = w, h
        else:
            tile_w, tile_h = tile_size
        # Build grid
        cols = grid_cols
        rows = int(np.ceil(len(frames) / cols))
        grid = []
        idx = 0
        for _ in range(rows):
            row_imgs = []
            for _ in range(cols):
                if idx < len(frames):
                    _, f = frames[idx]
                    f = cv2.resize(f, (tile_w, tile_h))
                else:
                    f = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                row_imgs.append(f)
                idx += 1
            grid.append(np.hstack(row_imgs))
        return np.vstack(grid)

    def _update_fps(self, cid: int, ts: float):
        last = self._last_ts_map.get(cid)
        if last is not None and ts > last:
            dt = max(1e-6, ts - last)
            fps = 1.0 / dt
            # simple EMA
            prev = self._fps_map.get(cid, fps)
            self._fps_map[cid] = 0.8 * prev + 0.2 * fps
        self._last_ts_map[cid] = ts

    def _handle_keys(self, key: int):
        if key in (ord('q'), 27):
            self._running = False
        elif key == ord('g'):
            self.single_view = not self.single_view
        elif key == ord('n'):
            self.current_cam_index = (self.current_cam_index + 1) % max(1, len(self.camera_ids))
        elif key == ord('p'):
            self.current_cam_index = (self.current_cam_index - 1) % max(1, len(self.camera_ids))
        elif key == ord('c'):
            self.show_conf = not self.show_conf
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('d') or key == ord('t'):
            self.tracking_mode = not self.tracking_mode
        elif key == ord('+'):
            self.font_scale = min(3.0, self.font_scale + 0.1)
        elif key == ord('-'):
            self.font_scale = max(0.4, self.font_scale - 0.1)
        elif key in (ord('F'), ord('f')) and (key == ord('F')):
            # uppercase F toggles fullscreen
            self.fullscreen = not self.fullscreen
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                pass

    def run(self):
        self._running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while self._running:
            # Wait for the next pipeline tick (new det/track message)
            try:
                self._tick_event.wait(timeout=self.period_s)
                self._tick_event.clear()
            except Exception:
                pass

            t0 = time.time()
            frames: List[Tuple[int, np.ndarray]] = []
            # Build frames list depending on view mode
            if self.single_view and self.camera_ids:
                cid = self.camera_ids[self.current_cam_index]
                item = self.latest_store.latest(cid)
                if item is not None:
                    frame, ts = item
                    self._update_fps(cid, ts)
                    frame = self._draw_overlays(frame, cid)
                    frames.append((cid, frame))
            else:
                for cid in self.camera_ids:
                    item = self.latest_store.latest(cid)
                    if item is None:
                        continue
                    frame, ts = item
                    self._update_fps(cid, ts)
                    frame = self._draw_overlays(frame, cid)
                    frames.append((cid, frame))

            # Render
            if self.single_view and frames:
                # show the single frame resized to window
                cid, frame = frames[0]
                cv2.imshow(self.window_name, frame)
            else:
                grid = self._tile_frames(frames, grid_cols=4)
                cv2.imshow(self.window_name, grid)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._handle_keys(key)

            # No busy rate control needed; we sync to pipeline via event with a small timeout
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

