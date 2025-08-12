import logging
import os
import sys
# Ensure 'cv' directory is on sys.path so 'GPU' package imports work
_cv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'cv'))
if _cv_path not in sys.path:
    sys.path.append(_cv_path)

import queue
import time
from typing import List

from collections import deque
from GPU.prod.config.runtime import default_config
from GPU.prod.buffers.latest_frame_store import LatestFrameStore
from GPU.prod.capture.camera_worker import CameraWorker
from GPU.prod.detection.detector_worker import DetectorWorker
from GPU.prod.tracking.tracker_manager import TrackerManager
from GPU.prod.tracking.tracking_orchestrator import TrackingOrchestrator


def setup_logging(level: str = 'INFO'):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_detection_only(active_cameras: List[int], duration_s: float = 60.0, show_gui: bool = False):
    cfg = default_config(active_cameras)
    setup_logging(cfg.log_level)
    logger = logging.getLogger("app")

    # Resolve model path to absolute to avoid CWD issues
    if not os.path.isabs(cfg.model_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        cfg.model_path = os.path.abspath(os.path.join(repo_root, cfg.model_path))

    # Latest frame store
    store = LatestFrameStore(camera_ids=cfg.active_cameras, depth=cfg.latest_store_depth)

    # Queues
    det_out = queue.Queue(maxsize=1000)

    # Optional GUI
    gui = None
    if show_gui:
        try:
            from GPU.prod.gui.gui_display import GridGUIThread
            gui = GridGUIThread(window_name="Detections", latest_store=store, camera_ids=cfg.active_cameras,
                                 tracking_mode=False, target_hz=8.0)
            gui.start()
        except Exception as e:
            logger.warning(f"GUI launch failed: {e}")
            gui = None

    # Start camera threads
    cams = []
    for cid in cfg.active_cameras:
        cw = CameraWorker(camera_id=cid, latest_store=store, frame_skip=cfg.frame_skip, resize=cfg.resize,
                          debug=(cfg.log_level == 'DEBUG'))
        if cw.connect():
            cw.start()
            cams.append(cw)

    if not cams:
        logger.error("No cameras connected")
        return {'ok': False, 'reason': 'no_cameras'}

    # Start detector worker (1 worker for now)
    detector = DetectorWorker(cfg=cfg, latest_store=store, output_queue=det_out)
    detector.start()

    logger.info(f"Detection-only pipeline running for {duration_s}s...")

    # Monitor loop with duration
    start = time.time()
    last_log = start
    det_total = 0
    per_cam = {cid: 0 for cid in cfg.active_cameras}
    max_q = 0

    try:
        while time.time() - start < duration_s:
            try:
                msg = det_out.get(timeout=0.5)
                det_total += 1
                per_cam[msg['camera_id']] = per_cam.get(msg['camera_id'], 0) + 1
                if show_gui and gui is not None:
                    gui.push_msg(msg)
            except queue.Empty:
                pass
            max_q = max(max_q, det_out.qsize())
            if time.time() - last_log >= cfg.log_interval_s:
                elapsed = time.time() - start
                rate = det_total / max(1e-6, elapsed)
                logger.info(f"elapsed={elapsed:.1f}s total_msgs={det_total} rate={rate:.1f}/s q={det_out.qsize()} max_q={max_q}")
                last_log = time.time()
    finally:
        detector.stop()
        detector.join(timeout=2)
        for cw in cams:
            cw.stop()
            cw.join(timeout=2)

    elapsed = max(1e-6, time.time() - start)
    summary = {
        'ok': True,
        'mode': 'detection',
        'cameras': cfg.active_cameras,
        'duration_s': duration_s,
        'total_msgs': det_total,
        'avg_msgs_per_sec': det_total / elapsed,
        'per_camera_msgs': per_cam,
        'max_queue_size': max_q,
    }
    logger.info(f"SUMMARY: {summary}")
    return summary


def run_tracking(active_cameras: List[int], duration_s: float = 60.0, show_gui: bool = False,
                 reid_enabled: bool | None = None, db_enabled: bool | None = None,
                 batched_tracking: bool | None = None):
    from GPU.prod.config.runtime import default_config
    cfg = default_config(active_cameras)
    if reid_enabled is not None:
        cfg.reid_enabled = reid_enabled
    if db_enabled is not None:
        cfg.db_enabled = db_enabled
    if batched_tracking is not None:
        cfg.use_batched_tracking = batched_tracking
    setup_logging(cfg.log_level)
    logger = logging.getLogger("app")

    # Resolve model path to absolute to avoid CWD issues
    if not os.path.isabs(cfg.model_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        cfg.model_path = os.path.abspath(os.path.join(repo_root, cfg.model_path))

    store = LatestFrameStore(camera_ids=cfg.active_cameras, depth=cfg.latest_store_depth)

    # Per-camera tracker output
    trk_out = queue.Queue(maxsize=1000)

    # Detection output (for batched detector path)
    det_out = queue.Queue(maxsize=1000)

    # Optional DB writer (remote-only)
    from GPU.prod.db.mongo_writer import MongoWriterThread
    db_writer = None
    if getattr(cfg, 'db_enabled', True):
        db_writer = MongoWriterThread(
            latest_store=store,
            active_cameras=cfg.active_cameras,
            atlas_uri=cfg.db_uri,
            database=cfg.db_database,
            collection=cfg.db_collection,
            age_threshold=cfg.db_age_threshold,
            batch_interval_s=cfg.db_batch_interval_s,
        )
        db_writer.start()

    # Optional Redis-backed ReID pipeline (non-blocking)
    from GPU.prod.reid.redis_client import RedisClient
    from GPU.prod.reid.feature_worker import FeatureWorker
    from GPU.prod.reid.reid_worker import ReIDWorker
    from GPU.prod.reid.redis_writer import RedisWriter
    from GPU.prod.reid.association_router import AssociationRouter

    # Neighbor map from user memory
    neighbor_map = {1:[2,5],2:[1,3,6],3:[2,4,7],4:[3],5:[1,6,8],6:[5,7,2,9],7:[6,3,10],8:[5,9],9:[8,10,6],10:[9,11,7],11:[10]}

    reid_enabled = getattr(cfg, 'reid_enabled', False)
    router = None
    redis_client = None
    if reid_enabled:
        redis_client = RedisClient(cfg.redis_uri)
        redis_client.connect()

    # Optional GUI
    gui = None
    if show_gui:
        try:
            from GPU.prod.gui.gui_display import GridGUIThread
            gui = GridGUIThread(window_name="Tracking", latest_store=store, camera_ids=cfg.active_cameras,
                                 tracking_mode=True, target_hz=8.0)
            gui.start()
        except Exception as e:
            logger.warning(f"GUI launch failed: {e}")
            gui = None

    # Start camera threads
    cams = []
    for cid in cfg.active_cameras:
        cw = CameraWorker(camera_id=cid, latest_store=store, frame_skip=cfg.frame_skip, resize=cfg.resize,
                          debug=(cfg.log_level == 'DEBUG'))
        if cw.connect():
            cw.start()
            cams.append(cw)

    if not cams:
        logger.error("No cameras connected")
        return {'ok': False, 'reason': 'no_cameras'}

    # Choose tracking path based on config flag
    if getattr(cfg, 'use_batched_tracking', False):
        # Start detector worker (batched)
        detector = DetectorWorker(cfg=cfg, latest_store=store, output_queue=det_out)
        detector.start()
        # Start orchestrator to consume detections and emit tracks
        orchestrator = TrackingOrchestrator(cfg=cfg, det_queue=det_out, out_queue=trk_out, camera_ids=cfg.active_cameras, latest_store=store)
        orchestrator.start()
        tracker_manager = orchestrator  # for router compatibility
    else:
        # Resolve BoT-SORT YAML path relative to repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        botsort_yaml = os.path.abspath(os.path.join(repo_root, 'cv', 'GPU', 'configs', 'warehouse_botsort.yaml'))
        tm = TrackerManager(camera_ids=cfg.active_cameras, latest_store=store, output_queue=trk_out,
                            model_path=cfg.model_path, device=cfg.device, confidence=cfg.confidence,
                            botsort_yaml_path=botsort_yaml, log_interval_s=cfg.log_interval_s,
                            debug=(cfg.log_level == 'DEBUG'))
        tm.start()
        tracker_manager = tm

    logger.info(f"Tracking pipeline running for {duration_s}s...")

    # AFTER trackers started, optionally bring up ReID pipeline (needs tracker_manager ref)
    if reid_enabled and redis_client is not None:
        import queue as qmod
        feat_in = qmod.Queue(maxsize=200)
        feat_out = qmod.Queue(maxsize=200)
        reid_out = qmod.Queue(maxsize=200)
        rw_in = qmod.Queue(maxsize=500)
        from GPU.prod.reid.feature_worker import FeatureWorker
        from GPU.prod.reid.reid_worker import ReIDWorker
        from GPU.prod.reid.redis_writer import RedisWriter
        from GPU.prod.reid.association_router import AssociationRouter
        feat_workers = [FeatureWorker(store, feat_in, feat_out, cfg.yolo_cls_model_path, device=cfg.device, name=f'FeatureWorker-{i+1}') for i in range(max(1, cfg.feature_workers))]
        for w in feat_workers: w.start()
        reid_workers = [ReIDWorker(feat_out, reid_out, redis_client, neighbor_map, cfg.reid_same_cam_window_s, cfg.reid_neighbor_window_s, cfg.reid_topk, cfg.reid_similarity_threshold, name=f'ReIDWorker-{i+1}') for i in range(max(1, cfg.reid_workers))]
        for w in reid_workers: w.start()
        redis_writer = RedisWriter(rw_in, redis_client)
        redis_writer.start()
        router = AssociationRouter(cfg, store, neighbor_map, db_writer, redis_client, tracker_manager, feat_workers, reid_workers, feat_in, feat_out, reid_out, rw_in)
        router.start()

    # Monitor loop with duration
    start = time.time()
    last = start
    total_msgs = 0
    per_cam = {cid: 0 for cid in cfg.active_cameras}
    max_q = 0

    # System Processing FPS (post-tracker) sliding window metrics
    system_window_s = float(getattr(cfg, 'system_fps_window_s', 10.0))
    system_log_interval_s = float(getattr(cfg, 'system_fps_log_interval_s', 2.0))
    sys_last_log = start
    per_cam_post_times = {cid: deque() for cid in cfg.active_cameras}  # deque of completion timestamps
    per_cam_lat = {cid: deque() for cid in cfg.active_cameras}         # deque of (completion_ts, latency_ms)

    def _prune_deques(now_ts: float):
        for cid in cfg.active_cameras:
            dq = per_cam_post_times[cid]
            while dq and (now_ts - dq[0]) > system_window_s:
                dq.popleft()
            lq = per_cam_lat[cid]
            while lq and (now_ts - lq[0][0]) > system_window_s:
                lq.popleft()

    def _percentiles(values, ps=(50, 95)):
        if not values:
            return {p: 0.0 for p in ps}
        vals = sorted(values)
        out = {}
        n = len(vals)
        for p in ps:
            # nearest-rank method
            k = max(1, int(round(p/100.0 * n)))
            out[p] = float(vals[k-1])
        return out

    try:
        while time.time() - start < duration_s:
            try:
                msg = trk_out.get(timeout=0.5)
                total_msgs += 1
                now_ts = time.time()
                cid = msg['camera_id']
                per_cam[cid] = per_cam.get(cid, 0) + 1

                # Record post-tracker completion and latency using capture ts carried as msg['ts']
                per_cam_post_times[cid].append(now_ts)
                cap_ts = float(msg.get('ts', now_ts))
                lat_ms = max(0.0, (now_ts - cap_ts) * 1000.0)
                per_cam_lat[cid].append((now_ts, lat_ms))

                # Prune to window and compute sliding-window SystemFPS for this camera for GUI
                _prune_deques(now_ts)
                sys_fps_cam = len(per_cam_post_times[cid]) / max(1e-6, system_window_s)
                msg['sys_fps'] = sys_fps_cam

                # Route to Redis ReID pipeline (if enabled) else direct to DB
                try:
                    if router is not None:
                        router.enqueue(msg)
                    else:
                        if db_writer is not None:
                            db_writer.enqueue(msg)
                    if total_msgs % 50 == 0:
                        logger.info(f"Enqueue: msg #{total_msgs}, tracks={len(msg.get('tracks', []))}")
                except Exception as e:
                    logger.error(f"Enqueue failed: {e}")
                if show_gui and gui is not None:
                    gui.push_msg(msg)
            except queue.Empty:
                pass
            now_loop = time.time()
            max_q = max(max_q, trk_out.qsize())
            # Existing periodic summary
            if now_loop - last >= cfg.log_interval_s:
                elapsed = now_loop - start
                rate = total_msgs / max(1e-6, elapsed)
                logger.info(f"elapsed={elapsed:.1f}s total_msgs={total_msgs} rate={rate:.1f}/s q={trk_out.qsize()} max_q={max_q}")
                last = now_loop
            # New: System Processing FPS and latency log
            if now_loop - sys_last_log >= system_log_interval_s:
                _prune_deques(now_loop)
                per_cam_sysfps = {cid: (len(per_cam_post_times[cid]) / max(1e-6, system_window_s)) for cid in cfg.active_cameras}
                total_sysfps = sum(per_cam_sysfps.values())
                per_cam_p = {}
                for cid in cfg.active_cameras:
                    vals = [v for _, v in per_cam_lat[cid]]
                    pct = _percentiles(vals)
                    per_cam_p[cid] = {'p50_ms': round(pct[50], 1), 'p95_ms': round(pct[95], 1)}
                logger.info(f"SystemFPS(post-tracker, window={system_window_s:.0f}s): total={total_sysfps:.2f} per_cam={per_cam_sysfps} lat_ms={per_cam_p}")
                sys_last_log = now_loop
    finally:
        # Stop tracking path
        if getattr(cfg, 'use_batched_tracking', False):
            try:
                orchestrator.stop()
                orchestrator.join(timeout=2)
            except Exception:
                pass
            try:
                detector.stop()
                detector.join(timeout=2)
            except Exception:
                pass
        else:
            tm.stop()
        for cw in cams:
            cw.stop()
            cw.join(timeout=2)
        # Stop ReID pipeline
        if router is not None:
            try:
                router.stop()
                router.join(timeout=2)
            except Exception:
                pass
        # Stop DB writer last
        try:
            if db_writer is not None:
                db_writer.stop()
                db_writer.join(timeout=2)
        except Exception:
            pass
        logger.info("Stopped.")

    # Build and print a detailed final report
    elapsed = max(1e-6, time.time() - start)
    total_fps = total_msgs / elapsed

    # Per-camera rates at the app dequeue point (DB/event rate)
    per_cam_fps = {cid: (cnt / elapsed) for cid, cnt in per_cam.items()}

    # Compute final System Processing FPS (post-tracker) over the sliding window and latency percentiles
    now_final = time.time()
    _prune_deques(now_final)
    per_cam_sysfps_final = {cid: (len(per_cam_post_times[cid]) / max(1e-6, system_window_s)) for cid in cfg.active_cameras}
    total_sysfps_final = sum(per_cam_sysfps_final.values())
    per_cam_lat_p_final = {}
    for cid in cfg.active_cameras:
        vals = [v for _, v in per_cam_lat[cid]]
        pct = _percentiles(vals)
        per_cam_lat_p_final[cid] = {'p50_ms': round(pct[50], 1), 'p95_ms': round(pct[95], 1)}

    # Compose a structured log with ingest/detect/track where available
    logger.info("==== FINAL PERFORMANCE REPORT ====")
    logger.info(f"Duration: {elapsed:.1f}s | Cameras: {cfg.active_cameras}")
    logger.info(f"App dequeue FPS (total): {total_fps:.2f}")
    logger.info(f"App dequeue FPS (per camera): {per_cam_fps}")
    logger.info(f"SystemFPS(post-tracker, window={system_window_s:.0f}s): total={total_sysfps_final:.2f} per_cam={per_cam_sysfps_final} lat_ms={per_cam_lat_p_final}")
    logger.info("Notes: Ingest/Detection/Tracking FPS are logged during the run per component.\n"
                "- Ingest: CameraWorker[CID] ingest_fps=...\n"
                "- Detection: DetectorWorker total_det_msgs_per_s=... (or per_cam via counts)\n"
                "- Tracking: CameraTracker[CID] trk_fps=...\n"
                "- System Processing FPS: frames exiting tracker regardless of DB gating\n"
                "- End-to-end (DB): MongoWriter E2E_system_fps=... (frames reaching DB stage)")

    summary = {
        'ok': True,
        'mode': 'tracking',
        'cameras': cfg.active_cameras,
        'duration_s': duration_s,
        'total_msgs': total_msgs,
        'avg_msgs_per_sec': total_fps,
        'per_camera_msgs': per_cam,
        'max_queue_size': max_q,
    }
    logger.info(f"SUMMARY: {summary}")
    return summary


if __name__ == '__main__':
    # Example run: subset of cameras first
    run_detection_only(active_cameras=[8, 9])

