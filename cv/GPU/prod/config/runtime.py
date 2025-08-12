import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RuntimeConfig:
    # Cameras
    active_cameras: List[int] = None  # set via default_config(...)

    # Capture
    target_fps_per_camera: float = 5.0
    frame_skip: int = 4  # for ~20-21 FPS input
    resize: Tuple[int, int] = (1280, 720)
    latest_store_depth: int = 2  # configurable to 5

    # Detection / Tracking
    model_path: str = os.path.join('cv', 'GPU', 'custom_yolo.pt')
    device: str = 'cuda:0'
    confidence: float = 0.5
    max_batch: int = 12
    batch_window_ms: int = 12
    detection_batch_mode: str = 'latest_full'  # 'latest_full' or 'new_only'

    # Tracking association thresholds (aligned with warehouse_botsort.yaml/test)
    match_thresh: float = 0.5
    min_hits: int = 10
    max_age: int = 250
    track_buffer: int = 200
    fuse_score: bool = True

    # New: enable batched-detection-based tracking orchestrator
    use_batched_tracking: bool = False
    # New: enable batched-detection-based tracking orchestrator
    use_batched_tracking: bool = False

    # ReID / Redis (feature-based cross-camera ID persistence)
    reid_enabled: bool = True
    redis_uri: str = 'redis://localhost:6379/0'
    reid_similarity_threshold: float = 0.5
    reid_same_cam_window_s: int = 120
    reid_neighbor_window_s: int = 300
    reid_topk: int = 100
    feature_workers: int = 4
    reid_workers: int = 2
    redis_writer_workers: int = 1
    yolo_cls_model_path: str = 'yolov8n-cls.pt'

    # Database (remote-only Atlas)
    db_enabled: bool = True
    db_uri: str = 'mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/'
    db_database: str = 'WARP'
    db_collection: str = 'detections'
    db_age_threshold: int = 1
    db_batch_interval_s: float = 2.0

    # Logging
    log_level: str = 'INFO'  # 'DEBUG' to enable more logs
    log_interval_s: float = 2.0  # periodic summary

    # System Processing FPS (post-tracker) reporting
    system_fps_window_s: float = 10.0  # sliding window length
    system_fps_log_interval_s: float = 2.0  # log cadence for SystemFPS


# Simple helper to build a default config

def default_config(cameras: List[int]) -> RuntimeConfig:
    cfg = RuntimeConfig(active_cameras=cameras)
    return cfg

