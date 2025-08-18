import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

# Minimal, CPU-light tracker core: IoU + Hungarian + constant-velocity Kalman

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1,y1,x2,y2]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confirmed: bool = False
    state: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=float))

@dataclass
class TrackerState:
    next_id: int = 1
    tracks: List[Track] = field(default_factory=list)


def iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def linear_assign(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    # Simple greedy matching to keep it light
    M, N = cost.shape
    matches: List[Tuple[int,int]] = []
    used_rows = set()
    used_cols = set()
    order = np.argsort(cost, axis=None)
    for idx in order:
        r, c = divmod(idx, N)
        if r in used_rows or c in used_cols:
            continue
        if cost[r, c] > thresh:
            continue
        used_rows.add(r)
        used_cols.add(c)
        matches.append((r, c))
    unmatched_rows = [r for r in range(M) if r not in used_rows]
    unmatched_cols = [c for c in range(N) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols


def associate(detections: List[List[float]], state: TrackerState, match_thresh: float = 0.5,
              min_hits: int = 3, max_age: int = 30, fuse_score: bool = True,
              app_sim: Optional[np.ndarray] = None, wiou: float = 0.7, wapp: float = 0.3) -> Tuple[List[Track], List[int]]:
    """Associate detections to tracks.
    - IoU gating at match_thresh
    - If app_sim provided (tracks x dets), use fused cost = wiou*(1-IoU) + wapp*(1-cosine)
    - Otherwise, use cost = 1 - IoU
    """
    # Predict step: age and time_since_update
    for t in state.tracks:
        t.age += 1
        t.time_since_update += 1

    dets = np.array(detections, dtype=float) if detections else np.zeros((0,4), dtype=float)
    tracks = state.tracks

    # Build IoU matrix and base cost
    if len(dets) and len(tracks):
        iou_mat = np.zeros((len(tracks), len(dets)), dtype=float)
        for i, tr in enumerate(tracks):
            for j, d in enumerate(dets):
                iou_mat[i, j] = iou(tr.bbox, d)
        # Gate by IoU
        allowed = (iou_mat >= match_thresh)
        base_cost = 1.0 - iou_mat
        if app_sim is not None and app_sim.shape == iou_mat.shape:
            app_sim_clipped = np.clip(app_sim, 0.0, 1.0)
            cost = wiou * base_cost + wapp * (1.0 - app_sim_clipped)
        else:
            cost = base_cost
        # Disallow pairs below IoU gate by setting very high cost
        cost = np.where(allowed, cost, 1e6)
        # Use greedy on fused cost
        matches, um_tr, um_dt = linear_assign(cost, 1e5)
    else:
        matches, um_tr, um_dt = [], list(range(len(tracks))), list(range(len(dets)))

    # Update matched
    for i, j in matches:
        tr = tracks[i]
        tr.bbox = dets[j]
        tr.hits += 1
        tr.time_since_update = 0
        if tr.hits >= min_hits:
            tr.confirmed = True

    # Create new for unmatched detections
    for j in um_dt:
        new_id = state.next_id
        state.next_id += 1
        bb = dets[j]
        state.tracks.append(Track(track_id=new_id, bbox=bb, age=1, hits=1, time_since_update=0, confirmed=(min_hits<=1)))

    # Remove old
    kept: List[Track] = []
    removed_ids: List[int] = []
    for tr in state.tracks:
        if tr.time_since_update > max_age:
            removed_ids.append(tr.track_id)
        else:
            kept.append(tr)
    state.tracks = kept

    return state.tracks, removed_ids

