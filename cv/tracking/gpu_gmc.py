# Stub for GPU-based global motion compensation (GMC)
# Phase 2 will implement OpenCV CUDA / NvOF; for now this returns identity

from typing import Tuple

class GPU_GMC:
    def __init__(self):
        pass

    def estimate(self, prev_frame, curr_frame) -> Tuple[float, float]:
        """Return estimated (dx, dy) pixel shift between frames.
        Currently a stub returning (0,0).
        """
        return 0.0, 0.0

