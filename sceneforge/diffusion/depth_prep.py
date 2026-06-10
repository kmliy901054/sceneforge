"""Depth z-buffer -> ControlNet control image (ARCHITECTURE.md §7.3).

The depth ControlNet (diffusers/controlnet-depth-sdxl-1.0) was trained on
DPT-MiDaS *relative inverse depth* (disparity ~ 1/z), min/max normalized — not
a linear flip. In disparity space, near-range (object-vs-table) contrast is
amplified and the far floor compressed, which is exactly what label fidelity
needs. ``mode="disparity"`` is the default (confirmed by the M1 A/B);
``mode="linear"`` (the v1.0 flip) is retained for comparison.

NEVER-RESIZE CONTRACT (§4.4 / §7.3): the depth map must be rendered at the
diffusion target resolution; this function refuses to resample. Resampling
softens silhouette edges and costs mask IoU downstream.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

DepthMode = Literal["disparity", "linear"]

__all__ = ["depth_to_control", "DepthMode"]


def depth_to_control(
    depth_m: np.ndarray,
    resolution: int = 768,
    mode: DepthMode = "disparity",
) -> Image.Image:
    """Convert a metric z-buffer into an 8-bit RGB ControlNet control image.

    Args:
        depth_m: (H, W) float depth in meters; 0 = no hit (§4.3). Must already
            be exactly ``resolution`` x ``resolution`` — never resized here.
        resolution: target diffusion resolution (asserted, not applied).
        mode: ``"disparity"`` (MiDaS-style 1/z, percentile [2, 98] normalized,
            invalid -> 0; DEFAULT) or ``"linear"`` (legacy inverted linear
            normalization, M1 A/B loser, kept for comparison).

    Returns:
        PIL RGB image, near = bright, far/invalid = black (MiDaS semantics).
    """
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be (H, W); got shape {depth_m.shape}")
    if depth_m.shape != (resolution, resolution):
        raise ValueError(
            f"never-resize contract violated: depth is {depth_m.shape}, expected "
            f"({resolution}, {resolution}) — render depth at the target resolution (§7.3)"
        )

    depth_m = depth_m.astype(np.float32, copy=False)
    valid = depth_m > 0
    if not valid.any():
        out = np.zeros_like(depth_m)
    elif mode == "disparity":
        disp = 1.0 / np.maximum(depth_m, 1e-6)
        lo, hi = np.percentile(disp[valid], [2, 98])
        out = np.clip((disp - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        out[~valid] = 0.0  # no-hit = infinitely far (MiDaS semantics)
    elif mode == "linear":
        lo, hi = np.percentile(depth_m[valid], [2, 98])
        out = np.clip((hi - depth_m) / max(hi - lo, 1e-9), 0.0, 1.0)
        out[~valid] = 0.0
    else:
        raise ValueError(f"unknown depth mode: {mode!r}")

    u8 = (out * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")
