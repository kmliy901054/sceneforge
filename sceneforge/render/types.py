"""sceneforge/render/types.py — Scene→Render data contracts (ARCHITECTURE.md §4.3).

Verbatim dataclasses from the architecture document. `ComposedScene` is produced
by `scene/compose.py`; `RenderResult` is produced by a render backend
(`pyrender_backend.py` primary, `numpy_backend.py` contingency stub §5.5) and
consumed by diffusion (§4.4) and labels (§4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class ComposedScene:
    """Renderer input: explicit (mesh, transform) pairs, never a trimesh.Scene.

    REVIEW FIX: pyrender 0.1.45 Scene.from_trimesh_scene DROPS node names
    (scene.py:583 — verified in source). The backend builds pyrender.Scene
    manually from these (mesh, transform) pairs, retaining Node handles for
    seg_node_map. Never use from_trimesh_scene.
    """

    instances: list[tuple[int, trimesh.Trimesh, np.ndarray]]  # (instance_id, mesh COPY, 4x4 T)
    static: list[tuple[str, trimesh.Trimesh, np.ndarray]]     # ("_floor"|"_table", mesh, T)
    glb_path: str                                             # viewer.glb (+Y-up, no floor)


@dataclass
class InstanceLabel:
    """One labeled instance (COCO-ready; all numeric fields python ints, §8.1)."""

    instance_id: int
    asset_id: str
    category: str
    is_target: bool
    bbox_xywh: tuple[int, int, int, int]  # python ints (json-safe), COCO order
    area_px: int                          # python int
    rle: dict                             # {"size": [H, W], "counts": str} ascii


@dataclass
class RenderResult:
    """One layout render: geometry products shared by every style variant (§4.5)."""

    width: int
    height: int
    color: np.ndarray        # (H,W,3) uint8 flat-shaded preview
    depth_m: np.ndarray      # (H,W) float32 meters; 0 = no hit
    seg_ids: np.ndarray      # (H,W) int32; 0=bg, k=instance_id k
    instances: list[InstanceLabel]  # area_px >= 200 only
    camera_pose: np.ndarray         # (4,4) float32
    K: np.ndarray                   # (3,3) float32
