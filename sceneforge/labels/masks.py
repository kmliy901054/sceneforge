"""sceneforge/labels/masks.py — seg ids → instance masks/bboxes/RLE (ARCHITECTURE.md §8.1)
plus the depth16.png on-disk codec (§3.1).

Per object: ``mask = (seg_ids == instance_id)``; instances with ``area < 200`` px are
skipped; bbox via ``cv2.boundingRect``; RLE via ``pycocotools.mask.encode`` with
``counts`` ascii-decoded. ALL numeric fields are cast to python ``int`` at label
construction — numpy int64 is not JSON-serializable (§8.1 review fix). Masks are
computed once per layout and shared by every style variant (§4.5: label transfer
is a no-op — the product's thesis).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from pycocotools import mask as mask_util

#: §8.1 — instances below this pixel area are dropped from RenderResult.instances.
MIN_AREA_PX = 200
#: §5.4 — target-visibility guard threshold (orchestrator bumps camera below this).
TARGET_MIN_AREA_PX = 1000


@dataclass
class _InstanceLabelShim:
    """Field-exact stand-in for ``sceneforge.render.types.InstanceLabel`` (§4.3).

    ``render/types.py`` is built concurrently; :func:`_label_cls` always prefers
    the real dataclass and falls back to this shim only while it is unavailable.
    """

    instance_id: int
    asset_id: str
    category: str
    is_target: bool
    bbox_xywh: tuple[int, int, int, int]  # python ints (json-safe), COCO order
    area_px: int                          # python int
    rle: dict                             # {"size": [H, W], "counts": str} ascii


def _label_cls() -> type:
    """Late import of the §4.3 InstanceLabel dataclass (duck-type fallback)."""
    try:
        from sceneforge.render.types import InstanceLabel
        return InstanceLabel
    except Exception:  # module not yet written by the render agent
        return _InstanceLabelShim


def encode_rle(mask: np.ndarray) -> dict:
    """Binary mask → COCO RLE ``{"size": [H, W], "counts": <ascii str>}`` (§4.3)."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return {
        "size": [int(s) for s in rle["size"]],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else str(counts),
    }


def extract_instances(seg_ids: np.ndarray, spec_or_objects: Any) -> list[Any]:
    """SEG-pass id image → ``list[InstanceLabel]`` (§8.1).

    Args:
        seg_ids: (H, W) integer array; 0 = background, k = instance_id k (§4.3).
        spec_or_objects: a ``SceneSpec`` (its ``.objects`` is used) or any iterable
            of objects exposing ``instance_id / asset_id / category / is_target``.

    Instances with fewer than :data:`MIN_AREA_PX` visible pixels are skipped.
    Every numeric field is a python ``int``; RLE counts are ascii ``str``.
    """
    objects = getattr(spec_or_objects, "objects", spec_or_objects)
    cls = _label_cls()
    labels: list[Any] = []
    for obj in objects:
        instance_id = int(obj.instance_id)
        mask = seg_ids == instance_id
        area = int(mask.sum())
        if area < MIN_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        labels.append(
            cls(
                instance_id=instance_id,
                asset_id=str(obj.asset_id),
                category=str(obj.category),
                is_target=bool(obj.is_target),
                bbox_xywh=(int(x), int(y), int(w), int(h)),
                area_px=area,
                rle=encode_rle(mask),
            )
        )
    return labels


def write_depth16(path: Union[str, Path], depth_m: np.ndarray) -> Path:
    """Write depth meters → ``depth16.png`` per the §3.1 on-disk contract.

    uint16 millimeters, ``depth_mm = round(depth_m·1000).clip(0, 65535)``,
    0 = no-hit; max representable 65.535 m. Round-trip asserted in test_masks.py.
    """
    path = Path(path)
    depth_mm = np.round(depth_m.astype(np.float64) * 1000.0).clip(0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise IOError(f"cv2.imwrite failed for {path}")
    return path


def read_depth16(path: Union[str, Path]) -> np.ndarray:
    """Read a §3.1 ``depth16.png`` → (H, W) float32 meters; 0 = no-hit."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read depth16 png: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 png, got {raw.dtype}")
    return raw.astype(np.float32) / 1000.0
