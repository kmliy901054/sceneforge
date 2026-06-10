"""sceneforge/labels/overlay.py — annotation overlays (ARCHITECTURE.md §8.2).

``draw_overlay(img, instances, mode)`` with modes ``"boxes" | "masks" | "both" |
"off"``: fixed per-category palette; boxes 3 px + label chips; masks alpha 0.35;
the target instance is drawn thicker and starred. Raw and overlay PNGs are
written at generation time by the orchestrator, so the UI toggle is a pure
cached-path swap.
"""
from __future__ import annotations

import hashlib
import logging
from colorsys import hsv_to_rgb
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_util

from sceneforge.labels.coco import CATEGORIES

logger = logging.getLogger(__name__)

OverlayMode = Literal["boxes", "masks", "both", "off"]
OVERLAY_MODES: tuple[str, ...] = ("boxes", "masks", "both", "off")

MASK_ALPHA = 0.35   # §8.2
BOX_PX = 3          # §8.2: boxes 3 px
TARGET_BOX_PX = 5   # §8.2: target thicker

#: Fixed per-category palette (§8.2), RGB, aligned with the §5.1 library order.
#: (Kelly/Trubetskoy-style maximally-distinct colors.)
_PALETTE_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),    # mug
    (60, 180, 75),    # bowl
    (255, 225, 25),   # plate
    (0, 130, 200),    # cup
    (245, 130, 48),   # bottle
    (145, 30, 180),   # can
    (70, 240, 240),   # box
    (240, 50, 230),   # book
    (210, 245, 60),   # pan
    (250, 190, 212),  # ball
    (0, 128, 128),    # pot
    (220, 190, 255),  # screwdriver
    (170, 110, 40),   # hammer
    (255, 250, 200),  # cutting_board
    (128, 0, 0),      # laptop
)
PALETTE: dict[str, tuple[int, int, int]] = dict(zip(CATEGORIES, _PALETTE_COLORS))


def category_color(category: str) -> tuple[int, int, int]:
    """RGB color for a category: fixed palette entry, else a deterministic
    hash-seeded hue (md5, not ``hash()`` — stable across processes)."""
    if category in PALETTE:
        return PALETTE[category]
    hue = hashlib.md5(category.encode("utf-8")).digest()[0] / 255.0
    r, g, b = hsv_to_rgb(hue, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _decode_mask(rle: dict) -> np.ndarray:
    """COCO RLE (ascii str counts, §4.3) → (H, W) uint8 binary mask."""
    return mask_util.decode(dict(rle))


def _draw_chip(arr: np.ndarray, text: str, anchor: tuple[int, int],
               color: tuple[int, int, int]) -> None:
    """Filled label chip with auto black/white text, clamped inside the image."""
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, fs, ft)
    h, w = arr.shape[:2]
    left = int(np.clip(anchor[0], 0, max(0, w - tw - 6)))
    top = int(np.clip(anchor[1] - th - baseline - 4, 0, max(0, h - th - baseline - 4)))
    cv2.rectangle(arr, (left, top), (left + tw + 6, top + th + baseline + 4),
                  tuple(int(c) for c in color), -1)
    txt_color = (0, 0, 0) if sum(color) / 3 > 130 else (255, 255, 255)
    cv2.putText(arr, text, (left + 3, top + th + 2), font, fs, txt_color, ft, cv2.LINE_AA)


def draw_overlay(img: Any, instances: Sequence[Any], mode: OverlayMode = "both") -> np.ndarray:
    """Render instance annotations onto a copy of ``img`` (§8.2).

    Args:
        img: (H, W, 3) uint8 RGB array (or anything ``np.asarray`` coerces, e.g.
            ``PIL.Image``). Never mutated.
        instances: §4.3 ``InstanceLabel``s (duck-typed: ``bbox_xywh``, ``rle``,
            ``category``, ``is_target``).
        mode: "off" returns an unmodified copy; "masks" alpha-blends at 0.35;
            "boxes" draws 3 px rectangles + label chips (target: 5 px, starred —
            ascii ``*`` since Hershey fonts cannot render '★'); "both" = both.
    """
    if mode not in OVERLAY_MODES:
        raise ValueError(f"mode must be one of {OVERLAY_MODES}, got {mode!r}")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.array(arr[..., :3], dtype=np.uint8, copy=True)
    if mode == "off":
        return arr

    if mode in ("masks", "both"):
        for inst in instances:
            mask = _decode_mask(inst.rle)
            if mask.shape != arr.shape[:2]:
                logger.warning("overlay: RLE size %s != image size %s — mask skipped",
                               mask.shape, arr.shape[:2])
                continue
            sel = mask.astype(bool)
            color = np.array(category_color(inst.category), dtype=np.float32)
            arr[sel] = (arr[sel].astype(np.float32) * (1.0 - MASK_ALPHA)
                        + color * MASK_ALPHA).astype(np.uint8)

    if mode in ("boxes", "both"):
        for inst in instances:
            x, y, w, h = (int(v) for v in inst.bbox_xywh)
            color = tuple(int(c) for c in category_color(inst.category))
            is_target = bool(inst.is_target)
            cv2.rectangle(arr, (x, y), (x + w, y + h), color,
                          TARGET_BOX_PX if is_target else BOX_PX)
            label = f"* {inst.category} *" if is_target else str(inst.category)
            _draw_chip(arr, label, (x, y), color)
    return arr
