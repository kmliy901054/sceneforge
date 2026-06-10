"""scripts/make_contact_sheet.py — M2 acceptance artifact (ARCHITECTURE.md §5.1, TASKS.md M2).

SELF-CONTAINED pyrender EGL contact sheet: renders all 15 procedural builders
into a 4x4 grid (15 assets + 1 title cell), each cell a color | depth | seg
triplet, written to outputs/contact_sheet.png.

Deliberately does NOT import sceneforge.render (owned by the render backend);
the orbit camera below inlines the normative §3.1 equations.

Usage:
    python scripts/make_contact_sheet.py [--out outputs/contact_sheet.png] [--size 224]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `sceneforge`

import sceneforge.compat  # noqa: F401,E402  — FIRST sceneforge import (§0): np.infty + EGL

import argparse  # noqa: E402
import math  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402

from sceneforge.assets.builders import BUILDERS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LABEL_H = 22  # px strip under each triplet for the asset name


def orbit_pose(look_at: np.ndarray, az_deg: float, el_deg: float, dist: float) -> np.ndarray:
    """§3.1 orbit camera: eye = look_at + d*[cos el cos az, cos el sin az, sin el];
    pyrender looks down -Z, columns [x_axis, y_axis, z_axis, eye]."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = look_at + dist * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    z_axis = eye - look_at
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x_axis, y_axis, z_axis, eye
    return pose


def render_asset(renderer: "pyrender.OffscreenRenderer", asset_id: str) -> tuple:
    """Render one builder: returns (color, depth_m, seg_red) arrays."""
    mesh = BUILDERS[asset_id]()
    mesh.visual.vertex_colors = (205, 205, 210, 255)
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.4, 0.4, 0.4))
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    look_at = mesh.bounds.mean(axis=0)
    dist = max(2.0 * float(np.linalg.norm(mesh.extents)), 0.20)
    pose = orbit_pose(look_at, az_deg=35.0, el_deg=30.0, dist=dist)
    scene.add(pyrender.PerspectiveCamera(yfov=math.radians(50.0)), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    color, depth = renderer.render(scene)
    seg, _ = renderer.render(
        scene, flags=pyrender.RenderFlags.SEG, seg_node_map={node: (1, 0, 0)}
    )
    return color, depth, seg[:, :, 0]


def make_cell(color: np.ndarray, depth: np.ndarray, seg: np.ndarray, name: str, size: int) -> np.ndarray:
    """One grid cell: [color | disparity-style depth | seg mask] + name strip (RGB)."""
    valid = depth > 0
    depth_vis = np.zeros_like(depth)
    if valid.any():
        disp = 1.0 / np.maximum(depth, 1e-6)
        lo, hi = disp[valid].min(), disp[valid].max()
        depth_vis = np.where(valid, (disp - lo) / max(hi - lo, 1e-9), 0.0)
    depth_rgb = np.repeat((depth_vis * 255).astype(np.uint8)[..., None], 3, axis=2)
    seg_rgb = np.zeros((size, size, 3), np.uint8)
    seg_rgb[seg > 0] = (70, 200, 120)
    triplet = np.concatenate([color[:, :, :3], depth_rgb, seg_rgb], axis=1)
    strip = np.full((_LABEL_H, triplet.shape[1], 3), 18, np.uint8)
    cv2.putText(strip, name, (6, _LABEL_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([triplet, strip], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_REPO_ROOT / "outputs" / "contact_sheet.png"))
    parser.add_argument("--size", type=int, default=224, help="per-pass render size (px)")
    args = parser.parse_args()

    size = args.size
    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        cells = [
            make_cell(*render_asset(renderer, asset_id), asset_id, size)
            for asset_id in BUILDERS
        ]
    finally:
        renderer.delete()

    # 16th cell: title card (4x4 grid for 15 assets).
    title = np.full_like(cells[0], 18)
    for i, line in enumerate(["SceneForge", "15 procedural assets", "color | depth | seg"]):
        cv2.putText(title, line, (10, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (235, 235, 235), 2, cv2.LINE_AA)
    cells.append(title)

    rows = [np.concatenate(cells[r * 4:(r + 1) * 4], axis=1) for r in range(4)]
    sheet = np.concatenate(rows, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        print(f"FAILED to write {out_path}", file=sys.stderr)
        return 1
    print(f"contact sheet: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
