"""sceneforge/scene/compose.py — SceneSpec → ComposedScene + viewer.glb (ARCHITECTURE.md §5.3).

``build(spec, library, out_dir) -> ComposedScene``:

- static entries: far floor (depth-background only, §3.1) + table slab with its
  TOP surface at z = 0;
- per ``ObjectSpec``: a COPIED mesh (``AssetLibrary.get_mesh`` hands out copies
  — §5.1 aliasing review fix) with per-instance vertex tint and
  ``T = trans(x, y, z) @ rotz(yaw)``;
- viewer export (§5.3 review fixes): a ``trimesh.Scene`` of table + objects
  only — NO floor (Model3D auto-frames the whole asset and the floor makes the
  scene read as a clump on a slab) — rotated −π/2 about X so world +Z maps to
  glTF +Y (otherwise the table renders standing on edge in three.js), written
  to ``<out_dir>/viewer.glb``.

The render-path geometry (``ComposedScene.instances``/``static``) stays
un-rotated and keeps the floor.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

from sceneforge.assets.library import AssetLibrary
from sceneforge.render.types import ComposedScene
from sceneforge.spec import ObjectSpec, SceneSpec

FLOOR_TOP_Z_M = -0.75      #: floor plane height (§3.1 — depth-map background only)
FLOOR_SIZE_M = 2.5         #: floor extent (§3.1: 2.5 × 2.5 m)
FLOOR_THICKNESS_M = 0.02
TABLE_THICKNESS_M = 0.04   #: slab thickness (§3.1); table TOP surface is z = 0
FLOOR_RGBA = (90, 90, 95, 255)
TABLE_RGBA = (168, 144, 118, 255)
VIEWER_GLB_NAME = "viewer.glb"

#: world +Z-up → glTF +Y-up for the Model3D viewer (§5.3).
_VIEWER_Y_UP = rotation_matrix(-math.pi / 2.0, [1.0, 0.0, 0.0])


def object_transform(obj: ObjectSpec) -> np.ndarray:
    """``T = trans(x_m, y_m, z_m) @ rotz(yaw_deg)`` per §5.3, (4,4) float64."""
    T = rotation_matrix(math.radians(obj.yaw_deg), [0.0, 0.0, 1.0])
    T[:3, 3] = [obj.x_m, obj.y_m, obj.z_m]
    return T


def build(
    spec: SceneSpec, library: AssetLibrary, out_dir: Union[str, Path]
) -> ComposedScene:
    """Compose one resolved layout into renderer-ready (mesh, transform) pairs.

    Writes ``<out_dir>/viewer.glb`` (table + objects only, +Y-up, no floor) and
    returns the ``ComposedScene`` (§4.3): explicit pairs, never a
    ``trimesh.Scene`` — the EGL backend builds ``pyrender.Scene`` manually and
    keeps Node handles for ``seg_node_map`` (§5.4).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    floor = trimesh.creation.box(
        extents=(FLOOR_SIZE_M, FLOOR_SIZE_M, FLOOR_THICKNESS_M)
    )
    floor.apply_translation([0.0, 0.0, FLOOR_TOP_Z_M - FLOOR_THICKNESS_M / 2.0])
    floor.visual.vertex_colors = FLOOR_RGBA

    table = trimesh.creation.box(
        extents=(spec.table.width_m, spec.table.depth_m, TABLE_THICKNESS_M)
    )
    table.apply_translation([0.0, 0.0, -TABLE_THICKNESS_M / 2.0])  # TOP at z = 0
    table.visual.vertex_colors = TABLE_RGBA

    static: list[tuple[str, trimesh.Trimesh, np.ndarray]] = [
        ("_floor", floor, np.eye(4)),
        ("_table", table, np.eye(4)),
    ]

    instances: list[tuple[int, trimesh.Trimesh, np.ndarray]] = []
    for obj in spec.objects:
        mesh = library.get_mesh(obj.asset_id, obj.scale)  # already a COPY (§5.1)
        mesh.visual.vertex_colors = (*obj.color_rgb, 255)  # per-instance tint
        instances.append((obj.instance_id, mesh, object_transform(obj)))

    glb_path = _export_viewer_glb(table, instances, out_dir / VIEWER_GLB_NAME)
    return ComposedScene(instances=instances, static=static, glb_path=str(glb_path))


def _export_viewer_glb(
    table: trimesh.Trimesh,
    instances: list[tuple[int, trimesh.Trimesh, np.ndarray]],
    path: Path,
) -> Path:
    """Export table + objects (NO floor) rotated to +Y-up as ``viewer.glb`` (§5.3).

    Meshes are copied into the viewer scene so the export can never alias the
    render-path geometry.
    """
    scene = trimesh.Scene()
    scene.add_geometry(table.copy(), node_name="_table", geom_name="_table")
    for instance_id, mesh, transform in instances:
        name = f"instance_{instance_id}"
        scene.add_geometry(
            mesh.copy(), transform=transform, node_name=name, geom_name=name
        )
    scene.apply_transform(_VIEWER_Y_UP)
    scene.export(str(path))
    return path
