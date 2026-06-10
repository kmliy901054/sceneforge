"""sceneforge/render/pyrender_backend.py — primary EGL backend (ARCHITECTURE.md §5.4).

Builds the ``pyrender.Scene`` MANUALLY from ``ComposedScene`` (mesh, transform)
pairs — ``Scene.from_trimesh_scene`` drops node names (verified in 0.1.45
source) and cannot feed ``seg_node_map``. Two passes per layout:

1. color + depth z-buffer (flat-shaded preview, ``depth_m`` float32 meters);
2. SEG pass with ``seg_node_map = {node: (instance_id, 0, 0)}`` — instance_id
   in the RED channel (64 ≪ 255); floor/table omitted → black background. SEG
   disables ``GL_MULTISAMPLE`` and the resolve blit is exact (verified in
   source), so IDs round-trip pixel-perfectly: ``seg_ids = seg[:, :, 0]``.

One persistent ``OffscreenRenderer`` is reused across renders and recreated
only on viewport size change. Instance labels are extracted by
``sceneforge.labels.masks.extract_instances`` (§8.1) when a ``SceneSpec`` is
supplied — imported lazily so the render path stays testable while the labels
module is developed concurrently.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import sceneforge.compat  # noqa: F401  (np.infty + PYOPENGL_PLATFORM before pyrender)
import numpy as np
import pyrender

from sceneforge.render import camera as camera_math
from sceneforge.render.types import ComposedScene, InstanceLabel, RenderResult
from sceneforge.spec import CameraSpec

if TYPE_CHECKING:  # pragma: no cover
    from sceneforge.spec import SceneSpec

_AMBIENT_LIGHT = 0.4
_DIRECTIONAL_INTENSITY = 3.0
_BG_COLOR = (0.0, 0.0, 0.0, 0.0)


def _extract_instances(seg_ids: np.ndarray, spec: "SceneSpec") -> list[InstanceLabel]:
    """Late-bound call into ``sceneforge.labels.masks.extract_instances`` (§5.4 step 4).

    The labels module is owned by a concurrent workstream; a thin lazy import
    keeps this backend importable and testable on its own, with a clear error
    if labels are requested before that module lands.
    """
    try:
        from sceneforge.labels import masks as _masks
    except ImportError as exc:
        raise RuntimeError(
            "render_scene(spec=...) needs sceneforge.labels.masks.extract_instances "
            "(ARCHITECTURE.md §5.4 step 4 / §8.1), which is not available yet. "
            "Call render_scene without spec and extract labels separately, or "
            "implement sceneforge/labels/masks.py."
        ) from exc
    extract = getattr(_masks, "extract_instances", None)
    if extract is None:
        raise RuntimeError(
            "sceneforge.labels.masks exists but does not define extract_instances "
            "(expected per ARCHITECTURE.md §8.1)."
        )
    return extract(seg_ids, spec)


class PyrenderBackend:
    """EGL (or osmesa, §5.5 fallback #1) offscreen renderer with SEG instance pass."""

    name = "pyrender"

    def __init__(self) -> None:
        self._renderer: Optional[pyrender.OffscreenRenderer] = None
        self._size: Optional[tuple[int, int]] = None

    # -- offscreen renderer lifecycle (§5.4: persistent, recreated on size change) --
    def _offscreen(self, width: int, height: int) -> pyrender.OffscreenRenderer:
        if self._renderer is None or self._size != (width, height):
            if self._renderer is not None:
                self._renderer.delete()
            self._renderer = pyrender.OffscreenRenderer(width, height)
            self._size = (width, height)
        return self._renderer

    def close(self) -> None:
        """Release the GL context (idempotent)."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
            self._size = None

    # ------------------------------- main entry -------------------------------
    def render_scene(
        self,
        composed: ComposedScene,
        camera: CameraSpec,
        width: int,
        height: int,
        *,
        spec: "Optional[SceneSpec]" = None,
    ) -> RenderResult:
        """Render one layout → RenderResult (§4.3). Two passes per §5.4.

        ``spec`` (the resolved SceneSpec) enables instance-label extraction via
        labels.masks (§8.1); without it ``RenderResult.instances`` is left empty
        for the caller to fill.
        """
        scene = pyrender.Scene(bg_color=_BG_COLOR, ambient_light=_AMBIENT_LIGHT)

        # Manual scene build (NEVER from_trimesh_scene): keep Node handles.
        for _name, mesh, transform in composed.static:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=transform)
        node_for_instance: dict[int, pyrender.Node] = {}
        for instance_id, mesh, transform in composed.instances:
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=transform)
            node_for_instance[instance_id] = node

        pose = camera_math.pose_from_orbit(camera)
        scene.add(
            pyrender.PerspectiveCamera(yfov=math.radians(camera.yfov_deg)), pose=pose
        )
        scene.add(
            pyrender.DirectionalLight(intensity=_DIRECTIONAL_INTENSITY), pose=pose
        )

        renderer = self._offscreen(width, height)

        # Pass 1: flat-shaded color preview + z-buffer depth (meters, 0 = no hit).
        color, depth = renderer.render(scene)
        color = np.ascontiguousarray(color[:, :, :3], dtype=np.uint8)
        depth_m = np.ascontiguousarray(depth, dtype=np.float32)

        # Pass 2: SEG — instance_id in RED channel; floor/table omitted from the
        # map are skipped by pyrender → black background (§12-B).
        seg_node_map = {
            node: (instance_id, 0, 0)
            for instance_id, node in node_for_instance.items()
        }
        seg, _ = renderer.render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map
        )
        seg_ids = seg[:, :, 0].astype(np.int32)

        instances: list[InstanceLabel] = (
            _extract_instances(seg_ids, spec) if spec is not None else []
        )

        return RenderResult(
            width=width,
            height=height,
            color=color,
            depth_m=depth_m,
            seg_ids=seg_ids,
            instances=instances,
            camera_pose=pose.astype(np.float32),
            K=camera_math.intrinsics(width, height, camera.yfov_deg).astype(np.float32),
        )

    # --------------------------------- probe ----------------------------------
    def probe(self, size: int = 64) -> None:
        """Render a tiny one-box scene incl. SEG pass to validate the GL platform.

        Used by ``get_renderer`` (§5.5) before committing to a backend; raises
        on any GL failure or if the SEG pass produces no instance pixels.
        """
        import trimesh

        box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        box.apply_translation([0.0, 0.0, 0.15])  # rest on z=0
        composed = ComposedScene(
            instances=[(1, box, np.eye(4))], static=[], glb_path=""
        )
        result = self.render_scene(composed, CameraSpec(), size, size)
        if not (result.seg_ids == 1).any():
            raise RuntimeError("SEG probe rendered no instance pixels")
