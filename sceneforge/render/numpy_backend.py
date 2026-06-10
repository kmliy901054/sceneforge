"""sceneforge/render/numpy_backend.py — CONTINGENCY STUB (ARCHITECTURE.md §5.5).

NOT IMPLEMENTED by design. EGL+SEG was re-verified working on this machine
(2026-06-10); a second full renderer guards a disproven failure mode, and a
Python triangle loop realistically runs 5–12 s/frame — demo-degrading even if
built. The fallback chain is: EGL → osmesa retry (``get_renderer``) → golden-
run replay → THIS backend, implemented only if the M1 step-0 768² EGL assert
fails (paid for by cutting builders 15→10, cut-order #8).

Every entry point raises ``NotImplementedError`` carrying the implementation
recipe and the activation checklist below, so the ``RenderResult`` contract
stays alive without shipping dead code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from sceneforge.render.types import ComposedScene, RenderResult
    from sceneforge.spec import CameraSpec, SceneSpec

RECIPE = """\
numpy_backend is a CONTINGENCY STUB (ARCHITECTURE.md §5.5) — not implemented.

Implementation recipe (z-buffer scanline rasterizer, ~150 lines):
  1. Camera: pose = camera.pose_from_orbit(spec_camera); world→camera transform
     is inv(pose). K = camera.intrinsics(W, H, yfov_deg).
  2. For every (instance_id, mesh, T) in composed.instances AND every static
     (floor/table) entry: vertices_w = (T @ homog(mesh.vertices)); transform to
     camera frame; keep triangles fully in front of the camera (z_c < -znear).
  3. Project per-vertex with the NORMATIVE §3.1 equations — EXACTLY these, the
     classic cross-backend mirror bug lives here:
         u = cx + fx * x_c / (-z_c)
         v = cy - fy * y_c / (-z_c)
  4. Rasterize each triangle over its integer pixel bbox with barycentric
     coverage; interpolate 1/z (perspective-correct); nearest-depth wins the
     z-buffer (float32 meters as -z_c; 0 = no hit).
  5. seg_ids: winning instance_id per pixel for instance meshes; static
     floor/table write DEPTH ONLY, never seg (background stays 0) — mirrors
     the EGL SEG pass where unmapped nodes are skipped.
  6. color: flat-shaded preview from face normals · light direction (uint8);
     cosmetic only.
  7. Return RenderResult(width, height, color, depth_m, seg_ids, instances,
     camera_pose=pose.astype(float32), K=K.astype(float32)) — identical
     contract to pyrender_backend.render_scene.

Activation checklist (MANDATORY before this backend may serve a run, §5.5):
  [ ] M1 step-0 768x768 EGL color+depth+SEG assert failed AND the osmesa
      retry failed (otherwise do not build this).
  [ ] Cross-backend equivalence vs the EGL backend on the M1 scenes:
      per-instance mask IoU >= 0.98 and mean |depth diff| < 2 mm.
  [ ] seg ids round-trip exactly (byte-identical across two renders).
  [ ] Budget honesty: expect 5-12 s/frame in pure Python; cut builders 15→10
      (cut-order #8) to pay for the implementation time.
"""


class NumpyBackend:
    """Contingency CPU rasterizer — every method raises NotImplementedError."""

    name = "numpy-stub"

    def render_scene(
        self,
        composed: "ComposedScene",
        camera: "CameraSpec",
        width: int,
        height: int,
        *,
        spec: "Optional[SceneSpec]" = None,
    ) -> "RenderResult":
        raise NotImplementedError(RECIPE)

    def probe(self, size: int = 64) -> None:
        raise NotImplementedError(RECIPE)


def render_scene(
    composed: "ComposedScene",
    camera: "CameraSpec",
    width: int,
    height: int,
    *,
    spec: "Optional[SceneSpec]" = None,
) -> "RenderResult":
    """Module-level mirror of the backend interface — raises with the recipe."""
    raise NotImplementedError(RECIPE)
