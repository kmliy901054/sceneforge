"""sceneforge/render/camera.py — orbit pose + intrinsics (ARCHITECTURE.md §3.1).

Conventions (NORMATIVE)
-----------------------
World frame: right-handed, +Z up; meters and degrees. Table TOP surface is the
plane z = 0. Camera orbit: ``azimuth_deg`` CCW from +X around +Z;
``elevation_deg`` above the z = 0 plane; ``distance_m`` from ``look_at``
(``look_at`` and ``yfov`` are SERVER-PINNED, never LLM-settable)::

    eye = look_at + d * [cos(el)*cos(az), cos(el)*sin(az), sin(el)]

Pyrender convention (camera looks down its −Z, up ≈ +Z world)::

    z_axis = normalize(eye − look_at)
    x_axis = normalize(cross([0, 0, 1], z_axis))
    y_axis = cross(z_axis, x_axis)
    pose columns = [x_axis, y_axis, z_axis, eye]

Elevation is clamped to [10, 80]° (enforced by ``CameraSpec``), so ``z_axis``
is never parallel to world +Z and the cross product is always well-defined.

Intrinsics (square images): ``yfov_deg = 50`` →
``fx = fy = H / (2·tan(yfov/2))``, ``cx = W/2``, ``cy = H/2``.

Projection equations (NORMATIVE — the classic cross-backend mirror bug lives
here). For a camera-frame point ``(x_c, y_c, z_c)`` with ``z_c < 0`` in front
of the camera::

    u = cx + fx * x_c / (−z_c)
    v = cy − fy * y_c / (−z_c)

Image frame is COCO standard: origin top-left, bbox ``[x, y, w, h]`` pixels.
Any alternative backend (``numpy_backend.py``) must reproduce these equations
EXACTLY; acceptance is the §5.5 contingency-activation checklist —
per-instance mask IoU ≥ 0.98 and mean |depth diff| < 2 mm vs the EGL backend.
"""
from __future__ import annotations

import math

import numpy as np

from sceneforge.spec import CameraSpec

_WORLD_UP = np.array([0.0, 0.0, 1.0])


def eye_from_orbit(camera: CameraSpec) -> np.ndarray:
    """Orbit eye position, (3,) float64: ``look_at + d·[cos el·cos az, cos el·sin az, sin el]``."""
    az = math.radians(camera.azimuth_deg)
    el = math.radians(camera.elevation_deg)
    offset = camera.distance_m * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    return np.asarray(camera.look_at, dtype=np.float64) + offset


def pose_from_orbit(camera: CameraSpec) -> np.ndarray:
    """World-from-camera pose, (4,4) float64, per the §3.1 orbit equations.

    Columns of the rotation block are [x_axis, y_axis, z_axis]; translation is
    the eye. The camera looks down its −Z toward ``camera.look_at``.
    """
    look_at = np.asarray(camera.look_at, dtype=np.float64)
    eye = eye_from_orbit(camera)
    z_axis = eye - look_at
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(_WORLD_UP, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def intrinsics(width: int, height: int, yfov_deg: float = 50.0) -> np.ndarray:
    """Pinhole K, (3,3) float64: ``fx = fy = H/(2·tan(yfov/2))``, ``cx = W/2``, ``cy = H/2``."""
    f = height / (2.0 * math.tan(math.radians(yfov_deg) / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
