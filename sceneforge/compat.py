"""sceneforge/compat.py — P0 environment shims (ARCHITECTURE.md §0).

``import sceneforge.compat`` is line 1 of ``app.py`` and of every script/test
entrypoint — BEFORE anything that may import pyrender or PyOpenGL.

Two repairs, verified working on this machine (2026-06-10):

1. ``PYOPENGL_PLATFORM=egl`` must be in the environment before the first
   pyrender/PyOpenGL import for headless rendering. Set via ``setdefault`` so an
   explicit operator override (e.g. ``osmesa``) wins; the osmesa retry on EGL
   failure is handled by ``sceneforge.render.get_renderer`` (§5.5).
2. pyrender 0.1.45 references ``np.infty`` (``pyrender/mesh.py:87``), which was
   removed in numpy 2.0. Restore the alias — do NOT downgrade numpy.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402  (env var must be set first)

if not hasattr(np, "infty"):  # numpy >= 2.0 removed the alias
    np.infty = np.inf  # type: ignore[attr-defined]
