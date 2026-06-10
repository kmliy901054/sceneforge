"""sceneforge/render — headless render backends (ARCHITECTURE.md §5.4/§5.5).

``get_renderer(cfg)`` implements the §5.5 fallback policy:

  1. primary: pyrender on the current ``PYOPENGL_PLATFORM`` (EGL, set by
     ``sceneforge.compat``);
  2. fallback #1: one retry with ``PYOPENGL_PLATFORM=osmesa`` (env-var flip +
     purge/re-import of OpenGL + pyrender);
  3. loud ``RendererUnavailable`` pointing at fallback #2 (golden-run replay)
     and the contingency recipe in ``numpy_backend.py``.

The chosen backend is cached process-wide (its OffscreenRenderer is persistent
per §5.4).
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Optional

import sceneforge.compat  # noqa: F401  (np.infty + PYOPENGL_PLATFORM before pyrender)

from sceneforge.render.types import ComposedScene, InstanceLabel, RenderResult

if TYPE_CHECKING:  # pragma: no cover
    from sceneforge.render.pyrender_backend import PyrenderBackend

__all__ = [
    "ComposedScene",
    "InstanceLabel",
    "RenderResult",
    "RendererUnavailable",
    "get_renderer",
]

logger = logging.getLogger(__name__)

_GL_MODULE_PREFIXES = ("OpenGL", "pyrender", "sceneforge.render.pyrender_backend")

_backend: "Optional[PyrenderBackend]" = None


class RendererUnavailable(RuntimeError):
    """No GL backend could be initialized (EGL and osmesa both failed)."""


def _purge_gl_modules() -> None:
    """Drop OpenGL/pyrender (and our backend module) so a re-import re-reads
    PYOPENGL_PLATFORM — PyOpenGL picks its platform at import time."""
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in _GL_MODULE_PREFIXES):
            del sys.modules[name]


def _probe_backend(platform: str, *, reimport: bool) -> "PyrenderBackend":
    """Import the pyrender backend under ``platform`` and smoke-test it
    (tiny offscreen render including a SEG pass)."""
    os.environ["PYOPENGL_PLATFORM"] = platform
    if reimport:
        _purge_gl_modules()
    module = importlib.import_module("sceneforge.render.pyrender_backend")
    backend = module.PyrenderBackend()
    backend.probe()
    return backend


def get_renderer(cfg: Any = None) -> "PyrenderBackend":
    """Return the process-wide render backend (§5.5 policy: EGL → osmesa → loud).

    ``cfg`` (AppConfig) is accepted per the §2 interface; the current backend
    needs nothing from it — resolution and camera arrive per ``render_scene``
    call. Raises :class:`RendererUnavailable` if both GL platforms fail.
    """
    global _backend
    if _backend is not None:
        return _backend

    first = os.environ.get("PYOPENGL_PLATFORM", "egl")
    attempts: list[tuple[str, Exception]] = []

    try:
        _backend = _probe_backend(first, reimport=False)
        logger.info("render backend ready on platform %r", first)
        return _backend
    except Exception as exc:  # noqa: BLE001 — any GL init failure routes to retry
        attempts.append((first, exc))
        logger.warning("renderer init failed on %r: %s", first, exc)

    if first != "osmesa":
        try:
            _backend = _probe_backend("osmesa", reimport=True)
            logger.warning("EGL unavailable — running on osmesa (§5.5 fallback #1)")
            return _backend
        except Exception as exc:  # noqa: BLE001
            attempts.append(("osmesa", exc))
            logger.warning("renderer init failed on 'osmesa': %s", exc)

    detail = "; ".join(f"{p}: {type(e).__name__}: {e}" for p, e in attempts)
    raise RendererUnavailable(
        "No working GL platform for headless rendering (ARCHITECTURE.md §5.5). "
        f"Attempts — {detail}. Next steps: (fallback #2) load the committed "
        "golden-run replay directory for the demo; (contingency) implement the "
        "numpy rasterizer per the recipe + activation checklist in "
        "sceneforge/render/numpy_backend.py (NORMATIVE projection equations in "
        "sceneforge/render/camera.py §3.1)."
    )
