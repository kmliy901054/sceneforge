"""tests/test_viewsweep.py — feature B camera-grid contract (CPU-only).

The sweep's pure pieces (grid product, azimuth wrap, §3.1 clamps, server-pinned
yfov/look_at) are tested by importing the script module directly; the full
GPU path is exercised by running the script for real (see sweep_meta.json of a
run). No GL/LLM/diffusion is touched here.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (repo rule, §0)

import importlib.util
from pathlib import Path

import pytest

from sceneforge.spec import CameraSpec

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_viewsweep.py"
_spec = importlib.util.spec_from_file_location("forge_viewsweep", _SCRIPT)
viewsweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viewsweep)


def test_parse_float_list():
    assert viewsweep.parse_float_list("0,45, -90") == [0.0, 45.0, -90.0]
    assert viewsweep.parse_float_list(None) is None
    assert viewsweep.parse_float_list("  ") is None


def test_default_grid_evenly_spaced_azimuths():
    base = CameraSpec(azimuth_deg=35.0, elevation_deg=30.0, distance_m=1.1)
    grid = viewsweep.build_view_grid(base, n_views=8)
    assert len(grid) == 8
    assert grid[0].azimuth_deg == pytest.approx(35.0)
    azs = [c.azimuth_deg for c in grid]
    assert len(set(round(a, 3) for a in azs)) == 8         # all distinct
    steps = {round((azs[i + 1] - azs[i]) % 360.0, 3) for i in range(7)}
    assert steps == {45.0}                                  # even 360/8 spacing
    for cam in grid:                                        # same scene framing
        assert cam.elevation_deg == 30.0 and cam.distance_m == 1.1
        assert cam.yfov_deg == base.yfov_deg and cam.look_at == base.look_at
        assert -180.0 <= cam.azimuth_deg <= 180.0


def test_explicit_lists_form_full_product_with_defaults():
    base = CameraSpec(azimuth_deg=0.0, elevation_deg=40.0, distance_m=1.3)
    grid = viewsweep.build_view_grid(
        base, n_views=99, azimuths=[0.0, 90.0], elevations=[20.0, 60.0],
        distances=[1.0, 2.0])
    assert len(grid) == 8  # 2 × 2 × 2 product; n_views ignored
    combos = {(c.azimuth_deg, c.elevation_deg, c.distance_m) for c in grid}
    assert len(combos) == 8
    # an unset axis falls back to the base camera's value
    grid2 = viewsweep.build_view_grid(base, azimuths=[10.0, 20.0, 30.0])
    assert len(grid2) == 3
    assert all(c.elevation_deg == 40.0 and c.distance_m == 1.3 for c in grid2)


def test_grid_clamps_and_wraps_into_spec_bounds():
    base = CameraSpec()
    grid = viewsweep.build_view_grid(
        base, azimuths=[270.0, -270.0], elevations=[5.0, 85.0],
        distances=[0.1, 9.0])
    for cam in grid:  # every CameraSpec validates — clamped BEFORE construction
        assert cam.azimuth_deg in (-90.0, 90.0)             # wrapped
        assert cam.elevation_deg in (10.0, 80.0)            # §3.1 clamp
        assert cam.distance_m in (0.5, 2.5)
    # ±180° is the same azimuth; both wrap to the canonical -180 (in-bounds)
    assert viewsweep.wrap_azimuth(180.0) == pytest.approx(-180.0)
    assert viewsweep.wrap_azimuth(-180.0) == pytest.approx(-180.0)
    assert viewsweep.wrap_azimuth(45.0) == pytest.approx(45.0)


def test_bad_view_count_raises():
    with pytest.raises(ValueError):
        viewsweep.build_view_grid(CameraSpec(), n_views=0)
