"""tests/test_builders.py — §5.1 builders + AssetLibrary (ARCHITECTURE.md §5.1, §12-A).

Covers: all 15 builders nonempty/watertight-ish/floored/≤6k faces, determinism
(vertex hash across two calls), scale linearity, card schema, library copy
semantics (the two-mugs-two-colors aliasing review fix), and footprint_radius.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0): np.infty shim + EGL env

import hashlib

import numpy as np
import pytest
import trimesh

from sceneforge.assets.builders import BUILDERS
from sceneforge.assets.library import AssetLibrary

ASSET_IDS = list(BUILDERS)
KITCHEN_SET = ["mug", "bowl", "plate", "cup", "bottle", "can", "box", "book", "pan", "ball"]


def _vertex_hash(mesh: trimesh.Trimesh) -> str:
    return hashlib.sha256(np.asarray(mesh.vertices, dtype=np.float64).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def library() -> AssetLibrary:
    return AssetLibrary()


# --------------------------------------------------------------------- builders

def test_registry_has_15_builders_table_order():
    assert len(BUILDERS) == 15
    assert ASSET_IDS[:10] == KITCHEN_SET  # §5.1 table order; cut-order #8 core first


@pytest.mark.parametrize("asset_id", ASSET_IDS)
def test_builder_mesh_quality(asset_id):
    mesh = BUILDERS[asset_id]()
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert 0 < len(mesh.faces) <= 6000
    assert np.isfinite(mesh.vertices).all()
    # rests exactly on the tabletop plane z=0 (§3.1)
    assert abs(mesh.bounds[0, 2]) <= 1e-6
    # watertight-ish: consistently wound primitives, nondegenerate volume
    assert mesh.is_winding_consistent
    assert mesh.convex_hull.volume > 0


@pytest.mark.parametrize("asset_id", ASSET_IDS)
def test_builder_deterministic(asset_id):
    a = BUILDERS[asset_id](1.0)
    b = BUILDERS[asset_id](1.0)
    assert _vertex_hash(a) == _vertex_hash(b)
    assert np.array_equal(a.faces, b.faces)


@pytest.mark.parametrize("asset_id", ASSET_IDS)
def test_builder_scale_scales_bounds(asset_id):
    base = BUILDERS[asset_id](1.0)
    for scale in (0.85, 1.2):
        scaled = BUILDERS[asset_id](scale)
        assert np.allclose(scaled.extents, scale * base.extents, rtol=1e-6, atol=1e-9)
        assert abs(scaled.bounds[0, 2]) <= 1e-6  # still floored after scaling


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_builder_rejects_bad_scale(bad_scale):
    with pytest.raises(ValueError):
        BUILDERS["mug"](bad_scale)


# ---------------------------------------------------------------------- cards

def test_cards_cover_builders_with_required_fields(library):
    cards = library.cards()
    assert {c["asset_id"] for c in cards} == set(BUILDERS)
    for card in cards:
        for field in ("asset_id", "name", "description", "synonyms", "affordances",
                      "nominal_height_m"):
            assert field in card, f"{card['asset_id']}: missing {field}"
        assert len(card["synonyms"]) >= 4, card["asset_id"]
        assert len(card["affordances"]) >= 1, card["asset_id"]
        assert card["nominal_height_m"] > 0


def test_card_nominal_height_matches_builder(library):
    for card in library.cards():
        built = BUILDERS[card["asset_id"]]().extents[2]
        assert built == pytest.approx(card["nominal_height_m"], rel=0.15), card["asset_id"]


# -------------------------------------------------------------------- library

def test_get_mesh_returns_distinct_copies(library):
    a = library.get_mesh("mug")
    b = library.get_mesh("mug")
    assert a is not b
    assert _vertex_hash(a) == _vertex_hash(b)


def test_tint_does_not_alias_cached_canonical(library):
    """Two mugs, two colors (§5.1 review fix): tinting one copy must not bleed."""
    red = library.get_mesh("mug")
    red.visual.vertex_colors = (255, 0, 0, 255)
    other = library.get_mesh("mug")
    assert not np.array_equal(
        np.asarray(red.visual.vertex_colors), np.asarray(other.visual.vertex_colors)
    )
    assert _vertex_hash(other) == _vertex_hash(red)  # geometry untouched, only color


def test_transform_does_not_mutate_cached_canonical(library):
    moved = library.get_mesh("can")
    moved.apply_translation([1.0, 2.0, 3.0])
    fresh = library.get_mesh("can")
    assert abs(fresh.bounds[0, 2]) <= 1e-6
    assert _vertex_hash(fresh) != _vertex_hash(moved)


@pytest.mark.parametrize("asset_id", ASSET_IDS)
def test_footprint_radius_positive_and_scales(asset_id, library):
    r1 = library.footprint_radius(asset_id)
    assert r1 > 0
    assert library.footprint_radius(asset_id, 2.0) == pytest.approx(2.0 * r1, rel=1e-6)
    # circumradius bounds the XY extents of the mesh
    mesh = library.get_mesh(asset_id)
    assert r1 >= 0.5 * max(mesh.extents[0], mesh.extents[1]) - 1e-9


def test_unknown_asset_raises(library):
    with pytest.raises(KeyError):
        library.get_mesh("warp_drive")
    with pytest.raises(KeyError):
        library.footprint_radius("warp_drive")
