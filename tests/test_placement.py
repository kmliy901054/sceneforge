"""tests/test_placement.py — placement + composition tests (ARCHITECTURE.md §5.2/§5.3).

Covers: clamp into the inset table rect; collision-free output (pairwise circle
check, radii + 8 mm); determinism (same seed → identical placements) and
idempotency; target immobility; drop-lowest-priority-non-target on unresolvable
layouts (+ grounding_log entry); z_m = 0; two-mugs-two-colors tint isolation
through compose (§5.1 aliasing review fix); viewer.glb exists, loads via
trimesh, contains NO floor geometry and is +Y-up while the render path keeps
the floor un-rotated. No EGL/GPU required.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0)

import math

import numpy as np
import pytest
import trimesh

from sceneforge.assets.library import AssetLibrary
from sceneforge.scene import compose
from sceneforge.scene.placement import MARGIN_M, resolve_placements
from sceneforge.spec import ObjectSpec, SceneSpec, StyleSpec, TableSpec

RED = (255, 0, 0)
GREEN = (0, 255, 0)
_TOL = 1e-9


# ---------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def library() -> AssetLibrary:
    return AssetLibrary()


def obj(
    instance_id: int,
    asset_id: str,
    x: float,
    y: float,
    *,
    yaw: float = 0.0,
    scale: float = 1.0,
    target: bool = False,
    color: tuple[int, int, int] = (180, 180, 180),
) -> ObjectSpec:
    return ObjectSpec(
        instance_id=instance_id,
        asset_id=asset_id,
        category=asset_id,
        requested=asset_id,
        x_m=x,
        y_m=y,
        yaw_deg=yaw,
        scale=scale,
        is_target=target,
        color_rgb=color,
    )


def make_spec(
    objects: list[ObjectSpec], seed: int = 42, table: TableSpec | None = None
) -> SceneSpec:
    return SceneSpec(
        task="pick the red mug from a cluttered kitchen table",
        seed=seed,
        table=table or TableSpec(),
        objects=objects,
        styles=[
            StyleSpec(name="clean_lab", prompt="a photo of mug on a table, in a clean lab")
        ],
    )


def cluttered_spec(seed: int = 42) -> SceneSpec:
    """Heavily overlapping pile (incl. coincident centers → seeded tie-break)."""
    return make_spec(
        [
            obj(1, "mug", 0.0, 0.0, target=True, color=RED),
            obj(2, "bowl", 0.0, 0.0),       # coincident with the target
            obj(3, "bottle", 0.02, 0.01),
            obj(4, "can", 0.02, 0.01),      # coincident with the bottle
            obj(5, "book", -0.03, 0.02),
            obj(6, "plate", 0.9, 0.6),      # out of bounds → clamped into the pile side
        ],
        seed=seed,
    )


def assert_collision_free(spec: SceneSpec, library: AssetLibrary) -> None:
    """Pairwise 2D circle check: dist ≥ r_i + r_j + 8 mm for every pair (§5.2)."""
    objects = spec.objects
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            min_dist = (
                library.footprint_radius(a.asset_id, a.scale)
                + library.footprint_radius(b.asset_id, b.scale)
                + MARGIN_M
            )
            dist = math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
            assert dist >= min_dist - _TOL, (
                f"instances {a.instance_id}/{b.instance_id}: {dist:.4f} < {min_dist:.4f}"
            )


def assert_in_inset_rect(spec: SceneSpec, library: AssetLibrary) -> None:
    for o in spec.objects:
        r = library.footprint_radius(o.asset_id, o.scale)
        assert abs(o.x_m) <= spec.table.width_m / 2 - r + _TOL
        assert abs(o.y_m) <= spec.table.depth_m / 2 - r + _TOL


# ------------------------------------------------------------- placement ----
def test_clamps_into_table_rect_inset_by_footprint(library):
    spec = make_spec(
        [obj(1, "mug", 0.0, 0.0, target=True), obj(2, "plate", 5.0, -5.0)]
    )
    resolved = resolve_placements(spec, library)
    assert_in_inset_rect(resolved, library)
    assert_collision_free(resolved, library)


def test_collision_free_and_z_zero(library):
    resolved = resolve_placements(cluttered_spec(), library)
    assert len(resolved.objects) == 6  # default table fits all six (nothing dropped)
    assert_collision_free(resolved, library)
    assert_in_inset_rect(resolved, library)
    assert all(o.z_m == 0.0 for o in resolved.objects)


def test_determinism_same_seed_identical_placements(library):
    a = resolve_placements(cluttered_spec(seed=7), library)
    b = resolve_placements(cluttered_spec(seed=7), library)
    assert [(o.instance_id, o.x_m, o.y_m, o.z_m) for o in a.objects] == [
        (o.instance_id, o.x_m, o.y_m, o.z_m) for o in b.objects
    ]


def test_idempotent_on_resolved_output(library):
    once = resolve_placements(cluttered_spec(), library)
    twice = resolve_placements(once, library)
    assert [(o.x_m, o.y_m) for o in twice.objects] == [
        (o.x_m, o.y_m) for o in once.objects
    ]
    assert twice.grounding_log == once.grounding_log


def test_input_spec_never_mutated(library):
    spec = cluttered_spec()
    before = spec.model_dump()
    resolve_placements(spec, library)
    assert spec.model_dump() == before


def test_target_never_moves(library):
    # Target starts inside the inset rect → the step-1 clamp is a no-op and the
    # collision sweeps must move every OTHER object around it (§5.2).
    spec = make_spec(
        [
            obj(1, "mug", 0.10, 0.05, target=True, color=RED),
            obj(2, "bowl", 0.10, 0.05),     # coincident with the target
            obj(3, "can", 0.12, 0.06),
            obj(4, "cup", 0.08, 0.04),
        ]
    )
    resolved = resolve_placements(spec, library)
    target = next(o for o in resolved.objects if o.is_target)
    assert (target.x_m, target.y_m) == (0.10, 0.05)
    assert_collision_free(resolved, library)


def test_drops_lowest_priority_non_target_and_logs(library):
    # Min-size table + three big cutting boards: geometrically impossible to
    # separate → both non-targets must be dropped, target retained (§5.2 step 3).
    spec = make_spec(
        [
            obj(1, "cutting_board", 0.0, 0.0, scale=1.2, target=True),
            obj(2, "cutting_board", 0.01, 0.0, scale=1.2),
            obj(3, "cutting_board", -0.01, 0.0, scale=1.2),
        ],
        table=TableSpec(width_m=0.6, depth_m=0.5),
    )
    resolved = resolve_placements(spec, library)
    assert [o.instance_id for o in resolved.objects] == [1]
    assert resolved.objects[0].is_target
    drops = [e for e in resolved.grounding_log if e.get("event") == "placement_drop"]
    assert len(drops) == 2
    # Lowest-priority (last in list order) non-target goes first.
    assert [d["instance_id"] for d in drops] == [3, 2]
    assert all(d["asset_id"] == "cutting_board" for d in drops)
    assert_collision_free(resolved, library)


# ----------------------------------------------------------------- compose --
def test_two_mugs_two_colors_tint_isolation(library, tmp_path):
    # §5.1 review fix: in-place tinting of a CACHED mesh would alias colors
    # across instances — two mugs at the same scale must keep distinct tints.
    spec = resolve_placements(
        make_spec(
            [
                obj(1, "mug", -0.15, 0.0, target=True, color=RED),
                obj(2, "mug", 0.15, 0.0, color=GREEN),
            ]
        ),
        library,
    )
    composed = compose.build(spec, library, tmp_path)
    colors = {
        iid: np.asarray(mesh.visual.vertex_colors) for iid, mesh, _ in composed.instances
    }
    assert np.all(colors[1][:, :3] == RED)
    assert np.all(colors[2][:, :3] == GREEN)
    # The library canonical stays untinted (fresh copies are not red/green).
    canonical = np.asarray(library.get_mesh("mug").visual.vertex_colors)
    assert not np.all(canonical[:, :3] == RED)
    assert not np.all(canonical[:, :3] == GREEN)


def test_compose_transforms_and_statics(library, tmp_path):
    spec = resolve_placements(
        make_spec(
            [
                obj(1, "mug", 0.10, 0.05, yaw=90.0, target=True, color=RED),
                obj(2, "book", -0.20, -0.10),
            ]
        ),
        library,
    )
    composed = compose.build(spec, library, tmp_path)

    # Render path keeps floor + table statics, un-rotated (+Z-up, §5.3).
    names = [name for name, _, _ in composed.static]
    assert names == ["_floor", "_table"]
    floor, table = composed.static[0][1], composed.static[1][1]
    assert floor.bounds[1, 2] == pytest.approx(compose.FLOOR_TOP_Z_M)
    assert table.bounds[1, 2] == pytest.approx(0.0)              # table TOP at z = 0
    assert table.extents[:2] == pytest.approx(
        [spec.table.width_m, spec.table.depth_m]
    )

    # T = trans(x, y, z) @ rotz(yaw) per instance (§5.3).
    by_id = {iid: T for iid, _, T in composed.instances}
    mug = next(o for o in spec.objects if o.instance_id == 1)
    T = by_id[1]
    assert T[:3, 3] == pytest.approx([mug.x_m, mug.y_m, 0.0])
    yaw = math.radians(mug.yaw_deg)
    expected_r = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(T[:3, :3], expected_r, atol=1e-12)


def test_viewer_glb_exists_loads_and_has_no_floor(library, tmp_path):
    spec = resolve_placements(
        make_spec(
            [
                obj(1, "mug", -0.15, 0.0, target=True, color=RED),
                obj(2, "bottle", 0.15, 0.10),
                obj(3, "bowl", 0.0, -0.15),
            ]
        ),
        library,
    )
    composed = compose.build(spec, library, tmp_path)

    glb_path = tmp_path / "viewer.glb"
    assert composed.glb_path == str(glb_path)
    assert glb_path.is_file() and glb_path.stat().st_size > 0

    loaded = trimesh.load(str(glb_path))
    assert isinstance(loaded, trimesh.Scene)
    # Table + one geometry per object — and NO floor (§5.3).
    assert len(loaded.geometry) == 1 + len(spec.objects)
    assert not any("floor" in name.lower() for name in loaded.geometry)
    # The 2.5 m floor would dominate any extent; table width must be the max.
    assert float(loaded.extents.max()) == pytest.approx(spec.table.width_m, rel=1e-3)
    # +Y-up (§5.3): heights map to Y — far smaller than the table footprint.
    extents = loaded.extents  # [width(X), height(Y), depth(Z)] after the −π/2 X-rotation
    assert extents[1] < 0.5
    assert extents[0] == pytest.approx(spec.table.width_m, rel=1e-3)
    assert extents[2] == pytest.approx(spec.table.depth_m, rel=1e-3)
