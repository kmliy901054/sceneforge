"""sceneforge/scene/placement.py — deterministic placement resolver (ARCHITECTURE.md §5.2).

``resolve_placements(spec, library) -> SceneSpec`` — deterministic, idempotent,
no LLM round-trips (§4.2):

1. Clamp each ``(x_m, y_m)`` into the table rect inset by the object's
   footprint radius (XY circumradius from ``AssetLibrary.footprint_radius``).
2. Collision = 2D circle overlap (radii + 8 mm margin). Resolve over ≤ 40
   iterations, pushing overlapping pairs apart along their center line — the
   target NEVER moves; every object is re-clamped each iteration; a seeded
   ``default_rng(spec.seed)`` breaks coincident-center ties.
3. Still overlapping after 40 iterations → drop the lowest-priority (last in
   list order) non-target object involved in a remaining collision, append a
   ``grounding_log`` entry, and re-resolve.
4. ``z_m = 0`` for all objects. Geometry errors NEVER go back to the LLM.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

from numpy.random import Generator, default_rng

from sceneforge.assets.library import AssetLibrary
from sceneforge.spec import ObjectSpec, SceneSpec

logger = logging.getLogger(__name__)

MARGIN_M = 0.008      #: 8 mm circle-collision margin (§5.2)
MAX_ITERATIONS = 40   #: pair-separation sweep budget per §5.2
_TIE_EPS = 1e-9       #: centers closer than this are "coincident" → seeded rng direction
_PUSH_EPS = 1e-6      #: extra separation so resolved pairs are strictly clear of the margin

_Radii = dict[int, float]
_Clamp = Callable[[ObjectSpec], None]


def resolve_placements(spec: SceneSpec, library: AssetLibrary) -> SceneSpec:
    """Clamp + collision-separate all object placements per ARCHITECTURE.md §5.2.

    Returns a NEW ``SceneSpec`` (the input is never mutated): every object is
    inside the table rect inset by its footprint radius, pairwise circle
    collisions (radii + 8 mm) are resolved with the target pinned in place,
    ``z_m == 0`` everywhere, and any dropped object is recorded in
    ``grounding_log``. Pure function of ``(spec, library)`` — same seed gives
    byte-identical placements; re-running on its own output is a no-op.
    """
    out = spec.model_copy(deep=True)
    rng = default_rng(out.seed)
    radii: _Radii = {
        o.instance_id: library.footprint_radius(o.asset_id, o.scale) for o in out.objects
    }
    half_w = out.table.width_m / 2.0
    half_d = out.table.depth_m / 2.0

    def clamp(obj: ObjectSpec) -> None:
        r = radii[obj.instance_id]
        obj.x_m = _clamp_axis(obj.x_m, half_w, r)
        obj.y_m = _clamp_axis(obj.y_m, half_d, r)

    for obj in out.objects:           # step 1 (all objects, target included) + step 4
        clamp(obj)
        obj.z_m = 0.0

    objects = list(out.objects)
    while not _separate(objects, radii, clamp, rng):       # steps 2–3
        victim = _lowest_priority_collider(objects, radii)
        if victim is None:  # pragma: no cover — collisions always involve a non-target
            break
        objects.remove(victim)
        logger.warning(
            "placement: dropping instance %d (%s) — unresolved collision",
            victim.instance_id, victim.asset_id,
        )
        out.grounding_log.append(
            {
                "event": "placement_drop",
                "instance_id": victim.instance_id,
                "requested": victim.requested,
                "asset_id": victim.asset_id,
                "reason": f"unresolved collision after {MAX_ITERATIONS} placement iterations",
            }
        )
    out.objects = objects
    return out


# --------------------------------------------------------------------- helpers
def _clamp_axis(value: float, half_extent: float, radius: float) -> float:
    """Clamp one coordinate into ``[-half_extent + radius, half_extent - radius]``.

    A degenerate inset (object footprint wider than the half-extent) collapses
    to the table center — placement never raises on oversized objects.
    """
    lo, hi = -half_extent + radius, half_extent - radius
    if lo > hi:
        return 0.0
    return min(max(value, lo), hi)


def _min_dist(a: ObjectSpec, b: ObjectSpec, radii: _Radii) -> float:
    return radii[a.instance_id] + radii[b.instance_id] + MARGIN_M


def _separate(
    objects: list[ObjectSpec], radii: _Radii, clamp: _Clamp, rng: Generator
) -> bool:
    """≤ 40 sweeps pushing overlapping pairs apart along their center line (§5.2 step 2).

    The target never moves (its partner absorbs the whole push); non-target
    pairs split the push evenly; coincident centers take a seeded random
    direction; every object is re-clamped after each sweep. Returns True iff
    the layout is collision-free.
    """
    for _ in range(MAX_ITERATIONS):
        moved = False
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                min_dist = _min_dist(a, b, radii)
                dx, dy = b.x_m - a.x_m, b.y_m - a.y_m
                dist = math.hypot(dx, dy)
                if dist >= min_dist:
                    continue
                if dist <= _TIE_EPS:  # coincident centers: seeded rng breaks the tie
                    theta = float(rng.uniform(0.0, 2.0 * math.pi))
                    ux, uy = math.cos(theta), math.sin(theta)
                else:
                    ux, uy = dx / dist, dy / dist
                push = (min_dist - dist) + _PUSH_EPS
                if a.is_target:                       # target never moves
                    b.x_m += ux * push
                    b.y_m += uy * push
                elif b.is_target:
                    a.x_m -= ux * push
                    a.y_m -= uy * push
                else:
                    a.x_m -= ux * push / 2.0
                    a.y_m -= uy * push / 2.0
                    b.x_m += ux * push / 2.0
                    b.y_m += uy * push / 2.0
                moved = True
        if not moved:
            return True
        for obj in objects:  # re-clamp each iter (target was never moved → no-op)
            clamp(obj)
    return not _any_collision(objects, radii)


def _any_collision(objects: list[ObjectSpec], radii: _Radii) -> bool:
    return _lowest_priority_collider(objects, radii, target_ok=True) is not None


def _lowest_priority_collider(
    objects: list[ObjectSpec], radii: _Radii, target_ok: bool = False
) -> Optional[ObjectSpec]:
    """Last-in-list (lowest-priority) object still in collision (§5.2 step 3).

    With ``target_ok=False`` (the drop path) the target is never returned —
    every colliding pair contains at least one non-target since exactly one
    target exists.
    """
    colliding: set[int] = set()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if math.hypot(b.x_m - a.x_m, b.y_m - a.y_m) < _min_dist(a, b, radii):
                colliding.update((i, j))
    for idx in sorted(colliding, reverse=True):
        if target_ok or not objects[idx].is_target:
            return objects[idx]
    return None
