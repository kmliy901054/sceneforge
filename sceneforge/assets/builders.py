"""sceneforge/assets/builders.py — 15 procedural trimesh asset builders (ARCHITECTURE.md §5.1, §12-A).

Each ``build_<name>(scale: float = 1.0) -> trimesh.Trimesh`` is a deterministic
pure function of ``scale``: a union (concatenation) of trimesh primitives in
canonical pose — resting on z=0, centered in XY, front = +X, ≤ 6k triangles.
Every builder ends with ``mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])``
so ``min(vertices.z) == 0`` exactly (the tabletop plane, §3.1).

ControlNet-depth re-textures everything downstream, so only the depth
silhouette matters (§12-A) — low-poly lathe/primitive shapes are sufficient.
The kitchen set #1–10 is the protected core (cut-order #8).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

import numpy as np
import trimesh
from trimesh import creation
from trimesh.transformations import rotation_matrix

__all__ = ["BUILDERS"] + [
    f"build_{n}" for n in (
        "mug", "bowl", "plate", "cup", "bottle", "can", "box", "book", "pan",
        "ball", "pot", "screwdriver", "hammer", "cutting_board", "laptop",
    )
]

_MAX_FACES = 6000


def _finalize(parts: Sequence[trimesh.Trimesh], scale: float) -> trimesh.Trimesh:
    """Concatenate parts, apply uniform ``scale``, floor min-z to exactly 0 (§5.1)."""
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    mesh = trimesh.util.concatenate(list(parts)) if len(parts) > 1 else parts[0]
    mesh.apply_scale(float(scale))
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    assert len(mesh.faces) <= _MAX_FACES, f"builder exceeded {_MAX_FACES} faces"
    return mesh


def _rot(angle_deg: float, axis: Sequence[float]) -> np.ndarray:
    return rotation_matrix(math.radians(angle_deg), axis)


def _revolve(profile: List[List[float]], sections: int) -> trimesh.Trimesh:
    """Revolve a closed (radius, z) profile about +Z (watertight when both ends at r=0)."""
    return creation.revolve(np.asarray(profile, dtype=np.float64), sections=sections)


# --------------------------------------------------------------------------- #
# 1–10: kitchen set (protected core, §12.7 cut-order #8)
# --------------------------------------------------------------------------- #

def build_mug(scale: float = 1.0) -> trimesh.Trimesh:
    """#1 mug: cylinder body (r .042, h .095) + torus loop handle on +X."""
    body = creation.cylinder(radius=0.042, height=0.095, sections=32)
    body.apply_translation([0.0, 0.0, 0.0475])
    handle = creation.torus(
        major_radius=0.028, minor_radius=0.007, major_sections=24, minor_sections=12
    )
    handle.apply_transform(_rot(90.0, [1, 0, 0]))  # ring into the XZ plane
    handle.apply_translation([0.046, 0.0, 0.050])  # half-embedded in the +X wall
    return _finalize([body, handle], scale)


def build_bowl(scale: float = 1.0) -> trimesh.Trimesh:
    """#2 bowl: revolved hollow bowl profile (r .08, h .055)."""
    profile = [
        [0.000, 0.000], [0.050, 0.000], [0.075, 0.028], [0.080, 0.055],
        [0.072, 0.055], [0.066, 0.030], [0.044, 0.010], [0.000, 0.010],
    ]
    return _finalize([_revolve(profile, sections=48)], scale)


def build_plate(scale: float = 1.0) -> trimesh.Trimesh:
    """#3 plate: revolved shallow dish profile (r .11, h .02)."""
    profile = [
        [0.000, 0.000], [0.070, 0.000], [0.102, 0.012], [0.110, 0.020],
        [0.103, 0.020], [0.094, 0.008], [0.065, 0.006], [0.000, 0.006],
    ]
    return _finalize([_revolve(profile, sections=48)], scale)


def build_cup(scale: float = 1.0) -> trimesh.Trimesh:
    """#4 cup: revolved truncated-cone tumbler, hollow (h .11)."""
    profile = [
        [0.0000, 0.000], [0.0320, 0.000], [0.0420, 0.110],
        [0.0355, 0.110], [0.0285, 0.012], [0.0000, 0.012],
    ]
    return _finalize([_revolve(profile, sections=40)], scale)


def build_bottle(scale: float = 1.0) -> trimesh.Trimesh:
    """#5 bottle: cylinder body + cone shoulder + neck cylinder + cap (h .24)."""
    body = creation.cylinder(radius=0.035, height=0.150, sections=32)
    body.apply_translation([0.0, 0.0, 0.075])
    shoulder = creation.cone(radius=0.035, height=0.080, sections=32)
    shoulder.apply_translation([0.0, 0.0, 0.150])
    neck = creation.cylinder(radius=0.012, height=0.070, sections=24)
    neck.apply_translation([0.0, 0.0, 0.185])
    cap = creation.cylinder(radius=0.016, height=0.030, sections=24)
    cap.apply_translation([0.0, 0.0, 0.225])
    return _finalize([body, shoulder, neck, cap], scale)


def build_can(scale: float = 1.0) -> trimesh.Trimesh:
    """#6 can: cylinder (r .033, h .115)."""
    return _finalize([creation.cylinder(radius=0.033, height=0.115, sections=32)], scale)


def build_box(scale: float = 1.0) -> trimesh.Trimesh:
    """#7 box: .18 x .12 x .08 cuboid."""
    return _finalize([creation.box([0.18, 0.12, 0.08])], scale)


def build_book(scale: float = 1.0) -> trimesh.Trimesh:
    """#8 book: .21 x .15 x .03 cuboid."""
    return _finalize([creation.box([0.21, 0.15, 0.03])], scale)


def build_pan(scale: float = 1.0) -> trimesh.Trimesh:
    """#9 pan: revolved dish (r .11) + box handle along +X."""
    profile = [
        [0.000, 0.000], [0.092, 0.000], [0.105, 0.020], [0.110, 0.040],
        [0.103, 0.040], [0.097, 0.018], [0.086, 0.008], [0.000, 0.008],
    ]
    dish = _revolve(profile, sections=48)
    handle = creation.box([0.130, 0.026, 0.012])
    handle.apply_translation([0.165, 0.0, 0.034])  # overlaps the rim at x ~ .10
    return _finalize([dish, handle], scale)


def build_ball(scale: float = 1.0) -> trimesh.Trimesh:
    """#10 ball: icosphere(3) of radius .05."""
    return _finalize([creation.icosphere(subdivisions=3, radius=0.05)], scale)


# --------------------------------------------------------------------------- #
# 11–15: extended set (cut-order #8 if time bites)
# --------------------------------------------------------------------------- #

def build_pot(scale: float = 1.0) -> trimesh.Trimesh:
    """#11 pot: annulus wall + bottom disk (r .09, h .12) + stub handles on ±X."""
    wall = creation.annulus(r_min=0.082, r_max=0.090, height=0.120, sections=32)
    wall.apply_translation([0.0, 0.0, 0.060])
    bottom = creation.cylinder(radius=0.090, height=0.012, sections=32)
    bottom.apply_translation([0.0, 0.0, 0.006])
    parts = [wall, bottom]
    for sx in (1.0, -1.0):
        h = creation.box([0.025, 0.045, 0.014])
        h.apply_translation([sx * 0.0975, 0.0, 0.100])
        parts.append(h)
    return _finalize(parts, scale)


def build_screwdriver(scale: float = 1.0) -> trimesh.Trimesh:
    """#12 screwdriver: capsule handle + thin shaft, lying flat along +X (l ~.21)."""
    handle = creation.capsule(height=0.060, radius=0.015, count=[16, 16])
    handle.apply_transform(_rot(90.0, [0, 1, 0]))  # axis Z -> X
    handle.apply_translation([-0.060, 0.0, 0.0])   # spans x ~ [-.105, -.015]
    shaft = creation.cylinder(radius=0.0045, height=0.115, sections=16)
    shaft.apply_transform(_rot(90.0, [0, 1, 0]))
    shaft.apply_translation([0.0425, 0.0, 0.0])    # spans x ~ [-.015, +.10]; tip = front
    return _finalize([handle, shaft], scale)


def build_hammer(scale: float = 1.0) -> trimesh.Trimesh:
    """#13 hammer: cylinder handle + crossways box head at +X, lying flat (l .25)."""
    handle = creation.cylinder(radius=0.013, height=0.210, sections=20)
    handle.apply_transform(_rot(90.0, [0, 1, 0]))  # axis along X
    head = creation.box([0.040, 0.110, 0.032])
    head.apply_translation([0.125, 0.0, 0.0])
    return _finalize([handle, head], scale)


def build_cutting_board(scale: float = 1.0) -> trimesh.Trimesh:
    """#14 cutting_board: .30 x .20 x .018 slab."""
    return _finalize([creation.box([0.30, 0.20, 0.018])], scale)


def build_laptop(scale: float = 1.0) -> trimesh.Trimesh:
    """#15 laptop: two thin boxes hinged open 110 deg (footprint .32 wide x .22 deep).

    Base lies flat (front edge at +X); the screen rotates about the hinge line
    (the base's -X top edge, along Y) 110 deg open from the base plane.
    """
    base = creation.box([0.22, 0.32, 0.008])
    base.apply_translation([0.0, 0.0, 0.004])  # spans z [0, .008]
    screen = creation.box([0.22, 0.32, 0.006])
    screen.apply_translation([0.11, 0.0, 0.0])           # hinge edge at local x=0
    screen.apply_transform(_rot(-110.0, [0, 1, 0]))      # lift +X up, lean past vertical
    screen.apply_translation([-0.11, 0.0, 0.008])        # hinge onto the base's back edge
    return _finalize([base, screen], scale)


# Registry: asset_id -> builder, in the §5.1 table order (1-based COCO category order, §8.3).
BUILDERS: Dict[str, Callable[[float], trimesh.Trimesh]] = {
    "mug": build_mug,
    "bowl": build_bowl,
    "plate": build_plate,
    "cup": build_cup,
    "bottle": build_bottle,
    "can": build_can,
    "box": build_box,
    "book": build_book,
    "pan": build_pan,
    "ball": build_ball,
    "pot": build_pot,
    "screwdriver": build_screwdriver,
    "hammer": build_hammer,
    "cutting_board": build_cutting_board,
    "laptop": build_laptop,
}
