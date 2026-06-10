"""sceneforge/spec.py — THE data contract. Do not change field names casually.

VERIFIED ON THIS MACHINE: Ollama 0.20.3 format=json_schema enforces structure and
types but NOT numeric minimum/maximum (first live response violated ge=10).
Therefore: clamp_to_bounds() runs on the raw dict BEFORE model_validate(); the
repair loop handles ONLY structural/semantic failures (missing fields,
exactly_one_target, list lengths).

Layers per ARCHITECTURE.md §3.2: LLM-facing models (strict, all-required,
minimal), clamp_to_bounds (numeric sanitation Ollama can't do), and the resolved
contract models. slugify() lives here too (§6.6 rule 1) because style slugs are
the canonical key for filenames/COCO/dropdowns and multiple modules need it.
"""
from __future__ import annotations

import re
from itertools import count
from typing import Iterable, Literal, Union, get_args, get_origin

import annotated_types
from pydantic import BaseModel, Field, field_validator

# Active only at ladder levels with guidance_scale > 1 (L2/L3) — diffusers turns
# CFG on iff guidance_scale > 1; at Lightning cfg=0 negative prompts are dead code.
NEGATIVE_PROMPT = "blurry, deformed, duplicate objects, lowres, watermark, cartoon, painting"


class LLMCameraSpec(BaseModel):                  # what the LLM may control
    azimuth_deg: float = Field(35.0, ge=-180, le=180)
    elevation_deg: float = Field(30.0, ge=10, le=80)
    distance_m: float = Field(1.1, ge=0.5, le=2.5)


class CameraSpec(LLMCameraSpec):                 # resolved: server-pinned fields
    yfov_deg: float = Field(50.0, ge=25, le=80)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.05)


class TableSpec(BaseModel):
    width_m: float = Field(1.2, ge=0.6, le=2.0)
    depth_m: float = Field(0.8, ge=0.5, le=1.5)


# ---------- what the LLM emits (all-required for format=json_schema) ----------
class PlannedObject(BaseModel):
    description: str = Field(..., min_length=2, max_length=80)   # "red ceramic mug" — RAG input
    x_m: float = Field(..., ge=-1.0, le=1.0)
    y_m: float = Field(..., ge=-0.75, le=0.75)
    yaw_deg: float = Field(..., ge=-180, le=180)
    size: Literal["small", "medium", "large"]                    # → scale 0.85/1.0/1.2
    is_target: bool


class LLMScenePlan(BaseModel):
    scene_summary: str = Field(..., max_length=200)
    objects: list[PlannedObject] = Field(..., min_length=2, max_length=8)
    camera: LLMCameraSpec

    @field_validator("objects")
    @classmethod
    def exactly_one_target(cls, v: list[PlannedObject]) -> list[PlannedObject]:
        if sum(o.is_target for o in v) != 1:
            raise ValueError("exactly one object must have is_target=true")
        return v


class LLMStyle(BaseModel):                       # NO negative_prompt — dead at L0/L1,
    name: str = Field(..., min_length=2, max_length=40)   # wasted styler tokens (review fix)
    prompt: str = Field(..., min_length=10, max_length=250)  # 250 chars ≈ CLIP-77 budget


class StyleSet(BaseModel):
    styles: list[LLMStyle] = Field(..., min_length=2, max_length=6)


# ---------- resolved contract ----------
class StyleSpec(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9_]{2,40}$")  # SLUG — canonical key for
                                                          # filenames/COCO/dropdowns (§6.6)
    prompt: str                                            # post-validated: categories in
                                                           # first clause, ≤77 CLIP tokens
    negative_prompt: str = NEGATIVE_PROMPT                 # constant; L2/L3 only


class ObjectSpec(BaseModel):
    instance_id: int = Field(..., ge=1, le=64)
    asset_id: str
    category: str                                 # == asset_id for v1
    requested: str
    x_m: float
    y_m: float
    yaw_deg: float
    scale: float = Field(1.0, ge=0.5, le=1.6)
    z_m: float = 0.0
    is_target: bool = False
    color_rgb: tuple[int, int, int] = (180, 180, 180)


class SceneSpec(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    task: str
    seed: int = 42
    table: TableSpec = TableSpec()
    objects: list[ObjectSpec] = Field(..., min_length=1, max_length=8)
    camera: CameraSpec = CameraSpec()
    styles: list[StyleSpec] = Field(..., min_length=1, max_length=6)
    grounding_log: list[dict] = []   # [{requested, asset_id, score, method}] + camera bumps
                                     # + dropped objects + style-prompt rewrites


# ---------- numeric sanitation (between Ollama and pydantic) ----------
def _bounds_of(field_metadata: list) -> tuple[float | None, float | None]:
    """Extract (ge, le) bounds from a FieldInfo's annotated-types metadata."""
    lo: float | None = None
    hi: float | None = None
    for meta in field_metadata:
        if isinstance(meta, annotated_types.Ge):
            lo = float(meta.ge)  # type: ignore[arg-type]
        elif isinstance(meta, annotated_types.Le):
            hi = float(meta.le)  # type: ignore[arg-type]
    return lo, hi


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """Return the BaseModel subclass inside `annotation` (unwraps Optional/Union)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) is Union:  # Optional[Model] / Model | None
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def clamp_to_bounds(raw: dict, model: type[BaseModel]) -> dict:
    """Coerce every numeric leaf into its Field ge/le bounds (recurses into nested
    models and lists by reading model_fields metadata). Same philosophy as placement
    clamping: geometry/range problems are fixed deterministically, never sent back
    to the LLM. Unit-tested in test_spec.py.

    Non-numeric values, missing fields, unknown keys and wrong-typed leaves are
    left untouched — those are structural failures for the repair loop (§6.3).
    Returns a new dict; `raw` is not mutated.
    """
    out = dict(raw)
    for name, field in model.model_fields.items():
        if name not in out:
            continue
        value = out[name]

        nested = _nested_model(field.annotation)
        if nested is not None:
            if isinstance(value, dict):
                out[name] = clamp_to_bounds(value, nested)
            continue

        if get_origin(field.annotation) is list:
            args = get_args(field.annotation)
            item_model = _nested_model(args[0]) if args else None
            if item_model is not None and isinstance(value, list):
                out[name] = [
                    clamp_to_bounds(v, item_model) if isinstance(v, dict) else v
                    for v in value
                ]
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        lo, hi = _bounds_of(field.metadata)
        if lo is not None and value < lo:
            value = type(out[name])(lo)
        if hi is not None and value > hi:
            value = type(out[name])(hi)
        out[name] = value
    return out


# ---------- style-name slugs (§6.6 rule 1) ----------
_SLUG_RUN = re.compile(r"[^a-z0-9]+")
_SLUG_MAX = 40           # StyleSpec.name pattern: ^[a-z0-9_]{2,40}$
_SLUG_FALLBACK = "style"  # degenerate input (slug shorter than the pattern's min 2)


def slugify(name: str, taken: Iterable[str] = ()) -> str:
    """Slugify a style name per ARCHITECTURE.md §6.6 rule 1.

    Lowercase, replace non-[a-z0-9] runs with "_", truncate to 40; on collision
    with `taken`, suffix "_2"/"_3"/... (uniqueness enforced post-slug). The slug
    is THE canonical key for filenames, COCO attributes and the style dropdown,
    and always matches StyleSpec's ^[a-z0-9_]{2,40}$ pattern.
    """
    base = _SLUG_RUN.sub("_", name.lower()).strip("_")[:_SLUG_MAX].rstrip("_")
    if len(base) < 2:
        base = _SLUG_FALLBACK
    taken_set = set(taken)
    if base not in taken_set:
        return base
    for n in count(2):
        suffix = f"_{n}"
        candidate = base[: _SLUG_MAX - len(suffix)].rstrip("_") + suffix
        if candidate not in taken_set:
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover
