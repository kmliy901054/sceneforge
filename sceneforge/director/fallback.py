"""sceneforge/director/fallback.py — deterministic LLM fallbacks (§6.5).

Every director entry point degrades here on ``DirectorUnavailable`` or repair
exhaustion: the demo completes end-to-end with the Ollama server stopped.

- ``template_plan(task, seed)``: pure function of (task, seed) — tokenize the
  task, difflib the tokens against card synonyms to pick the target, add 2–4
  round-robin distractors, place everything on a seeded golden-angle ring
  (r ∈ [0.10, 0.30]), fixed camera (35, 30, 1.1).
- ``DEFAULT_STYLES``: 4 hardcoded category-first prompts with an ``{objects}``
  placeholder filled at use time (§6.6 post-validation fills it).
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Mapping, Optional, Sequence

import numpy as np

from sceneforge.spec import (
    LLMCameraSpec,
    LLMScenePlan,
    LLMStyle,
    PlannedObject,
    StyleSet,
    StyleSpec,
)

logger = logging.getLogger(__name__)

GOLDEN_ANGLE_DEG = 137.50776405003785          # placement ring step (§6.5)
RING_R_MIN, RING_R_MAX = 0.10, 0.30
DISTRACTOR_ROTATION = ("bowl", "bottle", "can", "book")   # round-robin pool (§6.5)
FIXED_CAMERA = LLMCameraSpec(azimuth_deg=35.0, elevation_deg=30.0, distance_m=1.1)
_MATCH_CUTOFF = 0.6
_WORD_RE = re.compile(r"[a-z]+")

#: Built-in synonym table for the 15 builders (§5.1) — used when the cards file
#: is unavailable so the no-Ollama, no-assets path still works. The cards file,
#: when present, takes precedence (richer synonyms, single source of truth).
_BUILTIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mug": ("mug", "coffee mug", "coffee cup"),
    "bowl": ("bowl", "dish"),
    "plate": ("plate", "saucer"),
    "cup": ("cup", "glass", "tumbler"),
    "bottle": ("bottle", "water bottle", "flask"),
    "can": ("can", "soda can", "tin"),
    "box": ("box", "carton", "package", "block", "cube"),
    "book": ("book", "novel", "notebook"),
    "pan": ("pan", "frying pan", "skillet"),
    "ball": ("ball", "sphere", "orb"),
    "pot": ("pot", "cooking pot", "saucepan"),
    "screwdriver": ("screwdriver",),
    "hammer": ("hammer", "mallet"),
    "cutting_board": ("cutting board", "chopping board"),
    "laptop": ("laptop", "notebook computer", "computer"),
}

#: Color words a fallback target description may carry through to grounding —
#: kept in sync with director.COLOR_WORDS so e.g. "red mug" tints correctly.
_COLOR_RE = re.compile(
    r"\b(red|blue|green|yellow|white|black|orange|brown)\b", re.IGNORECASE
)


def _synonym_table(cards: Optional[Sequence[Mapping]]) -> dict[str, tuple[str, ...]]:
    """asset_id → synonyms, from explicit cards, the cards file, or the builtin."""
    if cards is None:
        try:
            from sceneforge.config import get_config

            cards_path = get_config().paths.cards_file
            cards = json.loads(cards_path.read_text(encoding="utf-8"))
        except Exception:
            return _BUILTIN_SYNONYMS
    table: dict[str, tuple[str, ...]] = {}
    for card in cards:
        name = str(card["name"])
        table[name] = (name.replace("_", " "),) + tuple(
            str(s).lower() for s in card.get("synonyms", ())
        )
    return table or _BUILTIN_SYNONYMS


def _pick_target(task: str, synonyms: dict[str, tuple[str, ...]]) -> tuple[str, str]:
    """(asset_id, description) for the task's object — difflib over synonyms (§6.5)."""
    tokens = _WORD_RE.findall(task.lower())
    best_asset, best_score = "box", 0.0
    for asset_id, names in synonyms.items():
        for name in names:
            for token in tokens:
                score = difflib.SequenceMatcher(None, token, name).ratio()
                if score > best_score:
                    best_asset, best_score = asset_id, score
    if best_score < _MATCH_CUTOFF:
        best_asset = "box"
    color = _COLOR_RE.search(task)
    description = best_asset.replace("_", " ")
    if color:
        description = f"{color.group(1).lower()} {description}"
    return best_asset, description


def template_plan(
    task: str, seed: int, cards: Optional[Sequence[Mapping]] = None
) -> LLMScenePlan:
    """Deterministic no-LLM scene plan — pure function of ``(task, seed)`` (§6.5).

    Target from difflib over card synonyms (+ a color word from the task if
    present), 2–4 distractors round-robin from §6.5's pool (target excluded),
    seeded golden-angle ring placement r ∈ [0.10, 0.30], fixed camera
    (az 35, el 30, dist 1.1). Always validates: exactly one target, 3–5 objects.
    """
    rng = np.random.default_rng(seed)
    synonyms = _synonym_table(cards)
    target_asset, target_desc = _pick_target(task, synonyms)

    n_distractors = int(rng.integers(2, 5))                 # 2..4 (§6.5)
    pool = [a for a in DISTRACTOR_ROTATION if a != target_asset]
    start = int(rng.integers(0, len(pool)))
    distractors = [pool[(start + i) % len(pool)] for i in range(n_distractors)]

    theta0 = float(rng.uniform(0.0, 360.0))
    descriptions = [target_desc] + [a.replace("_", " ") for a in distractors]
    objects: list[PlannedObject] = []
    n = len(descriptions)
    for i, desc in enumerate(descriptions):
        theta = np.deg2rad(theta0 + i * GOLDEN_ANGLE_DEG)
        r = RING_R_MIN if n == 1 else RING_R_MIN + (RING_R_MAX - RING_R_MIN) * i / (n - 1)
        x = float(np.clip(r * np.cos(theta), -0.5, 0.5))    # usable tabletop (§6.2)
        y = float(np.clip(r * np.sin(theta), -0.35, 0.35))
        objects.append(
            PlannedObject(
                description=desc,
                x_m=round(x, 3),
                y_m=round(y, 3),
                yaw_deg=round(float(rng.uniform(-180.0, 180.0)), 1),
                size="medium",
                is_target=(i == 0),
            )
        )

    summary = f"template fallback: {task}"[:200]
    return LLMScenePlan(scene_summary=summary, objects=objects, camera=FIXED_CAMERA)


#: 4 hardcoded styles (§6.5) — already category-first; ``{objects}`` is replaced
#: with the grounded category list during style post-validation (§6.6).
DEFAULT_STYLES: list[StyleSpec] = [
    StyleSpec(
        name="rustic_kitchen",
        prompt=(
            "a photo of {objects} on a table, in a rustic farmhouse kitchen, warm "
            "morning window light, worn wooden surface, shallow depth of field, 35mm photo"
        ),
    ),
    StyleSpec(
        name="garage_workbench",
        prompt=(
            "a photo of {objects} on a table, on a cluttered garage workbench, cool "
            "fluorescent lighting, scratched metal surface, photorealistic, high detail"
        ),
    ),
    StyleSpec(
        name="clean_lab",
        prompt=(
            "a photo of {objects} on a table, in a bright clean laboratory, diffuse "
            "white light, matte epoxy counter, sharp focus, professional photograph"
        ),
    ),
    StyleSpec(
        name="outdoor_picnic",
        prompt=(
            "a photo of {objects} on a table, at an outdoor picnic, dappled afternoon "
            "sunlight, checkered cloth, natural bokeh background, DSLR photo"
        ),
    ),
]


def default_style_set(n_styles: int = 4) -> StyleSet:
    """First ``min(n, 4)`` DEFAULT_STYLES as an LLM-shaped StyleSet (≥ 2).

    The hardcoded fallback is a 4-style set (§1); asking for more than 4 yields
    4 — never duplicated prompts.
    """
    n = max(2, min(int(n_styles), len(DEFAULT_STYLES)))
    return StyleSet(
        styles=[LLMStyle(name=s.name, prompt=s.prompt) for s in DEFAULT_STYLES[:n]]
    )
