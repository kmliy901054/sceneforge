"""sceneforge/director/director.py — plan/style LLM calls, clamp→validate→repair,
grounding, and style post-validation (ARCHITECTURE.md §6.2–§6.6).

Flow per forge: ``plan_scene`` (planner chat, §6.3 repair loop) → ``make_styles``
(styler chat) → ``ground_plan`` (RAG fill + server-pinned camera + §6.6 style
post-validation) → resolved ``SceneSpec``.

Every LLM-touching path degrades deterministically (``fallback.py``); repair
transcripts accumulate in a ``DirectorLog`` the orchestrator flushes to
``outputs/runs/<id>/director_log.md`` (NOT docs/agent_log.md — §6.3).
"""
from __future__ import annotations

import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from pydantic import ValidationError

from sceneforge.config import AppConfig
from sceneforge.director.fallback import default_style_set, template_plan
from sceneforge.director.ollama_client import DirectorUnavailable, chat_structured
from sceneforge.director.rag import EmbeddingIndex
from sceneforge.spec import (
    CameraSpec,
    LLMScenePlan,
    LLMStyle,
    ObjectSpec,
    SceneSpec,
    StyleSet,
    StyleSpec,
    TableSpec,
    clamp_to_bounds,
    slugify,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3            # attempt 0..2 (§6.3): 1 initial + ≤2 repair rounds
PLAN_TEMPERATURE = 0.7      # §6.2 planner
STYLE_TEMPERATURE = 0.95    # §6.2 styler (seed = run seed + 1)

# ---------------------------------------------------------------- §6.2 prompts
PLANNER_SYSTEM = """\
You are SceneForge's scene planner for robot tabletop-manipulation tasks.

World model (follow it exactly):
- tabletop at z=0, x in [-0.5, 0.5], y in [-0.35, 0.35] usable; units meters, angles degrees.
- place 2-8 objects as (x_m, y_m, yaw_deg) on the tabletop; spread objects, no stacking.
- exactly ONE object has is_target=true: the object the task manipulates.
- short concrete object descriptions like "red ceramic mug" (material/color + noun).
- camera is an orbit at the table center: azimuth_deg, elevation_deg, distance_m.

Respond with JSON only, matching the provided schema.

Example task: "push the blue bowl to the edge of the cluttered workbench"
Example response:
{"scene_summary": "a blue bowl among hand tools on a workbench tabletop",
 "objects": [
  {"description": "blue plastic bowl", "x_m": -0.10, "y_m": 0.05, "yaw_deg": 0.0, "size": "medium", "is_target": true},
  {"description": "yellow screwdriver", "x_m": 0.18, "y_m": -0.15, "yaw_deg": 70.0, "size": "small", "is_target": false},
  {"description": "metal can of nails", "x_m": 0.32, "y_m": 0.20, "yaw_deg": 0.0, "size": "medium", "is_target": false},
  {"description": "wooden hammer", "x_m": -0.35, "y_m": -0.22, "yaw_deg": -35.0, "size": "medium", "is_target": false}],
 "camera": {"azimuth_deg": -25.0, "elevation_deg": 40.0, "distance_m": 1.3}}
"""

# NOTE: the few-shot task deliberately differs from the canonical demo task
# ("pick the red mug...") — with an identical example the model parrots the
# example verbatim instead of planning (observed live on gemma4:e4b).

STYLER_SYSTEM = """\
You are SceneForge's style director. Invent N visually distinct PHOTOREAL
environments for the SAME physical scene.

Rules (follow them exactly):
- each prompt MUST begin with the objects: "a photo of {categories} on a table, in <environment> ..."
- environment, surface, lighting and camera-realism details come AFTER the object clause.
- at most 40 words per prompt; photographic vocabulary (lens, light, surface, focus).
- NO identity-changing style words (cartoon, painting, illustration, anime, render, sketch, drawing).
- names: short, 2-4 lowercase words.

Respond with JSON only, matching the provided schema.
"""

# ----------------------------------------------------------- director_log (§6.3)
class DirectorLog:
    """Markdown accumulator for runtime LLM transcripts/repairs (§6.3, §4.7).

    The orchestrator flushes it to ``outputs/runs/<id>/director_log.md``;
    ``meta`` carries machine-readable facts (source=llm|fallback, repair counts)
    for ForgeEvent payloads and run_meta.json.
    """

    def __init__(self) -> None:
        self.entries: list[str] = []
        self.meta: dict[str, Any] = {}

    def add(self, title: str, body: str = "") -> None:
        entry = f"## {title}\n"
        if body:
            entry += f"\n{body.rstrip()}\n"
        self.entries.append(entry)

    def render(self) -> str:
        return "# Director log\n\n" + "\n".join(self.entries) if self.entries else ""

    def flush(self, path: Union[str, Path]) -> Path:
        """Append accumulated entries to ``path`` and clear the accumulator."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.render()
        if text:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        self.entries.clear()
        return path


def _fence(obj: Any) -> str:
    """JSON code fence for log bodies."""
    try:
        body = json.dumps(obj, indent=2, default=str)
    except TypeError:
        body = repr(obj)
    return f"```json\n{body}\n```"


# ------------------------------------------------------- §6.3 plan repair loop
def plan_scene(
    task: str,
    seed: int,
    cfg: AppConfig,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    log: Optional[DirectorLog] = None,
) -> LLMScenePlan:
    """Planner chat with the §6.3 clamp→validate→repair loop (≤ 2 repair rounds).

    clamp_to_bounds runs on the raw dict BEFORE validation (Ollama does NOT
    enforce numeric ranges — verified); the repair loop handles only
    structural/semantic failures. DirectorUnavailable or exhaustion →
    ``fallback.template_plan`` — never raises. ``log.meta["plan_source"]`` is
    "llm" | "fallback"; transcripts accumulate in ``log``.
    """
    log = log if log is not None else DirectorLog()
    schema = LLMScenePlan.model_json_schema()
    user_msg = f'Task: "{task}"\nPlan the scene. Return JSON only.'

    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = chat_structured(
                cfg.ollama.planner_model,
                PLANNER_SYSTEM,
                user_msg,
                schema,
                seed=seed,
                temperature=PLAN_TEMPERATURE,
                keep_alive=cfg.ollama.keep_alive,
                num_ctx=cfg.ollama.num_ctx,
                on_token=on_token,
                host=cfg.ollama.host,
            )
        except DirectorUnavailable as exc:
            log.add(f"plan_scene attempt {attempt}: director unavailable", str(exc))
            break
        raw = clamp_to_bounds(raw, LLMScenePlan)
        try:
            plan = LLMScenePlan.model_validate(raw)
        except ValidationError as exc:
            log.add(
                f"plan_scene attempt {attempt}: validation failed (repair round)",
                _fence(raw) + "\n\nErrors:\n" + _fence(exc.errors()),
            )
            user_msg += (
                f"\nYour previous JSON failed validation:\n{exc.errors()}"
                "\nReturn corrected JSON only."
            )
            continue
        log.meta["plan_source"] = "llm"
        log.meta["plan_repairs"] = attempt
        log.add(f"plan_scene attempt {attempt}: valid plan", _fence(raw))
        return plan

    log.meta["plan_source"] = "fallback"
    log.meta["plan_repairs"] = MAX_ATTEMPTS - 1
    log.add("plan_scene: deterministic template fallback (§6.5)")
    return template_plan(task, seed)


def make_styles(
    task: str,
    plan: LLMScenePlan,
    n_styles: int,
    cfg: AppConfig,
    *,
    seed: int = 42,
    on_token: Optional[Callable[[str], None]] = None,
    log: Optional[DirectorLog] = None,
) -> StyleSet:
    """Styler chat (§6.2: temp 0.95, seed = run seed + 1) with the same
    clamp→validate→repair loop; falls back to ``DEFAULT_STYLES`` (§6.5).

    Returns at most ``n_styles`` styles (extras trimmed deterministically);
    ``log.meta["style_source"]`` is "llm" | "fallback".
    """
    log = log if log is not None else DirectorLog()
    schema = StyleSet.model_json_schema()
    objects = ", ".join(o.description for o in plan.objects)
    user_msg = (
        f'Task: "{task}"\nObjects on the table: {objects}.\n'
        f"Invent exactly {n_styles} distinct environments (N={n_styles}). Return JSON only."
    )

    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = chat_structured(
                cfg.ollama.planner_model,
                STYLER_SYSTEM,
                user_msg,
                schema,
                seed=seed + 1,
                temperature=STYLE_TEMPERATURE,
                keep_alive=cfg.ollama.keep_alive,
                num_ctx=cfg.ollama.num_ctx,
                on_token=on_token,
                host=cfg.ollama.host,
            )
        except DirectorUnavailable as exc:
            log.add(f"make_styles attempt {attempt}: director unavailable", str(exc))
            break
        raw = clamp_to_bounds(raw, StyleSet)
        try:
            styles = StyleSet.model_validate(raw)
        except ValidationError as exc:
            log.add(
                f"make_styles attempt {attempt}: validation failed (repair round)",
                _fence(raw) + "\n\nErrors:\n" + _fence(exc.errors()),
            )
            user_msg += (
                f"\nYour previous JSON failed validation:\n{exc.errors()}"
                "\nReturn corrected JSON only."
            )
            continue
        if len(styles.styles) > n_styles:
            log.add(f"make_styles: trimmed {len(styles.styles)} → {n_styles} styles")
            styles = StyleSet(styles=styles.styles[: max(2, n_styles)])
        log.meta["style_source"] = "llm"
        log.meta["style_repairs"] = attempt
        log.add(f"make_styles attempt {attempt}: valid styles", _fence(raw))
        return styles

    log.meta["style_source"] = "fallback"
    log.add("make_styles: DEFAULT_STYLES fallback (§6.5)")
    return default_style_set(n_styles)


# --------------------------------------------------------------- color rules
#: Simple color words parsed from object descriptions → vertex tint.
COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "red": (200, 40, 40),
    "blue": (50, 80, 200),
    "green": (50, 160, 60),
    "yellow": (230, 200, 50),
    "white": (240, 240, 240),
    "black": (40, 40, 40),
    "orange": (235, 140, 40),
    "brown": (130, 90, 50),
}
_COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.IGNORECASE)

#: Stable per-category default palette (used when the description names no color).
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "mug": (70, 110, 180),
    "bowl": (200, 160, 60),
    "plate": (235, 235, 230),
    "cup": (160, 200, 220),
    "bottle": (60, 140, 90),
    "can": (180, 60, 60),
    "box": (150, 110, 70),
    "book": (90, 70, 150),
    "pan": (60, 60, 65),
    "ball": (220, 120, 40),
    "pot": (120, 120, 125),
    "screwdriver": (200, 80, 30),
    "hammer": (110, 80, 60),
    "cutting_board": (190, 150, 100),
    "laptop": (70, 70, 75),
}
DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 180)  # ObjectSpec default

SIZE_SCALE: dict[str, float] = {"small": 0.85, "medium": 1.0, "large": 1.2}  # §3.2


def parse_color(description: str) -> Optional[tuple[int, int, int]]:
    """First simple color word in the description → rgb, else None."""
    m = _COLOR_RE.search(description)
    return COLOR_WORDS[m.group(1).lower()] if m else None


def color_for(description: str, category: str) -> tuple[int, int, int]:
    """Description color word wins; else the stable per-category palette."""
    return parse_color(description) or CATEGORY_COLORS.get(category, DEFAULT_COLOR)


# -------------------------------------------------- §6.6 style post-validation
#: Identity-changing words stripped deterministically — the styler system
#: prompt alone is unenforced, and negative_prompt is dead at cfg=0 (§6.6).
IDENTITY_BLACKLIST: tuple[str, ...] = (
    "cel shaded",
    "pixel art",
    "vector art",
    "low poly",
    "oil painting",
    "3d render",
    "cartoon",
    "painting",
    "illustration",
    "illustrated",
    "anime",
    "manga",
    "rendered",
    "rendering",
    "render",
    "sketch",
    "drawing",
    "drawn",
    "comic",
    "watercolor",
    "claymation",
    "cgi",
    "stylized",
)
_BLACKLIST_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in IDENTITY_BLACKLIST) + r")\b",
    re.IGNORECASE,
)

CLIP_MAX_TOKENS = 77     # SDXL CLIP encoders hard-truncate here (incl. BOS/EOS)
CONTENT_BUDGET = 75      # 77 minus BOS/EOS
EARLY_WINDOW = 60        # categories must appear within the first ~60 tokens (§6.6)
_WORDS_PER_TOKEN = 0.75  # CLIP averages ~1.33 tokens/word — approximation fallback


@lru_cache(maxsize=1)
def _clip_tokenizer():
    """CLIPTokenizer from the SDXL repo (cached, CPU, tiny — §6.6), or None.

    None → graceful word-count approximation (weights may not be on disk yet;
    the M1 snapshot_download owns fetching them).
    """
    try:
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
        )
    except Exception as exc:
        logger.warning(
            "CLIP tokenizer unavailable (%s); using word-count approximation", exc
        )
        return None


def count_clip_tokens(text: str) -> int:
    """CLIP token count of ``text`` (no special tokens); word-based approximation
    (~1.33 tokens/word, rounded up) if the tokenizer is not on disk."""
    tok = _clip_tokenizer()
    if tok is None:
        return math.ceil(len(text.split()) / _WORDS_PER_TOKEN)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _clip_prefix(text: str, n_tokens: int) -> str:
    """The first ``n_tokens`` CLIP tokens of ``text``, decoded back to a string
    (word-count approximation when the tokenizer is unavailable)."""
    tok = _clip_tokenizer()
    if tok is None:
        n_words = max(1, int(n_tokens * _WORDS_PER_TOKEN))
        return " ".join(text.split()[:n_words])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= n_tokens:
        return text
    return tok.decode(ids[:n_tokens]).strip()


def _tidy(text: str) -> str:
    """Collapse whitespace/comma debris left by word stripping."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",(\s*,)+", ",", text)
    return text.strip(" ,.").strip()


def strip_identity_words(prompt: str) -> str:
    """Remove blacklist words (§6.6 rule 2), tidying leftover punctuation."""
    return _tidy(_BLACKLIST_RE.sub("", prompt))


def _environment_tail(prompt: str) -> str:
    """The prompt minus its leading object clause (§6.6 rebuild rule).

    Strips a leading "a photo of ... on a/the table," clause when present; a
    prompt that merely starts "a photo of ..." loses its first clause (it names
    the wrong objects); a prompt with no object clause at all IS the tail.
    Empty tails get a neutral photoreal filler.
    """
    m = re.match(r"\s*a\s+photo(?:graph)?\s+of\b.*?\bon\s+(?:a|the)\s+table[,.]?\s*",
                 prompt, re.IGNORECASE)
    if m:
        tail = prompt[m.end():]
    elif re.match(r"\s*a\s+photo(?:graph)?\s+of\b", prompt, re.IGNORECASE):
        comma = prompt.find(",")
        tail = prompt[comma + 1:] if comma != -1 else ""
    else:
        tail = prompt
    tail = _tidy(tail)
    return tail or "in a photorealistic environment, natural light, sharp focus"


def postvalidate_styles(
    styles: Union[StyleSet, Iterable[LLMStyle]],
    categories: Sequence[str],
    grounding_log: list[dict],
) -> list[StyleSpec]:
    """§6.6 deterministic style post-validation → resolved ``StyleSpec`` list.

    Per style: (1) slugify name + post-slug uniqueness; (2) strip
    identity-changing words; (3) fill the fallback ``{objects}`` placeholder;
    (4) category coverage within the first ~60 CLIP tokens — if ANY category is
    missing/late, REBUILD as ``"a photo of {cats} on a table, " + environment
    tail`` (§12.6 dispute 7: append lands past CLIP-77 truncation); (5) trim to
    the 77-token budget. Rewrites and per-style token counts are appended to
    ``grounding_log`` (the orchestrator copies token counts to run_meta.json).
    """
    items = list(styles.styles if isinstance(styles, StyleSet) else styles)
    seen: list[str] = []                       # de-duplicated, order-preserving
    for c in categories:
        if c not in seen:
            seen.append(c)
    cat_phrases = [c.replace("_", " ") for c in seen]
    cats_clause = ", ".join(cat_phrases)

    taken: set[str] = set()
    out: list[StyleSpec] = []
    for style in items:
        slug = slugify(style.name, taken)
        taken.add(slug)

        prompt = style.prompt.replace("{objects}", cats_clause)
        stripped = strip_identity_words(prompt)
        if stripped != prompt:
            grounding_log.append(
                {"event": "style_blacklist_strip", "style": slug,
                 "before": prompt, "after": stripped}
            )
        prompt = stripped

        early = _clip_prefix(prompt, EARLY_WINDOW).lower()
        missing = [
            p for p in cat_phrases
            if not re.search(rf"\b{re.escape(p)}\b", early)
        ]
        if missing:
            rebuilt = f"a photo of {cats_clause} on a table, {_environment_tail(prompt)}"
            grounding_log.append(
                {"event": "style_prompt_rebuild", "style": slug,
                 "missing": missing, "before": prompt, "after": rebuilt}
            )
            prompt = rebuilt

        if count_clip_tokens(prompt) > CONTENT_BUDGET:
            trimmed = _tidy(_clip_prefix(prompt, CONTENT_BUDGET))
            grounding_log.append(
                {"event": "style_prompt_trim", "style": slug,
                 "before": prompt, "after": trimmed}
            )
            prompt = trimmed

        grounding_log.append(
            {"event": "style_tokens", "style": slug,
             "clip_tokens": count_clip_tokens(prompt)}
        )
        out.append(StyleSpec(name=slug, prompt=prompt))
    return out


# ------------------------------------------------------------- §4.2 grounding
def ground_plan(
    plan: LLMScenePlan,
    styles: Union[StyleSet, Iterable[LLMStyle]],
    task: str,
    seed: int,
    library: Any,
    rag: EmbeddingIndex,
) -> SceneSpec:
    """Resolve an LLM plan + styles into THE contract ``SceneSpec`` (§4.2).

    Fills ``asset_id/category/scale/color_rgb/instance_id`` (RAG grounding,
    size→scale 0.85/1.0/1.2, color-word-or-palette tint), pins ``look_at``/
    ``yfov`` (server-pinned, §3.1), slugifies + post-validates styles (§6.6),
    and records every decision in ``grounding_log``. Deterministic, idempotent,
    no LLM calls. ``library`` (AssetLibrary, duck-typed) double-checks that
    grounded ids are buildable; unknown ids degrade to "box" with a log entry.
    """
    known: Optional[set[str]] = None
    if library is not None:
        try:
            known = {str(c["name"]) for c in library.cards()}
        except Exception:  # library still under construction — RAG ids are trusted
            known = None

    grounding_log: list[dict] = []
    objects: list[ObjectSpec] = []
    for i, planned in enumerate(plan.objects, start=1):
        asset_id, score, method = rag.ground(planned.description)
        if known is not None and asset_id not in known:
            grounding_log.append(
                {"event": "unknown_asset", "requested": planned.description,
                 "asset_id": asset_id, "replaced_with": "box"}
            )
            asset_id, score, method = "box", 0.0, "default"
        grounding_log.append(
            {"requested": planned.description, "asset_id": asset_id,
             "score": round(float(score), 4), "method": method}
        )
        objects.append(
            ObjectSpec(
                instance_id=i,
                asset_id=asset_id,
                category=asset_id,                      # == asset_id for v1 (§3.2)
                requested=planned.description,
                x_m=planned.x_m,
                y_m=planned.y_m,
                yaw_deg=planned.yaw_deg,
                scale=SIZE_SCALE[planned.size],
                z_m=0.0,
                is_target=planned.is_target,
                color_rgb=color_for(planned.description, asset_id),
            )
        )

    # Server-pinned camera fields: CameraSpec defaults supply yfov_deg=50 and
    # look_at=(0,0,0.05); the LLM only ever sets az/el/dist (§3.1).
    camera = CameraSpec(**plan.camera.model_dump())

    categories = [o.category for o in objects]
    style_specs = postvalidate_styles(styles, categories, grounding_log)

    return SceneSpec(
        task=task,
        seed=seed,
        table=TableSpec(),
        objects=objects,
        camera=camera,
        styles=style_specs,
        grounding_log=grounding_log,
    )
