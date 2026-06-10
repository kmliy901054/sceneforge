"""sceneforge/augment/restyle.py — ROSIE-style background restyler for REAL frames.

Productized from the validated probe prototype
(experiments/vla_probe/code/{make_masks.py, make_variants_s.py}); the probe
measured that restyling ONLY the far background moves OpenVLA/SpatialVLA
actions ~8-10× above the JPEG floor, so this is targeted augmentation: train on
N appearance worlds of the same episode to push that leak down.

Per-frame pipeline (``restyle_frames``):

1. Depth-Anything-V2-Small relative depth (RGB-estimated — explicitly NOT the
   §3.1 metric ground-truth contract; recorded as such in provenance.json).
2. Percentile near/far split: the nearest ``keep_percentile`` % of pixels
   (robot + workspace) are KEPT; threshold exposed. A lower-image row prior
   (Bridge-style tabletop occupies the lower frame) and a keep-mask dilation
   protect robot/object borders, as in the probe.
3. Temporal smoothing of the raw threshold mask across consecutive frames:
   per-pixel MAJORITY VOTE (median) over a centered ``smooth_window``-frame
   window, so single-frame depth flicker cannot pop the mask.
4. SDXL depth-ControlNet restyle via the repo ``ForgePipeline`` (Lightning
   ladder, §7), generated at the source aspect ratio; FIXED seed + prompt per
   style across all frames of the episode for temporal stability.
5. Bitwise-exact composite: ``np.where(keep, original, generated)`` — near
   pixels are byte-identical to the input (asserted at write time).

Outputs under ``out_dir``: ``frames/<style_slug>/<frame>.png``,
``masks/<frame>_{keep,depth}.png``, ``audit_sheet.jpg``, ``provenance.json``,
and (for video input) ``video_<style_slug>.mp4`` via ffmpeg.

Style prompts: explicit ``style_prompts`` > LLM styles via the existing
director (``use_llm_styles=True``, falls back deterministically) >
``director.fallback.DEFAULT_STYLES``.

VRAM discipline (§10): the depth model is loaded, run, and freed BEFORE
diffusion; the SDXL burst runs inside ``gpu.phase("diffusion")`` (Ollama
eviction barrier) unless the caller injects an already-managed ``pipeline``.
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from sceneforge import gpu
from sceneforge.config import AppConfig, get_config
from sceneforge.diffusion.pipeline import ForgePipeline, aspect_size
from sceneforge.spec import NEGATIVE_PROMPT, slugify

logger = logging.getLogger(__name__)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

#: Probe-validated defaults (make_masks.py: PCT=28 → keep nearest 72%,
#: KEEP_BELOW=0.62, DILATE=8) + the 5-frame smoothing window from the spec.
DEFAULT_KEEP_PERCENTILE = 72.0
DEFAULT_KEEP_BELOW_FRAC = 0.62
DEFAULT_DILATE_PX = 8
DEFAULT_SMOOTH_WINDOW = 5

#: Object clause for the fallback DEFAULT_STYLES' ``{objects}`` placeholder —
#: real frames have no grounded category list.
OBJECTS_CLAUSE = "a robot arm and objects"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

StyleList = list[tuple[str, str]]  # [(slug, prompt)]


# ================================================================ pure pieces
def normalize_nearness(depth: np.ndarray) -> np.ndarray:
    """Relative depth map → near-ness in [0, 1] (1 = near), float32.

    Depth-Anything predicts MiDaS-style relative inverse depth (larger =
    nearer), so a min/max normalization preserves that polarity.
    """
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    return (d - lo) / max(hi - lo, 1e-8)


def build_keep_masks(
    nearness: Sequence[np.ndarray],
    keep_percentile: float = DEFAULT_KEEP_PERCENTILE,
    keep_below_frac: Optional[float] = DEFAULT_KEEP_BELOW_FRAC,
    dilate_px: int = DEFAULT_DILATE_PX,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> list[np.ndarray]:
    """Per-frame boolean KEEP masks (True = near = robot/workspace, untouched).

    Steps (probe-validated order, smoothing inserted between threshold and
    priors): (1) per-frame percentile threshold — keep the nearest
    ``keep_percentile`` % of pixels; (2) temporal majority vote over a centered
    ``smooth_window``-frame window (edge-clamped, so 1-2 frame episodes
    degrade to the raw mask); (3) ``keep_below_frac`` row prior — rows below
    that fraction of the height are always kept; (4) keep-mask dilation by
    ``dilate_px`` to protect robot/object borders.

    Pure CPU function of its inputs — unit-tested without any model.
    """
    if not 0.0 < keep_percentile < 100.0:
        raise ValueError(f"keep_percentile must be in (0, 100); got {keep_percentile}")
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError(f"smooth_window must be a positive odd int; got {smooth_window}")

    raw = []
    for d in nearness:
        if d.ndim != 2:
            raise ValueError(f"nearness maps must be (H, W); got {d.shape}")
        thr = np.percentile(d, 100.0 - keep_percentile)
        raw.append(d >= thr)

    half = smooth_window // 2
    n = len(raw)
    stack = np.stack(raw).astype(np.uint8)  # (N, H, W)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    ) if dilate_px > 0 else None

    masks: list[np.ndarray] = []
    for i in range(n):
        window = stack[max(0, i - half): min(n, i + half + 1)]
        keep = window.mean(axis=0) >= 0.5  # per-pixel majority vote (median)
        if keep_below_frac is not None:
            keep[int(keep_below_frac * keep.shape[0]):, :] = True
        if kernel is not None:
            keep = cv2.dilate(keep.astype(np.uint8) * 255, kernel) > 0
        masks.append(keep.astype(bool))
    return masks


def composite_exact(
    original: np.ndarray, generated: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """Bitwise-exact composite: KEEP pixels from ``original``, rest from
    ``generated``. Pure integer select (``np.where``) — no float math, so near
    pixels are byte-identical to the input (the probe's verified invariant).
    """
    if original.shape != generated.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {generated.shape}")
    if keep.shape != original.shape[:2]:
        raise ValueError(f"mask shape {keep.shape} != frame {original.shape[:2]}")
    return np.where(keep[..., None], original, generated)


def control_from_nearness(nearness: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Near-ness map → 8-bit RGB ControlNet control at generation size.

    Near = bright, far = dark (MiDaS semantics, what the depth ControlNet was
    trained on — §7.3). Estimated-depth caveat: this is an RGB-derived relative
    map, NOT rendered ground truth, so the §7.3 never-resize contract does not
    apply (labels never come from this control).
    """
    u8 = np.clip(np.round(nearness * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")
    return img.resize(size, Image.BICUBIC)


# ================================================================ I/O helpers
def _collect_frames(
    input_path: Union[str, Path]
) -> tuple[list[np.ndarray], list[str], Optional[float], bool]:
    """``input_dir_or_video`` → (RGB uint8 frames, frame names, fps, is_video)."""
    input_path = Path(input_path)
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"no image frames ({'/'.join(_IMAGE_EXTS)}) in {input_path}")
        frames, names = [], []
        for p in files:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"cannot read frame {p}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            names.append(p.stem)
        return frames, names, None, False

    if input_path.suffix.lower() in _VIDEO_EXTS:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"cannot open video {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        frames, names = [], []
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            names.append(f"frame{i:06d}")
            i += 1
        cap.release()
        if not frames:
            raise IOError(f"video {input_path} decoded 0 frames")
        return frames, names, fps, True

    raise ValueError(
        f"{input_path}: expected a directory of frames or a video file "
        f"({'/'.join(_VIDEO_EXTS)})"
    )


def estimate_depth(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Depth-Anything-V2-Small relative depth per frame, resized to frame size.

    Loads the model, runs every frame, then frees it (VRAM discipline — the
    depth model must be gone before the SDXL burst). Returns raw relative maps
    (larger = nearer); callers normalize via :func:`normalize_nearness`.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info("loading depth model %s (device=%s)", DEPTH_MODEL_ID, device)
    pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_ID,
        device=device,
        torch_dtype=torch.float32,
    )
    try:
        out: list[np.ndarray] = []
        for frame in frames:
            h, w = frame.shape[:2]
            pred = pipe(Image.fromarray(frame))["predicted_depth"]
            depth = np.asarray(pred.squeeze().float().cpu(), dtype=np.float32)
            out.append(cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC))
        return out
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================================================ style picks
def _normalize_styles(
    style_prompts: Union[Sequence[Any], Mapping[str, str]], n_styles: int
) -> StyleList:
    """Explicit ``style_prompts`` → [(slug, prompt)] (first ``n_styles``).

    Accepts a mapping name→prompt, or a sequence of strings / (name, prompt)
    pairs / dicts with "name"+"prompt" keys (StyleSpec-shaped objects too).
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(style_prompts, Mapping):
        pairs = [(str(k), str(v)) for k, v in style_prompts.items()]
    else:
        for i, item in enumerate(style_prompts):
            if isinstance(item, str):
                pairs.append((f"style_{i + 1}", item))
            elif isinstance(item, Mapping):
                pairs.append((str(item["name"]), str(item["prompt"])))
            elif hasattr(item, "name") and hasattr(item, "prompt"):
                pairs.append((str(item.name), str(item.prompt)))
            else:
                name, prompt = item  # (name, prompt) tuple
                pairs.append((str(name), str(prompt)))
    taken: set[str] = set()
    out: StyleList = []
    for name, prompt in pairs[:n_styles]:
        slug = slugify(name, taken)
        taken.add(slug)
        out.append((slug, prompt))
    if not out:
        raise ValueError("style_prompts produced no styles")
    return out


def _default_styles(n_styles: int) -> StyleList:
    """First ``n_styles`` of the director's DEFAULT_STYLES (≤ 4), ``{objects}``
    filled with the generic robot-workspace clause."""
    from sceneforge.director.fallback import DEFAULT_STYLES

    return [
        (s.name, s.prompt.replace("{objects}", OBJECTS_CLAUSE))
        for s in DEFAULT_STYLES[: max(1, min(n_styles, len(DEFAULT_STYLES)))]
    ]


def _llm_styles(task: str, n_styles: int, seed: int, cfg: AppConfig) -> tuple[StyleList, str]:
    """Style prompts via the existing director (§6) → ([(slug, prompt)], source).

    ``make_styles`` already degrades to DEFAULT_STYLES on DirectorUnavailable /
    repair exhaustion; identity-changing words are stripped deterministically
    (negative prompts are dead at cfg=0, §6.6).
    """
    from sceneforge.director.director import DirectorLog, make_styles, strip_identity_words
    from sceneforge.director.fallback import template_plan

    log = DirectorLog()
    style_set = make_styles(task, template_plan(task, seed), n_styles, cfg,
                            seed=seed, log=log)
    taken: set[str] = set()
    out: StyleList = []
    for s in style_set.styles[:n_styles]:
        slug = slugify(s.name, taken)
        taken.add(slug)
        prompt = strip_identity_words(s.prompt.replace("{objects}", OBJECTS_CLAUSE))
        out.append((slug, prompt))
    return out, str(log.meta.get("style_source", "fallback"))


# ================================================================ audit sheet
def _audit_sheet(
    frames: Sequence[np.ndarray],
    nearness: Sequence[np.ndarray],
    keeps: Sequence[np.ndarray],
    composites: Mapping[str, Sequence[np.ndarray]],
    path: Path,
    max_rows: int = 8,
    cell_h: int = 200,
) -> Path:
    """Mask audit sheet: rows = frames, cols = original | depth | keep-audit
    (red = restyled region) | one composite per style. JPEG, probe-style."""
    n = len(frames)
    idxs = list(range(n)) if n <= max_rows else [
        int(round(i * (n - 1) / (max_rows - 1))) for i in range(max_rows)
    ]
    style_names = list(composites)
    rows = []
    for i in idxs:
        cells = []
        audit = frames[i].copy()
        far = ~keeps[i]
        audit[far] = (0.45 * audit[far] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        depth_viz = np.repeat(
            (np.clip(nearness[i], 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2
        )
        for cell in (frames[i], depth_viz, audit, *(composites[s][i] for s in style_names)):
            h, w = cell.shape[:2]
            cells.append(cv2.resize(cell, (int(w * cell_h / h), cell_h)))
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=90)
    return path


def _assemble_video(frame_dir: Path, out_path: Path, fps: float) -> bool:
    """Reassemble ``frame%06d.png`` → mp4 via ffmpeg; False if ffmpeg missing/fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found — skipping video assembly for %s", out_path)
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
           "-i", str(frame_dir / "frame%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return out_path.is_file()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg failed for %s: %s", out_path, exc)
        return False


# ==================================================================== main API
def restyle_frames(
    input_dir_or_video: Union[str, Path],
    out_dir: Union[str, Path],
    n_styles: int = 4,
    keep_percentile: float = DEFAULT_KEEP_PERCENTILE,
    style_prompts: Union[Sequence[Any], Mapping[str, str], None] = None,
    seed: int = 42,
    *,
    keep_below_frac: Optional[float] = DEFAULT_KEEP_BELOW_FRAC,
    dilate_px: int = DEFAULT_DILATE_PX,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    use_llm_styles: bool = False,
    task: str = "a robot arm manipulating objects on a cluttered table",
    cfg: Optional[AppConfig] = None,
    pipeline: Any = None,
    depth_fn: Optional[Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]] = None,
) -> dict:
    """Restyle the background of a real robot episode into ``n_styles`` worlds.

    Args:
        input_dir_or_video: directory of frames (sorted by name) or a video.
        out_dir: output root (created): ``frames/<style>/``, ``masks/``,
            ``audit_sheet.jpg``, ``provenance.json``, ``video_<style>.mp4``.
        n_styles: number of appearance worlds.
        keep_percentile: the nearest X % of pixels (per frame) are KEPT —
            the exposed near/far threshold (probe default 72 → restyle the
            farthest 28 %).
        style_prompts: explicit prompts (strings / (name, prompt) / mapping);
            None → LLM styles when ``use_llm_styles`` else DEFAULT_STYLES.
        seed: base seed; style k generates with FIXED ``seed + k`` on every
            frame (temporal stability).
        keep_below_frac: rows below this height fraction are always kept
            (workspace spatial prior; None disables).
        dilate_px: keep-mask dilation protecting robot/object borders.
        smooth_window: odd temporal majority-vote window (5 = spec default).
        use_llm_styles: ask the director for prompts (falls back deterministically).
        task: task hint for LLM style synthesis.
        cfg: AppConfig (default: process config).
        pipeline: injectable, ALREADY-LOADED ForgePipeline-like object (tests /
            callers managing VRAM themselves — no gpu.phase, no load/unload
            here). None → own ForgePipeline inside ``gpu.phase("diffusion")``.
        depth_fn: injectable depth estimator ``frames -> relative depth maps``
            (tests); None → Depth-Anything-V2-Small.

    Returns:
        The provenance dict (also written to ``out_dir/provenance.json``).
    """
    t_start = time.monotonic()
    cfg = cfg if cfg is not None else get_config()
    out_dir = Path(out_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames, names, fps, is_video = _collect_frames(input_dir_or_video)
    h, w = frames[0].shape[:2]
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("all frames of an episode must share one size")

    # ---- 1+2+3: depth → percentile masks → temporal smoothing --------------
    t0 = time.monotonic()
    raw_depth = list((depth_fn or estimate_depth)(frames))
    nearness = [normalize_nearness(d) for d in raw_depth]
    depth_s = time.monotonic() - t0
    keeps = build_keep_masks(nearness, keep_percentile, keep_below_frac,
                             dilate_px, smooth_window)
    restyle_fracs = [round(float(1.0 - k.mean()), 4) for k in keeps]
    for name, near, keep in zip(names, nearness, keeps):
        cv2.imwrite(str(masks_dir / f"{name}_keep.png"),
                    keep.astype(np.uint8) * 255)
        cv2.imwrite(str(masks_dir / f"{name}_depth.png"),
                    (np.clip(near, 0, 1) * 65535).astype(np.uint16))

    # ---- styles -------------------------------------------------------------
    style_source = "explicit"
    if style_prompts is not None:
        styles = _normalize_styles(style_prompts, n_styles)
    elif use_llm_styles:
        styles, style_source = _llm_styles(task, n_styles, seed, cfg)
    else:
        styles, style_source = _default_styles(n_styles), "default"
    style_seeds = {slug: seed + k for k, (slug, _) in enumerate(styles)}

    # ---- 4+5: SDXL depth-ControlNet restyle + exact composite ---------------
    gen_w, gen_h = aspect_size(w, h, cfg.gen.resolution)
    controls = [control_from_nearness(near, (gen_w, gen_h)) for near in nearness]

    owns_pipeline = pipeline is None
    composites: dict[str, list[np.ndarray]] = {}
    gen_seconds: list[float] = []

    def _burst(pipe: Any) -> None:
        for slug, prompt in styles:  # FIXED seed+prompt per style across frames
            gen_seed = style_seeds[slug]
            style_dir = out_dir / "frames" / slug
            style_dir.mkdir(parents=True, exist_ok=True)
            composites[slug] = []
            for name, frame, control, keep in zip(names, frames, controls, keeps):
                t1 = time.monotonic()
                img = pipe.generate(control, prompt, NEGATIVE_PROMPT, gen_seed,
                                    cond_scale=cfg.gen.cond_scale,
                                    steps=cfg.gen.steps, size=(gen_w, gen_h))
                gen_seconds.append(time.monotonic() - t1)
                gen_np = np.asarray(img.resize((w, h), Image.BICUBIC).convert("RGB"))
                comp = composite_exact(frame, gen_np, keep)
                # the probe's verified invariant, re-asserted on every frame:
                assert np.array_equal(comp[keep], frame[keep]), \
                    "near-pixel composite must be bitwise exact"
                comp_bgr = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(style_dir / f"{name}.png"), comp_bgr):
                    raise IOError(f"failed to write {style_dir / f'{name}.png'}")
                composites[slug].append(comp.astype(np.uint8))
            logger.info("style %r done (%d frames, seed=%d)", slug, len(names), gen_seed)

    t0 = time.monotonic()
    if owns_pipeline:
        with gpu.phase("diffusion", cfg=cfg):  # Ollama eviction barrier (§10.3)
            pipeline = ForgePipeline(resolution=cfg.gen.resolution)
            pipeline.load(int(cfg.gen.level[1]))
            try:
                _burst(pipeline)
            finally:
                pipeline.unload()
    else:
        _burst(pipeline)  # caller manages VRAM/load state
    diffusion_s = time.monotonic() - t0

    # ---- audit sheet + provenance -------------------------------------------
    audit_path = _audit_sheet(frames, nearness, keeps, composites,
                              out_dir / "audit_sheet.jpg")

    videos: dict[str, Optional[str]] = {}
    if is_video:
        for slug, _ in styles:
            out_mp4 = out_dir / f"video_{slug}.mp4"
            ok = _assemble_video(out_dir / "frames" / slug, out_mp4, fps or 10.0)
            videos[slug] = str(out_mp4) if ok else None

    provenance = {
        "tool": "sceneforge.augment.restyle_frames",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_dir_or_video),
        "is_video": is_video,
        "fps": fps,
        "n_frames": len(frames),
        "frame_size": [int(w), int(h)],
        "frame_names": names,
        "params": {
            "n_styles": len(styles),
            "keep_percentile": keep_percentile,
            "keep_below_frac": keep_below_frac,
            "dilate_px": dilate_px,
            "smooth_window": smooth_window,
            "seed": seed,
        },
        "depth_model": DEPTH_MODEL_ID if depth_fn is None else "injected depth_fn",
        "depth_semantics": (
            "RGB-estimated RELATIVE near-ness (1 = near), min/max normalized — "
            "NOT the ARCHITECTURE.md §3.1 metric depth16 contract; masks/*_depth.png "
            "is uint16 nearness*65535"
        ),
        "style_source": style_source,
        "styles": [
            {"name": slug, "prompt": prompt, "seed": style_seeds[slug]}
            for slug, prompt in styles
        ],
        "generation": {
            "pipeline": "ForgePipeline" if owns_pipeline else type(pipeline).__name__,
            "level": cfg.gen.level,
            "steps": cfg.gen.steps,
            "cond_scale": cfg.gen.cond_scale,
            "size": [gen_w, gen_h],
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "restyle_frac_per_frame": restyle_fracs,
        "near_pixels_bitwise_identical": True,  # asserted per composite above
        "timings_s": {
            "depth": round(depth_s, 3),
            "diffusion": round(diffusion_s, 3),
            "per_image": round(float(np.mean(gen_seconds)), 3) if gen_seconds else None,
            "total": round(time.monotonic() - t_start, 3),
        },
        "outputs": {
            "frames_dir": str(out_dir / "frames"),
            "masks_dir": str(masks_dir),
            "audit_sheet": str(audit_path),
            "videos": videos,
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
