"""sceneforge/ui/handlers.py — Gradio event handlers (ARCHITECTURE.md §11).

``on_forge`` is a generator over ``ForgeRun.run()`` events: planner tokens
stream into the Code panel (batched ~0.2 s so Gradio is never flooded), the
layout-0 GLB + depth control land BEFORE the style LLM call, the gallery fills
live per ``image`` event, and the status line updates on every event — the UI
is never silent. Generation order is asserted layout-major here (§4.8).

``on_reforge`` makes no LLM calls; ``on_toggle`` is a pure cached-path swap
(§8.2); ``on_export`` builds the COCO zip; ``on_timer`` feeds the VRAM footer.

``on_video_augment`` (tab 2) is a generator: it runs
``sceneforge.augment.restyle_frames`` in a worker thread — reusing the
ForgeRun singleton's SDXL pipeline inside ``gpu.phase("diffusion")`` so the
augmenter never double-loads SDXL next to the forge tenant — and streams a
status line by polling the written frame files (~1 s ticks), then returns the
side-by-side original/restyled videos, the mask audit sheet and a zip bundle.
"""
from __future__ import annotations

import sceneforge.compat  # noqa: F401  — first import (§0)

import logging
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import gradio as gr
import requests

from sceneforge import gpu
from sceneforge.config import get_config
from sceneforge.orchestrator import ForgeRun, export_coco

logger = logging.getLogger(__name__)

#: plan_token UI flush interval (§11 demo pacing — batch tokens ~0.2 s).
TOKEN_FLUSH_S = 0.2

DEFAULT_TASK = "pick the red mug from a cluttered kitchen table"

_runner: Optional[ForgeRun] = None
_runner_lock = threading.Lock()

#: §10.5 — footer shows a "warming up" banner while the pre-warm thread runs.
WARMING = False


def get_runner() -> ForgeRun:
    """Process-wide ForgeRun singleton (renderer/SDXL/OWLv2 anchor tenants)."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = ForgeRun()
        return _runner


def prewarm() -> None:
    """§10.5 cold-start pre-warm: (1) gemma load+immediate-unload (page cache →
    3 s warm reloads), (2) eager ForgePipeline cuda build, (3) OWLv2 to CPU."""
    global WARMING
    WARMING = True
    try:
        cfg = get_config()
        try:  # (1) gemma prime: /api/generate with keep_alive=0 loads then evicts
            requests.post(f"{cfg.ollama.host}/api/generate",
                          json={"model": cfg.ollama.planner_model, "keep_alive": 0},
                          timeout=120)
        except requests.RequestException as exc:
            logger.warning("pre-warm: gemma prime skipped (%s)", exc)
        runner = get_runner()
        runner.ensure_pipeline()       # (2) SDXL on cuda (30–90 s first build)
        runner.ensure_scorer_cpu()     # (3) OWLv2 weights resident on CPU
        logger.info("pre-warm complete: %s", gpu.snapshot())
    except Exception:  # noqa: BLE001 — pre-warm must never kill the app
        logger.exception("pre-warm failed (continuing; first forge will be slow)")
    finally:
        WARMING = False


# ------------------------------------------------------------------- helpers
def _gallery_items(state: dict, mode: str) -> list[tuple[str, str]]:
    """Layout-major gallery list (rows=layout, cols=style) for overlay ``mode``."""
    items: list[tuple[str, str]] = []
    styles = state.get("styles") or []
    for entry in sorted(state.get("layouts", []), key=lambda e: e["layout_idx"]):
        for slug in styles:
            paths = entry.get("images", {}).get(slug)
            if paths:
                items.append((paths.get(mode) or paths["off"],
                              f"layout {entry['layout_idx']} · {slug}"))
    return items


def _fidelity_value(fid: Optional[dict]) -> Optional[dict]:
    """gr.Label value: dict of 0–1 confidences (§11 fidelity meter)."""
    if not fid:
        return None
    return {
        "fidelity_adj": float(min(1.0, max(0.0, fid["fidelity_adj_mean"]))),
        "match_rate": float(min(1.0, max(0.0, fid["match_rate"]))),
        "hallucination_rate": float(min(1.0, max(0.0, fid["hallucination_rate"]))),
    }


# ------------------------------------------------------------------ on_forge
#: on_forge output order — keep in sync with blocks.py wiring.
FORGE_OUTPUTS = ("status_md", "plan_stream", "spec_json", "source_md",
                 "layout_3d", "control_img", "gallery", "fidelity_lbl",
                 "style_dd", "run_state", "download")


def on_forge(task: str, n_layouts: float, n_styles: float, seed: float,
             overlay_mode: str) -> Iterator[tuple]:
    """Generator handler for the FORGE button (§11 demo pacing)."""
    runner = get_runner()
    task = (task or "").strip() or DEFAULT_TASK
    out: dict[str, Any] = {
        "status_md": "**planning…**",
        "plan_stream": "",
        "spec_json": None,
        "source_md": "",
        "layout_3d": None,
        "control_img": None,
        "gallery": [],
        "fidelity_lbl": None,
        "style_dd": gr.update(),
        "run_state": None,
        "download": gr.update(),
    }

    def tup() -> tuple:
        return tuple(out[k] for k in FORGE_OUTPUTS)

    yield tup()

    token_buf: list[str] = []
    last_flush = 0.0
    state: dict = {"layouts": [], "styles": []}
    last_layout, last_style = -1, -1

    for event in runner.run(task, int(n_layouts), int(n_styles), int(seed)):
        kind, p = event.kind, event.payload

        if kind == "plan_token":
            token_buf.append(p["text"])
            now = time.monotonic()
            if now - last_flush >= TOKEN_FLUSH_S:  # batch tokens ~0.2 s (§11)
                out["plan_stream"] += "".join(token_buf)
                token_buf.clear()
                last_flush = now
                yield tup()
            continue
        if token_buf:  # flush any trailing tokens before the next stage
            out["plan_stream"] += "".join(token_buf)
            token_buf.clear()

        if kind == "status":
            stage = p.get("stage", "")
            out["status_md"] = f"**{stage}**"
            if stage == "styles":
                state["styles"] = p.get("styles", [])
                slugs = state["styles"]
                out["style_dd"] = gr.update(
                    choices=slugs, value=slugs[0] if slugs else None)
                out["status_md"] = (f"**styles ready** ({p.get('source', '?')}): "
                                    + ", ".join(slugs))
        elif kind == "spec":
            out["spec_json"] = p["spec"]
            out["source_md"] = (f"planner: **{p.get('source', '?')}**"
                                f" · repairs: {p.get('repairs', 0)}")
        elif kind == "layout":
            entry = dict(p)
            state["layouts"].append(entry)
            if p["layout_idx"] == 0:
                out["layout_3d"] = p["glb"]
                out["control_img"] = p["control"]
        elif kind == "image":
            li, si = int(p["layout_idx"]), int(p["style_idx"])
            # PINNED layout-major order (§4.8) — rows=layout/cols=style.
            assert (li == last_layout and si == last_style + 1) or (
                li > last_layout and si == 0), (
                f"generation order not layout-major: got (layout {li}, style "
                f"{si}) after (layout {last_layout}, style {last_style})")
            last_layout, last_style = li, si
            entry = next((e for e in state["layouts"]
                          if e["layout_idx"] == li), None)
            if entry is not None:
                entry.setdefault("images", {})[p["style"]] = p["overlays"]
            out["gallery"] = _gallery_items(state, overlay_mode)
            out["status_md"] = (f"**forging {p['index']}/{p['total']}** · "
                                f"{p['gen_seconds']:.2f} s/img")
        elif kind == "fidelity":
            state["fidelity"] = p
            out["fidelity_lbl"] = _fidelity_value(p)
            out["status_md"] = (f"**scored** · fidelity {p['fidelity_adj_mean']:.2f}"
                                f" · halluc {p['hallucination_rate']:.2f}"
                                f" · kept {p['n_kept']}/{p['n_kept'] + p['n_quarantined']}")
        elif kind == "done":
            state = p["state"]  # canonical JSON-able path dict (§4.8)
            out["run_state"] = state
            out["gallery"] = _gallery_items(state, overlay_mode)
            t = p.get("timings", {})
            out["status_md"] = (f"**done** · run `{p['run_id']}` · "
                                f"total {t.get('total_s', 0):.1f} s · "
                                f"first visual {t.get('first_visual_s', 0):.1f} s")
        elif kind == "error":
            if p.get("fatal"):
                out["status_md"] = f"**FORGE FAILED** — {p.get('message', '')}"
                yield tup()
                return
            out["status_md"] = (f"**layout {p.get('layout_idx')} skipped** — "
                                f"{p.get('message', '')}")
        yield tup()


# ---------------------------------------------------------------- on_reforge
def on_reforge(state: Optional[dict], az: float, el: float, dist: float,
               style_name: Optional[str], reforge_seed: float
               ) -> tuple[Any, Any, Any, str]:
    """Re-Forge ⚡ — no LLM calls; ≤ 6 s slider-to-image (§4.8/§11).

    Returns (reforge AnnotatedImage value, layout_3d glb, control image,
    status line).
    """
    if not state or not state.get("layouts"):
        return None, gr.update(), gr.update(), "**re-forge needs a forge run first**"
    t0 = time.monotonic()
    try:
        glb, img_path, _overlays, annotated = get_runner().reforge(
            state,
            camera={"azimuth_deg": az, "elevation_deg": el, "distance_m": dist},
            seed=int(reforge_seed) if reforge_seed is not None else None,
            style_name=style_name or None,
        )
    except Exception as exc:  # noqa: BLE001 — surface, never crash
        logger.exception("re-forge failed")
        return None, gr.update(), gr.update(), f"**re-forge failed** — {exc}"
    dt = time.monotonic() - t0
    return (annotated, glb, gr.update(),
            f"**re-forged** in {dt:.1f} s (az {az:.0f}° el {el:.0f}° d {dist:.2f} m)")


def on_reseed() -> int:
    """🎲 — a fresh random generation seed for the next re-forge."""
    return random.randint(0, 2**31 - 1)


# ----------------------------------------------------------------- on_toggle
def on_toggle(state: Optional[dict], mode: str) -> list[tuple[str, str]]:
    """Overlay radio change — pure cached-path swap (§8.2), no recompute."""
    if not state:
        return []
    return _gallery_items(state, mode)


# ----------------------------------------------------------------- on_export
def on_export(state: Optional[dict], include_quarantined: bool) -> tuple[Any, str]:
    """Export COCO → dataset.zip path into the DownloadButton (§8.3)."""
    if not state or not state.get("run_dir"):
        return gr.update(), "**export needs a forge run first**"
    try:
        zip_path = export_coco(state["run_dir"],
                               include_quarantined=bool(include_quarantined))
    except Exception as exc:  # noqa: BLE001
        logger.exception("export failed")
        return gr.update(), f"**export failed** — {exc}"
    return gr.update(value=str(zip_path)), f"**exported** `{zip_path}`"


# --------------------------------------------------------- on_video_augment
#: video-augment tab slot count (blocks.py builds this many original/restyled
#: player pairs; the styles slider is capped here).
VIDEO_AUGMENT_MAX_STYLES = 4

#: on_video_augment output order — keep in sync with blocks.py wiring:
#: (status, audit, download, orig_0, restyled_0, …, orig_3, restyled_3).


def _frames_to_video(frames_dir: Path, out_path: Path, fps: float) -> Optional[str]:
    """Assemble a directory of frames into an mp4 for the tab's players
    (display only — datasets go through lerobot_io). Frames are cropped to
    even dimensions for yuv420p; returns None if assembly fails."""
    try:
        from sceneforge.augment.lerobot_io import write_video
        from sceneforge.augment.restyle import _collect_frames

        frames, _names, _fps, _is_video = _collect_frames(frames_dir)
        frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
        write_video(frames, out_path, fps or 10.0)
        return str(out_path)
    except Exception:  # noqa: BLE001 — display video is best-effort
        logger.exception("frame→video assembly failed for %s", frames_dir)
        return None


def on_video_augment(video_file: Optional[str], frames_dir: str,
                     n_styles: float, styles_text: str, keep_pct: float,
                     window: float) -> Iterator[tuple]:
    """Generator handler for the Video Augment tab's AUGMENT button.

    Streams status while restyle_frames runs in a worker thread; progress is
    read off the filesystem (masks/*_keep.png = frame count, frames/<style>/*
    = composites done) so the UI is never silent during the SDXL burst.
    """
    n = max(1, min(int(n_styles), VIDEO_AUGMENT_MAX_STYLES))
    hidden = gr.update(value=None, visible=False)
    out: dict[str, Any] = {"status": "**starting…**", "audit": None,
                           "download": gr.update(),
                           "slots": [hidden] * (2 * VIDEO_AUGMENT_MAX_STYLES)}

    def tup() -> tuple:
        return (out["status"], out["audit"], out["download"], *out["slots"])

    src = video_file or (frames_dir or "").strip()
    if not src:
        out["status"] = "**upload a video or set a frames directory first**"
        yield tup()
        return
    src_path = Path(src)
    if not src_path.exists():
        out["status"] = f"**input not found** — `{src_path}`"
        yield tup()
        return
    is_video_input = src_path.is_file()

    style_prompts = [l.strip() for l in (styles_text or "").splitlines()
                     if l.strip()] or None
    win = int(window)
    win = win + 1 if win % 2 == 0 else win  # build_keep_masks wants odd

    cfg = get_config()
    run_dir = (Path(cfg.paths.runs_dir) /
               f"video_augment_{time.strftime('%Y%m%d_%H%M%S')}")
    out["status"] = (f"**preparing** (pipeline load + decode + depth)… "
                     f"→ `{run_dir.name}`")
    yield tup()

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            from sceneforge.augment import restyle_frames

            runner = get_runner()
            # §10.3 discipline: Ollama eviction barrier + the SHARED SDXL
            # pipeline (no second ForgePipeline next to the forge tenant).
            with gpu.phase("diffusion", cfg=cfg):
                runner.ensure_pipeline()
                result["prov"] = restyle_frames(
                    str(src_path), run_dir, n_styles=n,
                    keep_percentile=float(keep_pct),
                    style_prompts=style_prompts, smooth_window=win,
                    cfg=cfg, pipeline=runner.pipeline)
        except Exception as exc:  # noqa: BLE001 — surfaced in the status line
            logger.exception("video augment failed")
            result["error"] = exc

    worker = threading.Thread(target=_worker, name="video-augment", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=1.0)
        n_frames = len(list((run_dir / "masks").glob("*_keep.png"))) \
            if (run_dir / "masks").is_dir() else 0
        if n_frames:
            done = len(list((run_dir / "frames").glob("*/*.png")))
            out["status"] = (f"**restyling** {done}/{n_frames * n} frames "
                             f"· {n} style(s) · keep {keep_pct:.0f}%")
        yield tup()
    if "error" in result:
        out["status"] = f"**AUGMENT FAILED** — {result['error']}"
        yield tup()
        return

    prov = result["prov"]
    fps = prov.get("fps") or 10.0
    out["audit"] = prov["outputs"]["audit_sheet"]
    original_mp4 = (str(src_path) if is_video_input else
                    _frames_to_video(src_path, run_dir / "video_original.mp4",
                                     fps))
    videos = prov["outputs"].get("videos") or {}
    slots: list = []
    for i, style in enumerate(prov["styles"][:VIDEO_AUGMENT_MAX_STYLES]):
        slug = style["name"]
        restyled = videos.get(slug) or _frames_to_video(
            run_dir / "frames" / slug, run_dir / f"video_{slug}.mp4", fps)
        slots.append(gr.update(value=original_mp4, visible=True,
                               label=f"original · {slug}"))
        slots.append(gr.update(value=restyled, visible=True,
                               label=f"restyled · {slug} (seed {style['seed']})"))
    while len(slots) < 2 * VIDEO_AUGMENT_MAX_STYLES:
        slots.append(hidden)
    out["slots"] = slots

    try:
        zip_path = shutil.make_archive(str(run_dir), "zip", run_dir)
        out["download"] = gr.update(value=zip_path)
    except OSError as exc:
        logger.warning("augment zip failed: %s", exc)

    t = prov.get("timings_s", {})
    out["status"] = (f"**done** · {prov['n_frames']} frames × "
                     f"{len(prov['styles'])} style(s) · "
                     f"depth {t.get('depth', 0):.1f} s · "
                     f"diffusion {t.get('diffusion', 0):.1f} s · "
                     f"restyled frac "
                     f"{sum(prov['restyle_frac_per_frame']) / max(1, len(prov['restyle_frac_per_frame'])):.2f}"
                     f" · `{run_dir}`")
    yield tup()


# ------------------------------------------------------------------ on_timer
def on_timer() -> str:
    """VRAM footer via gr.Timer(2.0) (§11) — never raises."""
    try:
        cfg = get_config()
        snap = gpu.snapshot()
        used = snap["total_gb"] - snap["free_gb"]
        models = ", ".join(m["name"] for m in snap["ollama_models"]) or "none"
        s_img = cfg.gen.s_per_img
        import os

        parts = [
            f"VRAM {used:.1f}/{snap['total_gb']:.0f} GB (free {snap['free_gb']:.1f})",
            f"backend pyrender/{os.environ.get('PYOPENGL_PLATFORM', 'egl')}",
            f"mode {cfg.vram.mode}",
            f"{s_img:.2f} s/img ({cfg.gen.level})" if s_img else cfg.gen.level,
            f"ollama: {models}",
        ]
        if WARMING:
            parts.append("⏳ warming up…")
        return " · ".join(parts)
    except Exception as exc:  # noqa: BLE001 — footer must never crash the app
        return f"footer unavailable: {exc}"
