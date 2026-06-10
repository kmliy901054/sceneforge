"""sceneforge/orchestrator.py — ForgeRun: the demo-paced pipeline (ARCHITECTURE.md §1, §4.7, §4.8).

Pipeline order is tuned for demo pacing (first visual ≤ ~5 s after FORGE):

    plan (tokens streamed) → clamp+validate → ground+place →
    RENDER LAYOUT 0 AND EMIT GLB + DEPTH CONTROL BEFORE THE STYLE LLM CALL →
    style call → remaining layouts → diffusion (layout-major, grid fills live) →
    fidelity.

``ForgeRun.run(...) -> Iterator[ForgeEvent]`` — event payloads are JSON-able
paths/scalars ONLY (the UI ``gr.State`` holds a dict of paths, never arrays).
Generation order is PINNED layout-major (outer loop layouts, inner loop styles)
so the gallery grid reads rows=layout / cols=style — asserted in ``on_forge``.

``ForgeRun.reforge(state, camera, seed, style_name, nudge)`` makes ZERO LLM
calls: re-render depth + 1 diffusion call; identical latency in both VRAM modes
(SDXL stays loaded). Target ≤ 6 s slider-to-image.

Run artifacts per §4.7 under ``outputs/runs/<run_id>/``:
    run_meta.json (task, config snapshot, timings, vram log, style CLIP-token
    counts) · director_log.md (runtime LLM transcripts, flushed from
    DirectorLog) · layout_<k>/{spec.json, viewer.glb, depth16.png, control.png,
    seg_ids.png, labels.json, img_<slug>.png, overlay_<slug>_<mode>.png} ·
    fidelity.json · coco/ + dataset.zip (at export).

Overlay PNGs for ALL modes are written at generation time so the UI radio
toggle is a pure cached-path swap (§8.2).
"""
from __future__ import annotations

import json
import logging
import queue as queue_mod
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Literal, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image

from sceneforge import gpu
from sceneforge.assets.library import AssetLibrary
from sceneforge.config import AppConfig, get_config
from sceneforge.diffusion.depth_prep import depth_to_control
from sceneforge.diffusion.pipeline import ForgePipeline
from sceneforge.director.director import (
    DirectorLog,
    ground_plan,
    make_styles,
    plan_scene,
    postvalidate_styles,
)
from sceneforge.director.fallback import default_style_set
from sceneforge.director.ollama_client import embed as ollama_embed
from sceneforge.director.rag import EmbeddingIndex
from sceneforge.labels import masks as masks_mod
from sceneforge.labels.coco import export_zip
from sceneforge.labels.overlay import draw_overlay
from sceneforge.render.types import InstanceLabel, RenderResult
from sceneforge.scene import compose
from sceneforge.scene.placement import resolve_placements
from sceneforge.spec import CameraSpec, SceneSpec

logger = logging.getLogger(__name__)

EventKind = Literal[
    "plan_token", "spec", "layout", "status", "image", "fidelity", "done", "error"
]

#: Overlay variants written to disk at generation time (§8.2); "off" is the raw image.
OVERLAY_FILE_MODES: tuple[str, ...] = ("boxes", "masks", "both")

#: §5.4 target-visibility guard: camera bump amounts (re-render ONCE).
GUARD_ELEVATION_BUMP_DEG = 15.0
GUARD_DISTANCE_FACTOR = 1.15

#: Nudge step for re-forge target moves (§11: ±x/±y 5 cm).
NUDGE_STEP_M = 0.05


# ----------------------------------------------------------- §4.5/§4.6/§4.8 shapes
@dataclass
class ForgeEvent:
    """One orchestrator event (§4.8). ``payload`` is JSON-able paths/scalars only."""

    kind: EventKind
    payload: dict


@dataclass
class GeneratedImage:
    """One diffusion output (§4.5)."""

    path: str
    layout_idx: int
    style_name: str
    seed: int
    gen_seconds: float


@dataclass
class LayoutRecord:
    """One rendered layout (§4.6): geometry products shared by every style."""

    layout_idx: int
    spec: SceneSpec
    render: RenderResult
    control_path: str
    glb_path: str


class _LayoutSkipped(Exception):
    """Internal: target-visibility guard failed twice — skip this layout (§5.4)."""


# ================================================================== ForgeRun
class ForgeRun:
    """Owns the per-process heavy singletons (renderer / SDXL pipeline / OWLv2
    scorer / asset library / RAG index) and drives forge + re-forge.

    All collaborators are injectable for tests; defaults are lazily built on
    first use so importing this module stays cheap.
    """

    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        *,
        library: Optional[AssetLibrary] = None,
        rag: Optional[EmbeddingIndex] = None,
        renderer: Any = None,
        pipeline: Any = None,
        scorer: Any = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else get_config()
        self._library = library
        self._rag = rag
        self._renderer = renderer
        self._pipeline = pipeline
        self._scorer = scorer
        self._pipe_lock = threading.Lock()

    # ------------------------------------------------------------ lazy deps
    @property
    def library(self) -> AssetLibrary:
        if self._library is None:
            self._library = AssetLibrary(cards_path=self.cfg.paths.cards_file,
                                         glb_dir=self.cfg.paths.glb_dir)
        return self._library

    @property
    def rag(self) -> EmbeddingIndex:
        if self._rag is None:
            cfg = self.cfg

            def _embed(texts: Sequence[str]) -> np.ndarray:
                # ALWAYS keep_alive=0 for embeds (§6.1 — server pins forever otherwise)
                return ollama_embed(cfg.ollama.embed_model, texts,
                                    keep_alive=0, host=cfg.ollama.host)

            self._rag = EmbeddingIndex.load_or_build(
                cfg.paths.cards_file, cfg.paths.embed_cache, embed_fn=_embed)
        return self._rag

    @property
    def renderer(self) -> Any:
        if self._renderer is None:
            from sceneforge.render import get_renderer  # lazy: touches GL

            self._renderer = get_renderer(self.cfg)
        return self._renderer

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = ForgePipeline(resolution=self.cfg.gen.resolution)
        return self._pipeline

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            from sceneforge.eval.fidelity import Owlv2Scorer  # lazy: transformers

            self._scorer = Owlv2Scorer(
                device="cuda" if torch.cuda.is_available() else "cpu",
                keep_threshold=self.cfg.eval.keep_threshold,
                det_threshold=self.cfg.eval.detect_threshold,
                gate_area_px=self.cfg.eval.min_gate_area_px,
            )
        return self._scorer

    @property
    def gen_level(self) -> int:
        return int(self.cfg.gen.level[1])

    def ensure_pipeline(self) -> None:
        """Eagerly build the SDXL pipeline on cuda (§10.5 pre-warm step 2).

        Lock-guarded so the pre-warm thread and a concurrent forge cannot
        construct two pipelines.
        """
        with self._pipe_lock:
            if getattr(self.pipeline, "pipe", None) is None:
                self.pipeline.load(self.gen_level)

    def ensure_scorer_cpu(self) -> None:
        """Instantiate OWLv2 processor+model once, weights to CPU (§10.5 step 3).

        ``phase("eval")`` moves the model to cuda for scoring and back after.
        """
        scorer = self.scorer
        if getattr(scorer, "model", None) is None and hasattr(scorer, "load"):
            device = scorer.device
            scorer.device = "cpu"
            try:
                scorer.load()
            finally:
                scorer.device = device

    # ============================================================== forge run
    def run(
        self,
        task: str,
        n_layouts: int = 2,
        n_styles: int = 4,
        seed: int = 42,
        style_ref: Optional[Image.Image] = None,
        style_ref_scale: Optional[float] = None,
    ) -> Iterator[ForgeEvent]:
        """Run one forge end-to-end, yielding §4.8 events in the §1 demo order.

        Event contract (asserted in tests/test_orchestrator.py):
        ``plan_token*`` → ``spec`` → ``layout``(0) → ``status``(stage="styles")
        → ``layout``(1..) → ``image``\\ * (PINNED layout-major) → ``fidelity``
        → ``done``. Recoverable per-layout failures emit ``error`` events and
        skip the layout; a fatal failure emits a final ``error`` event.

        ``style_ref``: optional PIL style-reference image. PRECEDENCE: the LLM
        still writes the per-style text prompts exactly as without a
        reference; the IP-Adapter then ADDS the reference's visual style on
        top of each text prompt during diffusion (image- and text-conditioning
        are combined in the UNet; ``style_ref_scale`` — default
        ``cfg.gen.ip_adapter_scale`` — balances the two, 0.0 = text only).
        The reference is enabled for THIS run's diffusion phase only and
        disabled afterwards, so later runs/re-forges start unstyled.
        """
        try:
            yield from self._run(task, int(n_layouts), int(n_styles), int(seed),
                                 style_ref=style_ref, style_ref_scale=style_ref_scale)
        except Exception as exc:  # noqa: BLE001 — never crash the UI handler
            logger.exception("forge run failed")
            yield ForgeEvent("error", {"fatal": True, "message": f"{type(exc).__name__}: {exc}"})

    def _run(self, task: str, n_layouts: int, n_styles: int, seed: int,
             style_ref: Optional[Image.Image] = None,
             style_ref_scale: Optional[float] = None) -> Iterator[ForgeEvent]:
        cfg = self.cfg
        n_layouts = max(1, min(n_layouts, 4))
        n_styles = max(2, min(n_styles, 6))
        res = cfg.gen.resolution

        t_start = time.monotonic()
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{seed}"
        run_dir = Path(cfg.paths.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        log = DirectorLog()
        vram_log: list[dict] = []
        timings: dict[str, Any] = {}
        state: dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "task": task,
            "seed": seed,
            "resolution": res,
            "n_layouts": n_layouts,
            "n_styles": n_styles,
            "styles": [],
            "layouts": [],
            "fidelity": None,
        }
        layouts: list[LayoutRecord] = []
        images: list[GeneratedImage] = []

        # ------------------------------------------------------ spec + render
        with gpu.phase("spec", cfg=cfg, vram_log=vram_log):
            yield ForgeEvent("status", {"stage": "planning"})

            t0 = time.monotonic()
            plan = None
            tokens: "queue_mod.Queue[Any]" = queue_mod.Queue()
            _DONE = object()
            box: dict[str, Any] = {}

            def _plan_worker() -> None:
                try:
                    box["plan"] = plan_scene(task, seed, cfg,
                                             on_token=tokens.put, log=log)
                except Exception as exc:  # plan_scene never raises by contract
                    box["error"] = exc
                finally:
                    tokens.put(_DONE)

            worker = threading.Thread(target=_plan_worker, daemon=True)
            worker.start()
            while True:
                item = tokens.get()
                if item is _DONE:
                    break
                yield ForgeEvent("plan_token", {"text": str(item)})
            worker.join()
            if "error" in box:
                raise box["error"]
            plan = box["plan"]
            timings["plan_s"] = round(time.monotonic() - t0, 3)

            # Ground + place layout 0 (placeholder styles — the style LLM call
            # is deliberately AFTER the first visual, §1).
            yield ForgeEvent("status", {"stage": "grounding"})
            t0 = time.monotonic()
            spec0 = ground_plan(plan, default_style_set(n_styles), task, seed,
                                self.library, self.rag)
            spec0 = resolve_placements(spec0, self.library)
            timings["ground_s"] = round(time.monotonic() - t0, 3)
            yield ForgeEvent("spec", {
                "spec": spec0.model_dump(mode="json"),
                "source": log.meta.get("plan_source", "fallback"),
                "repairs": log.meta.get("plan_repairs", 0),
            })

            # ---- layout 0: render + emit GLB + control BEFORE the style call
            yield ForgeEvent("status", {"stage": "rendering layout 0"})
            t0 = time.monotonic()
            try:
                rec0 = self._render_layout(spec0, 0, run_dir)
                layouts.append(rec0)
                state["layouts"].append(self._layout_state(rec0))
                timings["layout0_render_s"] = round(time.monotonic() - t0, 3)
                timings["first_visual_s"] = round(time.monotonic() - t_start, 3)
                yield ForgeEvent("layout", self._layout_state(rec0))
            except _LayoutSkipped as exc:
                yield ForgeEvent("error", {
                    "recoverable": True, "layout_idx": 0, "message": str(exc)})

            # ---- style LLM call (second call, after the first visual)
            yield ForgeEvent("status", {"stage": "styling"})
            t0 = time.monotonic()
            style_set = make_styles(task, plan, n_styles, cfg, seed=seed, log=log)
            categories = [o.category for o in spec0.objects]
            style_specs = postvalidate_styles(style_set, categories,
                                              spec0.grounding_log)
            timings["style_s"] = round(time.monotonic() - t0, 3)
            spec0.styles = style_specs
            if layouts:  # rewrite layout-0 spec.json with the real styles
                layouts[0] = LayoutRecord(0, spec0, layouts[0].render,
                                          layouts[0].control_path, layouts[0].glb_path)
                self._write_spec(spec0, run_dir / "layout_0")
            state["styles"] = [s.name for s in style_specs]
            yield ForgeEvent("status", {
                "stage": "styles",
                "styles": [s.name for s in style_specs],
                "source": log.meta.get("style_source", "fallback"),
            })

            # ---- remaining layouts
            t0 = time.monotonic()
            for k in range(1, n_layouts):
                yield ForgeEvent("status", {"stage": f"rendering layout {k}"})
                spec_k = self._vary_layout(spec0, k, seed)
                spec_k = resolve_placements(spec_k, self.library)
                try:
                    rec = self._render_layout(spec_k, k, run_dir)
                except _LayoutSkipped as exc:
                    yield ForgeEvent("error", {
                        "recoverable": True, "layout_idx": k, "message": str(exc)})
                    continue
                layouts.append(rec)
                state["layouts"].append(self._layout_state(rec))
                yield ForgeEvent("layout", self._layout_state(rec))
            timings["layouts_rest_render_s"] = round(time.monotonic() - t0, 3)

        log.flush(run_dir / "director_log.md")  # §4.7 — runtime transcripts

        if not layouts:
            raise RuntimeError("every layout failed the target-visibility guard")

        # ------------------------------------------------------------ diffusion
        total_imgs = len(layouts) * len(style_specs)
        t_diff = time.monotonic()
        ip_scale = float(style_ref_scale if style_ref_scale is not None
                         else cfg.gen.ip_adapter_scale)
        style_ref_meta: Optional[dict] = None
        if style_ref is not None:
            ref_path = run_dir / "style_ref.png"  # provenance (§4.7 artifacts)
            style_ref.convert("RGB").save(ref_path)
            style_ref_meta = {"path": str(ref_path), "scale": ip_scale,
                              "adapter": "h94/IP-Adapter sdxl vit-h"}
            state["style_ref"] = style_ref_meta
        try:
            with gpu.phase("diffusion", cfg=cfg, vram_log=vram_log):
                self.ensure_pipeline()
                if style_ref is not None:
                    # IP-Adapter ADDS the reference's visual style on top of
                    # the LLM style prompts — enabled for this run only.
                    yield ForgeEvent("status", {
                        "stage": f"style reference on (scale {ip_scale:.2f})"})
                    self.pipeline.enable_style_reference(style_ref, scale=ip_scale)
                done_count = 0
                for rec in layouts:  # PINNED layout-major (§4.8): outer layouts...
                    control = Image.open(rec.control_path).convert("RGB")
                    layout_dir = run_dir / f"layout_{rec.layout_idx}"
                    lstate = self._state_entry(state, rec.layout_idx)
                    for style_idx, style in enumerate(style_specs):  # ...inner styles
                        gen_seed = seed + rec.layout_idx * 1000 + style_idx  # §7.2 seed law
                        t0 = time.monotonic()
                        img = self.pipeline.generate(
                            control, style.prompt, style.negative_prompt, gen_seed,
                            cond_scale=cfg.gen.cond_scale, steps=cfg.gen.steps)
                        gen_s = time.monotonic() - t0
                        paths = self._write_image_set(img, rec.render.instances,
                                                      layout_dir, style.name)
                        images.append(GeneratedImage(paths["off"], rec.layout_idx,
                                                     style.name, gen_seed,
                                                     round(gen_s, 3)))
                        if lstate is not None:
                            lstate["images"][style.name] = paths
                        done_count += 1
                        yield ForgeEvent("status", {
                            "stage": f"forging {done_count}/{total_imgs}"})
                        yield ForgeEvent("image", {
                            "layout_idx": rec.layout_idx,
                            "style": style.name,
                            "style_idx": style_idx,
                            "path": paths["off"],
                            "overlays": paths,
                            "seed": gen_seed,
                            "gen_seconds": round(gen_s, 3),
                            "index": done_count,
                            "total": total_imgs,
                        })
                timings["diffusion_peak_vram"] = self._peak_vram()
        finally:
            if style_ref is not None:
                # later runs / re-forges start unstyled (run() docstring)
                self.pipeline.disable_style_reference()
        timings["diffusion_s"] = round(time.monotonic() - t_diff, 3)
        timings["diffusion_s_per_img"] = round(
            (time.monotonic() - t_diff) / max(1, len(images)), 3)

        # -------------------------------------------------------------- fidelity
        yield ForgeEvent("status", {"stage": "scoring"})
        t0 = time.monotonic()
        with gpu.phase("eval", cfg=cfg, vram_log=vram_log):
            report = self._score(images, layouts)
        timings["eval_s"] = round(time.monotonic() - t0, 3)
        fid_payload = self._fidelity_payload(report, run_dir)
        state["fidelity"] = fid_payload
        yield ForgeEvent("fidelity", fid_payload)

        # ------------------------------------------------------------ run meta
        timings["total_s"] = round(time.monotonic() - t_start, 3)
        meta = {
            "run_id": run_id,
            "task": task,
            "seed": seed,
            "n_layouts": n_layouts,
            "n_styles": n_styles,
            "config": cfg.model_dump(mode="json"),
            "style_ref": style_ref_meta,  # None unless a reference was used
            "timings": timings,
            "vram_log": vram_log,
            "style_clip_tokens": [
                e for e in spec0.grounding_log if e.get("event") == "style_tokens"
            ],
            "director": dict(log.meta),
            "images": [asdict(g) for g in images],
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")

        yield ForgeEvent("status", {"stage": "done"})
        yield ForgeEvent("done", {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "timings": timings,
            "state": state,
        })

    # =============================================================== re-forge
    def reforge(
        self,
        state: dict,
        camera: Union[dict, Sequence[float], None] = None,
        seed: Optional[int] = None,
        style_name: Optional[str] = None,
        nudge: Optional[str] = None,
    ) -> tuple[str, str, dict, tuple]:
        """§4.8 re-forge: NO LLM calls — re-render depth + ONE diffusion call.

        Args:
            state: the JSON-able run-state dict from the ``done`` event.
            camera: ``{"azimuth_deg", "elevation_deg", "distance_m"}`` (or an
                (az, el, dist) triple); None keeps the layout's camera.
            seed: explicit generation seed; None/negative → the §7.2 seed law.
            style_name: style slug; None → the layout's first style.
            nudge: one of "+x"/"-x"/"+y"/"-y" — moves the TARGET ±5 cm (§11).

        Returns:
            ``(glb_path, image_path, overlays, annotated)`` where ``overlays``
            maps mode→path ("off"/"boxes"/"masks"/"both", §8.2) and
            ``annotated`` is the ``gr.AnnotatedImage`` value
            ``(image_path, [(mask, category), ...])`` so labels visibly track
            the camera (§11).
        """
        if not state or not state.get("layouts"):
            raise ValueError("re-forge needs a completed forge run (empty state)")
        layout_idx = int(state.get("reforge_layout", state["layouts"][0]["layout_idx"]))
        lstate = self._state_entry(state, layout_idx) or state["layouts"][0]
        spec = SceneSpec.model_validate_json(
            Path(lstate["spec_json"]).read_text(encoding="utf-8"))

        if nudge:
            dx, dy = {
                "+x": (NUDGE_STEP_M, 0.0), "-x": (-NUDGE_STEP_M, 0.0),
                "+y": (0.0, NUDGE_STEP_M), "-y": (0.0, -NUDGE_STEP_M),
            }[nudge]
            for obj in spec.objects:
                if obj.is_target:
                    obj.x_m += dx
                    obj.y_m += dy
            spec = resolve_placements(spec, self.library)

        if camera is not None:
            if isinstance(camera, dict):
                az = camera.get("azimuth_deg", spec.camera.azimuth_deg)
                el = camera.get("elevation_deg", spec.camera.elevation_deg)
                dist = camera.get("distance_m", spec.camera.distance_m)
            else:
                az, el, dist = camera
            # look_at / yfov stay SERVER-PINNED (§3.1) — CameraSpec defaults.
            spec.camera = CameraSpec(
                azimuth_deg=float(np.clip(az, -180, 180)),
                elevation_deg=float(np.clip(el, 10, 80)),
                distance_m=float(np.clip(dist, 0.5, 2.5)),
            )

        out_dir = Path(lstate["dir"]) / "reforge"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = self.cfg.gen.resolution

        composed = compose.build(spec, self.library, out_dir)
        render = self.renderer.render_scene(composed, spec.camera, res, res, spec=spec)
        render = self._guard_target(spec, composed, render, res)[0]

        control = depth_to_control(render.depth_m, res, self.cfg.gen.depth_mode)
        control_path = out_dir / "control.png"
        control.save(control_path)

        styles = {s.name: s for s in spec.styles}
        style = styles.get(style_name) if style_name else None
        if style is None:
            style = spec.styles[0]
        style_idx = [s.name for s in spec.styles].index(style.name)
        if seed is None or int(seed) < 0:
            gen_seed = spec.seed + layout_idx * 1000 + style_idx  # §7.2 seed law
        else:
            gen_seed = int(seed)

        self.ensure_pipeline()
        img = self.pipeline.generate(
            control, style.prompt, style.negative_prompt, gen_seed,
            cond_scale=self.cfg.gen.cond_scale, steps=self.cfg.gen.steps)
        paths = self._write_image_set(img, render.instances, out_dir, style.name)

        annotations = []
        for inst in render.instances:
            mask = masks_mod.mask_util.decode(dict(inst.rle)).astype(np.float32)
            label = f"{inst.category} (target)" if inst.is_target else inst.category
            annotations.append((mask, label))
        annotated = (paths["off"], annotations)
        return composed.glb_path, paths["off"], paths, annotated

    # ================================================================ helpers
    def _render_layout(self, spec: SceneSpec, layout_idx: int, run_dir: Path) -> LayoutRecord:
        """Compose + render one layout, run the §5.4 target-visibility guard,
        and write the §4.7 per-layout artifacts. Raises ``_LayoutSkipped`` if
        the target is still invisible after one camera bump."""
        res = self.cfg.gen.resolution
        layout_dir = run_dir / f"layout_{layout_idx}"
        composed = compose.build(spec, self.library, layout_dir)
        render = self.renderer.render_scene(composed, spec.camera, res, res, spec=spec)

        render, bumped = self._guard_target(spec, composed, render, res)
        if not self._target_visible(spec, render):
            raise _LayoutSkipped(
                f"layout {layout_idx}: target invisible (area < "
                f"{masks_mod.TARGET_MIN_AREA_PX} px) even after camera bump "
                f"(el +{GUARD_ELEVATION_BUMP_DEG}, dist ×{GUARD_DISTANCE_FACTOR})")
        if bumped:
            spec.grounding_log.append({
                "event": "camera_bump", "layout_idx": layout_idx,
                "elevation_deg": spec.camera.elevation_deg,
                "distance_m": spec.camera.distance_m,
            })

        self._write_spec(spec, layout_dir)
        masks_mod.write_depth16(layout_dir / "depth16.png", render.depth_m)
        control = depth_to_control(render.depth_m, res, self.cfg.gen.depth_mode)
        control_path = layout_dir / "control.png"
        control.save(control_path)
        cv2.imwrite(str(layout_dir / "seg_ids.png"),
                    render.seg_ids.clip(0, 65535).astype(np.uint16))
        (layout_dir / "labels.json").write_text(json.dumps({
            "layout_idx": layout_idx,
            "width": render.width,
            "height": render.height,
            "instances": [asdict(i) for i in render.instances],
        }), encoding="utf-8")
        return LayoutRecord(layout_idx, spec, render, str(control_path),
                            composed.glb_path)

    def _guard_target(self, spec: SceneSpec, composed: Any, render: RenderResult,
                      res: int) -> tuple[RenderResult, bool]:
        """§5.4 guard: target must appear with area ≥ 1000 px; else bump the
        camera (el +15 clamp 80, dist ×1.15 clamp 2.5) and re-render ONCE."""
        if self._target_visible(spec, render):
            return render, False
        cam = spec.camera
        spec.camera = CameraSpec(
            azimuth_deg=cam.azimuth_deg,
            elevation_deg=min(80.0, cam.elevation_deg + GUARD_ELEVATION_BUMP_DEG),
            distance_m=min(2.5, cam.distance_m * GUARD_DISTANCE_FACTOR),
            yfov_deg=cam.yfov_deg,
            look_at=cam.look_at,
        )
        logger.warning("target-visibility guard: bumping camera to el=%.0f dist=%.2f",
                       spec.camera.elevation_deg, spec.camera.distance_m)
        render = self.renderer.render_scene(composed, spec.camera, res, res, spec=spec)
        return render, True

    @staticmethod
    def _target_visible(spec: SceneSpec, render: RenderResult) -> bool:
        target_id = next(o.instance_id for o in spec.objects if o.is_target)
        return any(
            i.instance_id == target_id and i.area_px >= masks_mod.TARGET_MIN_AREA_PX
            for i in render.instances
        )

    @staticmethod
    def _vary_layout(spec0: SceneSpec, k: int, seed: int) -> SceneSpec:
        """Deterministic layout-k variation of the grounded plan: jitter object
        poses and orbit the camera. Pure function of (spec0, k, seed)."""
        rng = np.random.default_rng(seed + 9973 * k)
        spec = spec0.model_copy(deep=True)
        for obj in spec.objects:
            obj.x_m += float(rng.uniform(-0.08, 0.08))
            obj.y_m += float(rng.uniform(-0.06, 0.06))
            yaw = obj.yaw_deg + float(rng.uniform(-45.0, 45.0))
            obj.yaw_deg = (yaw + 180.0) % 360.0 - 180.0
        cam = spec.camera
        az = (cam.azimuth_deg + 65.0 * k + float(rng.uniform(-10, 10)) + 180.0) % 360.0 - 180.0
        spec.camera = CameraSpec(
            azimuth_deg=az,
            elevation_deg=float(np.clip(cam.elevation_deg + rng.uniform(-5, 10), 10, 80)),
            distance_m=float(np.clip(cam.distance_m * rng.uniform(0.95, 1.1), 0.5, 2.5)),
            yfov_deg=cam.yfov_deg,
            look_at=cam.look_at,
        )
        spec.grounding_log.append({"event": "layout_variation", "layout_idx": k})
        return spec

    @staticmethod
    def _write_spec(spec: SceneSpec, layout_dir: Path) -> None:
        layout_dir.mkdir(parents=True, exist_ok=True)
        (layout_dir / "spec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8")

    def _write_image_set(self, img: Image.Image, instances: Sequence[Any],
                         dest_dir: Path, slug: str) -> dict[str, str]:
        """Write the raw image + overlay PNGs for ALL modes (§8.2 — the UI
        radio toggle is a pure cached-path swap). Returns mode→path."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        raw_path = dest_dir / f"img_{slug}.png"
        img.save(raw_path)
        paths = {"off": str(raw_path)}
        arr = np.asarray(img.convert("RGB"))
        for mode in OVERLAY_FILE_MODES:
            over = draw_overlay(arr, instances, mode)  # type: ignore[arg-type]
            path = dest_dir / f"overlay_{slug}_{mode}.png"
            Image.fromarray(over).save(path)
            paths[mode] = str(path)
        return paths

    @staticmethod
    def _layout_state(rec: LayoutRecord) -> dict:
        """JSON-able per-layout state entry (paths/scalars only, §4.8)."""
        layout_dir = Path(rec.control_path).parent
        cam = rec.spec.camera
        return {
            "layout_idx": rec.layout_idx,
            "dir": str(layout_dir),
            "spec_json": str(layout_dir / "spec.json"),
            "glb": rec.glb_path,
            "control": rec.control_path,
            "depth16": str(layout_dir / "depth16.png"),
            "labels": str(layout_dir / "labels.json"),
            "n_instances": len(rec.render.instances),
            "camera": {"azimuth_deg": cam.azimuth_deg,
                       "elevation_deg": cam.elevation_deg,
                       "distance_m": cam.distance_m},
            "images": {},
        }

    @staticmethod
    def _state_entry(state: dict, layout_idx: int) -> Optional[dict]:
        for entry in state.get("layouts", []):
            if int(entry.get("layout_idx", -1)) == layout_idx:
                return entry
        return None

    def _peak_vram(self) -> dict:
        try:
            return {k: round(float(v), 2)
                    for k, v in self.pipeline.peak_vram().items()}
        except Exception:  # mocked pipelines in tests
            return {}

    # --------------------------------------------------------------- fidelity
    def _score(self, images: Sequence[GeneratedImage],
               layouts: Sequence[LayoutRecord]) -> Any:
        """§9 wiring inside phase("eval"): move the (possibly CPU-pre-warmed,
        §10.5) OWLv2 model to its device, score, then park it back on CPU."""
        from sceneforge.eval.fidelity import GTInstance  # cheap, no weights

        by_idx = {rec.layout_idx: rec for rec in layouts}
        gts = [
            [GTInstance(category=i.category, bbox_xywh=tuple(i.bbox_xywh),
                        area_px=i.area_px, is_target=i.is_target)
             for i in by_idx[g.layout_idx].render.instances]
            for g in images
        ]
        scorer = self.scorer
        model = getattr(scorer, "model", None)
        if (model is not None and hasattr(model, "to")
                and str(scorer.device).startswith("cuda") and torch.cuda.is_available()):
            try:
                if next(model.parameters()).device.type != "cuda":
                    model.to(scorer.device)  # CPU-pre-warmed weights → cuda (§10.5)
            except StopIteration:  # pragma: no cover
                pass
        try:
            return scorer.score_batch([g.path for g in images], gts)
        finally:
            model = getattr(scorer, "model", None)
            if model is not None and hasattr(model, "to") and torch.cuda.is_available():
                model.to("cpu")  # OWLv2 lives in phase("eval") only (§10.3)
                torch.cuda.empty_cache()

    def _fidelity_payload(self, report: Any, run_dir: Path) -> dict:
        """FidelityReport → JSON-able payload + ``fidelity.json`` (§4.7)."""
        per_image = [asdict(s) for s in report.per_image]
        adj = [s["fidelity_adj"] for s in per_image]
        payload = {
            "fidelity_adj_mean": round(float(np.mean(adj)), 4) if adj else 0.0,
            "match_rate": round(float(report.match_rate), 4),
            "mean_matched_iou": round(float(report.mean_matched_iou), 4),
            "hallucination_rate": round(float(report.hallucination_rate), 4),
            "kept": [str(p) for p in report.kept],
            "quarantined": [str(p) for p in report.quarantined],
            "n_kept": len(report.kept),
            "n_quarantined": len(report.quarantined),
            "n_gate_eligible": int(getattr(report, "n_gate_eligible", 0)),
            "n_gt_total": int(getattr(report, "n_gt_total", 0)),
            "path": str(run_dir / "fidelity.json"),
        }
        (run_dir / "fidelity.json").write_text(
            json.dumps({**payload, "per_image": per_image}, indent=2),
            encoding="utf-8")
        return payload


# ============================================================== COCO export
def export_coco(run_dir: Union[str, Path], include_quarantined: bool = False) -> Path:
    """Export ``<run_dir>`` → ``dataset.zip`` (§8.3) from on-disk artifacts only.

    Reconstructs layouts from ``layout_<k>/{spec.json, labels.json}`` and images
    from ``img_<slug>.png`` files; quarantined images (``fidelity.json``) are
    excluded unless ``include_quarantined``. Returns the zip path for
    ``gr.DownloadButton``.
    """
    run_dir = Path(run_dir)
    layouts = []
    images = []
    for layout_dir in sorted(run_dir.glob("layout_*")):
        labels_path = layout_dir / "labels.json"
        spec_path = layout_dir / "spec.json"
        if not labels_path.is_file() or not spec_path.is_file():
            continue
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        instances = [
            InstanceLabel(
                instance_id=int(i["instance_id"]),
                asset_id=str(i["asset_id"]),
                category=str(i["category"]),
                is_target=bool(i["is_target"]),
                bbox_xywh=tuple(int(v) for v in i["bbox_xywh"]),
                area_px=int(i["area_px"]),
                rle=dict(i["rle"]),
            )
            for i in meta["instances"]
        ]
        layout_idx = int(meta["layout_idx"])
        depth16 = layout_dir / "depth16.png"  # feature C: RGB-D export source
        layouts.append(SimpleNamespace(
            layout_idx=layout_idx,
            spec=json.loads(spec_path.read_text(encoding="utf-8")),
            render=SimpleNamespace(width=int(meta["width"]),
                                   height=int(meta["height"]),
                                   instances=instances),
            depth16_path=str(depth16) if depth16.is_file() else None,
        ))
        for img_path in sorted(layout_dir.glob("img_*.png")):
            slug = img_path.stem[len("img_"):]
            images.append(SimpleNamespace(path=str(img_path),
                                          layout_idx=layout_idx,
                                          style_name=slug, seed=0,
                                          gen_seconds=0.0))
    if not layouts or not images:
        raise FileNotFoundError(f"no exportable layouts/images under {run_dir}")

    keep = None
    fidelity_summary = None
    fid_path = run_dir / "fidelity.json"
    if fid_path.is_file():
        fid = json.loads(fid_path.read_text(encoding="utf-8"))
        fidelity_summary = {k: fid.get(k) for k in (
            "fidelity_adj_mean", "match_rate", "mean_matched_iou",
            "hallucination_rate", "n_kept", "n_quarantined")}
        if not include_quarantined:
            quarantined = {str(Path(p)) for p in fid.get("quarantined", [])}
            keep = [g.path for g in images if str(Path(g.path)) not in quarantined]
    return export_zip(run_dir, layouts, images, keep,
                      fidelity_summary=fidelity_summary)
