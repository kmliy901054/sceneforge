"""scripts/forge_viewsweep.py — viewpoint-randomized dataset export (v2 feature B).

The anti-viewpoint-brittleness tool: literature says camera-viewpoint shift is
the catastrophic factor for VLAs, so this exports the SAME scene from many
calibrated viewpoints. ONE plan (LLM or fallback) produces ONE SceneSpec; the
scene is composed ONCE; the identical geometry is rendered across a camera grid
(azimuth × elevation × distance) × styles; ONE COCO zip is exported in which
every image record carries pinhole intrinsics ``K``, the 4×4 world-from-camera
``pose`` and the orbit view params under ``"sceneforge_camera"`` (plus
``cameras.json`` and per-view ground-truth ``depth/view_<k>_depth16.png`` —
labels/coco.py feature C).

Per view: depth control is re-rendered (never resized, §7.3) and the §5.4
target-visibility guard runs (bump elevation/distance once, else skip the view
loudly). Generation seed law: ``seed + view_idx*1000 + style_idx`` (§7.2).

Usage:
    python scripts/forge_viewsweep.py --task "pick the red mug from a cluttered
        kitchen table" [--views 8] [--styles 2] [--seed 42] [--no-llm]
    python scripts/forge_viewsweep.py --spec spec.json --azimuths 0,90,180,-90
        --elevations 25,55 --distances 1.0,1.6

Camera grid: explicit ``--azimuths/--elevations/--distances`` lists form a full
product (unset axes use the plan camera's value); otherwise ``--views`` N
azimuths evenly spaced around the table at the plan's elevation/distance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `sceneforge`

import sceneforge.compat  # noqa: F401,E402  — FIRST sceneforge import (§0)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402
from itertools import product  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from sceneforge.spec import CameraSpec, SceneSpec  # noqa: E402

logger = logging.getLogger("forge_viewsweep")


# ----------------------------------------------------------- grid (pure, tested)
def parse_float_list(text: Optional[str]) -> Optional[list[float]]:
    """``"0,45, -90"`` → [0.0, 45.0, -90.0]; None/empty → None."""
    if text is None or not text.strip():
        return None
    return [float(tok) for tok in text.split(",") if tok.strip()]


def wrap_azimuth(az: float) -> float:
    """Wrap to the CameraSpec azimuth domain [-180, 180]."""
    return float((az + 180.0) % 360.0 - 180.0)


def build_view_grid(
    base: CameraSpec,
    n_views: int = 8,
    azimuths: Optional[Sequence[float]] = None,
    elevations: Optional[Sequence[float]] = None,
    distances: Optional[Sequence[float]] = None,
) -> list[CameraSpec]:
    """The camera grid for one sweep — pure function, unit-tested CPU-only.

    Explicit lists → full ``azimuth × elevation × distance`` product (unset
    axes default to the base camera's value). No lists → ``n_views`` azimuths
    evenly spaced (360°/n) starting at the base azimuth. Elevation/distance
    are clamped into the §3.1 orbit bounds; ``yfov``/``look_at`` stay
    SERVER-PINNED from the base camera (never per-view).
    """
    if azimuths or elevations or distances:
        az_list = [wrap_azimuth(a) for a in (azimuths or [base.azimuth_deg])]
        el_list = list(elevations or [base.elevation_deg])
        d_list = list(distances or [base.distance_m])
    else:
        if n_views < 1:
            raise ValueError(f"--views must be >= 1; got {n_views}")
        az_list = [wrap_azimuth(base.azimuth_deg + i * 360.0 / n_views)
                   for i in range(n_views)]
        el_list = [base.elevation_deg]
        d_list = [base.distance_m]
    return [
        CameraSpec(
            azimuth_deg=wrap_azimuth(az),
            elevation_deg=float(np.clip(el, 10.0, 80.0)),
            distance_m=float(np.clip(d, 0.5, 2.5)),
            yfov_deg=base.yfov_deg,          # server-pinned (§3.1)
            look_at=base.look_at,
        )
        for az, el, d in product(az_list, el_list, d_list)
    ]


# ------------------------------------------------------------------ sweep run
def run_sweep(args: argparse.Namespace) -> dict:
    import cv2
    from sceneforge import gpu
    from sceneforge.config import get_config
    from sceneforge.diffusion.depth_prep import depth_to_control
    from sceneforge.director.director import DirectorLog, ground_plan, make_styles, plan_scene
    from sceneforge.director.fallback import default_style_set, template_plan
    from sceneforge.labels import masks as masks_mod
    from sceneforge.labels.coco import export_zip
    from sceneforge.orchestrator import ForgeRun, GeneratedImage
    from sceneforge.scene import compose
    from sceneforge.scene.placement import resolve_placements

    t_start = time.monotonic()
    cfg = get_config()
    runner = ForgeRun(cfg)
    res = cfg.gen.resolution
    log = DirectorLog()
    timings: dict = {}

    # ---------------------------------------- ONE plan → ONE resolved SceneSpec
    t0 = time.monotonic()
    if args.spec:
        spec = SceneSpec.model_validate_json(
            Path(args.spec).read_text(encoding="utf-8"))
        task = spec.task
        seed = int(args.seed) if args.seed is not None else spec.seed
        spec.seed = seed
        plan_source = "spec_file"
        style_source = "spec_file"
    else:
        task = args.task
        seed = int(args.seed) if args.seed is not None else 42
        with gpu.phase("spec", cfg=cfg):
            if args.no_llm:
                plan = template_plan(task, seed)
                style_set = default_style_set(args.styles)
                log.meta.update(plan_source="fallback", style_source="fallback")
            else:
                plan = plan_scene(task, seed, cfg, log=log)
                style_set = make_styles(task, plan, args.styles, cfg,
                                        seed=seed, log=log)
            spec = ground_plan(plan, style_set, task, seed,
                               runner.library, runner.rag)
        plan_source = log.meta.get("plan_source", "fallback")
        style_source = log.meta.get("style_source", "fallback")
    spec = resolve_placements(spec, runner.library)
    styles = spec.styles[: max(1, int(args.styles))]
    spec.styles = styles
    timings["plan_s"] = round(time.monotonic() - t0, 3)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{seed}_viewsweep"
    run_dir = Path(args.out) if args.out else Path(cfg.paths.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.flush(run_dir / "director_log.md")

    grid = build_view_grid(
        spec.camera,
        n_views=args.views,
        azimuths=parse_float_list(args.azimuths),
        elevations=parse_float_list(args.elevations),
        distances=parse_float_list(args.distances),
    )
    logger.info("task=%r seed=%d → %d views × %d styles (planner=%s)",
                task, seed, len(grid), len(styles), plan_source)

    # ------------------------------- compose ONCE, render the SAME scene per view
    t0 = time.monotonic()
    composed = compose.build(spec, runner.library, run_dir)  # writes viewer.glb
    layouts: list[SimpleNamespace] = []
    view_meta: list[dict] = []
    for k, cam in enumerate(grid):
        view_spec = spec.model_copy(deep=True)
        view_spec.camera = cam
        render = runner.renderer.render_scene(composed, cam, res, res, spec=view_spec)
        render, bumped = runner._guard_target(view_spec, composed, render, res)
        if not runner._target_visible(view_spec, render):
            logger.warning("view %d SKIPPED: target invisible (area < %d px) "
                           "even after camera bump", k, masks_mod.TARGET_MIN_AREA_PX)
            view_meta.append({"view_idx": k,
                              "camera": cam.model_dump(mode="json"),
                              "skipped": True, "bumped": bumped})
            continue

        view_dir = run_dir / f"view_{k}"
        view_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "spec.json").write_text(view_spec.model_dump_json(indent=2),
                                            encoding="utf-8")
        depth16_path = masks_mod.write_depth16(view_dir / "depth16.png",
                                               render.depth_m)
        control = depth_to_control(render.depth_m, res, cfg.gen.depth_mode)
        control_path = view_dir / "control.png"
        control.save(control_path)
        cv2.imwrite(str(view_dir / "seg_ids.png"),
                    render.seg_ids.clip(0, 65535).astype(np.uint16))
        (view_dir / "labels.json").write_text(json.dumps({
            "layout_idx": k,
            "width": render.width,
            "height": render.height,
            "instances": [asdict(i) for i in render.instances],
        }), encoding="utf-8")

        layouts.append(SimpleNamespace(
            layout_idx=k,
            name=f"view_{k}",                  # → view_<k>_* names in the zip
            spec=view_spec,
            render=render,                     # K/pose/depth_m for cameras.json
            depth16_path=str(depth16_path),
            control_path=str(control_path),
        ))
        view_meta.append({
            "view_idx": k,
            "camera": view_spec.camera.model_dump(mode="json"),
            "skipped": False,
            "bumped": bumped,
            "n_instances": len(render.instances),
            "target_area_px": next(
                (i.area_px for i in render.instances if i.is_target), 0),
        })
    timings["render_s"] = round(time.monotonic() - t0, 3)
    if not layouts:
        raise RuntimeError("every view failed the target-visibility guard")

    # --------------------------------------------- diffusion: views × styles
    from PIL import Image

    images: list[GeneratedImage] = []
    t0 = time.monotonic()
    with gpu.phase("diffusion", cfg=cfg):
        runner.ensure_pipeline()
        total = len(layouts) * len(styles)
        for rec in layouts:                                # view-major
            control_img = Image.open(rec.control_path).convert("RGB")
            for style_idx, style in enumerate(styles):
                gen_seed = seed + rec.layout_idx * 1000 + style_idx  # §7.2
                t1 = time.monotonic()
                img = runner.pipeline.generate(
                    control_img, style.prompt, style.negative_prompt, gen_seed,
                    cond_scale=cfg.gen.cond_scale, steps=cfg.gen.steps)
                gen_s = time.monotonic() - t1
                path = run_dir / f"view_{rec.layout_idx}" / f"img_{style.name}.png"
                img.save(path)
                images.append(GeneratedImage(str(path), rec.layout_idx,
                                             style.name, gen_seed, round(gen_s, 3)))
                logger.info("forged %d/%d (view %d, %s, %.2fs)",
                            len(images), total, rec.layout_idx, style.name, gen_s)
    timings["diffusion_s"] = round(time.monotonic() - t0, 3)
    timings["diffusion_s_per_img"] = round(timings["diffusion_s"] / len(images), 3)

    # ------------------------------------------------ ONE COCO zip (RGB-D + cams)
    zip_path = export_zip(run_dir, layouts, images)
    timings["total_s"] = round(time.monotonic() - t_start, 3)

    meta = {
        "tool": "scripts/forge_viewsweep.py",
        "task": task,
        "seed": seed,
        "plan_source": plan_source,
        "style_source": style_source,
        "styles": [{"name": s.name, "prompt": s.prompt} for s in styles],
        "grid": {
            "requested_views": args.views,
            "azimuths": args.azimuths,
            "elevations": args.elevations,
            "distances": args.distances,
            "n_views_total": len(grid),
            "n_views_rendered": len(layouts),
        },
        "views": view_meta,
        "n_images": len(images),
        "timings": timings,
        "run_dir": str(run_dir),
        "dataset_zip": str(zip_path),
    }
    (run_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render ONE SceneSpec across a camera grid × styles and "
                    "export ONE COCO zip with per-image K/pose/depth.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", help="natural-language task → one plan")
    src.add_argument("--spec", help="path to a resolved SceneSpec spec.json")
    parser.add_argument("--views", type=int, default=8,
                        help="N azimuths evenly spaced (ignored when explicit "
                             "lists are given)")
    parser.add_argument("--styles", type=int, default=2)
    parser.add_argument("--azimuths", default=None, help="comma list, degrees")
    parser.add_argument("--elevations", default=None,
                        help="comma list, degrees (clamped to [10, 80])")
    parser.add_argument("--distances", default=None,
                        help="comma list, meters (clamped to [0.5, 2.5])")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true",
                        help="deterministic template plan + DEFAULT_STYLES")
    parser.add_argument("--out", default=None, help="override the run directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    meta = run_sweep(args)
    print(json.dumps({k: meta[k] for k in (
        "task", "seed", "plan_source", "style_source", "grid", "n_images",
        "timings", "run_dir", "dataset_zip")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
