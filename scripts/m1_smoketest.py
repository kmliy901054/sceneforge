"""M1 go/no-go smoketest (ARCHITECTURE.md §13-M1, docs/TASKS.md M1). THROWAWAY harness.

SELF-CONTAINED: imports from sceneforge ONLY compat, diffusion.depth_prep,
diffusion.pipeline (L0 recipe) and the minimal OWLv2 scorer (eval.fidelity).
No builders, no placement, no director — 4 inline scenes from raw trimesh
primitives x 3 seeds = 12 renders with an inline §3.1 orbit camera, inline SEG
pass and inline mask->bbox.

Stages:
  0. weights check (5 HF repos, fp16 patterns, verify+resume), import asserts,
     768x768 EGL color+depth+SEG assert.
  1. 12 renders (box / can / composed-mug / sphere+plate; medium-large only,
     3-4 objects per scene -> >=30 gate-eligible GT at area>=1000).
  2. Ollama sweep-unload, then depth A/B (disparity vs linear) at L0: 24 images.
  3. OWLv2 scoring of both arms -> A/B winner.
  4. Gate at L0 (match_rate>=0.70 AND mean_matched_iou>=0.65 over gate-eligible
     GT); on FAIL walk the ladder L1 -> L2 -> L3 until pass.
  5. VRAM mode pick (§10.1-10.3): sequential vs coresident (gemma4:e4b@4096),
     rule: coresident only if min-device-free >= guard_gb (3.0) THROUGHOUT.
  6. outputs/m1/metrics.json + comparison grid + overlay contact sheet; freeze
     gen.level / gen.depth_mode / gen.s_per_img / vram.mode into sceneforge.yaml.

Run: cd /home/pairlab/DGAN/sceneforge && python scripts/m1_smoketest.py
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sceneforge.compat  # noqa: F401,E402  — MUST precede any pyrender import (§0)

import json
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests
import torch
import trimesh
from PIL import Image, ImageDraw

import pyrender  # noqa: E402  (after compat)

from sceneforge.diffusion.depth_prep import depth_to_control
from sceneforge.diffusion.pipeline import LEVEL_PARAMS, ForgePipeline
from sceneforge.eval.fidelity import FidelityReport, GTInstance, Owlv2Scorer

REPO_ROOT = "/home/pairlab/DGAN/sceneforge"
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "m1")
OLLAMA = "http://localhost:11434"
RES = 768
GUARD_GB = 3.0
GATE_MATCH_RATE = 0.70
GATE_MEAN_IOU = 0.65
NEGATIVE_PROMPT = ("blurry, deformed, duplicate objects, lowres, watermark, "
                   "cartoon, painting")  # active only at L2/L3 (cfg>1), §7.2

# ---------------------------------------------------------------- step 0a: weights
ALLOW_PATTERNS: dict[str, list[str] | None] = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "**/*.json", "*.json", "**/*.txt", "**/*.fp16.safetensors"],
    "diffusers/controlnet-depth-sdxl-1.0": ["*.json", "*.fp16.safetensors"],
    "madebyollin/sdxl-vae-fp16-fix": ["config.json", "diffusion_pytorch_model.safetensors"],
    "ByteDance/SDXL-Lightning": [
        "sdxl_lightning_4step_unet.safetensors", "sdxl_lightning_4step_lora.safetensors"],
    "google/owlv2-base-patch16-ensemble": None,  # whole repo (~0.6 GB)
}
REQUIRED_FILES: dict[str, list[str]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "model_index.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors", "tokenizer/vocab.json",
        "tokenizer_2/vocab.json", "scheduler/scheduler_config.json"],
    "diffusers/controlnet-depth-sdxl-1.0": [
        "config.json", "diffusion_pytorch_model.fp16.safetensors"],
    "madebyollin/sdxl-vae-fp16-fix": [
        "config.json", "diffusion_pytorch_model.safetensors"],
    "ByteDance/SDXL-Lightning": [
        "sdxl_lightning_4step_unet.safetensors", "sdxl_lightning_4step_lora.safetensors"],
    "google/owlv2-base-patch16-ensemble": [
        "config.json", "preprocessor_config.json", "model.safetensors"],
}


def check_weights() -> None:
    """Idempotent verify+resume of all 5 repos; per-repo byte totals; fail fast."""
    from huggingface_hub import hf_hub_download, snapshot_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    for repo, files in REQUIRED_FILES.items():
        missing = []
        for f in files:
            try:
                hf_hub_download(repo, f, local_files_only=True)
            except Exception:
                missing.append(f)
        if missing:
            print(f"[weights] {repo}: {len(missing)} files missing -> resuming download")
            snapshot_download(repo, allow_patterns=ALLOW_PATTERNS[repo])
        path = snapshot_download(repo, allow_patterns=ALLOW_PATTERNS[repo],
                                 local_files_only=True)
        total = 0
        for root, _dirs, fnames in os.walk(path):
            for fn in fnames:
                fp = os.path.join(root, fn)
                if os.path.exists(fp):
                    total += os.path.getsize(os.path.realpath(fp))
        print(f"[weights] OK {repo}: {total / 1e9:.2f} GB")


def assert_imports() -> None:
    import transformers
    assert transformers.__version__.startswith("4."), transformers.__version__
    from transformers import Owlv2ForObjectDetection  # noqa: F401
    from diffusers import StableDiffusionXLControlNetPipeline  # noqa: F401
    print(f"[imports] transformers {transformers.__version__}: "
          "Owlv2ForObjectDetection + StableDiffusionXLControlNetPipeline import cleanly")


def assert_egl_768() -> None:
    """§0 item 3: full 768x768 EGL color+depth+SEG render must work."""
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.visual.vertex_colors = (200, 60, 50, 255)
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.4, 0.4, 0.4))
    node = scene.add(pyrender.Mesh.from_trimesh(box, smooth=False), pose=np.eye(4))
    cam_pose = orbit_pose(35.0, 30.0, 1.0, (0.0, 0.0, 0.05))
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(50.0)), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
    r = pyrender.OffscreenRenderer(RES, RES)
    try:
        color, depth = r.render(scene)
        seg, _ = r.render(scene, flags=pyrender.RenderFlags.SEG,
                          seg_node_map={node: (7, 0, 0)})
    finally:
        r.delete()
    ids = set(np.unique(seg[:, :, 0]).tolist())
    assert color.shape == (RES, RES, 3) and depth.shape == (RES, RES)
    assert depth.max() > 0 and ids == {0, 7}, f"SEG ids did not round-trip: {ids}"
    print(f"[egl] 768x768 color+depth+SEG OK (seg ids {sorted(ids)}, "
          f"box px={(seg[:, :, 0] == 7).sum()})")


# ------------------------------------------------------------- inline scene engine
def orbit_pose(az_deg: float, el_deg: float, dist_m: float,
               look_at: tuple[float, float, float]) -> np.ndarray:
    """§3.1 NORMATIVE orbit pose (pyrender looks down its -Z, up ~ +Z world)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    look = np.asarray(look_at, dtype=np.float64)
    eye = look + dist_m * np.array([np.cos(el) * np.cos(az),
                                    np.cos(el) * np.sin(az), np.sin(el)])
    z_axis = (eye - look) / np.linalg.norm(eye - look)
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x_axis, y_axis, z_axis, eye
    return pose


def _floored(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    return mesh


def make_mug(scale: float = 1.0) -> trimesh.Trimesh:
    body = trimesh.creation.cylinder(radius=0.048 * scale, height=0.105 * scale,
                                     sections=48)
    body.apply_translation([0, 0, 0.0525 * scale])
    handle = trimesh.creation.torus(major_radius=0.034 * scale,
                                    minor_radius=0.011 * scale)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([0.060 * scale, 0, 0.0525 * scale])
    return _floored(trimesh.util.concatenate([body, handle]))


PALETTE = [(204, 64, 52), (62, 96, 204), (72, 168, 92), (222, 178, 64),
           (158, 96, 192), (220, 220, 215)]

# scene -> (categories+meshes builder, anchor xy positions, prompt object clause)
def _scene_box() -> list[tuple[str, trimesh.Trimesh]]:
    sizes = [(0.18, 0.12, 0.08), (0.14, 0.14, 0.11), (0.16, 0.10, 0.07),
             (0.12, 0.12, 0.13)]
    return [("box", _floored(trimesh.creation.box(extents=s))) for s in sizes]


def _scene_can() -> list[tuple[str, trimesh.Trimesh]]:
    dims = [(0.034, 0.118), (0.042, 0.140), (0.038, 0.125), (0.048, 0.155)]
    out = []
    for r, h in dims:
        c = trimesh.creation.cylinder(radius=r, height=h, sections=40)
        out.append(("can", _floored(c)))
    return out


def _scene_mug() -> list[tuple[str, trimesh.Trimesh]]:
    return [("mug", make_mug(s)) for s in (1.0, 1.15, 1.3)]


def _scene_sphere_plate() -> list[tuple[str, trimesh.Trimesh]]:
    out = []
    for r in (0.055, 0.062):
        out.append(("ball", _floored(trimesh.creation.icosphere(subdivisions=3, radius=r))))
    for r in (0.105, 0.115):
        out.append(("plate", _floored(trimesh.creation.cylinder(radius=r, height=0.018,
                                                                sections=48))))
    return out


SCENES: list[dict] = [
    dict(name="box", build=_scene_box,
         anchors=[(-0.30, 0.10), (0.00, -0.18), (0.30, 0.12), (0.02, 0.16)],
         clause="four cardboard boxes"),
    dict(name="can", build=_scene_can,
         anchors=[(-0.30, 0.10), (0.00, -0.18), (0.30, 0.12), (0.02, 0.16)],
         clause="four aluminum soda cans"),
    dict(name="mug", build=_scene_mug,
         anchors=[(-0.26, 0.08), (0.06, -0.16), (0.28, 0.14)],
         clause="three ceramic mugs"),
    dict(name="sphere_plate", build=_scene_sphere_plate,
         anchors=[(0.00, -0.18), (0.04, 0.17), (-0.30, 0.08), (0.30, 0.10)],
         clause="two balls and two plates"),
]


@dataclass(eq=False)  # identity equality: fields hold numpy arrays
class M1Render:
    scene_name: str
    seed_idx: int
    prompt: str
    color: np.ndarray            # (H,W,3) uint8
    depth_m: np.ndarray          # (H,W) float32
    gt: list[GTInstance]


_RENDERER: pyrender.OffscreenRenderer | None = None


def get_renderer() -> pyrender.OffscreenRenderer:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = pyrender.OffscreenRenderer(RES, RES)
    return _RENDERER


def render_m1_scene(scene_idx: int, seed_idx: int) -> M1Render:
    """One inline render: manual pyrender scene, color/depth + SEG, mask->bbox."""
    sd = SCENES[scene_idx]
    rng = np.random.default_rng(1000 * scene_idx + seed_idx)
    objects = sd["build"]()

    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.4, 0.4, 0.4))
    # static: floor (depth background only) + table slab, EXCLUDED from seg map
    floor = trimesh.creation.box(extents=(2.5, 2.5, 0.02))
    floor.apply_translation([0, 0, -0.76])
    floor.visual.vertex_colors = (90, 90, 95, 255)
    table = trimesh.creation.box(extents=(1.2, 0.8, 0.04))
    table.apply_translation([0, 0, -0.02])
    table.visual.vertex_colors = (168, 144, 118, 255)
    for m in (floor, table):
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False), pose=np.eye(4))

    node_for_instance: dict[int, pyrender.Node] = {}
    categories: dict[int, str] = {}
    for i, (cat, mesh) in enumerate(objects):
        iid = i + 1
        mesh = mesh.copy()
        rgb = PALETTE[i % len(PALETTE)]
        mesh.visual.vertex_colors = (*rgb, 255)
        ax, ay = sd["anchors"][i]
        x = ax + rng.uniform(-0.025, 0.025)
        y = ay + rng.uniform(-0.025, 0.025)
        yaw = rng.uniform(-np.pi, np.pi)
        T = (trimesh.transformations.translation_matrix([x, y, 0.0])
             @ trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
        node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=T)
        node_for_instance[iid] = node
        categories[iid] = cat

    az = 20.0 + 33.0 * seed_idx + 5.0 * scene_idx
    el = 30.0 + 7.0 * seed_idx
    dist = 0.95 + 0.12 * seed_idx
    cam_pose = orbit_pose(az, el, dist, (0.0, 0.0, 0.05))
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(50.0)), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0),
              pose=orbit_pose(az + 40.0, 55.0, 1.2, (0.0, 0.0, 0.05)))

    r = get_renderer()
    color, depth = r.render(scene)
    seg_node_map = {node_for_instance[iid]: (iid, 0, 0) for iid in node_for_instance}
    seg, _ = r.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map)
    seg_ids = seg[:, :, 0].astype(np.int32)

    gt: list[GTInstance] = []
    for iid, cat in categories.items():
        mask = seg_ids == iid
        area = int(mask.sum())
        if area < 200:
            continue
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        gt.append(GTInstance(category=cat, bbox_xywh=bbox, area_px=area))

    prompt = (f"a photo of {sd['clause']} on a wooden table, in a bright kitchen, "
              "photorealistic, natural soft light, high detail")
    return M1Render(scene_name=sd["name"], seed_idx=seed_idx, prompt=prompt,
                    color=np.asarray(color), depth_m=np.asarray(depth, np.float32),
                    gt=gt)


# ------------------------------------------------------------------ ollama helpers
def ollama_ps() -> list[dict]:
    try:
        return requests.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", [])
    except Exception:
        return []


def ollama_sweep_unload(timeout_s: float = 15.0) -> bool:
    """§10.3 sweep: /api/generate keep_alive=0; HTTP 400 (embed models) ->
    /api/embed with empty input. Polls /api/ps until EMPTY."""
    for m in ollama_ps():
        name = m.get("name") or m.get("model")
        try:
            r = requests.post(f"{OLLAMA}/api/generate",
                              json={"model": name, "keep_alive": 0}, timeout=30)
            if r.status_code == 400:
                requests.post(f"{OLLAMA}/api/embed",
                              json={"model": name, "input": "", "keep_alive": 0},
                              timeout=30)
        except Exception as e:
            print(f"[ollama] unload {name} failed: {e}")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not ollama_ps():
            print("[ollama] sweep-unload complete (/api/ps empty)")
            return True
        time.sleep(0.5)
    print(f"[ollama] WARNING: models still resident after {timeout_s}s: "
          f"{[m.get('name') for m in ollama_ps()]}")
    return False


def ollama_load_gemma(keep_alive: str = "30m") -> tuple[float, dict]:
    """Load gemma4:e4b @ num_ctx 4096 (coresident arm); returns (load_s, ps_entry)."""
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": "gemma4:e4b", "prompt": "Reply with one word: ready.",
        "options": {"num_ctx": 4096, "num_predict": 8},
        "keep_alive": keep_alive, "stream": False}, timeout=300)
    r.raise_for_status()
    dt = time.time() - t0
    entry = next((m for m in ollama_ps()
                  if (m.get("name") or "").startswith("gemma4:e4b")), {})
    return dt, entry


def gemma_300tok_seconds() -> float:
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": "gemma4:e4b",
        "prompt": "Write a detailed plan for arranging six objects on a kitchen table.",
        "options": {"num_ctx": 4096, "num_predict": 300},
        "keep_alive": "30m", "stream": False}, timeout=300)
    r.raise_for_status()
    return time.time() - t0


def device_free_gb() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / (1 << 30)


# ------------------------------------------------------------------- experiment
def generate_burst(pipe: ForgePipeline, renders: list[M1Render], mode: str,
                   tag: str, negative: str = "") -> tuple[list[str], float, dict]:
    """12 images (1/render) with depth mode `mode`. Returns (paths, s/img, vram)."""
    gen_dir = os.path.join(OUT_DIR, "gen")
    os.makedirs(gen_dir, exist_ok=True)
    pipe.reset_peak()
    paths, times = [], []
    for ridx, rd in enumerate(renders):
        control = depth_to_control(rd.depth_m, RES, mode=mode)
        t0 = time.time()
        img = pipe.generate(control, rd.prompt, negative=negative,
                            seed=42 + ridx * 1000)
        times.append(time.time() - t0)
        p = os.path.join(gen_dir, f"{tag}_{rd.scene_name}_s{rd.seed_idx}.png")
        img.save(p)
        paths.append(p)
    vram = pipe.peak_vram()
    s_img = float(np.mean(times))
    print(f"[burst {tag}] {len(paths)} imgs, {s_img:.2f} s/img "
          f"(min {min(times):.2f} max {max(times):.2f}), vram {vram}")
    return paths, s_img, vram


def report_dict(r: FidelityReport) -> dict:
    return dict(match_rate=round(r.match_rate, 4),
                mean_matched_iou=round(r.mean_matched_iou, 4),
                hallucination_rate=round(r.hallucination_rate, 4),
                n_gate_eligible=r.n_gate_eligible, n_gt_total=r.n_gt_total,
                mean_fidelity=round(float(np.mean([s.fidelity for s in r.per_image])), 4),
                n_quarantined=len(r.quarantined))


def gate_pass(r: FidelityReport) -> bool:
    return r.match_rate >= GATE_MATCH_RATE and r.mean_matched_iou >= GATE_MEAN_IOU


def make_comparison_grid(renders: list[M1Render], cols: dict[str, list[str]],
                         path: str) -> None:
    """4 scenes (seed 0) x 4 images: flat render | control | gen A | gen B."""
    tile = 384
    headers = ["flat render", "depth control (disparity)", "L0 gen (disparity)",
               "L0 gen (linear)"]
    seed0 = [rd for rd in renders if rd.seed_idx == 0]
    grid = Image.new("RGB", (tile * 4, tile * len(seed0) + 24), (20, 20, 20))
    d = ImageDraw.Draw(grid)
    for c, h in enumerate(headers):
        d.text((c * tile + 8, 6), h, fill=(255, 255, 255))
    for row, rd in enumerate(seed0):
        control = depth_to_control(rd.depth_m, RES, mode="disparity")
        imgs = [Image.fromarray(rd.color), control,
                Image.open(cols["disparity"][renders.index(rd)]),
                Image.open(cols["linear"][renders.index(rd)])]
        for c, im in enumerate(imgs):
            grid.paste(im.resize((tile, tile), Image.LANCZOS),
                       (c * tile, row * tile + 24))
    grid.save(path)
    print(f"[grid] saved {path}")


def make_contact_sheet(renders: list[M1Render], paths: list[str],
                       report: FidelityReport, out_path: str) -> None:
    """12 tiles: generated image + GT boxes (green) + OWLv2 misses noted."""
    tile = 384
    grid = Image.new("RGB", (tile * 3, tile * 4), (20, 20, 20))
    for i, (rd, p, score) in enumerate(zip(renders, paths, report.per_image)):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        for pi in score.per_instance:
            x, y, w, h = pi["bbox_xywh"]
            col = (0, 255, 80) if pi["matched"] else (255, 60, 40)
            d.rectangle([x, y, x + w, y + h], outline=col, width=4)
            d.text((x + 4, y + 4), f"{pi['category']} iou={pi['iou']:.2f}", fill=col)
        d.text((8, 8), f"{rd.scene_name}/s{rd.seed_idx} fid={score.fidelity:.2f} "
               f"halluc={score.hallucination_count}", fill=(255, 255, 0))
        r, c = divmod(i, 3)
        grid.paste(im.resize((tile, tile), Image.LANCZOS), (c * tile, r * tile))
    grid.save(out_path)
    print(f"[sheet] saved {out_path}")


def freeze_yaml(level: str, depth_mode: str, s_per_img: float, vram_mode: str) -> None:
    """Targeted value replacement in sceneforge.yaml (preserves comments)."""
    p = os.path.join(REPO_ROOT, "sceneforge.yaml")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write("gen:\n  resolution: 768\n  steps: 4\n  guidance_scale: 0.0\n"
                    "  cond_scale: 0.85\n  control_guidance_end: 0.9\n"
                    f"  level: {level}\n  depth_mode: {depth_mode}\n"
                    f"  s_per_img: {s_per_img:.2f}\n"
                    f"vram:\n  mode: {vram_mode}\n  guard_gb: 3.0\n")
        return
    text = open(p).read()
    lvl = int(level[1])
    pp = LEVEL_PARAMS[lvl]
    subs = {
        r"(?m)^(  level: )\S+": rf"\g<1>{level}",
        r"(?m)^(  depth_mode: )\S+": rf"\g<1>{depth_mode}",
        r"(?m)^(  s_per_img: )\S+": rf"\g<1>{s_per_img:.2f}",
        r"(?m)^(  mode: )\S+": rf"\g<1>{vram_mode}",
        r"(?m)^(  steps: )\S+": rf"\g<1>{pp['steps']}",
        r"(?m)^(  guidance_scale: )\S+": rf"\g<1>{pp['guidance_scale']}",
        r"(?m)^(  cond_scale: )\S+": rf"\g<1>{pp['cond_scale']}",
        r"(?m)^(  control_guidance_end: )\S+": rf"\g<1>{pp['control_guidance_end']}",
    }
    for pat, rep in subs.items():
        text = re.sub(pat, rep, text)
    open(p, "w").write(text)
    try:  # validate the frozen file still parses into AppConfig
        from sceneforge.config import load_config
        cfg = load_config(p)
        assert cfg.gen.level == level and cfg.vram.mode == vram_mode
        print(f"[yaml] frozen: level={cfg.gen.level} depth_mode={cfg.gen.depth_mode} "
              f"s_per_img={cfg.gen.s_per_img} vram.mode={cfg.vram.mode}")
    except Exception as e:
        print(f"[yaml] WARNING: post-freeze validation failed: {e}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "renders"), exist_ok=True)
    metrics: dict = {"gate_rule": "match_rate(IoU>=0.5) >= 0.70 AND mean_matched_iou "
                     ">= 0.65 over GT instances with area_px >= 1000 (§9/§13-M1)",
                     "levels": {}, "timestamps": {"start": time.time()}}

    # ---- step 0
    check_weights()
    assert_imports()
    assert_egl_768()

    # ---- step 1: 12 renders
    renders: list[M1Render] = []
    for s in range(4):
        for k in range(3):
            rd = render_m1_scene(s, k)
            renders.append(rd)
            Image.fromarray(rd.color).save(
                os.path.join(OUT_DIR, "renders", f"{rd.scene_name}_s{k}_color.png"))
            depth_to_control(rd.depth_m, RES, "disparity").save(
                os.path.join(OUT_DIR, "renders", f"{rd.scene_name}_s{k}_control.png"))
    n_gt = sum(len(r.gt) for r in renders)
    n_elig = sum(1 for r in renders for g in r.gt if g.area_px >= 1000)
    print(f"[renders] 12 renders, {n_gt} GT instances, {n_elig} gate-eligible (>=30 needed)")
    metrics["n_gt_total"], metrics["n_gate_eligible_gt"] = n_gt, n_elig
    if _RENDERER is not None:
        _RENDERER.delete()  # release EGL context before the GPU-heavy phase

    # ---- step 2: sweep ollama, load L0, A/B burst
    ollama_sweep_unload()
    metrics["free_gb_before_sdxl"] = device_free_gb()
    pipe = ForgePipeline(device="cuda", resolution=RES)
    t0 = time.time()
    pipe.load(0)
    metrics["sdxl_load_s"] = round(time.time() - t0, 1)
    print(f"[pipeline] L0 loaded in {metrics['sdxl_load_s']}s, "
          f"free now {device_free_gb():.2f} GB")
    # warmup (first call pays CUDA/cudnn autotune; excluded from s/img)
    pipe.generate(depth_to_control(renders[0].depth_m, RES, "disparity"),
                  renders[0].prompt, seed=1)

    ab_paths: dict[str, list[str]] = {}
    ab_simg: dict[str, float] = {}
    ab_vram: dict[str, dict] = {}
    for mode in ("disparity", "linear"):
        ab_paths[mode], ab_simg[mode], ab_vram[mode] = generate_burst(
            pipe, renders, mode, tag=f"L0_{mode}")

    # ---- step 3: OWLv2 A/B scoring
    scorer = Owlv2Scorer(device="cuda")
    layouts = [r.gt for r in renders]
    ab_reports = {m: scorer.score_batch(ab_paths[m], layouts)
                  for m in ("disparity", "linear")}
    for m, rep in ab_reports.items():
        print(f"[A/B {m}] {report_dict(rep)}")
    # A/B rule: gate metrics decide; when tied within noise (rounded to 0.02),
    # LOWER hallucination_rate wins — duplicated objects poison the export (§9).
    def ab_key(rep: FidelityReport) -> tuple:
        return (round(rep.match_rate / 0.02) * 0.02,
                round(rep.mean_matched_iou / 0.02) * 0.02,
                -rep.hallucination_rate)

    winner = max(ab_reports, key=lambda m: ab_key(ab_reports[m]))
    metrics["depth_ab"] = {m: report_dict(r) for m, r in ab_reports.items()}
    metrics["depth_ab"]["winner"] = winner
    metrics["depth_ab"]["rule"] = ("match_rate, mean_matched_iou (each rounded to "
                                   "0.02), then lower hallucination_rate")
    print(f"[A/B] winner: {winner}")

    # ---- step 4: gate at L0, ladder walk on failure
    level_names = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
    chosen_level, chosen_report = "L0", ab_reports[winner]
    chosen_paths, chosen_simg = ab_paths[winner], ab_simg[winner]
    metrics["levels"]["L0"] = {**report_dict(ab_reports[winner]),
                               "s_per_img": round(ab_simg[winner], 2),
                               "vram": {k: round(v, 2) for k, v in ab_vram[winner].items()},
                               "gate_pass": gate_pass(ab_reports[winner])}
    if not gate_pass(ab_reports[winner]):
        for lvl in (1, 2, 3):
            name = level_names[lvl]
            print(f"[ladder] L{lvl - 1} failed gate -> walking to {name}")
            pipe.load(lvl)
            neg = NEGATIVE_PROMPT if LEVEL_PARAMS[lvl]["guidance_scale"] > 1 else ""
            paths, s_img, vram = generate_burst(pipe, renders, winner,
                                                tag=name, negative=neg)
            rep = scorer.score_batch(paths, layouts)
            ok = gate_pass(rep)
            metrics["levels"][name] = {**report_dict(rep), "s_per_img": round(s_img, 2),
                                       "vram": {k: round(v, 2) for k, v in vram.items()},
                                       "gate_pass": ok}
            print(f"[ladder {name}] {report_dict(rep)} gate_pass={ok}")
            if ok:
                chosen_level, chosen_report = name, rep
                chosen_paths, chosen_simg = paths, s_img
                break
        else:
            chosen_level, chosen_report = "L3", rep
            chosen_paths, chosen_simg = paths, s_img

    final_pass = gate_pass(chosen_report)
    metrics["chosen_level"] = chosen_level
    metrics["gate_pass"] = final_pass

    # ---- artifacts: grid + contact sheet
    make_comparison_grid(renders, ab_paths,
                         os.path.join(OUT_DIR, "comparison_grid.png"))
    make_contact_sheet(renders, chosen_paths, chosen_report,
                       os.path.join(OUT_DIR, "contact_sheet.png"))
    scorer.unload()  # keep the VRAM experiment clean

    # ---- step 5: VRAM mode pick (judge mandate #2)
    vram_exp: dict = {}
    ollama_sweep_unload()
    pipe.reset_peak()
    for i in range(3):
        pipe.generate(depth_to_control(renders[i].depth_m, RES, winner),
                      renders[i].prompt, seed=9000 + i)
    seq = pipe.peak_vram()
    vram_exp["sequential"] = {k: round(v, 2) for k, v in seq.items()}
    free_before = device_free_gb()
    torch.cuda.empty_cache()
    vram_exp["empty_cache_freed_gb"] = round(device_free_gb() - free_before, 2)

    load_s, entry = ollama_load_gemma(keep_alive="30m")
    fully_on_gpu = bool(entry) and entry.get("size_vram", 0) == entry.get("size", -1)
    vram_exp["gemma_load_s"] = round(load_s, 1)
    vram_exp["gemma_size_gb"] = round(entry.get("size", 0) / (1 << 30), 2)
    vram_exp["gemma_size_vram_gb"] = round(entry.get("size_vram", 0) / (1 << 30), 2)
    vram_exp["gemma_fully_on_gpu"] = fully_on_gpu
    vram_exp["gemma_300tok_s"] = round(gemma_300tok_seconds(), 1)
    pipe.reset_peak()
    for i in range(3):
        pipe.generate(depth_to_control(renders[i].depth_m, RES, winner),
                      renders[i].prompt, seed=9100 + i)
    cor = pipe.peak_vram()
    vram_exp["coresident"] = {k: round(v, 2) for k, v in cor.items()}
    entry_after = next((m for m in ollama_ps()
                        if (m.get("name") or "").startswith("gemma4:e4b")), {})
    vram_exp["gemma_resident_after_burst"] = bool(entry_after)
    vram_exp["gemma_size_vram_gb_after"] = round(
        entry_after.get("size_vram", 0) / (1 << 30), 2)
    ollama_sweep_unload()

    vram_mode = ("coresident"
                 if cor["min_free_gb"] >= GUARD_GB and fully_on_gpu
                 and bool(entry_after) else "sequential")
    vram_exp["rule"] = (f"coresident iff min-free-throughout >= guard_gb ({GUARD_GB}) "
                        "with gemma4:e4b@4096 fully on GPU and still resident post-burst")
    vram_exp["picked"] = vram_mode
    metrics["vram"] = vram_exp
    metrics["quality_mode_timing"] = "skipped (cut-order #3, §12.7)"
    print(f"[vram] sequential min_free={seq['min_free_gb']:.2f} GB | coresident "
          f"min_free={cor['min_free_gb']:.2f} GB fully_on_gpu={fully_on_gpu} "
          f"-> mode={vram_mode}")

    # ---- step 6: freeze + report
    freeze_yaml(chosen_level, winner, chosen_simg, vram_mode)
    metrics["chosen"] = dict(level=chosen_level, depth_mode=winner,
                             s_per_img=round(chosen_simg, 2), resolution=RES,
                             vram_mode=vram_mode)
    metrics["timestamps"]["end"] = time.time()
    metrics["timestamps"]["total_s"] = round(
        metrics["timestamps"]["end"] - metrics["timestamps"]["start"], 1)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] {os.path.join(OUT_DIR, 'metrics.json')}")

    final = chosen_report
    print("=" * 70)
    print(f"M1 RESULT: {'GO' if final_pass else 'NO-GO'} at {chosen_level} "
          f"({winner} depth) — match_rate={final.match_rate:.3f} "
          f"(gate >=0.70), mean_matched_iou={final.mean_matched_iou:.3f} "
          f"(gate >=0.65), hallucination_rate={final.hallucination_rate:.3f} "
          f"(informational), {final.n_gate_eligible} gate-eligible GT")
    print("=" * 70)
    pipe.unload()
    return 0 if final_pass else 1


if __name__ == "__main__":
    sys.exit(main())
