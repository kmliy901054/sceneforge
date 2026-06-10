"""tests/test_orchestrator.py — ForgeRun contract tests (ARCHITECTURE.md §4.8).

Event-order contract with mocked director/pipeline/scorer/renderer (no GPU, no
Ollama, no EGL):

    plan_token* → spec → layout(0) → status(stage="styles") → layout(1..) →
    image* (PINNED layout-major) → fidelity → done

plus: re-forge makes ZERO LLM calls (mocked director raises if called),
run_meta.json + §4.7 artifacts are written, overlay PNGs exist for ALL modes,
payloads are JSON-able, and the COCO export round-trips through pycocotools.
"""
import sceneforge.compat  # noqa: F401  — first import (§0)

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sceneforge import gpu, orchestrator
from sceneforge.assets.library import AssetLibrary
from sceneforge.config import load_config
from sceneforge.director.fallback import default_style_set, template_plan
from sceneforge.director.rag import EmbeddingIndex
from sceneforge.eval.fidelity import FidelityReport, ImageScore
from sceneforge.labels import masks as masks_mod
from sceneforge.render.types import RenderResult

RES = 256  # min cfg.gen.resolution — keeps fake arrays tiny
TASK = "pick the red mug from a cluttered kitchen table"


# ------------------------------------------------------------------- fakes
class FakeRenderer:
    """CPU stand-in for the EGL backend: every instance gets a 60×60 block
    (area 3600 ≥ the 1000 px target-visibility gate)."""

    def __init__(self) -> None:
        self.calls = 0

    def render_scene(self, composed, camera, width, height, *, spec=None):
        self.calls += 1
        seg = np.zeros((height, width), np.int32)
        for idx, (instance_id, _mesh, _T) in enumerate(composed.instances):
            row, col = divmod(idx, 3)
            y, x = 10 + row * 70, 5 + col * 70
            seg[y:y + 60, x:x + 60] = instance_id
        depth = np.full((height, width), 1.5, np.float32)
        color = np.zeros((height, width, 3), np.uint8)
        instances = (masks_mod.extract_instances(seg, spec)
                     if spec is not None else [])
        return RenderResult(
            width=width, height=height, color=color, depth_m=depth,
            seg_ids=seg, instances=instances,
            camera_pose=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
        )


class FakePipeline:
    def __init__(self) -> None:
        self.pipe = object()  # "already loaded" — ensure_pipeline must not rebuild
        self.level = 0
        self.resolution = RES
        self.seeds: list[int] = []

    def load(self, level: int) -> None:
        self.level = level
        self.pipe = object()

    def generate(self, control, prompt, negative="", seed=0,
                 cond_scale=None, steps=None):
        self.seeds.append(int(seed))
        return Image.new("RGB", (RES, RES), (seed % 251, 40, 80))

    def peak_vram(self):
        return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0, "min_free_gb": 0.0}


class FakeScorer:
    device = "cpu"
    model = None

    def score_batch(self, images, layouts):
        per = [ImageScore(path=str(p), fidelity=0.8, fidelity_adj=0.75,
                          hallucination_count=0) for p in images]
        return FidelityReport(
            per_image=per, match_rate=0.9, mean_matched_iou=0.7,
            hallucination_rate=0.0, kept=[s.path for s in per], quarantined=[],
            n_gate_eligible=len(per), n_gt_total=len(per),
        )


# ----------------------------------------------------------------- fixtures
@pytest.fixture()
def director_calls(monkeypatch):
    """Mock the director module functions; count calls (reforge must make 0)."""
    calls = {"plan": 0, "styles": 0}

    def fake_plan(task, seed, cfg, *, on_token=None, log=None):
        calls["plan"] += 1
        if on_token is not None:  # exercise the plan_token streaming path
            for piece in ('{"scene_summary": ', '"mock plan"}'):
                on_token(piece)
        if log is not None:
            log.meta["plan_source"] = "llm"
            log.add("mock plan_scene")
        return template_plan(task, seed)

    def fake_styles(task, plan, n_styles, cfg, *, seed=42, on_token=None, log=None):
        calls["styles"] += 1
        if log is not None:
            log.meta["style_source"] = "llm"
        return default_style_set(n_styles)

    monkeypatch.setattr(orchestrator, "plan_scene", fake_plan)
    monkeypatch.setattr(orchestrator, "make_styles", fake_styles)
    return calls


@pytest.fixture()
def runner(tmp_path, monkeypatch, director_calls):
    # phase barriers must not touch Ollama/CUDA in unit tests (§10.3)
    monkeypatch.setattr(gpu, "ollama_unload", lambda *a, **k: True)
    monkeypatch.setattr(gpu, "snapshot", lambda *a, **k: {
        "free_gb": 0.0, "total_gb": 0.0, "torch_alloc_gb": 0.0,
        "torch_reserved_gb": 0.0, "ollama_models": []})

    cfg = load_config()
    cfg.gen.resolution = RES
    cfg.paths.runs_dir = tmp_path / "runs"

    library = AssetLibrary(cards_path=cfg.paths.cards_file)
    rag = EmbeddingIndex(library.cards())  # difflib-only: no Ollama, no cache
    return orchestrator.ForgeRun(
        cfg, library=library, rag=rag, renderer=FakeRenderer(),
        pipeline=FakePipeline(), scorer=FakeScorer(),
    )


def _forge(runner, n_layouts=2, n_styles=4, seed=42):
    events = list(runner.run(TASK, n_layouts=n_layouts, n_styles=n_styles, seed=seed))
    assert events and events[-1].kind == "done", \
        f"run did not finish: kinds={[e.kind for e in events]}"
    return events


# -------------------------------------------------------------------- tests
def test_event_order_contract(runner, director_calls):
    events = _forge(runner)
    assert not any(e.kind == "error" for e in events)

    def first(pred):
        return next(i for i, e in enumerate(events) if pred(e))

    i_spec = first(lambda e: e.kind == "spec")
    i_layout0 = first(lambda e: e.kind == "layout" and e.payload["layout_idx"] == 0)
    i_styles = first(lambda e: e.kind == "status"
                     and e.payload.get("stage") == "styles")
    i_layout1 = first(lambda e: e.kind == "layout" and e.payload["layout_idx"] == 1)
    i_image = first(lambda e: e.kind == "image")
    i_fid = first(lambda e: e.kind == "fidelity")
    i_done = first(lambda e: e.kind == "done")

    # §1 demo-paced order: spec → layout 0 (GLB+control BEFORE the style call)
    # → styles → remaining layouts → images → fidelity → done.
    assert i_spec < i_layout0 < i_styles < i_layout1 < i_image < i_fid < i_done
    assert i_done == len(events) - 1

    # planner tokens streamed before the spec event
    tokens = [i for i, e in enumerate(events) if e.kind == "plan_token"]
    assert tokens and max(tokens) < i_spec
    assert director_calls == {"plan": 1, "styles": 1}

    # layout 0 event carries the first-visual artifacts (§1)
    p0 = events[i_layout0].payload
    assert Path(p0["glb"]).is_file() and Path(p0["control"]).is_file()


def test_images_layout_major_and_seed_law(runner):
    events = _forge(runner)
    imgs = [e.payload for e in events if e.kind == "image"]
    assert len(imgs) == 2 * 4
    # PINNED layout-major (§4.8): outer layouts, inner styles
    assert [(p["layout_idx"], p["style_idx"]) for p in imgs] == [
        (layout, style) for layout in range(2) for style in range(4)]
    # §7.2 seed law: seed + layout_idx*1000 + style_idx
    assert all(p["seed"] == 42 + p["layout_idx"] * 1000 + p["style_idx"]
               for p in imgs)
    # overlay PNGs for ALL modes written at generation time (§8.2)
    for p in imgs:
        for mode in ("off", "boxes", "masks", "both"):
            assert Path(p["overlays"][mode]).is_file(), (p["style"], mode)


def test_payloads_json_able_and_artifacts(runner):
    events = _forge(runner)
    for event in events:  # ForgeEvent payloads: JSON-able paths/scalars only
        json.dumps(event.payload)

    done = events[-1].payload
    run_dir = Path(done["run_dir"])
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == TASK
    assert meta["timings"]["total_s"] > 0
    assert meta["vram_log"], "phase barriers must log VRAM snapshots (§10.3)"
    assert meta["style_clip_tokens"], "per-style CLIP token counts (§4.7)"
    assert (run_dir / "director_log.md").is_file()
    assert (run_dir / "fidelity.json").is_file()

    for k in (0, 1):  # §4.7 per-layout artifacts
        d = run_dir / f"layout_{k}"
        for name in ("spec.json", "viewer.glb", "depth16.png", "control.png",
                     "seg_ids.png", "labels.json"):
            assert (d / name).is_file(), f"layout_{k}/{name} missing"

    fid = [e for e in events if e.kind == "fidelity"][0].payload
    assert 0.0 <= fid["fidelity_adj_mean"] <= 1.0
    assert done["state"]["styles"] and len(done["state"]["layouts"]) == 2


def test_reforge_makes_no_llm_calls(runner, monkeypatch):
    events = _forge(runner)
    state = events[-1].payload["state"]
    n_gen_before = len(runner._pipeline.seeds)

    def _boom(*a, **k):  # any director call during re-forge is a contract breach
        raise AssertionError("re-forge must not call the LLM director (§4.8)")

    monkeypatch.setattr(orchestrator, "plan_scene", _boom)
    monkeypatch.setattr(orchestrator, "make_styles", _boom)

    glb, image, overlays, annotated = runner.reforge(
        state,
        camera={"azimuth_deg": 50.0, "elevation_deg": 45.0, "distance_m": 1.3},
        seed=123,
        style_name=state["styles"][1],
    )
    assert Path(glb).is_file() and Path(image).is_file()
    for mode in ("off", "boxes", "masks", "both"):
        assert Path(overlays[mode]).is_file()
    # exactly ONE diffusion call, with the explicit seed
    assert runner._pipeline.seeds[n_gen_before:] == [123]
    # AnnotatedImage value: (image_path, [(mask, label), ...]) (§11)
    img_path, annotations = annotated
    assert img_path == image and annotations
    mask, label = annotations[0]
    assert mask.shape == (RES, RES) and isinstance(label, str)
    # camera change persisted into the re-render (labels track the camera)
    spec = json.loads(Path(state["layouts"][0]["spec_json"]).read_text())
    assert spec["camera"]["azimuth_deg"] != 50.0  # original spec untouched


def test_reforge_nudge_leaves_layout_spec_untouched(runner):
    """Nudge mutates a COPY of the layout spec (re-resolved placements land in
    layout_<k>/reforge/); the on-disk layout spec.json must stay intact."""
    events = _forge(runner)
    state = events[-1].payload["state"]
    spec_before = Path(state["layouts"][0]["spec_json"]).read_text()

    glb, image, _overlays, _annotated = runner.reforge(state, nudge="+x")
    assert "/reforge/" in image and Path(image).is_file()
    assert Path(state["layouts"][0]["spec_json"]).read_text() == spec_before


def test_export_coco_roundtrip(runner, tmp_path):
    events = _forge(runner)
    run_dir = Path(events[-1].payload["run_dir"])
    zip_path = orchestrator.export_coco(run_dir)
    assert zip_path.is_file() and zip_path.suffix == ".zip"

    ann_file = run_dir / "coco" / "annotations.json"
    assert ann_file.is_file()
    from pycocotools.coco import COCO

    coco = COCO(str(ann_file))
    assert len(coco.imgs) == 8
    ann_ids = list(coco.anns)
    assert len(set(ann_ids)) == len(ann_ids) and len(ann_ids) > 0
    for ann in coco.anns.values():  # every RLE decodes back to its recorded area
        mask = coco.annToMask(ann)
        assert int(mask.sum()) == int(ann["area"])
    for img in coco.imgs.values():
        assert img["width"] == RES and img["height"] == RES
        assert (run_dir / "coco" / "images" / img["file_name"]).is_file()
