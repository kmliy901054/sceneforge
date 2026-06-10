"""tests/test_ip_adapter.py — IP-Adapter style-reference feature (CPU, mocked).

Covers the full plumbing with NO weights and NO GPU work, mirroring the
test_gpu.py / test_orchestrator.py mock patterns:

  - ForgePipeline.enable_style_reference()/disable_style_reference() drive
    diffusers' load_ip_adapter()/set_ip_adapter_scale()/unload_ip_adapter()
    with the exact h94/IP-Adapter sdxl vit-h arguments (adapter weights in
    sdxl_models/, laion CLIP-ViT-H encoder in models/image_encoder — NOT the
    bigG one in sdxl_models/image_encoder);
  - generate() passes ip_adapter_image ONLY while a reference is enabled;
  - load-order contract: enable before load() raises; unload() keeps the
    reference but forgets the (dead) adapter so a rebuild re-installs it;
  - cfg.gen.ip_adapter_scale: default, yaml and env override;
  - ForgeRun.run(style_ref=...) enables before the first generate, disables
    after the last one, and records provenance (style_ref.png + run_meta);
  - ui.handlers.on_forge passes the accordion image/scale through to run();
  - UI smoke: build_app() constructs with the new components.
"""
import sceneforge.compat  # noqa: F401  — first import (§0)

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sceneforge.config import load_config
from sceneforge.diffusion import pipeline as pipeline_mod
from sceneforge.diffusion.pipeline import ForgePipeline

REF = Image.new("RGB", (64, 64), (200, 120, 30))
CONTROL = Image.new("RGB", (768, 768))


def make_pipeline() -> tuple[ForgePipeline, MagicMock, list[dict]]:
    """ForgePipeline with a MagicMock pipe (test_gpu.py pattern) recording
    every __call__'s kwargs."""
    fp = ForgePipeline(device="cpu", resolution=768)
    fp.level = 0  # pretend load(0) happened; no weights involved
    call_kwargs: list[dict] = []
    pipe = MagicMock(name="pipe")

    def fake_call(*args, **kwargs):
        call_kwargs.append(kwargs)
        out = MagicMock()
        out.images = [Image.new("RGB", (8, 8))]
        return out

    pipe.side_effect = fake_call
    fp.pipe = pipe
    return fp, pipe, call_kwargs


# ---------------------------------------------------------- pipeline plumbing
def test_enable_loads_adapter_with_vit_h_paths():
    fp, pipe, _ = make_pipeline()
    fp.enable_style_reference(REF, scale=0.6)

    pipe.load_ip_adapter.assert_called_once_with(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale.assert_called_once_with(0.6)
    assert fp.style_reference_enabled
    assert fp.style_ref is not None and fp.style_scale == 0.6


def test_reenable_only_swaps_scale_not_weights():
    fp, pipe, _ = make_pipeline()
    fp.enable_style_reference(REF, scale=0.6)
    other = Image.new("RGB", (32, 32), (5, 5, 250))
    fp.enable_style_reference(other, scale=0.9)

    pipe.load_ip_adapter.assert_called_once()  # weights load once per build
    assert [c.args[0] for c in pipe.set_ip_adapter_scale.call_args_list] == [0.6, 0.9]
    assert fp.style_scale == 0.9


def test_enable_before_load_raises():
    fp = ForgePipeline(device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        fp.enable_style_reference(REF)


def test_generate_passes_ip_adapter_image_only_when_enabled():
    fp, _pipe, call_kwargs = make_pipeline()

    fp.generate(CONTROL, "prompt", seed=1)
    assert "ip_adapter_image" not in call_kwargs[0]  # un-adapted pipe: no kwarg

    fp.enable_style_reference(REF, scale=0.5)
    fp.generate(CONTROL, "prompt", seed=1)
    assert call_kwargs[1]["ip_adapter_image"] is fp.style_ref
    # the §7.4 L0 params are untouched by the style reference
    assert call_kwargs[1]["num_inference_steps"] == 4
    assert call_kwargs[1]["guidance_scale"] == 0.0

    fp.disable_style_reference()
    fp.generate(CONTROL, "prompt", seed=1)
    assert "ip_adapter_image" not in call_kwargs[2]


def test_disable_unloads_and_is_idempotent():
    fp, pipe, _ = make_pipeline()
    fp.disable_style_reference()                 # nothing enabled — no-op
    pipe.unload_ip_adapter.assert_not_called()

    fp.enable_style_reference(REF, scale=0.6)
    fp.disable_style_reference()
    pipe.unload_ip_adapter.assert_called_once()  # attn processors restored
    assert fp.style_ref is None and not fp.style_reference_enabled

    fp.disable_style_reference()                 # idempotent
    pipe.unload_ip_adapter.assert_called_once()


def test_unload_keeps_reference_but_forgets_dead_adapter():
    """§7.5 unload kills the pipe (and the adapter inside its UNet) — the
    style reference must survive so the next load() re-installs it."""
    fp, _pipe, _ = make_pipeline()
    fp.enable_style_reference(REF, scale=0.7)
    fp.unload()
    assert fp.style_ref is not None and fp.style_scale == 0.7
    assert fp._ip_loaded is False

    # simulate the load() rebuild path: fresh pipe -> enable re-loads weights
    pipe2 = MagicMock(name="pipe2")
    fp.pipe, fp.level = pipe2, 0
    fp.enable_style_reference(fp.style_ref, fp.style_scale)
    pipe2.load_ip_adapter.assert_called_once()
    pipe2.set_ip_adapter_scale.assert_called_once_with(0.7)


def test_module_constants_match_hub_layout():
    assert pipeline_mod.IP_ADAPTER_REPO == "h94/IP-Adapter"
    assert pipeline_mod.IP_ADAPTER_SUBFOLDER == "sdxl_models"
    assert pipeline_mod.IP_ADAPTER_WEIGHT == "ip-adapter_sdxl_vit-h.safetensors"
    # vit-h pairs with the ViT-H encoder; "/" makes diffusers treat the path
    # as repo-root-relative (diffusers 0.38 loaders/ip_adapter.py)
    assert pipeline_mod.IP_ADAPTER_ENCODER_FOLDER == "models/image_encoder"


# ------------------------------------------------------------------- config
def test_config_flag_default_yaml_and_env():
    assert load_config(path="/nonexistent.yaml").gen.ip_adapter_scale == 0.6
    assert load_config().gen.ip_adapter_scale == 0.6  # sceneforge.yaml entry
    cfg = load_config(environ={"SCENEFORGE_GEN_IP_ADAPTER_SCALE": "0.25"})
    assert cfg.gen.ip_adapter_scale == 0.25


# ----------------------------------------------------- orchestrator plumbing
import test_orchestrator as to  # noqa: E402  — reuse the §4.8 fakes/fixtures


class StyleAwarePipeline(to.FakePipeline):
    """FakePipeline + style-reference recording (events interleaved with
    generate() calls so enable-before/disable-after is provable)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple] = []

    def enable_style_reference(self, image, scale=0.6):
        self.events.append(("enable", image, scale))

    def disable_style_reference(self):
        self.events.append(("disable",))

    def generate(self, *a, **k):
        self.events.append(("generate",))
        return super().generate(*a, **k)


@pytest.fixture()
def style_runner(tmp_path, monkeypatch, director_calls):
    from sceneforge import gpu, orchestrator
    from sceneforge.assets.library import AssetLibrary
    from sceneforge.director.rag import EmbeddingIndex

    monkeypatch.setattr(gpu, "ollama_unload", lambda *a, **k: True)
    monkeypatch.setattr(gpu, "snapshot", lambda *a, **k: {
        "free_gb": 0.0, "total_gb": 0.0, "torch_alloc_gb": 0.0,
        "torch_reserved_gb": 0.0, "ollama_models": []})
    cfg = load_config()
    cfg.gen.resolution = to.RES
    cfg.paths.runs_dir = tmp_path / "runs"
    library = AssetLibrary(cards_path=cfg.paths.cards_file)
    return orchestrator.ForgeRun(
        cfg, library=library, rag=EmbeddingIndex(library.cards()),
        renderer=to.FakeRenderer(), pipeline=StyleAwarePipeline(),
        scorer=to.FakeScorer())


# reuse test_orchestrator's director mock (module fixture, importable by name)
director_calls = to.director_calls


def test_run_with_style_ref_enables_then_disables(style_runner):
    events = list(style_runner.run(to.TASK, n_layouts=2, n_styles=4, seed=42,
                                   style_ref=REF))
    assert events[-1].kind == "done"
    ev = style_runner._pipeline.events
    assert ev[0][0] == "enable" and ev[-1] == ("disable",)
    assert [e[0] for e in ev[1:-1]] == ["generate"] * 8  # 2 layouts × 4 styles
    _, image, scale = ev[0]
    assert image is REF
    assert scale == style_runner.cfg.gen.ip_adapter_scale == 0.6  # cfg default

    # provenance: reference saved + recorded in run_meta.json and state (§4.7)
    done = events[-1].payload
    run_dir = Path(done["run_dir"])
    assert (run_dir / "style_ref.png").is_file()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["style_ref"]["scale"] == 0.6
    assert meta["style_ref"]["path"] == str(run_dir / "style_ref.png")
    assert done["state"]["style_ref"]["scale"] == 0.6
    json.dumps(done)  # payloads stay JSON-able (§4.8)
    # the status stream announces the reference (UI is never silent, §11)
    assert any(e.kind == "status" and "style reference" in e.payload.get("stage", "")
               for e in events)


def test_run_with_explicit_scale_overrides_config(style_runner):
    events = list(style_runner.run(to.TASK, seed=1, style_ref=REF,
                                   style_ref_scale=0.9))
    assert events[-1].kind == "done"
    assert style_runner._pipeline.events[0] == ("enable", REF, 0.9)


def test_run_without_style_ref_never_touches_adapter(style_runner):
    events = list(style_runner.run(to.TASK, seed=7))
    assert events[-1].kind == "done"
    assert all(e[0] == "generate" for e in style_runner._pipeline.events)
    meta = json.loads((Path(events[-1].payload["run_dir"]) / "run_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["style_ref"] is None
    assert "style_ref" not in events[-1].payload["state"]


# --------------------------------------------------------------- UI handlers
def test_on_forge_passes_style_ref_through(monkeypatch):
    from sceneforge.orchestrator import ForgeEvent
    from sceneforge.ui import handlers

    seen: dict = {}

    class StubRunner:
        def run(self, task, n_layouts, n_styles, seed, **kwargs):
            seen.update(kwargs, task=task)
            yield ForgeEvent("done", {
                "run_id": "r", "run_dir": "/tmp/r", "timings": {},
                "state": {"layouts": [], "styles": []}})

    monkeypatch.setattr(handlers, "get_runner", lambda: StubRunner())

    list(handlers.on_forge("task", 2, 4, 42, "boxes", REF, 0.7))
    assert seen["style_ref"] is REF and seen["style_ref_scale"] == 0.7

    seen.clear()
    list(handlers.on_forge("task", 2, 4, 42, "boxes"))  # accordion untouched
    assert seen["style_ref"] is None and seen["style_ref_scale"] is None


# ----------------------------------------------------------------- UI smoke
def test_build_app_has_style_reference_components():
    import gradio as gr

    from sceneforge.ui.blocks import build_app

    demo = build_app()
    comps = list(demo.blocks.values())
    images = [c for c in comps if isinstance(c, gr.Image)
              and getattr(c, "label", "") == "style reference (optional)"]
    assert len(images) == 1, "style reference gr.Image missing"
    assert images[0].type == "pil"  # handlers expect a PIL image
    sliders = [c for c in comps if isinstance(c, gr.Slider)
               and "IP-Adapter" in (getattr(c, "label", "") or "")]
    assert len(sliders) == 1, "IP-Adapter scale slider missing"
    assert sliders[0].value == 0.6  # cfg.gen.ip_adapter_scale default
