"""OOM-recovery state transitions for ForgePipeline (ARCHITECTURE.md §7.5).

Uses a mocked ``torch.cuda.OutOfMemoryError`` — no real OOM, no model loads,
no GPU allocation. Verifies the EXACT recovery sequence:
  pipe.to("cpu") -> gc.collect() -> torch.cuda.empty_cache() ->
  pipe.enable_model_cpu_offload() -> retry ONCE -> still OOM -> 640 px,
plus the latched ``force_sequential`` side effect.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

import sceneforge.compat  # noqa: F401  (import-first rule, §0)
from sceneforge.diffusion.pipeline import ForgePipeline

SENTINEL = Image.new("RGB", (8, 8), (1, 2, 3))
CONTROL = Image.new("RGB", (768, 768))


def make_pipeline(fail_times: int) -> tuple[ForgePipeline, MagicMock, dict]:
    """ForgePipeline with a mock pipe whose first `fail_times` calls OOM."""
    fp = ForgePipeline(device="cuda", resolution=768)
    fp.level = 0  # pretend load(0) happened; no weights involved
    calls = {"n": 0, "kwargs": []}
    pipe = MagicMock(name="pipe")

    def fake_call(*args, **kwargs):
        calls["n"] += 1
        calls["kwargs"].append(kwargs)
        if calls["n"] <= fail_times:
            raise torch.cuda.OutOfMemoryError("mock OOM")
        out = MagicMock()
        out.images = [SENTINEL]
        return out

    pipe.side_effect = fake_call
    fp.pipe = pipe
    return fp, pipe, calls


def test_happy_path_no_recovery():
    fp, pipe, calls = make_pipeline(fail_times=0)
    img = fp.generate(CONTROL, "a photo of a box on a table", seed=7)
    assert img is SENTINEL
    assert calls["n"] == 1
    assert fp.offloaded is False
    assert fp.force_sequential is False
    assert fp.resolution == 768
    pipe.to.assert_not_called()
    pipe.enable_model_cpu_offload.assert_not_called()
    # L0 defaults flowed through (§7.4)
    kw = calls["kwargs"][0]
    assert kw["num_inference_steps"] == 4
    assert kw["guidance_scale"] == 0.0
    assert kw["controlnet_conditioning_scale"] == 0.85
    assert kw["control_guidance_end"] == 0.9
    assert kw["width"] == kw["height"] == 768


def test_single_oom_recovers_via_cpu_offload():
    fp, pipe, calls = make_pipeline(fail_times=1)
    img = fp.generate(CONTROL, "prompt", seed=0)
    assert img is SENTINEL
    assert calls["n"] == 2                      # retry exactly ONCE
    assert fp.offloaded is True                 # V3 reached
    assert fp.force_sequential is True          # §7.5 side effect latched
    assert fp.resolution == 768                 # no 640 drop on first recovery
    pipe.to.assert_called_once_with("cpu")
    pipe.enable_model_cpu_offload.assert_called_once_with()
    # ORDER: failing call -> to("cpu") -> enable_model_cpu_offload -> retry
    names = [c[0] for c in pipe.mock_calls]
    assert names.index("to") < names.index("enable_model_cpu_offload")


def test_double_oom_drops_to_640():
    fp, pipe, calls = make_pipeline(fail_times=2)
    img = fp.generate(CONTROL, "prompt", seed=0)
    assert img is SENTINEL
    assert calls["n"] == 3
    assert fp.offloaded is True
    assert fp.force_sequential is True
    assert fp.resolution == 640                 # V4: remainder of burst at 640
    pipe.enable_model_cpu_offload.assert_called_once_with()  # offload only once
    assert calls["kwargs"][2]["width"] == calls["kwargs"][2]["height"] == 640
    # ...and the NEXT generate stays at 640 (remainder of the burst)
    fp.generate(CONTROL, "prompt", seed=1)
    assert calls["kwargs"][3]["width"] == 640


def test_third_oom_propagates():
    fp, _pipe, calls = make_pipeline(fail_times=3)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        fp.generate(CONTROL, "prompt", seed=0)
    assert calls["n"] == 3                      # no infinite retry loop


def test_generate_before_load_raises():
    fp = ForgePipeline()
    with pytest.raises(RuntimeError):
        fp.generate(CONTROL, "prompt")


def test_peak_vram_reports_all_three_numbers():
    fp, _pipe, _calls = make_pipeline(fail_times=0)
    fp.generate(CONTROL, "prompt", seed=0)
    stats = fp.peak_vram()
    assert set(stats) == {"max_allocated_gb", "max_reserved_gb", "min_free_gb"}
    assert all(isinstance(v, float) for v in stats.values())
