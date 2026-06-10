"""tests/test_augment.py — feature A acceptance, CPU-only with tiny synthetic data.

Covers: the mask/composite invariant (near pixels bitwise identical to the
input — the probe's verified property), percentile-threshold semantics, the
5-frame temporal majority vote, parameter validation, aspect_size rounding,
and the provenance/audit artifacts. The depth model and SDXL pipeline are both
injected fakes — no GPU, no downloads.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (repo rule, §0)

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from sceneforge.augment import restyle
from sceneforge.config import AppConfig
from sceneforge.diffusion.pipeline import aspect_size

H, W = 48, 64
N_FRAMES = 5


# --------------------------------------------------------------------- fakes
class FakePipeline:
    """ForgePipeline stand-in: returns a solid magenta image at the requested
    size and records (prompt, seed, size) per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple]] = []

    def generate(self, control, prompt, negative="", seed=0,
                 cond_scale=None, steps=None, size=None):
        self.calls.append((prompt, int(seed), tuple(size) if size else None))
        w, h = size if size is not None else (64, 64)
        assert control.size == (w, h)  # control arrives at generation size
        return Image.new("RGB", (w, h), (255, 0, 255))


def _frames(rng: np.random.Generator, n: int = N_FRAMES):
    return [rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8) for _ in range(n)]


def _depth_fn(frames):
    """Vertical gradient: bottom rows near (large = near, Depth-Anything polarity)."""
    col = np.linspace(0.0, 10.0, H, dtype=np.float32)[:, None]
    return [np.repeat(col, W, axis=1) for _ in frames]


def _run(tmp_path: Path, rng=None, **kw):
    rng = rng or np.random.default_rng(0)
    frames = _frames(rng)
    in_dir = tmp_path / "episode"
    in_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(in_dir / f"f{i:03d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    pipe = FakePipeline()
    kw.setdefault("style_prompts", ["an industrial workshop, photo",
                                    "a bright white laboratory, photo"])
    kw.setdefault("n_styles", 2)
    prov = restyle.restyle_frames(
        in_dir, tmp_path / "out", seed=7, cfg=AppConfig(),
        pipeline=pipe, depth_fn=_depth_fn, **kw)
    return frames, pipe, prov, tmp_path / "out"


# --------------------------------------------- THE invariant: exact composite
class TestCompositeInvariant:
    def test_near_pixels_bitwise_identical(self, tmp_path):
        frames, _pipe, prov, out = _run(tmp_path)
        for i, frame in enumerate(frames):
            keep = cv2.imread(str(out / "masks" / f"f{i:03d}_keep.png"),
                              cv2.IMREAD_GRAYSCALE) > 0
            assert keep.shape == (H, W) and keep.any() and (~keep).any()
            for style in ("style_1", "style_2"):
                png = out / "frames" / style / f"f{i:03d}.png"
                assert png.is_file() and png.stat().st_size > 0
                comp = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
                # bitwise identical inside the keep mask...
                assert np.array_equal(comp[keep], frame[keep])
                # ...and actually restyled outside it (fake gen = magenta)
                assert not np.array_equal(comp[~keep], frame[~keep])
        assert prov["near_pixels_bitwise_identical"] is True

    def test_fixed_seed_and_prompt_per_style_across_frames(self, tmp_path):
        _frames_, pipe, prov, _out = _run(tmp_path)
        assert len(pipe.calls) == N_FRAMES * 2
        by_prompt: dict[str, set[int]] = {}
        for prompt, seed, size in pipe.calls:
            by_prompt.setdefault(prompt, set()).add(seed)
            assert size == aspect_size(W, H, AppConfig().gen.resolution)
        # one fixed seed per style, distinct between styles (temporal stability)
        assert all(len(seeds) == 1 for seeds in by_prompt.values())
        assert len({next(iter(s)) for s in by_prompt.values()}) == 2
        assert {s["seed"] for s in prov["styles"]} == {7, 8}

    def test_composite_exact_validates_shapes(self):
        a = np.zeros((4, 4, 3), np.uint8)
        with pytest.raises(ValueError):
            restyle.composite_exact(a, np.zeros((5, 4, 3), np.uint8), np.zeros((4, 4), bool))
        with pytest.raises(ValueError):
            restyle.composite_exact(a, a, np.zeros((5, 4), bool))


# ------------------------------------------------------------- mask building
class TestKeepMasks:
    def test_percentile_threshold_semantics(self):
        rng = np.random.default_rng(3)
        near = [rng.random((40, 40)).astype(np.float32)]
        masks = restyle.build_keep_masks(near, keep_percentile=72.0,
                                         keep_below_frac=None, dilate_px=0,
                                         smooth_window=1)
        kept = float(masks[0].mean())
        assert kept == pytest.approx(0.72, abs=0.02)  # nearest 72 % kept

    def test_temporal_majority_vote_removes_flicker(self):
        base = np.zeros((20, 20), np.float32)
        base[12:, :] = 1.0                     # stable near block
        flicker = base.copy()
        flicker[:6, :] = 1.0                   # one-frame depth glitch
        seq = [base, base, flicker, base, base]
        masks = restyle.build_keep_masks(seq, keep_percentile=50.0,
                                         keep_below_frac=None, dilate_px=0,
                                         smooth_window=5)
        # the glitch frame is outvoted by its 4 neighbors
        assert np.array_equal(masks[2], masks[0])
        # without smoothing the glitch leaks through
        raw = restyle.build_keep_masks(seq, keep_percentile=50.0,
                                       keep_below_frac=None, dilate_px=0,
                                       smooth_window=1)
        assert not np.array_equal(raw[2], raw[0])

    def test_row_prior_and_dilation_grow_keep(self):
        rng = np.random.default_rng(4)
        near = [rng.random((30, 30)).astype(np.float32)]
        plain = restyle.build_keep_masks(near, 50.0, keep_below_frac=None,
                                         dilate_px=0, smooth_window=1)[0]
        prior = restyle.build_keep_masks(near, 50.0, keep_below_frac=0.5,
                                         dilate_px=0, smooth_window=1)[0]
        dilated = restyle.build_keep_masks(near, 50.0, keep_below_frac=None,
                                           dilate_px=2, smooth_window=1)[0]
        assert prior[15:, :].all()                       # rows below frac kept
        assert prior.sum() >= plain.sum()
        assert dilated.sum() > plain.sum()               # dilation only grows
        assert (dilated | plain).sum() == dilated.sum()  # superset

    def test_parameter_validation(self):
        near = [np.zeros((8, 8), np.float32)]
        with pytest.raises(ValueError):
            restyle.build_keep_masks(near, smooth_window=4)   # even window
        with pytest.raises(ValueError):
            restyle.build_keep_masks(near, keep_percentile=0.0)
        with pytest.raises(ValueError):
            restyle.build_keep_masks([np.zeros((8, 8, 3), np.float32)])


# ------------------------------------------------------------------ artifacts
class TestArtifacts:
    def test_audit_sheet_masks_and_provenance(self, tmp_path):
        _frames_, _pipe, prov, out = _run(tmp_path)
        sheet = out / "audit_sheet.jpg"
        assert sheet.is_file() and sheet.stat().st_size > 0
        for i in range(N_FRAMES):
            assert (out / "masks" / f"f{i:03d}_keep.png").is_file()
            assert (out / "masks" / f"f{i:03d}_depth.png").is_file()

        on_disk = json.loads((out / "provenance.json").read_text())
        assert on_disk == json.loads(json.dumps(prov))  # JSON round-trip clean
        assert on_disk["n_frames"] == N_FRAMES
        assert on_disk["params"]["keep_percentile"] == 72.0
        assert on_disk["params"]["smooth_window"] == 5
        assert len(on_disk["styles"]) == 2
        assert len(on_disk["restyle_frac_per_frame"]) == N_FRAMES
        assert all(0.0 <= f <= 1.0 for f in on_disk["restyle_frac_per_frame"])
        assert on_disk["is_video"] is False and on_disk["fps"] is None

    def test_explicit_style_prompts_win_and_are_capped(self, tmp_path):
        _f, pipe, prov, _out = _run(
            tmp_path, style_prompts=[{"name": "Neon Bar!", "prompt": "a neon bar, photo"},
                                     ("lab", "a lab, photo"),
                                     "a forest, photo"],
            n_styles=2)
        names = [s["name"] for s in prov["styles"]]
        assert names == ["neon_bar", "lab"]               # slugified, first 2 only
        assert prov["style_source"] == "explicit"
        assert len({p for p, _, _ in pipe.calls}) == 2

    def test_default_styles_fill_objects_clause(self, tmp_path):
        _f, _pipe, prov, _out = _run(tmp_path, style_prompts=None, n_styles=2)
        assert prov["style_source"] == "default"
        for s in prov["styles"]:
            assert "{objects}" not in s["prompt"]
            assert restyle.OBJECTS_CLAUSE in s["prompt"]


# ------------------------------------------------------------------ utilities
def test_aspect_size_rounds_to_64():
    assert aspect_size(640, 480, 768) == (768, 576)   # the Bridge 4:3 case
    assert aspect_size(640, 480, 1024) == (1024, 768)
    w, h = aspect_size(123, 457, 768)
    assert w % 64 == 0 and h % 64 == 0 and max(w, h) >= 704
    assert aspect_size(10, 2000, 640) == (64, 640)    # short side floors at 64
