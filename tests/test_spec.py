"""tests/test_spec.py — spec-layer unit tests (ARCHITECTURE.md §3.2, §6.6).

Covers: clamp_to_bounds (nested models, list elements, valid values untouched),
slugify (unicode, explicit collision suffixes), the exactly_one_target
validator, the NEGATIVE_PROMPT constant, and an AppConfig load/override smoke
test (config.py has no dedicated test file).
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0): np.infty shim + EGL env

import copy
import re

import pytest
from pydantic import ValidationError

from sceneforge.config import AppConfig, load_config
from sceneforge.spec import (
    NEGATIVE_PROMPT,
    LLMScenePlan,
    LLMStyle,
    SceneSpec,
    StyleSpec,
    clamp_to_bounds,
    slugify,
)


def make_raw_plan(camera: dict | None = None, objects: list[dict] | None = None) -> dict:
    """A structurally valid raw planner dict (as Ollama format=json_schema yields)."""
    return {
        "scene_summary": "a red mug among kitchen clutter",
        "objects": objects
        or [
            {
                "description": "red ceramic mug",
                "x_m": 0.10,
                "y_m": -0.20,
                "yaw_deg": 30.0,
                "size": "medium",
                "is_target": True,
            },
            {
                "description": "steel water bottle",
                "x_m": -0.30,
                "y_m": 0.25,
                "yaw_deg": -90.0,
                "size": "large",
                "is_target": False,
            },
        ],
        "camera": camera or {"azimuth_deg": 35.0, "elevation_deg": 30.0, "distance_m": 1.1},
    }


# ---------------------------------------------------------------- clamp_to_bounds
class TestClampToBounds:
    def test_nested_camera_bounds(self):
        # elevation_deg=0 against ge=10 is the verified live Ollama failure (§3.2).
        raw = make_raw_plan(camera={"azimuth_deg": -500.0, "elevation_deg": 0.0, "distance_m": 99.0})
        out = clamp_to_bounds(raw, LLMScenePlan)
        assert out["camera"] == {"azimuth_deg": -180.0, "elevation_deg": 10.0, "distance_m": 2.5}
        plan = LLMScenePlan.model_validate(out)  # post-clamp the model must validate
        assert plan.camera.elevation_deg == 10.0

    def test_list_elements_clamped(self):
        raw = make_raw_plan()
        raw["objects"][0]["x_m"] = 5.0       # ge=-1.0, le=1.0
        raw["objects"][0]["yaw_deg"] = 720.0  # le=180
        raw["objects"][1]["y_m"] = -3.0       # ge=-0.75
        out = clamp_to_bounds(raw, LLMScenePlan)
        assert out["objects"][0]["x_m"] == 1.0
        assert out["objects"][0]["yaw_deg"] == 180.0
        assert out["objects"][1]["y_m"] == -0.75
        LLMScenePlan.model_validate(out)

    def test_valid_values_untouched_and_input_not_mutated(self):
        raw = make_raw_plan()
        snapshot = copy.deepcopy(raw)
        out = clamp_to_bounds(raw, LLMScenePlan)
        assert out == snapshot       # in-bounds leaves pass through exactly
        assert raw == snapshot       # the input dict is never mutated

    def test_wrong_types_and_unknown_keys_left_for_repair_loop(self):
        raw = make_raw_plan(camera={"azimuth_deg": "very wide", "elevation_deg": 0.0, "distance_m": 1.1})
        raw["bogus_extra"] = {"x": 1}
        raw["objects"][0]["size"] = "medium"          # Literal str: not numeric, untouched
        out = clamp_to_bounds(raw, LLMScenePlan)
        assert out["camera"]["azimuth_deg"] == "very wide"   # structural → repair loop
        assert out["camera"]["elevation_deg"] == 10.0        # numeric sibling still clamped
        assert out["bogus_extra"] == {"x": 1}
        assert out["objects"][0]["size"] == "medium"

    def test_bools_never_clamped(self):
        raw = make_raw_plan()
        out = clamp_to_bounds(raw, LLMScenePlan)
        assert out["objects"][0]["is_target"] is True
        assert out["objects"][1]["is_target"] is False


# ------------------------------------------------------------------- validators
class TestExactlyOneTarget:
    def test_zero_targets_rejected(self):
        raw = make_raw_plan()
        raw["objects"][0]["is_target"] = False
        with pytest.raises(ValidationError, match="exactly one object"):
            LLMScenePlan.model_validate(raw)

    def test_two_targets_rejected(self):
        raw = make_raw_plan()
        raw["objects"][1]["is_target"] = True
        with pytest.raises(ValidationError, match="exactly one object"):
            LLMScenePlan.model_validate(raw)

    def test_one_target_accepted(self):
        plan = LLMScenePlan.model_validate(make_raw_plan())
        assert sum(o.is_target for o in plan.objects) == 1


def test_negative_prompt_constant_present():
    assert isinstance(NEGATIVE_PROMPT, str) and NEGATIVE_PROMPT
    assert NEGATIVE_PROMPT == (
        "blurry, deformed, duplicate objects, lowres, watermark, cartoon, painting"
    )
    # StyleSpec carries it by default (active only at L2/L3, guidance_scale > 1).
    style = StyleSpec(name="clean_lab", prompt="a photo of mug on a table, in a lab")
    assert style.negative_prompt == NEGATIVE_PROMPT


# ---------------------------------------------------------------------- slugify
SLUG_PATTERN = re.compile(r"^[a-z0-9_]{2,40}$")  # StyleSpec.name pattern (§3.2)


class TestSlugify:
    def test_basic(self):
        assert slugify("Rustic Kitchen") == "rustic_kitchen"
        assert slugify("  Clean -- Lab  ") == "clean_lab"

    def test_unicode_runs_collapse_to_single_underscore(self):
        slug = slugify("Café — Néon Diner!")
        assert slug == "caf_n_on_diner"
        assert SLUG_PATTERN.match(slug)

    def test_collision_suffixes(self):
        taken: set[str] = set()
        first = slugify("Rustic Kitchen", taken)
        taken.add(first)
        second = slugify("Rustic-Kitchen!", taken)   # same slug post-normalization
        taken.add(second)
        third = slugify("rustic kitchen", taken)
        assert (first, second, third) == ("rustic_kitchen", "rustic_kitchen_2", "rustic_kitchen_3")

    def test_truncation_to_40_including_collision_suffix(self):
        long_name = "Ultra Mega Hyper Long Style Name " * 3
        slug = slugify(long_name)
        assert len(slug) <= 40 and SLUG_PATTERN.match(slug)
        slug2 = slugify(long_name, taken={slug})
        assert len(slug2) <= 40 and slug2.endswith("_2") and slug2 != slug

    def test_degenerate_input_still_yields_valid_slug(self):
        slug = slugify("!!! ???")
        assert SLUG_PATTERN.match(slug)
        # And it must be usable as a StyleSpec name (the canonical key).
        StyleSpec(name=slug, prompt="a photo of box on a table, plain background")

    def test_slug_of_llm_style_name_always_fits_stylespec(self):
        llm = LLMStyle(name="Néon Cyberpunk Alley #7", prompt="a photo of mug on a table, neon alley")
        StyleSpec(name=slugify(llm.name), prompt=llm.prompt)


# ------------------------------------------------------------- contract smoke
def test_scenespec_resolved_contract_roundtrip():
    spec = SceneSpec(
        task="pick the red mug from a cluttered kitchen table",
        objects=[
            {
                "instance_id": 1,
                "asset_id": "mug",
                "category": "mug",
                "requested": "red ceramic mug",
                "x_m": 0.1,
                "y_m": -0.2,
                "yaw_deg": 30.0,
                "is_target": True,
                "color_rgb": (200, 40, 40),
            }
        ],
        styles=[{"name": "rustic_kitchen", "prompt": "a photo of mug on a table, rustic kitchen"}],
    )
    assert spec.schema_version == "1.1"
    assert spec.camera.look_at == (0.0, 0.0, 0.05)   # server-pinned (§3.1)
    assert spec.camera.yfov_deg == 50.0
    rt = SceneSpec.model_validate(spec.model_dump())
    assert rt == spec
    # model_json_schema() of the LLM-facing models is passed verbatim as format=.
    assert "properties" in LLMScenePlan.model_json_schema()


# ------------------------------------------------------------------ AppConfig
class TestAppConfig:
    def test_yaml_defaults_match_architecture(self):
        cfg = load_config()  # reads repo sceneforge.yaml
        assert cfg.gen.resolution == 768
        assert cfg.gen.steps == 4
        assert cfg.gen.guidance_scale == 0.0
        assert cfg.gen.cond_scale == 0.85
        assert cfg.gen.control_guidance_end == 0.9
        assert cfg.gen.level == "L0"
        assert cfg.gen.depth_mode == "disparity"
        # None until M1 freezes a measured value into sceneforge.yaml; positive after
        assert cfg.gen.s_per_img is None or cfg.gen.s_per_img > 0
        assert cfg.vram.mode == "sequential"
        assert cfg.vram.guard_gb == 3.0
        assert cfg.eval.keep_threshold == 0.45
        assert cfg.eval.detect_threshold == 0.15
        assert cfg.eval.halluc_threshold == 0.3
        assert cfg.eval.min_gate_area_px == 1000
        assert cfg.ollama.host == "http://localhost:11434"
        assert cfg.ollama.planner_model == "gemma4:e4b"
        assert cfg.ollama.quality_model == "qwen3.5:27b"
        assert cfg.ollama.embed_model == "embeddinggemma"
        assert cfg.ollama.num_ctx == 4096
        assert cfg.ollama.keep_alive == "5m"
        assert cfg.paths.cards_file.is_absolute()
        assert str(cfg.paths.runs_dir).endswith("outputs/runs")

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SCENEFORGE_GEN_RESOLUTION", "640")
        monkeypatch.setenv("SCENEFORGE_VRAM_MODE", "coresident")
        monkeypatch.setenv("SCENEFORGE_GEN_S_PER_IMG", "2.3")
        monkeypatch.setenv("SCENEFORGE_OLLAMA_NUM_CTX", "8192")
        cfg = load_config()
        assert cfg.gen.resolution == 640
        assert cfg.vram.mode == "coresident"
        assert cfg.gen.s_per_img == pytest.approx(2.3)
        assert cfg.ollama.num_ctx == 8192

    def test_env_null_clears_optional(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SCENEFORGE_GEN_S_PER_IMG", "null")
        assert load_config().gen.s_per_img is None

    def test_missing_yaml_falls_back_to_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "absent.yaml")
        assert isinstance(cfg, AppConfig)
        assert cfg.gen.resolution == 768
        assert cfg.vram.guard_gb == 3.0
