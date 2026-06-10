"""tests/test_director.py — director plan/style/grounding tests (ARCHITECTURE.md §6).

chat_structured is MOCKED throughout (no Ollama): bad-then-good JSON exercises
the §6.3 repair loop; out-of-bounds numerics prove clamp_to_bounds avoids a
repair round; DirectorUnavailable and exhaustion exercise the deterministic
fallbacks; §6.6 style post-validation covers slug collisions, the identity-word
blacklist, the category-first rebuild and the CLIP-77 trim; template_plan is
checked for determinism + exactly-one-target; ground_plan for field fills and
server-pinned camera.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0)

import copy

import pytest

from sceneforge.config import AppConfig
from sceneforge.director import director as director_mod
from sceneforge.director.director import (
    CATEGORY_COLORS,
    CONTENT_BUDGET,
    DirectorLog,
    count_clip_tokens,
    ground_plan,
    make_styles,
    plan_scene,
    postvalidate_styles,
    strip_identity_words,
)
from sceneforge.director.fallback import (
    DEFAULT_STYLES,
    default_style_set,
    template_plan,
)
from sceneforge.director.ollama_client import DirectorUnavailable
from sceneforge.spec import LLMScenePlan, LLMStyle, StyleSet

TASK = "pick the red mug from a cluttered kitchen table"

GOOD_PLAN = {
    "scene_summary": "a red mug among kitchen clutter",
    "objects": [
        {"description": "red ceramic mug", "x_m": 0.10, "y_m": -0.20,
         "yaw_deg": 30.0, "size": "medium", "is_target": True},
        {"description": "steel water bottle", "x_m": -0.30, "y_m": 0.25,
         "yaw_deg": -90.0, "size": "large", "is_target": False},
    ],
    "camera": {"azimuth_deg": 35.0, "elevation_deg": 30.0, "distance_m": 1.1},
}

GOOD_STYLES = {
    "styles": [
        {"name": "Rustic Kitchen",
         "prompt": "a photo of mug, bottle on a table, in a rustic kitchen, warm light"},
        {"name": "Clean Lab",
         "prompt": "a photo of mug, bottle on a table, in a clean lab, white light"},
    ]
}

CARDS = [
    {"name": "mug", "synonyms": ["coffee mug"], "description": "a mug",
     "affordances": ["drinking"], "height_m": 0.095},
    {"name": "bottle", "synonyms": ["water bottle"], "description": "a bottle",
     "affordances": ["drinking"], "height_m": 0.24},
    {"name": "bowl", "synonyms": ["dish"], "description": "a bowl",
     "affordances": ["holding"], "height_m": 0.055},
    {"name": "can", "synonyms": ["soda can"], "description": "a can",
     "affordances": ["drinking"], "height_m": 0.115},
    {"name": "book", "synonyms": ["novel"], "description": "a book",
     "affordances": ["reading"], "height_m": 0.03},
    {"name": "box", "synonyms": ["carton"], "description": "a box",
     "affordances": ["container"], "height_m": 0.08},
]


class ChatMock:
    """Scripted stand-in for chat_structured: returns queued dicts/exceptions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, model, system, user, schema, **kwargs):
        self.calls.append(
            {"model": model, "system": system, "user": user,
             "schema": schema, **kwargs}
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)


@pytest.fixture()
def cfg() -> AppConfig:
    return AppConfig()


class StubRag:
    """Deterministic rag.ground stand-in for ground_plan tests."""

    def ground(self, description: str):
        low = description.lower()
        for asset in ("mug", "bottle", "book", "can", "bowl"):
            if asset in low:
                return asset, 0.9, "embedding"
        return "box", 0.0, "default"


# ----------------------------------------------------------- §6.3 repair loop
class TestPlanSceneRepairLoop:
    def test_bad_then_good_json_repairs_once(self, monkeypatch, cfg):
        bad = copy.deepcopy(GOOD_PLAN)
        for obj in bad["objects"]:
            obj["is_target"] = False                 # zero targets → structural fail
        mock = ChatMock([bad, GOOD_PLAN])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()

        plan = plan_scene(TASK, seed=42, cfg=cfg, log=log)

        assert isinstance(plan, LLMScenePlan)
        assert sum(o.is_target for o in plan.objects) == 1
        assert len(mock.calls) == 2
        # Repair round feeds the validation errors back to the LLM (§6.3).
        assert "failed validation" in mock.calls[1]["user"]
        assert "exactly one object" in mock.calls[1]["user"]
        assert log.meta == {"plan_source": "llm", "plan_repairs": 1}
        assert any("repair round" in e for e in log.entries)

    def test_clamp_avoids_repair_round(self, monkeypatch, cfg):
        # elevation_deg=0 against ge=10 is the verified live Ollama failure —
        # clamp fixes it BEFORE validation, burning zero repair rounds (§3.2).
        oob = copy.deepcopy(GOOD_PLAN)
        oob["camera"] = {"azimuth_deg": -500.0, "elevation_deg": 0.0, "distance_m": 99.0}
        oob["objects"][0]["x_m"] = 5.0
        mock = ChatMock([oob])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()

        plan = plan_scene(TASK, seed=42, cfg=cfg, log=log)

        assert len(mock.calls) == 1                  # no repair round consumed
        assert plan.camera.elevation_deg == 10.0
        assert plan.camera.azimuth_deg == -180.0
        assert plan.camera.distance_m == 2.5
        assert plan.objects[0].x_m == 1.0
        assert log.meta["plan_repairs"] == 0

    def test_unavailable_falls_back_to_template(self, monkeypatch, cfg):
        mock = ChatMock([DirectorUnavailable("connection refused")])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()

        plan = plan_scene(TASK, seed=7, cfg=cfg, log=log)

        assert plan == template_plan(TASK, 7)        # deterministic fallback (§6.5)
        assert log.meta["plan_source"] == "fallback"
        assert len(mock.calls) == 1

    def test_exhaustion_falls_back_after_three_attempts(self, monkeypatch, cfg):
        bad = copy.deepcopy(GOOD_PLAN)
        bad["objects"] = []                          # min_length=2 → always invalid
        mock = ChatMock([bad, bad, bad])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()

        plan = plan_scene(TASK, seed=42, cfg=cfg, log=log)

        assert len(mock.calls) == 3                  # attempt 0..2 (§6.3)
        assert log.meta["plan_source"] == "fallback"
        assert isinstance(plan, LLMScenePlan)        # template plan still valid

    def test_planner_call_parameters(self, monkeypatch, cfg):
        mock = ChatMock([GOOD_PLAN])
        monkeypatch.setattr(director_mod, "chat_structured", mock)

        plan_scene(TASK, seed=42, cfg=cfg)

        call = mock.calls[0]
        assert call["model"] == "gemma4:e4b"
        assert call["seed"] == 42
        assert call["temperature"] == 0.7
        assert call["num_ctx"] == 4096               # standardized (§6.1)
        assert call["keep_alive"] == "5m"            # §12.6 dispute 1
        assert call["schema"] == LLMScenePlan.model_json_schema()


class TestMakeStyles:
    def test_bad_then_good(self, monkeypatch, cfg):
        bad = {"styles": [{"name": "x", "prompt": "too short"}]}  # min_length fails
        mock = ChatMock([bad, GOOD_STYLES])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()
        plan = LLMScenePlan.model_validate(GOOD_PLAN)

        styles = make_styles(TASK, plan, 2, cfg, seed=42, log=log)

        assert isinstance(styles, StyleSet)
        assert len(styles.styles) == 2
        assert len(mock.calls) == 2
        assert mock.calls[0]["seed"] == 43           # run seed + 1 (§6.2)
        assert mock.calls[0]["temperature"] == 0.95
        assert log.meta == {"style_source": "llm", "style_repairs": 1}

    def test_unavailable_falls_back_to_default_styles(self, monkeypatch, cfg):
        mock = ChatMock([DirectorUnavailable("ollama stopped")])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        log = DirectorLog()
        plan = LLMScenePlan.model_validate(GOOD_PLAN)

        styles = make_styles(TASK, plan, 4, cfg, log=log)

        assert [s.name for s in styles.styles] == [s.name for s in DEFAULT_STYLES]
        assert log.meta["style_source"] == "fallback"

    def test_extra_styles_trimmed(self, monkeypatch, cfg):
        five = {"styles": GOOD_STYLES["styles"] + [
            {"name": f"Style {i}",
             "prompt": f"a photo of mug, bottle on a table, environment {i}"}
            for i in range(3)
        ]}
        mock = ChatMock([five])
        monkeypatch.setattr(director_mod, "chat_structured", mock)
        plan = LLMScenePlan.model_validate(GOOD_PLAN)

        styles = make_styles(TASK, plan, 3, cfg)
        assert len(styles.styles) == 3


# ------------------------------------------------- §6.6 style post-validation
class TestStylePostValidation:
    CATS = ["mug", "bottle"]

    def test_category_first_rebuild(self):
        glog: list[dict] = []
        styles = [LLMStyle(name="Cozy Cabin",
                           prompt="in a cozy log cabin, warm firelight, wooden surface")]
        out = postvalidate_styles(styles, self.CATS, glog)

        assert out[0].name == "cozy_cabin"
        assert out[0].prompt.startswith("a photo of mug, bottle on a table, ")
        assert "cozy log cabin" in out[0].prompt     # environment tail preserved
        rebuilds = [e for e in glog if e.get("event") == "style_prompt_rebuild"]
        assert len(rebuilds) == 1 and rebuilds[0]["missing"] == ["mug", "bottle"]

    def test_category_first_prompt_kept_verbatim(self):
        glog: list[dict] = []
        prompt = "a photo of mug, bottle on a table, in a clean lab, white light"
        out = postvalidate_styles([LLMStyle(name="clean lab", prompt=prompt)],
                                  self.CATS, glog)
        assert out[0].prompt == prompt               # no rewrite needed
        assert not any(e.get("event") == "style_prompt_rebuild" for e in glog)

    def test_identity_blacklist_stripped(self):
        glog: list[dict] = []
        styles = [LLMStyle(
            name="Toon World",
            prompt="a photo of mug, bottle on a table, cartoon style, anime colors, soft light",
        )]
        out = postvalidate_styles(styles, self.CATS, glog)
        assert "cartoon" not in out[0].prompt
        assert "anime" not in out[0].prompt
        assert any(e.get("event") == "style_blacklist_strip" for e in glog)

    def test_strip_identity_words_unit(self):
        assert "painting" not in strip_identity_words("an oil painting of a mug")
        assert strip_identity_words("warm light, render, sharp") == "warm light, sharp"

    def test_slug_collision_suffixed(self):
        glog: list[dict] = []
        styles = [
            LLMStyle(name="Rustic Kitchen", prompt="a photo of mug, bottle on a table, one"),
            LLMStyle(name="rustic-kitchen!", prompt="a photo of mug, bottle on a table, two"),
        ]
        out = postvalidate_styles(styles, self.CATS, glog)
        assert [s.name for s in out] == ["rustic_kitchen", "rustic_kitchen_2"]

    def test_clip_77_trim(self):
        glog: list[dict] = []
        # Token-dense tail: rare BPE words tokenize to several pieces each, so
        # this is > 75 CLIP tokens while staying under LLMStyle's 250 chars.
        tail = " ".join(["qzvrk jxwpl"] * 17)
        styles = [LLMStyle(name="verbose", prompt=f"a photo of mug, bottle on a table, {tail}"[:250])]
        out = postvalidate_styles(styles, self.CATS, glog)
        assert count_clip_tokens(out[0].prompt) <= CONTENT_BUDGET
        assert out[0].prompt.startswith("a photo of mug, bottle on a table")
        assert any(e.get("event") == "style_prompt_trim" for e in glog)

    def test_token_counts_logged_per_style(self):
        glog: list[dict] = []
        postvalidate_styles(
            [LLMStyle(name="a b", prompt="a photo of mug, bottle on a table, lab")],
            self.CATS, glog)
        counts = [e for e in glog if e.get("event") == "style_tokens"]
        assert len(counts) == 1 and counts[0]["clip_tokens"] > 0

    def test_objects_placeholder_filled_from_default_styles(self):
        glog: list[dict] = []
        out = postvalidate_styles(default_style_set(4), self.CATS, glog)
        assert len(out) == 4
        for spec in out:
            assert "{objects}" not in spec.prompt
            assert spec.prompt.startswith("a photo of mug, bottle on a table, ")

    def test_tokenizer_fallback_word_approximation(self, monkeypatch):
        # Graceful degradation when SDXL tokenizer weights are not on disk (§6.6).
        monkeypatch.setattr(director_mod, "_clip_tokenizer", lambda: None)
        glog: list[dict] = []
        out = postvalidate_styles(
            [LLMStyle(name="approx", prompt="a photo of mug, bottle on a table, lab light")],
            self.CATS, glog)
        assert out[0].prompt.startswith("a photo of mug, bottle on a table")
        assert count_clip_tokens(out[0].prompt) > 0


# ------------------------------------------------------- §6.5 template fallback
class TestTemplatePlan:
    def test_deterministic_pure_function(self):
        a = template_plan(TASK, 42, cards=CARDS)
        b = template_plan(TASK, 42, cards=CARDS)
        assert a.model_dump() == b.model_dump()

    def test_exactly_one_target_and_valid(self):
        plan = template_plan(TASK, 42, cards=CARDS)
        LLMScenePlan.model_validate(plan.model_dump())
        assert sum(o.is_target for o in plan.objects) == 1
        assert plan.objects[0].is_target              # target placed first, near center

    def test_target_matches_task_with_color(self):
        plan = template_plan(TASK, 42, cards=CARDS)
        assert plan.objects[0].description == "red mug"

    def test_ring_within_usable_tabletop(self):
        for seed in (1, 2, 3, 42):
            plan = template_plan(TASK, seed, cards=CARDS)
            assert 3 <= len(plan.objects) <= 5        # target + 2–4 distractors
            for obj in plan.objects:
                assert -0.5 <= obj.x_m <= 0.5
                assert -0.35 <= obj.y_m <= 0.35

    def test_fixed_camera(self):
        plan = template_plan(TASK, 42, cards=CARDS)
        cam = plan.camera
        assert (cam.azimuth_deg, cam.elevation_deg, cam.distance_m) == (35.0, 30.0, 1.1)

    def test_no_cards_file_uses_builtin_synonyms(self):
        plan = template_plan("put the hammer in the box", 5, cards=None)
        assert sum(o.is_target for o in plan.objects) == 1


# ----------------------------------------------------------- §4.2 ground_plan
class TestGroundPlan:
    def test_fills_resolved_fields_and_pins_camera(self):
        plan = LLMScenePlan.model_validate(GOOD_PLAN)
        styles = StyleSet.model_validate(GOOD_STYLES)

        spec = ground_plan(plan, styles, TASK, seed=42, library=None, rag=StubRag())

        assert spec.task == TASK and spec.seed == 42
        assert [o.instance_id for o in spec.objects] == [1, 2]
        assert [o.asset_id for o in spec.objects] == ["mug", "bottle"]
        assert all(o.category == o.asset_id for o in spec.objects)   # v1 rule
        assert [o.scale for o in spec.objects] == [1.0, 1.2]         # medium/large
        assert all(o.z_m == 0.0 for o in spec.objects)
        assert spec.objects[0].is_target and not spec.objects[1].is_target
        # Server-pinned camera fields (§3.1): never LLM-settable.
        assert spec.camera.look_at == (0.0, 0.0, 0.05)
        assert spec.camera.yfov_deg == 50.0
        assert spec.camera.azimuth_deg == 35.0

    def test_color_word_wins_else_palette(self):
        plan = LLMScenePlan.model_validate(GOOD_PLAN)
        styles = StyleSet.model_validate(GOOD_STYLES)
        spec = ground_plan(plan, styles, TASK, seed=42, library=None, rag=StubRag())
        assert spec.objects[0].color_rgb == (200, 40, 40)            # "red ... mug"
        assert spec.objects[1].color_rgb == CATEGORY_COLORS["bottle"]  # no color word

    def test_grounding_log_records_decisions(self):
        plan = LLMScenePlan.model_validate(GOOD_PLAN)
        styles = StyleSet.model_validate(GOOD_STYLES)
        spec = ground_plan(plan, styles, TASK, seed=42, library=None, rag=StubRag())
        ground_entries = [e for e in spec.grounding_log if "requested" in e]
        assert ground_entries[0] == {
            "requested": "red ceramic mug", "asset_id": "mug",
            "score": 0.9, "method": "embedding",
        }
        token_entries = [e for e in spec.grounding_log if e.get("event") == "style_tokens"]
        assert len(token_entries) == len(spec.styles)

    def test_styles_postvalidated_into_slugs(self):
        plan = LLMScenePlan.model_validate(GOOD_PLAN)
        styles = StyleSet.model_validate(GOOD_STYLES)
        spec = ground_plan(plan, styles, TASK, seed=42, library=None, rag=StubRag())
        assert [s.name for s in spec.styles] == ["rustic_kitchen", "clean_lab"]
        for style in spec.styles:
            assert style.prompt.startswith("a photo of mug, bottle on a table")

    def test_library_membership_check_degrades_to_box(self):
        class TinyLibrary:
            def cards(self):
                return [{"name": "bottle"}]           # "mug" is NOT buildable

        plan = LLMScenePlan.model_validate(GOOD_PLAN)
        styles = StyleSet.model_validate(GOOD_STYLES)
        spec = ground_plan(plan, styles, TASK, seed=42,
                           library=TinyLibrary(), rag=StubRag())
        assert spec.objects[0].asset_id == "box"
        assert any(e.get("event") == "unknown_asset" for e in spec.grounding_log)


# ---------------------------------------------------------------- DirectorLog
class TestDirectorLog:
    def test_flush_appends_and_clears(self, tmp_path):
        log = DirectorLog()
        log.add("plan_scene attempt 0: validation failed (repair round)", "details")
        path = log.flush(tmp_path / "run" / "director_log.md")
        assert path.is_file()
        assert "repair round" in path.read_text(encoding="utf-8")
        assert log.entries == []
        log.flush(path)                               # empty flush is a no-op
        assert path.read_text(encoding="utf-8").count("# Director log") == 1
