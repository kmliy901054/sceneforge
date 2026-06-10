"""sceneforge/config.py — AppConfig (pydantic) ← sceneforge.yaml + env overrides.

Runtime configuration per ARCHITECTURE.md §2 (file tree) and §10. Defaults here
mirror `sceneforge.yaml`; M1 freezes its measured decisions (gen.level,
gen.depth_mode, gen.s_per_img, vram.mode) back into that file (§7.4, §13).

Precedence: pydantic field defaults < sceneforge.yaml < SCENEFORGE_* env vars.
Env override naming: SCENEFORGE_<SECTION>_<FIELD> (case-insensitive; "__" also
accepted as the separator), e.g. SCENEFORGE_GEN_RESOLUTION=640,
SCENEFORGE_VRAM_MODE=coresident, SCENEFORGE_OLLAMA_HOST=http://...:11434.
The literal strings "null"/"none" set a field to None (e.g. gen.s_per_img).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "sceneforge.yaml"
ENV_PREFIX = "SCENEFORGE_"


class GenConfig(BaseModel):
    """Diffusion generation defaults (§7.2); level/depth_mode/s_per_img frozen by M1."""

    resolution: int = Field(768, ge=256, le=1024)
    steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    cond_scale: float = Field(0.85, ge=0.0, le=2.0)
    control_guidance_end: float = Field(0.9, ge=0.0, le=1.0)
    level: Literal["L0", "L1", "L2", "L3", "L4"] = "L0"          # ladder §7.4
    depth_mode: Literal["disparity", "linear"] = "disparity"     # A/B winner §7.3
    s_per_img: Optional[float] = Field(None, gt=0)               # MEASURED by M1 (§7.4)
    ip_adapter_scale: float = Field(0.6, ge=0.0, le=1.5)         # style-reference strength


class VramConfig(BaseModel):
    """VRAM orchestration (§10.2/§10.3): one guard_gb everywhere, both modes implemented."""

    mode: Literal["sequential", "coresident"] = "sequential"     # sequential = expected default
    guard_gb: float = Field(3.0, ge=0.0)


class EvalConfig(BaseModel):
    """OWLv2 fidelity thresholds (§9)."""

    keep_threshold: float = Field(0.45, ge=0.0, le=1.0)    # quarantine if fidelity_adj below
    detect_threshold: float = Field(0.15, ge=0.0, le=1.0)  # post_process_object_detection
    halluc_threshold: float = Field(0.3, ge=0.0, le=1.0)   # FP score floor / IoU ceiling
    min_gate_area_px: int = Field(1000, ge=0)              # gate-eligible GT instances


class OllamaConfig(BaseModel):
    """Ollama client defaults (§6.1). Server runs OLLAMA_KEEP_ALIVE=-1 (verified):
    every request MUST pass an explicit keep_alive; embed calls always use 0."""

    host: str = "http://localhost:11434"
    planner_model: str = "gemma4:e4b"
    quality_model: str = "qwen3.5:27b"      # ALWAYS keep_alive=0, spec phase only (§10.4)
    embed_model: str = "embeddinggemma"
    num_ctx: int = Field(4096, ge=512)      # ONE value everywhere (§6.1 — ctx change = reload)
    keep_alive: Union[str, int] = "5m"      # director chats; phase barrier does eviction


class PathsConfig(BaseModel):
    """Repo-root-anchored artifact paths (§2, §4.7). Relative entries resolve
    against `root` at validation time, so loaded configs always hold absolute paths."""

    root: Path = REPO_ROOT
    runs_dir: Path = Path("outputs/runs")
    cards_file: Path = Path("assets/cards/asset_cards.json")
    glb_dir: Path = Path("assets/glb")
    embed_cache: Path = Path("assets/index/embed_cache.npz")

    @model_validator(mode="after")
    def _absolutize(self) -> "PathsConfig":
        for name in ("runs_dir", "cards_file", "glb_dir", "embed_cache"):
            p: Path = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, self.root / p)
        return self


class AppConfig(BaseModel):
    """Top-level runtime config: cfg.gen / cfg.vram / cfg.eval / cfg.ollama / cfg.paths."""

    gen: GenConfig = GenConfig()
    vram: VramConfig = VramConfig()
    eval: EvalConfig = EvalConfig()
    ollama: OllamaConfig = OllamaConfig()
    paths: PathsConfig = PathsConfig()

    def save(self, path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Path:
        """Write the config back to yaml (M1 freezes measured gen/vram values, §13)."""
        path = Path(path)
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path


def _parse_env_value(value: str) -> Any:
    """Map "null"/"none" to None; otherwise return the raw string for pydantic coercion."""
    return None if value.strip().lower() in ("null", "none", "") else value


def _env_overrides(environ: Any = None) -> dict[str, dict[str, Any]]:
    """Collect SCENEFORGE_<SECTION>_<FIELD> overrides into a nested dict."""
    environ = os.environ if environ is None else environ
    sections = {
        name: field.annotation
        for name, field in AppConfig.model_fields.items()
    }
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in environ.items():
        if not key.upper().startswith(ENV_PREFIX):
            continue
        rest = key[len(ENV_PREFIX):].lower().replace("__", "_")
        for section, section_model in sections.items():
            if not rest.startswith(section + "_"):
                continue
            field = rest[len(section) + 1:]
            if field in section_model.model_fields:  # type: ignore[union-attr]
                overrides.setdefault(section, {})[field] = _parse_env_value(value)
            else:
                logger.warning("Ignoring unknown config override %s", key)
            break
        else:
            logger.warning("Ignoring unrecognized SCENEFORGE_* variable %s", key)
    return overrides


def load_config(path: Union[str, Path, None] = None, *, environ: Any = None) -> AppConfig:
    """Build AppConfig: field defaults ← sceneforge.yaml ← SCENEFORGE_* env vars.

    A missing yaml file is not an error (pure defaults + env). Unknown yaml keys
    or env fields are warned about and ignored, never fatal.
    """
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    raw: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a yaml mapping, got {type(loaded).__name__}")
        raw = {k: v for k, v in loaded.items() if k in AppConfig.model_fields}
        for k in loaded.keys() - raw.keys():
            logger.warning("Ignoring unknown section %r in %s", k, path)
    for section, fields in _env_overrides(environ).items():
        sec = raw.setdefault(section, {})
        if isinstance(sec, dict):
            sec.update(fields)
    return AppConfig.model_validate(raw)


_config: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    """Process-wide cached AppConfig (orchestrator/UI convenience)."""
    global _config
    if _config is None or reload:
        _config = load_config()
    return _config
