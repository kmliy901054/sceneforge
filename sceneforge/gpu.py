"""sceneforge/gpu.py — VRAM orchestration (ARCHITECTURE.md §10).

``snapshot()`` (device-wide ``torch.cuda.mem_get_info`` + Ollama ``/api/ps``, no
pynvml), ``ollama_unload()`` (per-model ``/api/generate`` keep_alive=0 with the
HTTP-400 → ``/api/embed`` fallback for embedding models, then ``/api/ps`` polling
≤ 15 s), the ``phase()`` context manager (§10.3 barriers, §10.4 quality swap) and
the V0–V4 degrade-ladder constants (§10.2, ONE ``guard_gb`` from config).

The Ollama server runs ``OLLAMA_KEEP_ALIVE=-1`` (verified): any request omitting
``keep_alive`` pins its model in VRAM forever — every unload POST here passes
``keep_alive: 0`` explicitly. Mode decisions are driven by device-wide
``mem_get_info`` free memory, NOT ``max_memory_allocated`` (blind to the CUDA
context, reserved slack and Ollama — §10.1).
"""
from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, Optional

import requests
import torch

from sceneforge.config import AppConfig, get_config

logger = logging.getLogger(__name__)

GIB = 1024 ** 3
PhaseName = Literal["spec", "spec_quality", "diffusion", "eval"]

# ---------------------------------------------------------------- §10.2 ladder
#: Degrade ladder V0–V4 — single source of truth; the SAME cfg.vram.guard_gb is
#: used by the M1 mode-pick rule AND the runtime pre-burst guard.
VRAM_LEVELS: tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4")
VRAM_LADDER: dict[str, str] = {
    "V0": "coresident @768 (gemma 4k ctx resident; embeddinggemma unloaded for bursts)",
    "V1": "sequential @768 (expected default)",
    "V2": "+ enable_vae_tiling + enable_attention_slicing (pre-burst free < guard_gb)",
    "V3": "enable_model_cpu_offload sequential, ~2x slower (caught OOM → recovery §7.5)",
    "V4": "resolution 640 (OOM persists after V3 retry)",
}
#: Degraded resolution at ladder level V4 (§10.2 / §7.5).
V4_RESOLUTION = 640


def next_vram_level(level: str) -> str:
    """One step down the §10.2 ladder; V4 is terminal."""
    idx = VRAM_LEVELS.index(level)
    return VRAM_LEVELS[min(idx + 1, len(VRAM_LEVELS) - 1)]


def guard_gb(cfg: Optional[AppConfig] = None) -> float:
    """THE guard constant — always read from config (§10.2: one value everywhere)."""
    return (cfg if cfg is not None else get_config()).vram.guard_gb


def free_gb() -> float:
    """Device-wide free VRAM in GiB via ``torch.cuda.mem_get_info`` (0.0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    free_b, _total_b = torch.cuda.mem_get_info()
    return free_b / GIB


def under_guard(cfg: Optional[AppConfig] = None) -> bool:
    """§10.2 V2 trigger: pre-burst device free memory below guard_gb."""
    return free_gb() < guard_gb(cfg)


# ------------------------------------------------------------------- snapshot
def _host(host: Optional[str] = None) -> str:
    return host if host is not None else get_config().ollama.host


def _ps_models(host: str, timeout_s: float = 5.0) -> list[dict]:
    """GET ``/api/ps`` → list of resident-model dicts; unreachable server → []."""
    try:
        resp = requests.get(f"{host}/api/ps", timeout=timeout_s)
        return list(resp.json().get("models") or [])
    except (requests.RequestException, ValueError) as exc:
        logger.debug("ollama /api/ps unavailable (%s) — treating as empty", exc)
        return []


def _model_name(model: dict) -> str:
    return str(model.get("name") or model.get("model") or "")


def snapshot(host: Optional[str] = None) -> dict:
    """Point-in-time VRAM state (§10.3): ``{"free_gb", "total_gb", "torch_alloc_gb",
    "torch_reserved_gb", "ollama_models": [{name, size_gb, size_vram_gb}]}``.

    Uses device-wide ``mem_get_info`` + ``/api/ps`` (no pynvml). Logged on every
    phase enter/exit and polled by the UI footer Timer.
    """
    free = total = alloc = reserved = 0.0
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free, total = free_b / GIB, total_b / GIB
            alloc = torch.cuda.memory_allocated() / GIB
            reserved = torch.cuda.memory_reserved() / GIB
        except Exception as exc:  # never let the footer Timer crash the app
            logger.warning("snapshot: CUDA query failed: %s", exc)
    models = [
        {
            "name": _model_name(m),
            "size_gb": round(float(m.get("size", 0)) / GIB, 2),
            "size_vram_gb": round(float(m.get("size_vram", 0)) / GIB, 2),
        }
        for m in _ps_models(_host(host))
    ]
    return {
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
        "torch_alloc_gb": round(alloc, 2),
        "torch_reserved_gb": round(reserved, 2),
        "ollama_models": models,
    }


# -------------------------------------------------------------- ollama_unload
def ollama_unload(
    host: Optional[str] = None,
    *,
    timeout_s: float = 15.0,
    poll_s: float = 0.25,
    only: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Evict resident models from Ollama VRAM (§10.3 review fix).

    Iterate ``/api/ps``; per model POST ``/api/generate {"model", "keep_alive": 0}``;
    on HTTP 400 ("does not support generate" — embedding models) POST
    ``/api/embed {"model", "input": "", "keep_alive": 0}`` (verified live: /api/ps
    empties, reload works). Poll ``/api/ps`` until NO matching models remain
    (timeout 15 s; measured ~2 s). This sweep inherently covers embeddinggemma.

    Args:
        only: optional name predicate restricting the sweep — the coresident
            diffusion barrier unloads only embedding models, leaving gemma (§10.3).

    Returns True once no matching models are resident. An unreachable server
    counts as already-empty (the demo completes with Ollama stopped, §1).
    """
    h = _host(host)

    def matching() -> list[str]:
        names = (_model_name(m) for m in _ps_models(h, timeout_s=min(5.0, timeout_s)))
        return [n for n in names if only is None or only(n)]

    for name in matching():
        try:
            resp = requests.post(f"{h}/api/generate",
                                 json={"model": name, "keep_alive": 0}, timeout=10)
            if resp.status_code == 400:  # embedding model — generate unsupported
                requests.post(f"{h}/api/embed",
                              json={"model": name, "input": "", "keep_alive": 0},
                              timeout=10)
            elif resp.status_code >= 300:
                logger.warning("ollama_unload: /api/generate %s → HTTP %d",
                               name, resp.status_code)
        except requests.RequestException as exc:
            logger.warning("ollama_unload: unload POST failed for %s: %s", name, exc)

    deadline = time.monotonic() + timeout_s
    while True:
        remaining = matching()
        if not remaining:
            return True
        if time.monotonic() >= deadline:
            logger.warning("ollama_unload: still resident after %.0fs: %s",
                           timeout_s, remaining)
            return False
        time.sleep(poll_s)


# ----------------------------------------------------------------- phase mgr
@contextmanager
def phase(
    name: PhaseName,
    *,
    cfg: Optional[AppConfig] = None,
    pipe: Any = None,
    host: Optional[str] = None,
    vram_log: Optional[list] = None,
) -> Iterator[dict]:
    """§10.3 phase barriers as a context manager.

    - ``"spec"`` entry: ``gc.collect()`` + ``torch.cuda.empty_cache()`` — returns
      1–2 GB allocator slack to the driver so gemma places fully on GPU next to
      idle SDXL weights (without this, planning drops from the measured 139 tok/s).
    - ``"spec_quality"`` (§10.4, qwen swap): ``pipe.to("cpu")`` → gc + empty_cache
      on entry; on exit ``ollama_unload()`` (qwen calls run keep_alive=0) then
      ``pipe.to("cuda")``.
    - ``"diffusion"`` entry: ``ollama_unload()`` barrier; coresident mode leaves
      gemma resident and unloads only embedding models.
    - ``"eval"`` entry: ``ollama_unload()`` in BOTH modes (OWLv2 lives here only).

    Enter/exit ``snapshot()``s are appended to ``vram_log`` (the orchestrator
    persists it to ``run_meta.json``) and yielded as the mutable context record
    ``{"phase", "enter", "exit"}``.
    """
    cfg = cfg if cfg is not None else get_config()
    h = host if host is not None else cfg.ollama.host

    if name == "spec":
        gc.collect()
        torch.cuda.empty_cache()
    elif name == "spec_quality":
        if pipe is not None:
            pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    elif name == "diffusion":
        if cfg.vram.mode == "coresident":
            embed_model = cfg.ollama.embed_model
            ollama_unload(h, only=lambda n: embed_model in n or "embed" in n.lower())
        else:
            ollama_unload(h)
    elif name == "eval":
        ollama_unload(h)
    else:
        raise ValueError(f"unknown phase {name!r} (expected one of "
                         f"'spec', 'spec_quality', 'diffusion', 'eval')")

    record: dict = {"phase": name, "enter": snapshot(h)}
    if vram_log is not None:
        vram_log.append(record)
    logger.info("phase(%s) enter: %s", name, record["enter"])
    try:
        yield record
    finally:
        if name == "spec_quality":
            ollama_unload(h)  # wait for qwen (keep_alive=0) to clear, §10.4
            if pipe is not None:
                pipe.to("cuda")
        record["exit"] = snapshot(h)
        logger.info("phase(%s) exit: %s", name, record["exit"])
