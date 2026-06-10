"""sceneforge/director/ollama_client.py — thin structured-output Ollama client (§6.1).

Two entry points, both with EXPLICIT keep_alive on every request — the server
runs ``OLLAMA_KEEP_ALIVE=-1`` (verified): any request omitting ``keep_alive``
pins its model in VRAM forever.

- ``chat_structured``: one chat call with ``format=<json schema dict>`` (grammar
  enforces structure/types, NOT numeric bounds — §3.2), ``stream=True`` so raw
  chunks can be forwarded to the UI via ``on_token``, returns the parsed dict.
- ``embed``: batch embeddings, default ``keep_alive=0`` (§12.6 dispute 2 — the
  committed embed cache makes query-time embeds rare; 1.11 GB is not worth
  pinning).

Any connection error, timeout, HTTP error, or JSON-parse failure raises
``DirectorUnavailable`` so callers take the deterministic fallback (§6.5).
An Ollama exception must never reach the UI.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

#: Crude chars-per-token ratio used only for the §12.6(4) "raise rather than
#: silently truncate" prompt-size guard. gemma tokenizers average ~4 chars/token
#: on English prose; 3 is deliberately conservative.
_CHARS_PER_TOKEN = 3


class DirectorUnavailable(Exception):
    """Raised when the LLM director cannot produce a usable response
    (connection/timeout/HTTP/parse failure). Callers take the deterministic
    fallback (§6.1, §6.5) — this exception must never reach the UI.
    """


def _client(host: Optional[str], timeout_s: float):
    """Build an ``ollama.Client`` (import deferred so tests can run without it)."""
    import ollama

    if host is None:
        from sceneforge.config import get_config

        host = get_config().ollama.host
    return ollama.Client(host=host, timeout=timeout_s)


def chat_structured(
    model: str,
    system: str,
    user: str,
    schema: dict,
    *,
    seed: int,
    temperature: float = 0.7,
    keep_alive: Union[str, int] = "5m",
    num_ctx: int = 4096,
    timeout_s: float = 90,
    on_token: Optional[Callable[[str], None]] = None,
    host: Optional[str] = None,
) -> dict:
    """One structured chat call → parsed JSON dict (ARCHITECTURE.md §6.1).

    ``schema`` is a ``model_json_schema()`` dict passed verbatim as ``format=``
    (Ollama 0.20.3 grammar-enforces structure/types/$defs, NOT numeric min/max —
    callers must run ``clamp_to_bounds`` on the result, §3.2). Always streams;
    ``on_token`` (if given) receives each raw text chunk for the UI panel.

    Raises ``DirectorUnavailable`` on any failure, and ``ValueError`` if the
    prompt cannot fit ``num_ctx`` (§12.6 dispute 4: raise, never truncate).
    """
    approx_tokens = (len(system) + len(user)) // _CHARS_PER_TOKEN
    if approx_tokens > num_ctx:
        raise ValueError(
            f"prompt ~{approx_tokens} tokens exceeds num_ctx={num_ctx}; "
            "refusing to silently truncate (ARCHITECTURE.md §12.6.4)"
        )

    try:
        client = _client(host, timeout_s)
        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options={"temperature": temperature, "seed": seed, "num_ctx": num_ctx},
            keep_alive=keep_alive,
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            piece = chunk.message.content or ""
            if piece:
                parts.append(piece)
                if on_token is not None:
                    on_token(piece)
        text = "".join(parts)
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise TypeError(f"expected a JSON object, got {type(raw).__name__}")
        return raw
    except Exception as exc:  # uniform funnel: connection/HTTP/timeout/parse
        if isinstance(exc, DirectorUnavailable):
            raise
        logger.warning("chat_structured(%s) failed: %s", model, exc)
        raise DirectorUnavailable(f"chat({model}) failed: {exc}") from exc


def embed(
    model: str,
    texts: Sequence[str],
    *,
    keep_alive: Union[str, int] = 0,
    timeout_s: float = 30,
    host: Optional[str] = None,
) -> np.ndarray:
    """Embed ``texts`` → float32 array of shape (len(texts), D) (§6.1, §12-D).

    Default ``keep_alive=0``: the server pins models forever otherwise, and
    embeddinggemma (1.11 GB resident, measured) broke the v1.0 coresident
    budget. Raises ``DirectorUnavailable`` on any failure.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        client = _client(host, timeout_s)
        resp = client.embed(model=model, input=list(texts), keep_alive=keep_alive)
        vectors = np.asarray(resp.embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(f"unexpected embeddings shape {vectors.shape}")
        return vectors
    except Exception as exc:
        if isinstance(exc, DirectorUnavailable):
            raise
        logger.warning("embed(%s) failed: %s", model, exc)
        raise DirectorUnavailable(f"embed({model}) failed: {exc}") from exc
