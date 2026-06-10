"""sceneforge/director/rag.py — EmbeddingIndex: numpy cosine + difflib fallback (§6.4, §12-D).

A vector DB for ≤30 vectors is indefensible: documents are embedded once
(embeddinggemma via ``/api/embed``, ALWAYS ``keep_alive=0``), L2-normalized into
an ``(N, D) float32`` matrix, and a query is one matmul. Asymmetric prefixes
(embeddinggemma's trained formats):

- document: ``"title: {name} | text: {description}. Synonyms: ... Used for: ..."``
- query:    ``"task: search result | query: {description}"``

The matrix is cached in ``assets/index/embed_cache.npz`` keyed by the sha256 of
the cards FILE (cache is COMMITTED so cold demos never need the embed model).

``ground(description)``: top-1 cosine if score ≥ 0.35; else difflib over card
names+synonyms; else ``"box"``. ``embed()`` failure → difflib directly. Every
decision is returned as ``(asset_id, score, method)`` for the grounding_log.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from sceneforge.director.ollama_client import DirectorUnavailable

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], np.ndarray]

SCORE_FLOOR = 0.35          # cosine below this → difflib (§6.4: routes nonsense to "box")
DIFFLIB_CUTOFF = 0.6        # SequenceMatcher ratio floor for a synonym match
FALLBACK_ASSET = "box"      # the universal silhouette (§6.4)

_WORD_RE = re.compile(r"[a-z]+")


def document_text(card: Mapping) -> str:
    """Embeddinggemma DOCUMENT format for one asset card (§6.4)."""
    synonyms = ", ".join(card.get("synonyms", ()))
    affordances = ", ".join(card.get("affordances", ()))
    return (
        f"title: {card['name']} | text: {card['description']}. "
        f"Synonyms: {synonyms}. Used for: {affordances}."
    )


def query_text(description: str) -> str:
    """Embeddinggemma QUERY format (asymmetric to the document prefix, §6.4)."""
    return f"task: search result | query: {description}"


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cards_sha256(cards_path: Union[str, Path]) -> str:
    """sha256 hex digest of the cards FILE bytes — the npz cache key (§6.4)."""
    return hashlib.sha256(Path(cards_path).read_bytes()).hexdigest()


def save_cache(
    path: Union[str, Path],
    vectors: np.ndarray,
    asset_ids: Sequence[str],
    cards_sha: str,
) -> Path:
    """Write the L2-normalized document matrix to npz, keyed by ``cards_sha``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=np.asarray(vectors, dtype=np.float32),
        asset_ids=np.asarray(list(asset_ids)),
        cards_sha256=np.asarray(cards_sha),
    )
    return path


def load_cache(
    path: Union[str, Path], cards_sha: str
) -> Optional[tuple[np.ndarray, list[str]]]:
    """Load (vectors, asset_ids) if the cache exists AND its sha matches, else None."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["cards_sha256"]) != cards_sha:
                logger.info("embed cache %s is stale (cards sha changed)", path)
                return None
            return (
                np.asarray(data["vectors"], dtype=np.float32),
                [str(a) for a in data["asset_ids"]],
            )
    except Exception as exc:  # corrupt cache is never fatal — rebuild or difflib
        logger.warning("embed cache %s unreadable (%s); ignoring", path, exc)
        return None


class EmbeddingIndex:
    """Cosine grounding over asset cards, with difflib + 'box' fallbacks (§6.4).

    ``vectors`` may be None (no cache, no Ollama): ``ground()`` then goes
    straight to difflib — the demo completes with the Ollama server stopped.
    """

    def __init__(
        self,
        cards: Sequence[Mapping],
        vectors: Optional[np.ndarray] = None,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self.cards = list(cards)
        self.asset_ids: list[str] = [str(c["name"]) for c in self.cards]
        self.vectors = None if vectors is None else _l2_normalize(vectors)
        self.embed_fn = embed_fn

    # ------------------------------------------------------------ construction
    @classmethod
    def build(cls, cards: Sequence[Mapping], embed_fn: EmbedFn) -> "EmbeddingIndex":
        """Embed every card's document string → L2-normalized (N, D) float32."""
        docs = [document_text(c) for c in cards]
        vectors = _l2_normalize(embed_fn(docs))
        return cls(cards, vectors=vectors, embed_fn=embed_fn)

    @classmethod
    def load_or_build(
        cls,
        cards_path: Union[str, Path],
        cache_path: Union[str, Path],
        embed_fn: Optional[EmbedFn] = None,
    ) -> "EmbeddingIndex":
        """Cards file + sha-keyed npz cache → index (build+save on cache miss).

        Cache miss without ``embed_fn`` (or with Ollama down) degrades to a
        difflib-only index — never raises (§6.4).
        """
        cards_path = Path(cards_path)
        cards = json.loads(cards_path.read_text(encoding="utf-8"))
        sha = cards_sha256(cards_path)
        cached = load_cache(cache_path, sha)
        if cached is not None:
            vectors, asset_ids = cached
            index = cls(cards, vectors=vectors, embed_fn=embed_fn)
            if index.asset_ids != asset_ids:  # cards reordered but bytes hashed? impossible —
                logger.warning("embed cache asset order mismatch; rebuilding")  # belt+braces
            else:
                return index
        if embed_fn is None:
            return cls(cards, vectors=None, embed_fn=None)
        try:
            index = cls.build(cards, embed_fn)
        except DirectorUnavailable as exc:
            logger.warning("embed model unavailable (%s); difflib-only index", exc)
            return cls(cards, vectors=None, embed_fn=None)
        save_cache(cache_path, index.vectors, index.asset_ids, sha)
        return index

    # ---------------------------------------------------------------- grounding
    def ground(self, description: str) -> tuple[str, float, str]:
        """Map a free-text object description to ``(asset_id, score, method)``.

        method ∈ {"embedding", "difflib", "default"}; "default" means the 0.35
        floor and difflib both failed and the universal "box" was assigned.
        """
        if self.vectors is not None and self.embed_fn is not None:
            try:
                q = _l2_normalize(self.embed_fn([query_text(description)]))[0]
                scores = self.vectors @ q
                top = int(np.argmax(scores))
                if float(scores[top]) >= SCORE_FLOOR:
                    return self.asset_ids[top], float(scores[top]), "embedding"
            except DirectorUnavailable:
                logger.info("embed() unavailable; difflib grounding for %r", description)
        return self._ground_difflib(description)

    def _ground_difflib(self, description: str) -> tuple[str, float, str]:
        """Best SequenceMatcher ratio of any description token (or the whole
        phrase) against any card name/synonym; below 0.6 → ('box', 0.0, 'default')."""
        tokens = _WORD_RE.findall(description.lower())
        candidates = tokens + [description.lower().strip()]
        best_asset, best_score = None, 0.0
        for card in self.cards:
            names = [str(card["name"]).lower()] + [
                str(s).lower() for s in card.get("synonyms", ())
            ]
            for name in names:
                for cand in candidates:
                    score = difflib.SequenceMatcher(None, cand, name).ratio()
                    if score > best_score:
                        best_asset, best_score = str(card["name"]), score
        if best_asset is not None and best_score >= DIFFLIB_CUTOFF:
            return best_asset, float(best_score), "difflib"
        return FALLBACK_ASSET, 0.0, "default"
