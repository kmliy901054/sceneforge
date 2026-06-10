"""tests/test_rag.py — EmbeddingIndex grounding paths (ARCHITECTURE.md §6.4, §12-D).

All embeddings are MOCKED (no Ollama): a keyword→axis embed_fn exercises the
cosine path; raising DirectorUnavailable exercises the difflib path; gibberish
exercises the 0.35-floor → difflib → "box" default. Plus npz-cache round-trip
keyed by the cards-file sha256.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0)

import json

import numpy as np
import pytest

from sceneforge.director.ollama_client import DirectorUnavailable
from sceneforge.director.rag import (
    DIFFLIB_CUTOFF,
    SCORE_FLOOR,
    EmbeddingIndex,
    cards_sha256,
    document_text,
    load_cache,
    query_text,
    save_cache,
)

CARDS = [
    {
        "name": "mug",
        "description": "a ceramic coffee mug with a handle",
        "synonyms": ["coffee mug", "coffee cup"],
        "affordances": ["drinking", "pick and place"],
        "height_m": 0.095,
    },
    {
        "name": "bottle",
        "description": "a plastic water bottle with a cap",
        "synonyms": ["water bottle", "flask"],
        "affordances": ["drinking", "pouring"],
        "height_m": 0.24,
    },
    {
        "name": "book",
        "description": "a paperback book",
        "synonyms": ["novel", "paperback"],
        "affordances": ["reading"],
        "height_m": 0.03,
    },
    {
        "name": "box",
        "description": "a plain cardboard box",
        "synonyms": ["carton", "package"],
        "affordances": ["container"],
        "height_m": 0.08,
    },
]

DIM = 8
_KEYWORD_AXIS = {"mug": 0, "bottle": 1, "book": 2, "box": 3}


def keyword_embed(texts):
    """Deterministic mock embed_fn: a keyword lights up its axis; text with no
    keyword maps to a far-off axis (cosine ≈ 0 vs every document)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        low = text.lower()
        hit = False
        for word, axis in _KEYWORD_AXIS.items():
            if word in low:
                out[i, axis] += 1.0
                hit = True
        if not hit:
            out[i, DIM - 1] = 1.0
    return out


@pytest.fixture()
def index() -> EmbeddingIndex:
    return EmbeddingIndex.build(CARDS, keyword_embed)


# ----------------------------------------------------------- document/query text
def test_asymmetric_prefixes():
    doc = document_text(CARDS[0])
    assert doc.startswith("title: mug | text: a ceramic coffee mug")
    assert "Synonyms: coffee mug, coffee cup." in doc
    assert "Used for: drinking, pick and place." in doc
    q = query_text("red ceramic mug")
    assert q == "task: search result | query: red ceramic mug"
    assert not q.startswith("title:")  # asymmetric (§6.4)


def test_build_normalizes_vectors(index: EmbeddingIndex):
    assert index.vectors is not None
    assert index.vectors.dtype == np.float32
    assert index.vectors.shape == (len(CARDS), DIM)
    np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-6)


# ------------------------------------------------------------- grounding paths
class TestGroundEmbedding:
    def test_embedding_hit(self, index: EmbeddingIndex):
        asset_id, score, method = index.ground("red ceramic mug")
        assert asset_id == "mug"
        assert method == "embedding"
        assert score >= SCORE_FLOOR

    def test_each_card_self_grounds(self, index: EmbeddingIndex):
        for card in CARDS:
            asset_id, _, method = index.ground(f"a {card['name']} on the table")
            assert (asset_id, method) == (card["name"], "embedding")

    def test_low_cosine_falls_to_difflib(self, index: EmbeddingIndex):
        # "bottl" (typo) has no keyword axis → cosine ≈ 0 < 0.35 floor, but
        # difflib vs the "bottle" synonym is well above the 0.6 cutoff.
        asset_id, score, method = index.ground("bottl of soda")
        assert asset_id == "bottle"
        assert method == "difflib"
        assert score >= DIFFLIB_CUTOFF

    def test_nonsense_routes_to_box(self, index: EmbeddingIndex):
        asset_id, score, method = index.ground("xyzzy frobnicator")
        assert (asset_id, score, method) == ("box", 0.0, "default")


class TestGroundDifflib:
    def test_embed_failure_goes_straight_to_difflib(self):
        def broken_embed(texts):
            raise DirectorUnavailable("ollama stopped")

        # Vectors exist (cache hit) but query-time embed fails → difflib (§6.4).
        vectors = keyword_embed([document_text(c) for c in CARDS])
        idx = EmbeddingIndex(CARDS, vectors=vectors, embed_fn=broken_embed)
        asset_id, score, method = idx.ground("mug")
        assert (asset_id, method) == ("mug", "difflib")
        assert score == pytest.approx(1.0)

    def test_no_vectors_no_embed_fn_uses_difflib(self):
        idx = EmbeddingIndex(CARDS)  # cold demo, no cache, no Ollama
        asset_id, _, method = idx.ground("a novel to read")
        assert (asset_id, method) == ("book", "difflib")

    def test_difflib_matches_synonym_not_just_name(self):
        idx = EmbeddingIndex(CARDS)
        asset_id, _, method = idx.ground("a flask of tea")
        assert (asset_id, method) == ("bottle", "difflib")

    def test_difflib_nonsense_routes_to_box(self):
        idx = EmbeddingIndex(CARDS)
        asset_id, score, method = idx.ground("qwghlm zzyzx")
        assert (asset_id, score, method) == ("box", 0.0, "default")


# --------------------------------------------------------------- npz cache
class TestCache:
    def test_round_trip_keyed_by_sha(self, tmp_path, index: EmbeddingIndex):
        cards_file = tmp_path / "asset_cards.json"
        cards_file.write_text(json.dumps(CARDS), encoding="utf-8")
        sha = cards_sha256(cards_file)
        cache = tmp_path / "embed_cache.npz"

        save_cache(cache, index.vectors, index.asset_ids, sha)
        loaded = load_cache(cache, sha)
        assert loaded is not None
        vectors, asset_ids = loaded
        np.testing.assert_array_equal(vectors, index.vectors)
        assert asset_ids == [c["name"] for c in CARDS]

    def test_stale_sha_returns_none(self, tmp_path, index: EmbeddingIndex):
        cache = tmp_path / "embed_cache.npz"
        save_cache(cache, index.vectors, index.asset_ids, "deadbeef")
        assert load_cache(cache, "0123abcd") is None

    def test_missing_cache_returns_none(self, tmp_path):
        assert load_cache(tmp_path / "absent.npz", "deadbeef") is None

    def test_load_or_build_uses_cache_then_rebuilds_on_change(self, tmp_path):
        cards_file = tmp_path / "asset_cards.json"
        cards_file.write_text(json.dumps(CARDS), encoding="utf-8")
        cache = tmp_path / "embed_cache.npz"
        calls = {"n": 0}

        def counting_embed(texts):
            calls["n"] += 1
            return keyword_embed(texts)

        idx1 = EmbeddingIndex.load_or_build(cards_file, cache, counting_embed)
        assert idx1.vectors is not None and calls["n"] == 1
        idx2 = EmbeddingIndex.load_or_build(cards_file, cache, counting_embed)
        assert calls["n"] == 1                       # cache hit — no re-embed
        np.testing.assert_array_equal(idx1.vectors, idx2.vectors)

        # Cards file changes → sha changes → rebuild.
        cards_file.write_text(json.dumps(CARDS + [
            {"name": "plate", "description": "a dinner plate", "synonyms": ["dish"],
             "affordances": ["serving"], "height_m": 0.02}
        ]), encoding="utf-8")
        idx3 = EmbeddingIndex.load_or_build(cards_file, cache, counting_embed)
        assert calls["n"] == 2
        assert len(idx3.asset_ids) == len(CARDS) + 1

    def test_load_or_build_without_embed_fn_degrades_to_difflib(self, tmp_path):
        cards_file = tmp_path / "asset_cards.json"
        cards_file.write_text(json.dumps(CARDS), encoding="utf-8")
        idx = EmbeddingIndex.load_or_build(cards_file, tmp_path / "none.npz", None)
        assert idx.vectors is None
        assert idx.ground("mug")[2] == "difflib"     # still functional, no Ollama
