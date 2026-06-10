#!/usr/bin/env python
"""scripts/build_rag_index.py — (re)build the committed RAG embed cache (§6.4, §12-D).

Embeds every asset card's document string with embeddinggemma via Ollama
``/api/embed`` (ALWAYS ``keep_alive=0`` — the server runs OLLAMA_KEEP_ALIVE=-1)
and writes the L2-normalized matrix to ``assets/index/embed_cache.npz``, keyed
by the sha256 of the cards file. The cache is COMMITTED so cold demos never
need the embed model. Skips the rebuild when the cache is already current
(pass --force to override).

Usage:
    python scripts/build_rag_index.py [--force] [--config sceneforge.yaml]
"""
import sceneforge.compat  # noqa: F401  — import FIRST (§0)

import argparse
import json
import sys
from functools import partial

import numpy as np

from sceneforge.config import load_config
from sceneforge.director.ollama_client import DirectorUnavailable, embed
from sceneforge.director.rag import (
    EmbeddingIndex,
    cards_sha256,
    load_cache,
    save_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the cache sha is current")
    parser.add_argument("--config", default=None, help="path to sceneforge.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cards_path = cfg.paths.cards_file
    cache_path = cfg.paths.embed_cache
    if not cards_path.is_file():
        print(f"ERROR: cards file not found: {cards_path}", file=sys.stderr)
        return 2

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    sha = cards_sha256(cards_path)
    print(f"cards: {cards_path} ({len(cards)} cards, sha256 {sha[:12]}…)")

    if not args.force and load_cache(cache_path, sha) is not None:
        print(f"cache current: {cache_path} — nothing to do (use --force to rebuild)")
        return 0

    # keep_alive=0 on every embed call: embeddinggemma must never stay pinned.
    embed_fn = partial(embed, cfg.ollama.embed_model, keep_alive=0,
                       host=cfg.ollama.host)
    try:
        index = EmbeddingIndex.build(cards, embed_fn)
    except DirectorUnavailable as exc:
        print(f"ERROR: embed model unavailable: {exc}", file=sys.stderr)
        return 1

    assert index.vectors is not None
    norms = np.linalg.norm(index.vectors, axis=1)
    out = save_cache(cache_path, index.vectors, index.asset_ids, sha)
    print(f"wrote {out}: vectors {index.vectors.shape} float32, "
          f"|v| in [{norms.min():.4f}, {norms.max():.4f}]")
    print("COMMIT this file (assets/index/embed_cache.npz) — cold demos depend on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
