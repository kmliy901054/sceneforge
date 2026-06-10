"""sceneforge/assets/library.py — AssetLibrary: cards + builder map (ARCHITECTURE.md §5.1, §12-A).

Loads ``assets/cards/asset_cards.json``, maps ``asset_id -> builder``, and serves
meshes for placement/composition:

- ``get_mesh(asset_id, scale)``: LRU-cached **canonical untinted** mesh; callers
  always receive ``mesh.copy()`` — in-place tinting of a cached mesh would alias
  colors across instances (review fix; test: two mugs, two colors).
- ``footprint_radius(asset_id, scale)``: XY circumradius (max ||(x, y)|| over
  vertices of the canonical mesh), cached — exact bound for §5.2 circle collision.
- ``cards()``: card dicts for RAG grounding (§6.4).
- ``ingest_glb()``: optional additive GLB drop-in hook (cut-order #2).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import trimesh

from sceneforge.assets import builders as _builders

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARDS_PATH = _REPO_ROOT / "assets" / "cards" / "asset_cards.json"
DEFAULT_GLB_DIR = _REPO_ROOT / "assets" / "glb"

_CARD_FIELDS = ("asset_id", "name", "description", "synonyms", "affordances", "nominal_height_m")


class AssetLibrary:
    """Procedural-first asset library: 15 builders + cards, optional GLB ingest (§12-A)."""

    def __init__(
        self,
        cards_path: Path | str = DEFAULT_CARDS_PATH,
        glb_dir: Path | str = DEFAULT_GLB_DIR,
        ingest_glb: bool = False,
    ) -> None:
        self._cards_path = Path(cards_path)
        self._glb_dir = Path(glb_dir)
        raw = json.loads(self._cards_path.read_text(encoding="utf-8"))
        self._cards: Dict[str, dict] = {}
        for card in raw:
            missing = [f for f in _CARD_FIELDS if f not in card]
            if missing:
                raise ValueError(f"asset card {card.get('asset_id', '?')!r} missing fields {missing}")
            self._cards[card["asset_id"]] = card
        self._builders: Dict[str, Callable[[float], trimesh.Trimesh]] = dict(_builders.BUILDERS)
        unbacked = set(self._cards) - set(self._builders)
        if unbacked:
            raise ValueError(f"cards without builders: {sorted(unbacked)}")
        # LRU cache of CANONICAL untinted meshes; get_mesh hands out copies only.
        self._canonical = lru_cache(maxsize=128)(self._build_canonical)
        self._radius_cache: Dict[tuple, float] = {}
        if ingest_glb:
            self.ingest_glb()

    # ------------------------------------------------------------------ API

    def asset_ids(self) -> List[str]:
        """Registered asset ids in stable (builder-table §5.1) order."""
        return [aid for aid in self._builders if aid in self._cards]

    def cards(self) -> List[dict]:
        """All asset cards (shallow copies) for RAG grounding (§6.4)."""
        return [dict(self._cards[aid]) for aid in self.asset_ids()]

    def card(self, asset_id: str) -> dict:
        return dict(self._cards[asset_id])

    def get_mesh(self, asset_id: str, scale: float = 1.0) -> trimesh.Trimesh:
        """Return a COPY of the LRU-cached canonical untinted mesh.

        Callers may tint/transform the result freely — the cached canonical is
        never handed out directly (aliasing review fix, §5.1).
        """
        return self._canonical(asset_id, float(scale)).copy()

    def footprint_radius(self, asset_id: str, scale: float = 1.0) -> float:
        """XY circumradius of the canonical mesh at ``scale`` (cached), for §5.2."""
        key = (asset_id, float(scale))
        if key not in self._radius_cache:
            mesh = self._canonical(asset_id, float(scale))
            self._radius_cache[key] = float(
                np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1).max()
            )
        return self._radius_cache[key]

    # ------------------------------------------------------- GLB ingest hook

    def ingest_glb(self, glb_dir: Path | str | None = None) -> List[str]:
        """Optional drop-in CC0 GLB ingest (additive only, cut-order #2, §12-A).

        For each ``assets/glb/<id>.glb`` with a ``<id>.json`` sidecar card:
        rescale so the resting height equals the card's ``nominal_height_m``,
        recenter XY, floor min-z to 0, then register card + builder. GLBs
        without a sidecar card are skipped. Returns the ingested asset ids.
        """
        directory = Path(glb_dir) if glb_dir is not None else self._glb_dir
        ingested: List[str] = []
        if not directory.is_dir():
            return ingested
        for glb_path in sorted(directory.glob("*.glb")):
            sidecar = glb_path.with_suffix(".json")
            if not sidecar.is_file():
                continue
            card = json.loads(sidecar.read_text(encoding="utf-8"))
            card.setdefault("asset_id", glb_path.stem)
            if any(f not in card for f in _CARD_FIELDS):
                continue  # malformed sidecar: skip silently (additive hook only)
            asset_id = card["asset_id"]
            base = trimesh.load(str(glb_path), force="mesh")
            height = float(base.extents[2])
            if height <= 0:
                continue
            base.apply_scale(float(card["nominal_height_m"]) / height)
            cx, cy = base.bounds.mean(axis=0)[:2]
            base.apply_translation([-cx, -cy, -base.bounds[0, 2]])

            def _builder(scale: float = 1.0, _base: trimesh.Trimesh = base) -> trimesh.Trimesh:
                mesh = _base.copy()
                mesh.apply_scale(float(scale))
                mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
                return mesh

            self._builders[asset_id] = _builder
            self._cards[asset_id] = card
            ingested.append(asset_id)
        return ingested

    # --------------------------------------------------------------- private

    def _build_canonical(self, asset_id: str, scale: float) -> trimesh.Trimesh:
        builder = self._builders.get(asset_id)
        if builder is None:
            raise KeyError(f"unknown asset_id {asset_id!r}; known: {sorted(self._builders)}")
        return builder(scale)
