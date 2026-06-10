# SceneForge — Milestone Task Plan (M1–M6)

Companion to `docs/ARCHITECTURE.md` (v1.1 FINAL). Section references (§) point there.
Rules: M1 is self-contained throwaway code (shares only `compat.py`, `depth_prep.py`, the
`ForgePipeline` L0 recipe, and the minimal OWLv2 scorer with the package). Cut order and
Minimal Lovable Demo are pre-committed in §12.7 — cut from that list only, in that order.

---

## M1 — Day 1 AM: environment repair + quantitative go/no-go (judge mandate #1)

**Self-containment rule:** `scripts/m1_smoketest.py` must NOT import from `sceneforge/`
except `sceneforge/compat.py`, `sceneforge/diffusion/depth_prep.py`,
`sceneforge/diffusion/pipeline.py` (L0 recipe only), and a minimal scorer that is later
promoted to `sceneforge/eval/fidelity.py`. No builders, no placement, no director.

### Tasks
- [ ] **Step 0a — model weights FIRST** (`scripts/m1_smoketest.py::check_weights`):
      `huggingface_hub.snapshot_download` (fp16 `allow_patterns`) for all 5 repos —
      sdxl-base-1.0, controlnet-depth-sdxl-1.0, sdxl-vae-fp16-fix, SDXL-Lightning
      (4step UNet + 4step LoRA), owlv2-base-patch16-ensemble. Print per-repo byte
      totals; fail fast if incomplete. (Verified today: only the VAE is complete;
      ~17 GB outstanding.) Run BEFORE the transformers pin so failures are unambiguous.
- [ ] **Step 0b — transformers pin:** `pip install "transformers>=4.49,<5"` (4.51.3).
      Assert `from transformers import Owlv2ForObjectDetection` and
      `from diffusers import StableDiffusionXLControlNetPipeline` import cleanly.
- [ ] **Step 0c — `sceneforge/compat.py`:** `np.infty = np.inf` shim +
      `os.environ.setdefault("PYOPENGL_PLATFORM", "egl")`. Assert a full **768×768**
      EGL color+depth+SEG render (`RenderFlags.SEG` + `seg_node_map`) succeeds.
      If EGL fails: retry osmesa; if both fail, STOP — implement the numpy rasterizer
      contingency (§5.5) before anything else and cut builders 15→10 to pay for it.
- [ ] **Step 0d — `sceneforge/config.py` skeleton** with `vram.guard_gb = 3.0`,
      `gen.*` placeholders (M1 writes the final values).
- [ ] **Step 1 — inline scenes:** 4 hardcoded scenes from raw trimesh primitives
      (box, can-cylinder, composed mug w/ torus handle, sphere + plate) × 3 seeds =
      12 renders at 768². Inline orbit camera (the §3.1 equations), inline SEG pass,
      inline mask→bbox. Medium/large objects ONLY so ≥ 30 GT instances have
      area_px ≥ 1000 (gate-eligible).
- [ ] **Step 2a — `sceneforge/diffusion/depth_prep.py`:** both modes —
      `disparity` (1/z, percentile [2,98], invalid→0; expected winner, matches the
      ControlNet's MiDaS training distribution) and `linear` (legacy flip). A/B both
      over the 12 scenes at L0; freeze the measured winner as default.
- [ ] **Step 2b — `sceneforge/diffusion/pipeline.py`:** L0 recipe exactly per §7.1
      (Lightning 4-step UNet swap, Euler trailing, cfg=0, NO slicing/tiling).
      Generate 1 image/scene; log s/img and ALL THREE memory numbers per step via
      callback: `max_memory_allocated`, `max_memory_reserved`, min
      `torch.cuda.mem_get_info` free.
- [ ] **Step 3 — minimal OWLv2 scorer** (promoted to `eval/fidelity.py` in M4):
      recall IoU per GT + hallucination count (score ≥ 0.3, IoU < 0.3 vs all GT of
      category) per §9. Score all 12 images. Emit 12-tile overlay contact sheet.
- [ ] **Step 4 — ladder walk on NO-GO:** escalate L0 → L1 → L2 → L3 (→ L4 only if
      L3 fails; requires canny download), re-measure gate + s/img per level.
- [ ] **Step 5 — VRAM mode pick (judge mandate #2):**
      - [ ] Sequential baseline: full unload (`ollama_unload()` incl. embeddinggemma
            via `/api/embed` keep_alive=0), 3 generations, record min device free.
      - [ ] Coresident test: load gemma4:e4b via `/api/generate` with
            `options.num_ctx=4096`, `keep_alive="30m"`; embeddinggemma UNLOADED;
            3 generations; record min device free; afterwards assert
            `/api/ps size_vram == size` for gemma (proves no CPU offload).
      - [ ] Measure `torch.cuda.empty_cache()` effect on nvidia-smi free post-burst;
            time a gemma reload + 300-token generation (expect ~3 s warm).
      - [ ] One timed quality-mode forge-after-forge: `pipe.to("cpu")` → qwen3.5:27b
            call (keep_alive=0) → `pipe.to("cuda")`; record swap + call seconds.
      - [ ] Pick: `mode = "coresident"` iff min device free ≥ `guard_gb` (3.0)
            THROUGHOUT, else `"sequential"` (expected). Write `vram.mode`,
            `gen.level`, `gen.resolution=768`, `gen.depth_mode`, measured
            `s_per_img` into `sceneforge.yaml`.

### Acceptance
- [ ] `outputs/m1/metrics.json`: per-level {match_rate, mean_matched_iou,
      hallucination_rate, s_per_img, alloc/reserved/min-free}, depth-prep A/B result,
      both-mode VRAM table, chosen defaults, gate-eligibility rule (area ≥ 1000) noted.
- [ ] Contact sheet PNG; explicit `GO` / `NO-GO` line printed.
- [ ] **GO gate: `match_rate(IoU≥0.5) ≥ 0.70` AND `mean_matched_iou ≥ 0.65`** over
      gate-eligible GT instances. Hallucination rate reported (informational).
- [ ] If even L3 fails: pivot ruling per §13 recorded in `docs/agent_log.md`.

---

## M2 — Day 1 PM: assets + scene engine + masks (EGL only)

### Tasks
- [ ] `sceneforge/spec.py`: all models per §3.2 + `clamp_to_bounds` helper.
- [ ] `assets/cards/asset_cards.json`: 15 cards {name, description, synonyms,
      affordances, height_m}.
- [ ] `sceneforge/assets/builders.py`: 15 builders per §5.1 table (≤ 6k tris,
      rest-on-z0, pure functions of scale). Kitchen set #1–10 first (cut-order #8
      protects them).
- [ ] `sceneforge/assets/library.py`: cards + builder map, LRU canonical-mesh cache
      returning `mesh.copy()` to callers, `footprint_radius`, optional GLB ingest
      (cut-order #2 — skip if behind).
- [ ] `sceneforge/scene/placement.py`: `resolve_placements` per §5.2 (clamp, circle
      collision ≤ 40 iters, target priority, seeded ties, drop+log overflow).
- [ ] `sceneforge/scene/compose.py`: `build() -> ComposedScene` (mesh-copy + tint +
      transform tuples); `viewer.glb` export — table+objects only, NO floor,
      rotated −90° about X (+Z-up → glTF +Y-up).
- [ ] `sceneforge/render/types.py`: `RenderResult`, `InstanceLabel`, `ComposedScene`.
- [ ] `sceneforge/render/camera.py`: `pose_from_orbit`, `intrinsics`, NORMATIVE
      projection equations in docstring (§3.1).
- [ ] `sceneforge/render/pyrender_backend.py`: manual Scene build (NEVER
      `from_trimesh_scene` — drops node names), persistent OffscreenRenderer,
      color/depth pass + SEG pass with retained Node handles.
- [ ] `sceneforge/render/__init__.py`: `get_renderer` — EGL probe → osmesa retry →
      loud failure pointing at the contingency recipe.
- [ ] `sceneforge/render/numpy_backend.py`: stub raising `NotImplementedError` with
      the implementation recipe + cross-backend equivalence checklist (mask IoU
      ≥ 0.98, depth diff < 2 mm).
- [ ] `sceneforge/labels/masks.py`: `extract_instances` (area ≥ 200 filter, python-int
      casts, ascii RLE). depth16.png writer: uint16 mm, 0 = no-hit.
- [ ] Target-visibility guard helper (used by orchestrator in M4): target present
      with area ≥ 1000 else camera bump (+15° el, ×1.15 dist) + single re-render.
- [ ] `scripts/make_contact_sheet.py`: all 15 assets as color/depth/seg triplets.
- [ ] Tests: `tests/test_spec.py` (clamp_to_bounds incl. nested models, slugify),
      `tests/test_placement.py` (determinism, collision, two-mugs-two-colors tint
      aliasing case), `tests/test_masks.py` (seg→RLE→pycocotools round-trip,
      depth16 round-trip).

### Acceptance
- [ ] `pytest tests/test_spec.py tests/test_placement.py tests/test_masks.py` green.
- [ ] Contact sheet renders all 15 assets.
- [ ] Fixed SceneSpec → byte-identical `seg_ids` across 2 runs (determinism).
- [ ] Target-visibility guard demonstrated on a deliberately occluded layout.

---

## M3 — Day 2 AM: director + RAG + clamp/repair + fallbacks

### Tasks
- [ ] `sceneforge/director/ollama_client.py`: `chat_structured` (format=schema,
      explicit `keep_alive="5m"`, `num_ctx=4096` standardized, seed, `on_token`
      streaming callback, `DirectorUnavailable`), `embed` (default `keep_alive=0` —
      server runs `OLLAMA_KEEP_ALIVE=-1`).
- [ ] `sceneforge/director/director.py`: `PLANNER_SYSTEM` + few-shot, `STYLER_SYSTEM`
      (category-FIRST clause, ≤ 40 words), `plan_scene` with clamp→validate→repair
      (≤ 2 rounds, structural errors only), `make_styles`, `ground_plan` (RAG fill,
      pin look_at/yfov, instance ids).
- [ ] Style post-validation (§6.6): slugify + uniqueness, identity-word blacklist
      strip, CLIP-77 tokenizer check, category-first deterministic rebuild, token
      counts → run_meta, rewrites → grounding_log.
- [ ] `sceneforge/director/rag.py`: `EmbeddingIndex` (document/query prefixes,
      npz cache keyed by sha256, 0.35 floor, difflib fallback).
- [ ] `sceneforge/director/fallback.py`: `template_plan` (golden-angle ring),
      `DEFAULT_STYLES` (4 hardcoded, category-first with `{objects}` placeholder).
- [ ] `scripts/build_rag_index.py`; build and **commit** `assets/index/embed_cache.npz`.
- [ ] Runtime transcript logger → `outputs/runs/<id>/director_log.md` (NOT
      docs/agent_log.md).
- [ ] `tests/test_rag.py` (mocked embeddings + difflib path); repair-loop fixtures in
      `tests/test_spec.py` (incl. an out-of-bounds-numerics fixture proving clamp
      avoids a repair round).

### Acceptance
- [ ] 10 canned tasks → 10/10 valid `SceneSpec` via **LLM-or-fallback, zero
      exceptions**; ≥ 8/10 via the LLM path with ≤ 2 repair rounds (transcripts kept).
- [ ] Grounding hits expected asset for 10 scripted phrases ("something to drink
      from" → mug/cup/bottle).
- [ ] **With Ollama stopped:** full plan+style path completes via fallback, zero
      exceptions, `source="fallback"` surfaced.
- [ ] `embed_cache.npz` committed; rebuild only on cards-hash change verified.

---

## M4 — Day 2 PM: diffusion + labels + COCO + fidelity wiring (moved up from M6)

### Tasks
- [ ] `sceneforge/gpu.py`: `snapshot()` (mem_get_info + reserved + /api/ps),
      `ollama_unload()` (per-model `/api/generate` keep_alive=0; on HTTP 400 →
      `/api/embed {"input": "", "keep_alive": 0}`; poll /api/ps until EMPTY, 15 s
      timeout), `phase()` ctx manager: "spec" (gc + empty_cache on entry),
      "spec_quality" (pipe.to cpu/cuda swap §10.4), "diffusion" (unload barrier),
      "eval" (unload both modes). guard_gb constant from config.
- [ ] `sceneforge/diffusion/pipeline.py` (promote M1 code): ladder levels L0–L4
      behind `load(level)`, OOM recovery sequence §7.5 (to-cpu → gc → empty_cache →
      enable_model_cpu_offload → retry once → 640 px; force sequential for session).
- [ ] `sceneforge/eval/fidelity.py` (promote M1 scorer): `Owlv2Scorer.score_batch`,
      fidelity_adj = fidelity − 0.5·FP_rate, quarantine below `keep_threshold=0.45`,
      gate-eligibility (area ≥ 1000) split in the report.
- [ ] `sceneforge/labels/overlay.py`: 4 modes, palette, target starred; raw + overlay
      PNGs written at generation time.
- [ ] `sceneforge/labels/coco.py`: `CocoWriter` — global annotation-id counter,
      explicit width/height per image, python-int casts, `"sceneforge"` provenance
      key, zip export.
- [ ] `sceneforge/orchestrator.py`: `ForgeRun.run()` generator with demo-paced order
      (plan stream → spec → layout-0 GLB+control → styles → remaining layouts →
      diffusion layout-major → fidelity → done); target-visibility guard per layout;
      seed law `spec.seed + layout_idx*1000 + style_idx`; per-phase VRAM snapshots
      into `run_meta.json`. `ForgeRun.reforge()` fast path (no LLM).
- [ ] `tests/test_gpu.py`: OOM-recovery state transitions with mocked
      `torch.cuda.OutOfMemoryError`; ollama_unload 400-fallback with mocked HTTP.
- [ ] Extend `tests/test_masks.py`: synthetic 2-layout × 2-style export →
      `COCO(annotations.json)` loads, `annToMask` round-trips, unique ann ids,
      `json.dumps` succeeds.

### Acceptance
- [ ] 2 layouts × 4 styles end-to-end (sequential mode, incl. gemma reload) within
      `8 × s_img(M1 level) + 60 s` (≈ ≤ 90 s at L0).
- [ ] `dataset.zip` downloadable; COCO acceptance tests green.
- [ ] Fidelity report (recall + hallucination) produced per run; quarantine list in
      `fidelity.json`.
- [ ] `pytest tests/` fully green.

---

## M5 — Day 3: Gradio UI + re-forge + hardening + backup video

### Tasks
- [ ] `sceneforge/ui/blocks.py`: full layout per §11 (task row, sliders, status line,
      Model3D + control image, Re-Forge accordion, plan_stream `gr.Code`, spec JSON,
      gallery `columns=n_styles`, overlay radio, fidelity label, export/download,
      footer + `gr.Timer(2.0)`); verify exact 6.17.3 kwarg names at build time.
- [ ] `sceneforge/ui/handlers.py`: `on_forge` generator (plan_token streaming —
      fallback: typewriter-replay of validated JSON; layout-major assertion;
      stage-status updates on every event), `on_reforge` (AnnotatedImage with masks
      tracking camera), nudge `.then()` chain, `on_toggle` (cached-path swap),
      `on_export`.
- [ ] `app.py`: compat import line 1; pre-warm sequence §10.5 (gemma load/unload,
      eager ForgePipeline, OWLv2 to CPU) with "warming up" footer banner;
      `launch(server_name="0.0.0.0", server_port=7860)`.
- [ ] `scripts/demo_prep.sh`: pre-warm + one throwaway forge.
- [ ] Hardening: kill-Ollama-mid-demo drill (fallback path, visible source flag);
      one quality-mode forge-after-forge run (timed); 20-task soak (cut-order #10:
      reduce to 10 if behind).
- [ ] **Record rough backup demo video** — the scripted happy path IS the video;
      storyboard live-forge segment at 1 layout × 4 styles (~25 s), 2×4 grid as
      b-roll from a golden run. Commit a golden run directory for replay fallback.

### Acceptance
- [ ] Scripted 2-minute happy path runs 3× consecutively without intervention.
- [ ] Re-forge slider-to-image ≤ 6 s; labels visibly track camera in AnnotatedImage.
- [ ] First visual (layout-0 GLB + control) ≤ ~5 s after FORGE (warm state); UI never
      silent (status line always advancing).
- [ ] Kill-Ollama drill passes; soak completes without crash; backup video exists.

---

## M6 — Day 4: eval polish + export run + deliverables

### Tasks
- [ ] Quarantine UI: `include_q_cb` wiring, fidelity meter shows
      `fidelity · halluc rate`; threshold polish from real-run distribution.
- [ ] 100+ image export run; spot-check 10 annotations by eye against overlays.
- [ ] `README.md`: architecture diagram (§1), quickstart, M1 metrics table, honest-eval
      caveat (§9 lower-bound note), demo gif/screenshots.
- [ ] `requirements.txt` frozen from working env (pins per §13).
- [ ] `docs/agent_log.md`: build-process collaboration narrative (absorb existing
      `docs/WORKFLOW_LOG.md`); cite ONE runtime `director_log.md` transcript as an
      appendix exhibit.
- [ ] Final demo video: run `scripts/demo_prep.sh` + one throwaway forge FIRST (warm
      page cache + CUDA), then record: forge → grid fills → overlay wipe → re-forge
      camera nudge with labels tracking → export → fidelity meter → pi0/OpenVLA
      closing line.
- [ ] Repo hygiene: `.gitignore` outputs (except golden run + m1 metrics), license
      note for any GLB drop-ins actually shipped.

### Acceptance
- [ ] All course deliverables present: GitHub repo, README w/ architecture,
      requirements.txt, agent-collaboration log, demo video/screenshots, interactive
      Gradio UI.
- [ ] `pytest tests/` green; `python app.py` cold-boots to a working UI with the
      warming banner; golden-run replay loads with Ollama stopped.
