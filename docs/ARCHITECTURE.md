# SceneForge — Final Architecture (v1.1 FINAL, 2026-06-10)

Lead-architect document, post-adversarial-review. Every decision here is settled; implementers should not re-derive them. All paths are absolute under repo root `/home/pairlab/DGAN/sceneforge`. Changes vs v1.0 incorporate three adversarial reviews; disputes between reviewers are resolved in §12.6.

**Environment facts verified on this machine (2026-06-10):** Python 3.11 (conda env `dgan`), torch 2.6.0+cu124 (CUDA OK), diffusers 0.38.0, **transformers 5.10.2 (BROKEN with torch 2.6 — P0 below)**, gradio 6.17.3, pyrender 0.1.45 (`RenderFlags.SEG`=8192 present; `render(..., seg_node_map=)` supported; SEG pass disables `GL_MULTISAMPLE` so red-channel IDs round-trip exactly; **`Scene.from_trimesh_scene` drops node names — never use it, see §5.4**), trimesh 4.12.2, pydantic 2.13.4, numpy 2.4.4, opencv 4.13.0, pycocotools 2.0.11, ollama-python 0.6.2 (`chat()` accepts `format=` JSON-schema dict, `keep_alive=`, `stream=True`), Ollama server 0.20.3 with **`OLLAMA_KEEP_ALIVE=-1` in its systemd unit (verified — any request omitting `keep_alive` pins its model in VRAM forever)**. RTX 3090 24 GB, 125 GB RAM, 236 GB free disk.

**Measured VRAM/latency facts (live, this machine):** desktop 0.82 GB · gemma4:e4b @ num_ctx 8192 = **10.88 GB** resident (~10.3 GB @ 4096) · embeddinggemma resident = **1.11 GB** (currently pinned with `expires_at` year 2318 due to server keep_alive=-1) · gemma warm reload **3.0 s**, cold load **33.4 s** · planning throughput 139 tok/s · `keep_alive:0` unload request returns in 5 ms, `/api/ps` clears and VRAM frees within 2 s · `/api/generate` returns **HTTP 400 for embedding models** (must unload them via `/api/embed`) · **Ollama 0.20.3 grammar-constrains JSON structure/types but NOT numeric min/max** (verified: first planner response emitted `elevation_deg: 0` against `ge=10`) · diffusers 0.38: `do_classifier_free_guidance = guidance_scale > 1`, so `negative_prompt` is dead code at Lightning's `guidance_scale=0`.

---

## 0) P0 environment repairs (M1 step 0 — before any other code)

0. **Model weights are NOT all on disk yet** (verified: only `sdxl-vae-fp16-fix` complete; `controlnet-depth-sdxl-1.0` is config-only; SDXL-base, SDXL-Lightning, OWLv2 absent ≈ 17 GB to go). M1 step 0 runs `huggingface_hub.snapshot_download(repo, allow_patterns=[fp16 variant files])` for all **5 repos** (idempotent verify+resume), prints per-repo byte totals, and **fails fast** if incomplete. Run this BEFORE the transformers pin so failures are unambiguous.
1. `pip install "transformers>=4.49,<5"` (pin 4.51.3). transformers 5.10.2 calls `torch.float8_e8m0fnu` (needs torch≥2.7) → `Owlv2ForObjectDetection` and `StableDiffusionXLControlNetPipeline` imports both crash today (verified). 4.51.x has Owlv2 and works with diffusers 0.38 + torch 2.6. Do NOT upgrade torch.
2. pyrender 0.1.45 uses `np.infty` (`pyrender/mesh.py:87`; removed in numpy 2.0). Fix via shim, NOT a numpy downgrade: `sceneforge/compat.py` runs `np.infty = np.inf` *before* the first `import pyrender` (verified working).
3. `PYOPENGL_PLATFORM=egl` set in `compat.py` via `os.environ.setdefault` before pyrender import. M1 step 0 asserts a full **768×768** EGL color+depth+SEG render (not just 64×64).

Import rule: **`import sceneforge.compat` is line 1 of `app.py` and of every script.**

---

## 1) System overview

SceneForge turns one natural-language robot-manipulation task into a labeled, photoreal, domain-randomized COCO dataset. The trick: a local LLM plans a 3D scene as validated JSON; a deterministic headless renderer owns the geometry (depth + pixel-perfect instance masks); depth-ControlNet SDXL-Lightning re-textures that exact geometry into N style worlds; because geometry is preserved, the renderer's masks/bboxes transfer to every variant for free; OWLv2 spot-checks that claim per image (recall AND hallucination) and auto-filters drifters.

**Pipeline order is tuned for demo pacing** (first visual ≤ ~5 s after FORGE): plan (tokens streamed to UI) → clamp+validate → ground+place → **render layout 0 and emit GLB + depth control immediately** → style LLM call → render remaining layouts → diffusion (grid fills live) → fidelity.

```
 user task: "pick the red mug from a cluttered kitchen table"
        │
        ▼
┌─────────────────────────────  DIRECTOR (CPU+Ollama)  ─────────────────────────────┐
│ gemma4:e4b chat(format=json_schema, stream→UI)  ──►  raw dict                      │
│      │ clamp_to_bounds() (Ollama does NOT enforce numeric ranges — verified)       │
│      ▼                                                                             │
│ LLMScenePlan (pydantic) · ValidationError? → repair loop (structural only, ≤2)     │
│      │                                       → deterministic template fallback     │
│      ▼                                                                             │
│ RAG grounding: embeddinggemma /api/embed (keep_alive=0) → numpy cosine over cards  │
│ + style synthesis (2nd LLM call) → slugify names, enforce category-first prompts,  │
│   CLIP-77-token budget check (hardcoded 4-style fallback)                          │
└──────────────────────┬─────────────────────────────────────────────────────────────┘
                       ▼  SceneSpec (THE contract)
┌─────────────────────────────  SCENE ENGINE (CPU+EGL)  ─────────────────────────────┐
│ procedural trimesh builders → placement (snap + 2D collision resolve)              │
│ → pyrender EGL (manually-built Scene; node handles kept for seg_node_map):         │
│   depth z-buffer + SEG instance pass → target-visibility check (auto camera bump)  │
│ outputs/layout: depth16.png(mm) · seg ids · {mask RLE, bbox} · viewer.glb (+Y-up,  │
│                 no floor) · control.png 768×768 (MiDaS-style disparity)            │
│ contingency only: osmesa retry → golden-run replay → numpy rasterizer (stub)       │
└──────────────────────┬─────────────────────────────────────────────────────────────┘
                       ▼  RenderResult
┌────────────────────────────  DIFFUSION (GPU, phase-gated)  ────────────────────────┐
│ gpu.phase("diffusion"): ollama_unload() sweeps /api/ps incl. embed models          │
│ SDXL-base fp16 + controlnet-depth fp16 + vae-fp16-fix + Lightning 4-step UNet      │
│ Euler trailing, cfg=0, cond_scale=0.85, 768×768, no slicing/tiling by default      │
│ layouts × styles, layout-major order → images (~2-2.5 s each at L0)                │
│ OOM → exact recovery sequence (§7.5) → cpu_offload → 640px                         │
└──────────────────────┬─────────────────────────────────────────────────────────────┘
                       ▼  images share the layout's labels (label transfer is a no-op)
┌──────────────  LABELS & EVAL  ──────────────┐   ┌──────────────  UI (Gradio 6)  ───────────┐
│ overlay renderer (bbox+mask wipes)          │   │ task box · FORGE · streamed spec ·       │
│ COCO writer (RLE, global ann ids) + zip     │   │ Model3D viewer · style gallery (rows=    │
│ OWLv2: recall IoU vs GT + hallucination     │   │ layout, cols=style) · overlay toggle ·   │
│ count → fidelity_adj → auto-quarantine      │   │ re-forge sliders (cam/seed) · Export ·   │
└─────────────────────────────────────────────┘   │ stage-status + VRAM footer               │
                                                  └──────────────────────────────────────────┘
```

Run artifacts live in `outputs/runs/<run_id>/` (§4.7). Every LLM-touching path has a deterministic fallback; the demo completes end-to-end with the Ollama server stopped.

---

## 2) Module decomposition — exact file tree

```
/home/pairlab/DGAN/sceneforge/
├── README.md                      # architecture diagram, quickstart, M1 metrics, demo gif
├── requirements.txt               # exact pins (frozen at M6)
├── sceneforge.yaml                # runtime config (vram mode + gen defaults written by M1)
├── app.py                         # entry: python app.py → gradio on :7860 (runs pre-warm §10.5)
├── docs/
│   ├── proposal_sceneforge.json   # (exists)
│   ├── ARCHITECTURE.md            # this document
│   ├── TASKS.md                   # M1–M6 milestone checklist
│   └── agent_log.md               # BUILD-process collaboration narrative (course deliverable;
│                                  #   absorbs existing WORKFLOW_LOG.md; cites ONE runtime
│                                  #   director transcript as an appendix exhibit)
├── assets/
│   ├── cards/asset_cards.json     # 15 asset cards (§12-A)
│   ├── glb/                       # OPTIONAL drop-in CC0 GLBs (auto-ingested; cut-order #2)
│   └── index/embed_cache.npz      # embeddinggemma vectors, keyed by sha256 of cards file (COMMITTED)
├── outputs/runs/<run_id>/         # per-run artifacts incl. director_log.md (§4.7)
├── scripts/
│   ├── m1_smoketest.py            # SELF-CONTAINED go/no-go harness (§13-M1) — throwaway code
│   ├── build_rag_index.py         # (re)build embed cache
│   ├── make_contact_sheet.py      # 15 builders rendered, M2 acceptance artifact
│   └── demo_prep.sh               # pre-warm gemma + SDXL + one throwaway forge before recording
├── sceneforge/                    # python package
│   ├── __init__.py
│   ├── compat.py                  # np.infty shim + PYOPENGL_PLATFORM=egl; import FIRST
│   ├── config.py                  # AppConfig (pydantic) ← sceneforge.yaml + env overrides
│   ├── spec.py                    # LLM-facing + resolved pydantic models, clamp_to_bounds (§3)
│   ├── orchestrator.py            # ForgeRun: pipeline (demo-paced order §1), progress events
│   ├── gpu.py                     # snapshot(), ollama_unload(), phase() ctx mgr, OOM ladder (§10)
│   ├── assets/
│   │   ├── builders.py            # 15 procedural trimesh builders (§5.1)
│   │   └── library.py             # AssetLibrary: cards + builders + optional GLB ingest
│   ├── director/
│   │   ├── ollama_client.py       # chat_structured(stream cb), embed(keep_alive=0) (§6.1)
│   │   ├── rag.py                 # EmbeddingIndex: numpy cosine + difflib fallback (§6.4)
│   │   ├── director.py            # plan_scene(), make_styles(), clamp+repair (§6), style post-
│   │   │                          #   validation (slug/categories/CLIP-77) (§6.6)
│   │   └── fallback.py            # template_plan(), DEFAULT_STYLES (§6.5)
│   ├── scene/
│   │   ├── placement.py           # resolve_placements(): snap + collision (§5.2)
│   │   └── compose.py             # SceneSpec → ComposedScene + viewer.glb export (§5.3)
│   ├── render/
│   │   ├── __init__.py            # get_renderer(cfg): EGL → osmesa retry → loud failure
│   │   ├── types.py               # RenderResult, InstanceLabel, ComposedScene (§4.3)
│   │   ├── camera.py              # orbit pose + intrinsics + NORMATIVE projection eqns (§3.1)
│   │   ├── pyrender_backend.py    # EGL color/depth/SEG, manual Scene build (§5.4)
│   │   └── numpy_backend.py       # CONTINGENCY STUB (NotImplementedError + recipe) (§5.5)
│   ├── diffusion/
│   │   ├── depth_prep.py          # z-buffer → disparity control image (§7.3) [written in M1]
│   │   └── pipeline.py            # ForgePipeline load/generate/unload/OOM ladder (§7) [M1 seeds L0]
│   ├── labels/
│   │   ├── masks.py               # seg ids → masks/bboxes/RLE, int casts (§8.1)
│   │   ├── overlay.py             # draw_overlay(img, instances, mode) (§8.2)
│   │   └── coco.py                # CocoWriter (global ann ids) + export_zip (§8.3)
│   ├── eval/
│   │   └── fidelity.py            # Owlv2Scorer: recall + hallucination (§9) [seeded in M1]
│   └── ui/
│       ├── blocks.py              # build_app() → gr.Blocks (§11)
│       └── handlers.py            # on_forge / on_reforge / on_toggle / on_export
└── tests/
    ├── test_spec.py               # clamp_to_bounds, slugify, repair-loop fixtures
    ├── test_placement.py          # collision/snap determinism + two-mugs-two-colors tint test
    ├── test_masks.py              # seg→RLE round-trip, depth16 round-trip, COCO ids/json-able
    ├── test_rag.py                # grounding with mocked embeddings + difflib path
    └── test_gpu.py                # OOM-recovery state transitions with mocked OutOfMemoryError
```

---

## 3) SceneSpec schema (pydantic v2) + coordinate conventions

### 3.1 Conventions (normative)

- **World frame:** right-handed, **+Z up**. Units **meters**, angles **degrees**.
- **Table:** TOP surface is plane **z = 0**; tabletop spans `x ∈ [-w/2, +w/2]`, `y ∈ [-d/2, +d/2]`, default 1.2 × 0.8 m, slab thickness 0.04 m. A floor plane at z = −0.75 (2.5×2.5 m) is added **for depth-map rendering only** (fills background with far depth) — it is **excluded from viewer.glb** (§5.3). Objects rest ON the table: after placement, `min(vertices.z) == 0`.
- **Object pose:** `(x_m, y_m)` on tabletop, `yaw_deg` CCW about +Z. No tilt/roll.
- **Camera (orbit):** `azimuth_deg` CCW from +X around +Z; `elevation_deg` above z=0 plane; `distance_m` from `look_at`. `eye = look_at + d·[cos el·cos az, cos el·sin az, sin el]`. Pyrender convention (camera looks down its −Z, up ≈ +Z world): `z_axis = normalize(eye − look_at)`, `x_axis = normalize(cross([0,0,1], z_axis))`, `y_axis = cross(z_axis, x_axis)`; pose columns `[x_axis, y_axis, z_axis, eye]`. Elevation clamped `[10, 80]`°. **`look_at` and `yfov` are SERVER-PINNED, never LLM-settable** (review fix: a stray look_at empties every mask).
- **Intrinsics:** square images, `yfov_deg = 50` → `fx = fy = H / (2·tan(yfov/2))`, `cx = W/2`, `cy = H/2`.
- **Projection equations (NORMATIVE, in `camera.py` docstring — the classic cross-backend mirror bug lives here):** for camera-frame point `(x_c, y_c, z_c)` with `z_c < 0` in front of the camera: `u = cx + fx·x_c/(−z_c)`, `v = cy − fy·y_c/(−z_c)`. Any alternative backend must reproduce these exactly (acceptance: per-instance mask IoU ≥ 0.98 and mean |depth diff| < 2 mm vs the EGL backend — part of the contingency-activation checklist, §5.5).
- **Image frame:** COCO standard: origin top-left, bbox `[x, y, w, h]` pixels.
- **depth16.png encoding (on-disk contract):** uint16 **millimeters**, `depth_mm = round(depth_m·1000).clip(0, 65535)`, **0 = no-hit**; max representable 65.5 m. Round-trip asserted in `test_masks.py`.

### 3.2 Schema code (`sceneforge/spec.py`)

Three layers: **LLM-facing models** (strict, all-required, minimal — Ollama's `format=` constrains structure/types only), `clamp_to_bounds` (numeric sanitation Ollama can't do), and **resolved contract** models.

```python
"""sceneforge/spec.py — THE data contract. Do not change field names casually.

VERIFIED ON THIS MACHINE: Ollama 0.20.3 format=json_schema enforces structure and
types but NOT numeric minimum/maximum (first live response violated ge=10).
Therefore: clamp_to_bounds() runs on the raw dict BEFORE model_validate(); the
repair loop handles ONLY structural/semantic failures (missing fields,
exactly_one_target, list lengths)."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# Active only at ladder levels with guidance_scale > 1 (L2/L3) — diffusers turns
# CFG on iff guidance_scale > 1; at Lightning cfg=0 negative prompts are dead code.
NEGATIVE_PROMPT = "blurry, deformed, duplicate objects, lowres, watermark, cartoon, painting"

class LLMCameraSpec(BaseModel):                  # what the LLM may control
    azimuth_deg: float = Field(35.0, ge=-180, le=180)
    elevation_deg: float = Field(30.0, ge=10, le=80)
    distance_m: float = Field(1.1, ge=0.5, le=2.5)

class CameraSpec(LLMCameraSpec):                 # resolved: server-pinned fields
    yfov_deg: float = Field(50.0, ge=25, le=80)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.05)

class TableSpec(BaseModel):
    width_m: float = Field(1.2, ge=0.6, le=2.0)
    depth_m: float = Field(0.8, ge=0.5, le=1.5)

# ---------- what the LLM emits (all-required for format=json_schema) ----------
class PlannedObject(BaseModel):
    description: str = Field(..., min_length=2, max_length=80)   # "red ceramic mug" — RAG input
    x_m: float = Field(..., ge=-1.0, le=1.0)
    y_m: float = Field(..., ge=-0.75, le=0.75)
    yaw_deg: float = Field(..., ge=-180, le=180)
    size: Literal["small", "medium", "large"]                    # → scale 0.85/1.0/1.2
    is_target: bool

class LLMScenePlan(BaseModel):
    scene_summary: str = Field(..., max_length=200)
    objects: list[PlannedObject] = Field(..., min_length=2, max_length=8)
    camera: LLMCameraSpec

    @field_validator("objects")
    @classmethod
    def exactly_one_target(cls, v):
        if sum(o.is_target for o in v) != 1:
            raise ValueError("exactly one object must have is_target=true")
        return v

class LLMStyle(BaseModel):                       # NO negative_prompt — dead at L0/L1,
    name: str = Field(..., min_length=2, max_length=40)   # wasted styler tokens (review fix)
    prompt: str = Field(..., min_length=10, max_length=250)  # 250 chars ≈ CLIP-77 budget

class StyleSet(BaseModel):
    styles: list[LLMStyle] = Field(..., min_length=2, max_length=6)

# ---------- resolved contract ----------
class StyleSpec(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9_]{2,40}$")  # SLUG — canonical key for
                                                          # filenames/COCO/dropdowns (§6.6)
    prompt: str                                            # post-validated: categories in
                                                           # first clause, ≤77 CLIP tokens
    negative_prompt: str = NEGATIVE_PROMPT                 # constant; L2/L3 only

class ObjectSpec(BaseModel):
    instance_id: int = Field(..., ge=1, le=64)
    asset_id: str
    category: str                                 # == asset_id for v1
    requested: str
    x_m: float; y_m: float; yaw_deg: float
    scale: float = Field(1.0, ge=0.5, le=1.6)
    z_m: float = 0.0
    is_target: bool = False
    color_rgb: tuple[int, int, int] = (180, 180, 180)

class SceneSpec(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    task: str
    seed: int = 42
    table: TableSpec = TableSpec()
    objects: list[ObjectSpec] = Field(..., min_length=1, max_length=8)
    camera: CameraSpec = CameraSpec()
    styles: list[StyleSpec] = Field(..., min_length=1, max_length=6)
    grounding_log: list[dict] = []   # [{requested, asset_id, score, method}] + camera bumps
                                     # + dropped objects + style-prompt rewrites

def clamp_to_bounds(raw: dict, model: type[BaseModel]) -> dict:
    """Coerce every numeric leaf into its Field ge/le bounds (recurses into nested
    models and lists by reading model_fields metadata). Same philosophy as placement
    clamping: geometry/range problems are fixed deterministically, never sent back
    to the LLM. Unit-tested in test_spec.py."""
```

`LLMScenePlan.model_json_schema()` / `StyleSet.model_json_schema()` are passed verbatim as `format=` to `ollama.Client.chat`. `$defs`/`$ref` and tuple `prefixItems` ARE grammar-enforced (verified live); numeric bounds are NOT — hence `clamp_to_bounds`.

---

## 4) Data contracts between stages (exact shapes)

`H = W = cfg.gen.resolution = 768` default.

**4.1 Director → Grounding:** `plan_scene(task, seed, cfg) -> LLMScenePlan` (with optional `on_token` callback streaming raw tokens to the UI), `make_styles(task, plan, n_styles, cfg) -> StyleSet`.

**4.2 Grounding/Placement → Scene:** `ground_plan(plan, styles, task, seed, library, rag) -> SceneSpec` (fills `asset_id/category/scale/color_rgb/instance_id`, pins `look_at`/`yfov`, slugifies styles, post-validates prompts §6.6); `resolve_placements(spec, library) -> SceneSpec` (clamped, collision-separated, `z_m` set). Deterministic, idempotent.

**4.3 Scene → Render (`render/types.py`):**
```python
@dataclass
class ComposedScene:
    # REVIEW FIX: pyrender 0.1.45 Scene.from_trimesh_scene DROPS node names
    # (scene.py:583 — verified in source). The backend builds pyrender.Scene
    # manually from these (mesh, transform) pairs, retaining Node handles for
    # seg_node_map. Never use from_trimesh_scene.
    instances: list[tuple[int, trimesh.Trimesh, np.ndarray]]   # (instance_id, mesh COPY, 4x4 T)
    static:    list[tuple[str, trimesh.Trimesh, np.ndarray]]   # ("_floor"|"_table", mesh, T)
    glb_path: str                                              # viewer.glb (+Y-up, no floor)

@dataclass
class InstanceLabel:
    instance_id: int; asset_id: str; category: str; is_target: bool
    bbox_xywh: tuple[int, int, int, int]   # python ints (json-safe), COCO order
    area_px: int                            # python int
    rle: dict                               # {"size": [H, W], "counts": str} ascii

@dataclass
class RenderResult:
    width: int; height: int
    color: np.ndarray        # (H,W,3) uint8 flat-shaded preview
    depth_m: np.ndarray      # (H,W) float32 meters; 0 = no hit
    seg_ids: np.ndarray      # (H,W) int32; 0=bg, k=instance_id k
    instances: list[InstanceLabel]          # area_px >= 200 only
    camera_pose: np.ndarray  # (4,4) float32
    K: np.ndarray            # (3,3) float32
```
Renderer interface: `render_scene(composed, camera: CameraSpec, width, height) -> RenderResult`.

**4.4 Render → Diffusion:** `depth_to_control(depth_m, resolution, mode="disparity") -> PIL.Image` (§7.3 — render at target res, NEVER resize the control image). `ForgePipeline.generate(control, prompt, negative, seed, cond_scale, steps) -> PIL.Image`.

**4.5 Diffusion → Labels:** identity transfer — each image of layout L reuses L's `RenderResult.instances` unchanged. `GeneratedImage = dataclass(path, layout_idx, style_name, seed, gen_seconds)`.

**4.6 Labels/Eval:** `CocoWriter.export(run_dir, layouts, images, keep) -> zip_path`. `Owlv2Scorer.score_batch(images, layouts) -> FidelityReport` with `ImageScore = dataclass(path, fidelity, fidelity_adj, hallucination_count, per_instance)` and `FidelityReport = dataclass(per_image, match_rate, mean_matched_iou, hallucination_rate, kept, quarantined)` (§9). `LayoutRecord = dataclass(layout_idx, spec, render, control_path, glb_path)`.

**4.7 Run directory layout:**
```
outputs/runs/<run_id>/                  # run_id = YYYYmmdd_HHMMSS_<seed>
  run_meta.json                         # task, config snapshot, timings, vram log,
                                        #   style CLIP-token counts
  director_log.md                       # RUNTIME LLM transcripts/repairs (kept OUT of
                                        #   docs/agent_log.md — review fix)
  layout_<k>/spec.json  viewer.glb  depth16.png  control.png  seg_ids.png  labels.json
  layout_<k>/img_<style_slug>.png  overlay_<style_slug>.png
  fidelity.json
  coco/  annotations.json  images/...   # built at export; zipped to dataset.zip
```

**4.8 Orchestrator:** `ForgeRun.run(...) -> Iterator[ForgeEvent]`, `ForgeEvent = dataclass(kind: Literal["plan_token","spec","layout","status","image","fidelity","done","error"], payload: dict)`. Event order implements the demo-paced pipeline of §1; **generation order is PINNED layout-major** (outer loop layouts, inner loop styles) so the gallery grid reads rows=layout/cols=style — asserted in `on_forge`. `ForgeRun.reforge(state, camera, seed, style_name, nudge) -> (glb, image, overlay, annotated)` — no LLM calls; identical latency in both VRAM modes (SDXL stays loaded).

---

## 5) Scene engine design

### 5.1 Procedural asset builders (`assets/builders.py`)

`def build_<name>(scale: float = 1.0) -> trimesh.Trimesh` — union of trimesh primitives, canonical pose resting on z=0, centered, front = +X, ≤ 6k triangles, deterministic pure function of `scale`, final `mesh.apply_translation([0,0,-mesh.bounds[0,2]])`.

The 15 builders (cut order keeps the first 10 — the kitchen set — if time bites, §12.7):

| # | asset_id | construction | nominal size |
|---|---|---|---|
| 1 | mug | `creation.cylinder(r=.042,h=.095)` + torus handle | h .095 |
| 2 | bowl | `creation.revolve` bowl profile | r .08 h .055 |
| 3 | plate | revolve shallow profile | r .11 h .02 |
| 4 | cup | truncated-cone revolve | h .11 |
| 5 | bottle | cylinder body + cone shoulder + neck + cap | h .24 |
| 6 | can | `creation.cylinder(r=.033,h=.115)` | h .115 |
| 7 | box | `creation.box([.18,.12,.08])` | — |
| 8 | book | `creation.box([.21,.15,.03])` | — |
| 9 | pan | revolve dish + box handle | r .11 |
| 10 | ball | `creation.icosphere(3, .05)` | r .05 |
| 11 | pot | cylinder shell + stub handles | r .09 h .12 |
| 12 | screwdriver | shaft + capsule handle, lies flat | l .21 |
| 13 | hammer | handle + box head, lies flat | l .25 |
| 14 | cutting_board | `creation.box([.30,.20,.018])` | — |
| 15 | laptop | two thin boxes hinged 110° | .32×.22 |

`AssetLibrary` (`library.py`): loads `assets/cards/asset_cards.json`, maps `asset_id → builder`, exposes `get_mesh(asset_id, scale)` (LRU-cached **canonical untinted** mesh; **callers receive `mesh.copy()`** — review fix: in-place tinting of a cached mesh aliases colors across instances; test: two mugs, two colors), `footprint_radius(asset_id, scale)` (XY circumradius, cached), `cards()`. Optional GLB ingest (`assets/glb/<id>.glb` + sidecar card): rescale to `height_m`, recenter, floor to z=0, register — additive only, **cut-order #2**.

### 5.2 Placement (`scene/placement.py`) — deterministic, no LLM round-trips

1. Clamp `(x_m, y_m)` into table rect inset by footprint radius.
2. Collision = 2D circle overlap (radii + 8 mm margin); resolve ≤ 40 iterations, pushing pairs apart along center line (target never moves); re-clamp each iter; seeded `default_rng(spec.seed)` breaks ties.
3. Still overlapping → drop lowest-priority non-target object, log in `grounding_log`.
4. `z_m = 0` for all. Geometry errors NEVER go back to the LLM.

### 5.3 Composition (`scene/compose.py`)

`build(spec, library, out_dir) -> ComposedScene`: floor + table slab as `static` entries; per `ObjectSpec` a **copied** mesh with vertex tint and `T = trans(x,y,z) @ rotz(yaw)`. Viewer export (review fixes — two changes): build a `trimesh.Scene` of **table + objects only (no floor** — Model3D auto-frames the whole asset and the floor makes the scene read as a clump on a slab), apply `rotation_matrix(-pi/2, [1,0,0])` so world +Z → glTF +Y (otherwise the table renders standing on edge in three.js), export `viewer.glb`. Render-path geometry stays un-rotated and keeps the floor.

### 5.4 pyrender EGL backend (`render/pyrender_backend.py`) — primary

**Build the `pyrender.Scene` manually — `Scene.from_trimesh_scene` drops node names (verified in 0.1.45 source) and cannot feed `seg_node_map`.** One persistent `OffscreenRenderer(W, H)` reused; recreated on size change.

1. `scene = pyrender.Scene(bg_color=(0,0,0,0), ambient_light=0.4)`; for each static/instance entry: `node = scene.add(pyrender.Mesh.from_trimesh(mesh), pose=T)`, keeping `node_for_instance: dict[int, pyrender.Node]`. Add `DirectionalLight(intensity=3.0)` + `PerspectiveCamera(yfov=radians(yfov_deg))` at `camera.pose_from_orbit(...)`.
2. **Pass 1:** `color, depth = renderer.render(scene)` → `depth_m`.
3. **Pass 2:** `seg_node_map = {node_for_instance[iid]: (iid, 0, 0)}` (instance_id in RED channel; 64 ≪ 255; floor/table omitted → black background). SEG disables `GL_MULTISAMPLE`, resolve blit is exact (verified in source) → IDs round-trip pixel-perfectly. `seg_ids = seg[:,:,0].astype(np.int32)`.
4. `labels.masks.extract_instances(seg_ids, spec)`.

**Target-visibility guard (review fix — orchestrator, after each layout render):** the `is_target` instance must appear in `RenderResult.instances` with `area_px ≥ 1000`. If not: bump `elevation_deg += 15` (clamp ≤ 80), `distance_m ×= 1.15` (clamp ≤ 2.5), re-render ONCE, log to `grounding_log`. Still invisible → emit `ForgeEvent("error", recoverable=layout)` and skip the layout loudly — never generate images whose headline target is unlabeled.

### 5.5 Render fallback policy (review fix: numpy rasterizer DEMOTED to contingency)

EGL+SEG is re-verified working on this machine today; a second full renderer is 0.5–1 day guarding a disproven failure mode, and a Python triangle loop is realistically 5–12 s/frame (not <2 s) — demo-degrading even if built. Policy:
- **Fallback #1:** `get_renderer` retries once with `PYOPENGL_PLATFORM=osmesa` (env-var flip + re-import) on EGL failure.
- **Fallback #2:** golden-run replay — a committed known-good run directory the UI can load for the demo.
- **Contingency only:** `numpy_backend.py` ships as a stub raising `NotImplementedError` with the implementation recipe (z-buffer scanline; MUST follow the §3.1 normative projection equations; activation checklist requires cross-backend equivalence: per-instance mask IoU ≥ 0.98, mean |depth diff| < 2 mm). Implemented only if the M1 step-0 768² EGL assert fails — paid for by cutting builders 15→10.
- M2 acceptance does NOT include the numpy backend; the `RenderResult` interface keeps the contract alive.

---

## 6) LLM director design (`director/`)

### 6.1 Ollama client (`ollama_client.py`)

```python
def chat_structured(model, system, user, schema, *, seed, temperature=0.7,
                    keep_alive: str | int = "5m", num_ctx: int = 4096,
                    timeout_s: float = 90,
                    on_token: Callable[[str], None] | None = None) -> dict
def embed(model, texts, *, keep_alive: str | int = 0, timeout_s: float = 30) -> np.ndarray
```
- **The server runs `OLLAMA_KEEP_ALIVE=-1` (verified): every call MUST pass an explicit `keep_alive` or its model pins in VRAM forever.** `embed()` defaults `keep_alive=0` (the committed embed cache makes query-time embeds rare; 1.11 GB is not worth pinning).
- Director chat calls use `keep_alive="5m"` in BOTH VRAM modes (review fix: per-call `keep_alive=0` forced a measured 3 s gemma reload between the plan and style calls within one forge for zero benefit) — eviction is enforced solely by the `phase("diffusion")` unload barrier.
- **`num_ctx=4096` standardized for ALL gemma4:e4b calls** (plan/style/repair): planner prompts are small, this saves ~0.6 GB KV (coresident viability), and Ollama silently reloads a model whenever `num_ctx` changes between requests (3 s churn) — so one value everywhere.
- `stream=True` is supported together with `format=`; `on_token` receives raw text chunks for the UI spec-streaming panel.
- Any connection error/timeout/parse failure raises `DirectorUnavailable`; callers take the deterministic fallback. Never let an Ollama exception reach the UI.

### 6.2 Prompts strategy

**Planner** (model `gemma4:e4b`; quality mode swaps `qwen3.5:27b` — see §10.4 for the mandatory GPU swap): system prompt `PLANNER_SYSTEM` states the world model verbatim ("tabletop at z=0, x∈[-0.5,0.5], y∈[-0.35,0.35] usable; 2–8 objects as (x_m, y_m, yaw_deg); exactly ONE is_target; spread objects, no stacking; camera orbit az/el/dist at table center") + ONE few-shot example. `format=LLMScenePlan.model_json_schema()`, temp 0.7, seed = run seed.

**Styler** (`STYLER_SYSTEM`): "Invent N visually distinct photoreal environments for the SAME physical scene. Each prompt MUST begin with the objects: 'a photo of {categories} on a table, in <environment>...' — environment/surface/lighting/camera-realism after the object clause; ≤ 40 words; photographic vocabulary; no identity-changing style words." `format=StyleSet.model_json_schema()`, temp 0.95, seed +1. (Category-first ordering is mandatory because CLIP truncates at 77 tokens — a late category mention is silently cut, §6.6.)

### 6.3 Clamp → validate → repair loop (`director.plan_scene`)

```
attempt 0..2:
    raw = chat_structured(...)                    # DirectorUnavailable → fallback
    raw = clamp_to_bounds(raw, LLMScenePlan)      # REVIEW FIX: Ollama does NOT enforce
                                                  # numeric ranges (verified live) — clamp,
                                                  # don't burn repair rounds on bounds
    try: return LLMScenePlan.model_validate(raw)
    except ValidationError as e:                  # structural/semantic only now
        user_msg += f"\nYour previous JSON failed validation:\n{e.errors()}\nReturn corrected JSON only."
fallback: return fallback.template_plan(task, seed)
```
Repair transcripts go to `outputs/runs/<id>/director_log.md` (NOT `docs/agent_log.md` — review fix: the course log is a build-process narrative; one runtime transcript is cited there as an appendix exhibit).

### 6.4 RAG grounding (`rag.py`) — Decision D (§12-D)

`EmbeddingIndex.build(cards, embed_fn)`: document string `f"title: {name} | text: {description}. Synonyms: {...}. Used for: {affordances}."` (embeddinggemma's document format), L2-normalized `(N,768) float32` in `assets/index/embed_cache.npz` keyed by sha256 of the cards file — **cache is COMMITTED** so cold demos never need the embed model. Query: `f"task: search result | query: {description}"` → cosine = one matmul. `ground(description) -> (asset_id, score, method)`: top-1 if score ≥ 0.35; else difflib over synonyms; else `"box"`. `embed()` failure → difflib directly. All decisions recorded in `grounding_log`, shown in the UI.

### 6.5 Deterministic fallbacks (`fallback.py`)

- `template_plan(task, seed)`: tokenize task → difflib vs card synonyms → target + 2–4 distractors round-robin from `["bowl","bottle","can","book"]` → seeded golden-angle ring (r ∈ [0.10, 0.30]) → fixed camera (35, 30, 1.1). Pure function of `(task, seed)`.
- `DEFAULT_STYLES`: 4 hardcoded `StyleSpec`s (rustic_kitchen / garage_workbench / clean_lab / outdoor_picnic), prompts with `{objects}` placeholder filled at use time — already category-first.
Every director entry point: try LLM → on `DirectorUnavailable`/exhaustion → fallback; `payload["source"] = "llm" | "fallback"` surfaced in UI.

### 6.6 Style post-validation (`director.py`, deterministic — review fixes)

Applied to every `LLMStyle` (LLM or fallback) before it becomes a `StyleSpec`:
1. **Slugify name** (review fix: names flow into file paths and COCO `file_name`): lowercase, replace non-`[a-z0-9]` runs with `_`, truncate to 40, suffix `_2`/`_3` on collision; uniqueness enforced post-slug. The slug is THE canonical key (filenames, `style_dd` dropdown, COCO attributes).
2. **Strip identity-changing words** (deterministic blacklist: cartoon, painting, illustration, anime, render, sketch, drawing…) — the styler system prompt alone is unenforced, and negative_prompt can't help at cfg=0.
3. **Category coverage + CLIP-77 budget** (review fix: SDXL's CLIP encoders hard-truncate at 77 tokens; a category mentioned late is silently dropped): tokenize with `CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")` (cached, CPU, tiny). Every layout category must appear within the first ~60 tokens. If ANY is missing/late, deterministically REBUILD as `f"a photo of {', '.join(categories)} on a table, {environment_tail}"` where `environment_tail` is the LLM prompt minus its leading object clause; then trim to 77 tokens. (This resolves the reviewers' append-vs-first-clause tension: appending would land past the truncation point — rebuild-with-prefix guarantees placement. §12.6.) Token counts per style logged to `run_meta.json`; rewrites logged to `grounding_log`.

---

## 7) Diffusion stage design (`diffusion/pipeline.py`)

### 7.1 Pipeline assembly (exact recipe)

```python
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae,
    torch_dtype=torch.float16, variant="fp16")
# SDXL-Lightning 4-step UNet (full swap, not LoRA):
unet_cfg = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
unet = UNet2DConditionModel.from_config(unet_cfg).to(torch.float16)
unet.load_state_dict(load_file(hf_hub_download(
    "ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")))
pipe.unet = unet
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")   # REQUIRED for Lightning
pipe.to("cuda")
# REVIEW FIX: NO enable_attention_slicing() / enable_vae_tiling() by default —
# torch 2.6 SDPA is already memory-efficient; slicing adds 10-25% latency to save
# headroom we have ~10 GiB of in sequential mode, and it muddies M1's s/img + VRAM
# measurements. Both are re-enabled only at degrade level V2 (§10.2).
```
4-step LoRA (`sdxl_lightning_4step_lora.safetensors`) is plan-B for ladder L2 via `load_lora_weights` + `fuse_lora()` (a fused LoRA tolerates 6–8 steps better than the distilled full UNet).

### 7.2 Generation call (defaults from `cfg.gen`, frozen by M1)

```python
img = pipe(prompt=style.prompt, negative_prompt=style.negative_prompt,  # negative is a
           # NO-OP at L0/L1: diffusers enables CFG iff guidance_scale > 1 (verified in
           # installed 0.38 source) — documented here and in spec.py.
           image=control_image, num_inference_steps=4, guidance_scale=0.0,
           controlnet_conditioning_scale=0.85, control_guidance_end=0.9,
           width=768, height=768,
           generator=torch.Generator("cuda").manual_seed(seed)).images[0]
```
Seed law: `seed = spec.seed + layout_idx * 1000 + style_idx`. Batch size 1. Resolution 768 default; 1024 only behind cut-order #4 and only if M1 measured headroom.

### 7.3 Depth preprocessing (`depth_prep.py`) — the fidelity-critical 20 lines

**REVIEW FIX (default changed): the ControlNet was trained on DPT-MiDaS *relative inverse depth (disparity ≈ 1/z)*, min/max normalized — not a linear flip.** In disparity space near-range (object-vs-table) contrast is amplified and the far floor compressed, exactly what label fidelity needs. Default `mode="disparity"`:
`valid = depth_m > 0`; `disp = 1/np.maximum(depth_m, 1e-6)`; `lo, hi = np.percentile(disp[valid], [2, 98])`; `out = np.clip((disp − lo)/(hi − lo), 0, 1)`; `out[~valid] = 0.0` (far, MiDaS semantics) → uint8 ×3 channels at exact 768×768 (rendered at target res; **never resize** — resampling softens edges and costs IoU). `mode="linear"` (the v1.0 flip) is retained for the M1 A/B: both run over the 12 M1 scenes at L0 (~1 min); the winner is frozen into `depth_prep.py` defaults and `sceneforge.yaml`. The floor plane exists to fill the background with far depth instead of invalid zeros. Save `control.png` per layout.

### 7.4 Conditioning ladder (M1 walks it; per-level realistic s/img — review fix)

| level | recipe | est. s/img @768 (3090) |
|---|---|---|
| **L0** | Lightning-4step UNet, cond=0.85, cge=0.9, cfg=0 | 2–2.5 |
| **L1** | Lightning-4step, cond=1.0, cge=1.0, cfg=0 | 2–2.5 |
| **L2** | base UNet + fused 4-step LoRA, steps=8, cond=0.9, cfg=1.5 (CFG ON → 2× UNet evals) | 5–8 |
| **L3** | plain base SDXL, EulerDiscrete, steps=20, cfg=7.5, cond=0.8 | 12–15 |
| **L4** | (contingency, ~2.5 GB download) + controlnet-canny-sdxl via MultiControlNet, canny from `seg_ids` boundaries, scales [0.7 depth, 0.4 canny] | per-level + ~30% |

M1 records MEASURED s/img per attempted level into `sceneforge.yaml`; downstream time budgets (M4 acceptance) are computed from the chosen level's measured number, not a fixed constant.

### 7.5 OOM recovery (review fix — the v1.0 escape hatch crashed when invoked)

`enable_model_cpu_offload()` on a cuda-resident pipeline errors in diffusers 0.38 (accelerate hooks expect CPU residency), and post-OOM the allocator is fragmented. Exact sequence, implemented in `ForgePipeline.generate` and unit-tested with a mocked `OutOfMemoryError` (`tests/test_gpu.py`):
```
catch torch.cuda.OutOfMemoryError →
  pipe.to("cpu") → gc.collect() → torch.cuda.empty_cache() →
  pipe.enable_model_cpu_offload() → retry the generate call ONCE →
  still OOM → drop to 640 px for the remainder of the burst.
Side effects: force cfg.vram.mode = "sequential" for the rest of the session; UI warning.
```
`ForgePipeline` API: `load(level)`, `generate(...)`, `unload()` (`del` refs + `empty_cache` + `gc.collect`), `peak_vram()` reporting **all three**: `max_memory_allocated`, `max_memory_reserved`, min `mem_get_info` free.

---

## 8) Label transfer + COCO export

### 8.1 Masks (`labels/masks.py`)

Per `ObjectSpec`: `mask = (seg_ids == instance_id)`; skip if `area < 200` px; bbox via `cv2.boundingRect`; RLE via `pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))` with `counts` ascii-decoded. **All numeric fields cast to python `int` at `InstanceLabel` construction** (review fix: numpy int64 is not JSON-serializable). Masks computed once per layout, **shared by all style variants** — the product's thesis.

### 8.2 Overlays (`labels/overlay.py`)

`draw_overlay(img, instances, mode: Literal["boxes","masks","both","off"])`: fixed per-category palette; boxes 3 px + label chips; masks alpha 0.35; target thicker + starred. Raw and overlay PNGs written at generation time → UI toggle is a pure cached-path swap.

### 8.3 COCO writer (`labels/coco.py`) — review fixes baked in

- Categories: fixed 15-asset list, `category_id` = 1-based library order (stable).
- **Annotation ids: ONE global running counter across ALL images** (per-layout reuse breaks COCO indexing).
- **Every `images` entry carries explicit `width`/`height`** (768, 768).
- All ids/bbox/area are python ints; `json.dumps` of the whole document is asserted in tests.
- Annotation: `{id, image_id, category_id, bbox, area, segmentation: rle, iscrowd: 0, attributes: {is_target, layout_idx, style, instance_id}}`. Extra top-level `"sceneforge": {task, specs, fidelity_summary}` (loaders ignore unknown keys).
- Export: copy kept images → `coco/images/`, write `annotations.json`, `make_archive → dataset.zip`, return path for `gr.DownloadButton`.
- Acceptance: `COCO(annotations.json)` loads; `annToMask` round-trips every annotation; `len(set(ann_ids)) == len(anns)`; `json.dumps` succeeds — all in `test_masks.py` over a synthetic 2-layout × 2-style export.

---

## 9) OWLv2 fidelity eval (`eval/fidelity.py`)

- `google/owlv2-base-patch16-ensemble`, fp16, loaded only inside `gpu.phase("eval")`, unloaded after. Needs transformers<5 (P0). Minimal scorer is **written in M1** (self-contained harness) and promoted into this module.
- OWLv2's processor pads to square — our images are square 768², so padding is identity and `post_process_object_detection(threshold=0.15, target_sizes=[(768,768)])` returns usable pixel boxes directly. Documented in code; silently breaks for non-square gen.
- Queries: `[f"a photo of a {category}"]` per unique layout category.
- **Recall term:** per GT instance, `iou_i` = max IoU vs same-category detections (0 if none). `fidelity = mean_i(iou_i)`. Batch: `match_rate` = fraction GT with iou ≥ 0.5; `mean_matched_iou`.
- **Hallucination term (review fix — the v1.0 eval was blind to duplicated objects, the canonical ControlNet+Lightning failure, and duplicates poison the export as false negatives):** per image, count detections of in-layout categories with score ≥ 0.3 whose IoU with EVERY GT box of that category is < 0.3 → `hallucination_count`. `fidelity_adj = fidelity − 0.5 · hallucination_count / max(1, n_gt)`. Quarantine on `fidelity_adj < cfg.eval.keep_threshold` (default 0.45). `hallucination_rate` reported separately in `fidelity.json` and the UI meter (λ-penalty chosen over hard quarantine-at-1 because OWLv2 itself false-positives, §12.6).
- **Gate population (review fix):** all gate/threshold statistics computed over GT instances with `area_px ≥ 1000` (OWLv2 recall collapses on tiny/thin instances — a missed screwdriver shaft is a detector failure, not a label failure); smaller instances reported separately as informational, exclusion rule recorded in `metrics.json`.
- Honest-metric caveat (README): OWLv2 box jitter caps matched IoU at ~0.75–0.85 even on perfect labels → the eval is a *lower bound* on label validity; M1 gate calibrated accordingly.

---

## 10) VRAM orchestration policy (`gpu.py`)

### 10.1 Corrected budget table (MEASURED — review fix: v1.0 was 1–2 GB optimistic)

| line item | GB |
|---|---|
| desktop (Xorg + gnome-remote-desktop) | 0.82 |
| gemma4:e4b resident @ num_ctx 8192 / 4096 | 10.88 / ~10.3 |
| embeddinggemma resident (if not unloaded) | 1.11 |
| SDXL fp16 weights (UNet 5.1 + CN 2.5 + TEs 1.7 + VAE 0.2) | ≈ 9.5 |
| SDXL activations @768², no slicing | 2.5–3.5 |
| torch CUDA context | 0.5–0.8 |
| allocator reserved-vs-allocated slack | 0.5–1.5 |
| **diffusion device-level footprint** | **≈ 12–13.5** |

Measured free with gemma(8k)+embeddinggemma resident: **11.9 GiB < 12–13.5 GiB needed** → co-residency as originally configured is NOT viable. It becomes plausible only with (a) embeddinggemma unloaded during bursts, (b) gemma at num_ctx 4096, (c) decisions driven by device-wide `torch.cuda.mem_get_info` (NOT `max_memory_allocated`, which is blind to context, reserved slack, and Ollama). **Sequential@768 is the expected default; coresident is an M1-verified bonus.** Honest rationale correction: re-forge makes zero LLM calls, so coresident buys NOTHING on re-forge — it saves only the measured 3.0 s warm gemma reload per forge spec phase.

### 10.2 Degrade ladder (single source of truth; `guard_gb = 3.0` everywhere)

| level | config | trigger |
|---|---|---|
| V0 | coresident @768 (gemma 4k ctx resident; embeddinggemma unloaded for bursts) | only if M1 min device-free ≥ guard_gb THROUGHOUT |
| V1 | **sequential @768 (expected default)** | default |
| V2 | + `enable_vae_tiling` + `enable_attention_slicing` | pre-burst `mem_get_info` free < guard_gb |
| V3 | `enable_model_cpu_offload` sequential (~2× slower) | caught OOM → recovery §7.5 |
| V4 | resolution 640 | OOM persists after V3 retry |

One constant `cfg.vram.guard_gb = 3.0` is used by BOTH the M1 mode-pick rule and the runtime guard (v1.0 had 2.5 vs 3.0 — a measurement in between would pick a mode that then degrades on every burst).

### 10.3 Modes (`cfg.vram.mode: "coresident" | "sequential"`, both implemented — judge mandate #2)

- **sequential (expected default):** director calls use `keep_alive="5m"` (NOT 0 — review fix, saves a measured 3 s mid-phase reload); eviction enforced at `phase("diffusion")` entry by `ollama_unload()`.
- **coresident:** gemma4:e4b `keep_alive="30m"`, num_ctx 4096, stays resident through diffusion; embeddinggemma still unloaded before bursts. Guard: pre-burst free < guard_gb → auto-degrade that burst to sequential + UI warning.
- Hard rules in both modes: `qwen3.5:27b` / `gemma4:26b` ALWAYS `keep_alive=0`, spec phase only; OWLv2 only in `phase("eval")` (which also runs `ollama_unload()` in coresident mode); SDXL loads once per process and stays (anchor tenant) **except** the quality-mode swap (§10.4).

**`ollama_unload()` (review fix — the v1.0 version 400'd and ate the 15 s timeout every burst):** iterate `/api/ps`; per model POST `/api/generate {"model": m, "keep_alive": 0}`; on **HTTP 400 ("does not support generate" — embedding models)** POST `/api/embed {"model": m, "input": "", "keep_alive": 0}` (verified live: `/api/ps` empties, reload works). Poll `/api/ps` until NO models remain (timeout 15 s; measured ~2 s). This sweep inherently covers embeddinggemma.

**`phase()` contract:**
- `phase("spec")` entry: `gc.collect()` + `torch.cuda.empty_cache()` — returns 1–2 GB of allocator slack to the driver so Ollama can place gemma fully on GPU next to idle SDXL weights (review fix: without this, gemma partially offloads to CPU and planning drops from the measured 139 tok/s). M1 asserts full-GPU placement via `/api/ps size_vram == size`.
- `phase("diffusion")` entry: `ollama_unload()` (sequential; coresident leaves gemma but still unloads embeddinggemma) → wait `/api/ps`.
- `phase("eval")`: `ollama_unload()` (both modes) → load OWLv2 → unload after.
- All phases log enter/exit `snapshot()` to `run_meta.json`.

`snapshot() -> {"free_gb", "total_gb", "torch_alloc_gb", "torch_reserved_gb", "ollama_models": [...]}` via `torch.cuda.mem_get_info` + `/api/ps` (no pynvml).

### 10.4 Quality-mode (qwen3.5:27b) — enforced GPU swap (review fix)

qwen is 17.4 GB (~19 GB with KV). With SDXL resident there are only ~10 GiB free → Ollama silently part-offloads qwen to CPU and a 600-token plan takes 30–60 s, stalling a live demo. `phase("spec_quality")`: `pipe.to("cpu")` → `gc.collect()` → `torch.cuda.empty_cache()` → qwen call(s) with `keep_alive=0` → wait unload → `pipe.to("cuda")`. Each move ~5–10 s (125 GB RAM absorbs fp16 weights) — acceptable for an explicit toggle; the UI labels the checkbox "Quality LLM (adds ~20 s)". M1 includes one timed quality-mode forge-AFTER-forge run. Quality mode is cut-order #3 if time bites.

### 10.5 Cold-start pre-warm (review fix — measured 33.4 s cold gemma load; SDXL first construction 30–90 s)

`app.py` launch sequence (footer shows a "warming up" banner via the Timer until done): (1) load + immediately unload gemma4:e4b (populates page cache → subsequent reloads 3.0 s), (2) eagerly construct `ForgePipeline` on cuda, (3) instantiate OWLv2 processor+model once to CPU (moved to cuda only in `phase("eval")`). `scripts/demo_prep.sh` = pre-warm + one throwaway forge; "run one throwaway forge before recording" is on the M6 demo checklist. Doc figures: gemma reload **3 s warm / 33 s cold (measured)**.

---

## 11) Gradio UI map + event flow (`ui/blocks.py`, `ui/handlers.py`)

Gradio **6.17.3** — verified present: `Model3D, Gallery, AnnotatedImage, Image, JSON, Code, Timer, State, DownloadButton, Slider, Dropdown, Checkbox, Radio, Accordion, Row, Column, Markdown, Label`. Implementer re-checks exact kwarg names at build time (6.x renamed some vs 5.x).

```
gr.Blocks(title="SceneForge")
├── Row: task_tb (Textbox, scale=4) · forge_btn ("FORGE", variant=primary)
├── Row: n_layouts (Slider 1–4, v=2) · n_styles (Slider 2–6, v=4) · seed (Number 42)
│        · quality_cb (Checkbox "Quality LLM (adds ~20 s)")        [cut-order #3]
│        · res_dd (Dropdown [768,1024], v=768)                     [cut-order #4]
├── Row: status_md (Markdown stage line: "planning → rendering → forging 3/8 → scoring")
├── Row
│   ├── Column: layout_3d (Model3D ← viewer.glb) + control_img (Image, depth control)
│   │   └── Accordion "Re-Forge" (judge mandate #3):
│   │       az_sl (-180..180) · el_sl (10..80) · dist_sl (0.5..2.5)
│   │       nudge_dd + 4 nudge btns (±x/±y 5 cm)                   [cut-order #5]
│   │       style_dd (slug names) · reseed_btn ("🎲") · reforge_btn ("Re-Forge ⚡")
│   │       reforge_img (AnnotatedImage ← (image, [(mask, label), ...]))
│   └── Column: plan_stream (Code, planner tokens stream in live)  ← wow-line restored
│               spec_json (JSON, validated SceneSpec incl. grounding_log)
│               source_md (Markdown "planner: llm | fallback")
├── Row: gallery (Gallery, columns=n_styles — rows=layout/cols=style REQUIRES
│        layout-major generation order, pinned in orchestrator + asserted in on_forge)
│        · overlay_radio (Radio ["off","boxes","masks","both"], v="boxes") [cut #6: off/both]
├── Row: fidelity_lbl (Label "fidelity 0.87 · halluc 0.04") · include_q_cb  [cut #7]
│        · export_btn ("Export COCO") · download (DownloadButton)
└── Row: footer_md (VRAM x/24 GB · backend · mode · s/img · warming-up banner)
     + gr.Timer(2.0) → gpu.snapshot() → footer_md                  [cut #9: static line]
```

**Demo pacing (review fix — v1.0 had 20–40 s of dead air at forge start):**
1. `on_forge` is a generator over `ForgeRun.run()` events. `plan_token` events stream raw planner tokens into `plan_stream` (ollama `stream=True` works with `format=`; if streaming proves awkward at build time, fallback = typewriter-replay of the validated JSON while rendering proceeds — zero risk, same effect).
2. Pipeline order (§1) emits layout-0 `viewer.glb` + control image BEFORE the style LLM call — first visual lands seconds after planning, not after two LLM calls.
3. `status_md` updates on every event — the UI is never silent.
4. Gallery fills live per `image` event at the chosen level's measured cadence.

**State:** `run_state = gr.State()` — JSON-able dict of paths only, never arrays.

**Events:**
- `forge_btn.click(on_forge, ...)` — generator as above; queue on (Gradio 6 default), concurrency 1.
- `reforge_btn.click(on_reforge, ...)` — no LLM; re-render depth + 1 diffusion call; returns `AnnotatedImage` `(image_path, [(mask, category), ...])` so labels visibly track the camera. Target ≤ 6 s slider-to-image (identical in both VRAM modes). Nudge buttons mutate target `(x_m, y_m)` ±0.05 then `.then(on_reforge, ...)`.
- `overlay_radio.change(on_toggle, ...)` — pure cached-path swap.
- `export_btn.click(on_export, ...)` → zip path into `DownloadButton`.

`gr.Model3D` does NOT round-trip in-viewer camera moves to the server → re-forge is slider-driven (deterministic, demo-reliable); the viewer is visual confirmation.

---

## 12) Settled design decisions

### 12-A. Asset library → PROCEDURAL-FIRST (15 parametric trimesh builders), optional GLB drop-in, no required downloads
ControlNet-depth re-textures everything, so only the depth silhouette matters — a lathe-profile mug reads identically to a scanned YCB mug in a depth map; GLB curation carries download/licensing/scale/orientation risk that historically eats half a day. Builders are deterministic, watertight-ish, low-poly, with exact known footprints for collision. The hybrid hook (`assets/glb/` + sidecar cards) preserves upside at zero critical-path risk; it is cut-order #2. The kitchen-set 10 (§5.1 #1–10) are the protected core; #11–15 are cut-order #8.

### 12-B. Per-instance masks → pyrender `RenderFlags.SEG` + `seg_node_map`, VERIFIED — with manual Scene construction
SEG=8192 exists; `render(..., seg_node_map=)` works; `GL_MULTISAMPLE` is disabled during SEG and the resolve blit is exact, so red-channel IDs round-trip pixel-perfectly (verified in 0.1.45 source). **Mandatory implementation detail (review fix): `Scene.from_trimesh_scene` drops node names — the backend builds the pyrender scene manually from `ComposedScene` (mesh, transform) pairs, retaining `pyrender.Node` handles for `seg_node_map`.** Flow: instance_id in RED channel; floor/table omitted → background black; `seg_ids == iid → mask → cv2.boundingRect → pycocotools RLE (ascii counts) → COCO`.

### 12-C. SceneSpec schema → §3.2 code; conventions §3.1
Z-up, meters/degrees, tabletop z=0, orbit camera with **server-pinned look_at/yfov**, split contract (`LLMScenePlan`/`LLMStyle` = LLM-facing strict subset; `SceneSpec`/`StyleSpec` = resolved), `clamp_to_bounds` between Ollama and pydantic (numeric ranges are NOT grammar-enforced — verified), slugged style names as canonical keys, `grounding_log` for transparency. The LLM never invents `asset_id`s (RAG's job) and never controls resolved fields (`z_m`, `scale`, `instance_id`, `look_at`).

### 12-D. RAG → embeddinggemma via Ollama `/api/embed` + numpy cosine; NO chromadb
A vector DB for ≤30 vectors is indefensible; one normalized matmul is exact, dependency-free, testable. Asymmetric prefixes (document: `"title:.. | text:.."`, query: `"task: search result | query:.."`) cost nothing and measurably help. Cache committed (`embed_cache.npz`, keyed by cards-file sha256) so cold demos never need the embed model; difflib is the no-Ollama fallback; 0.35 cosine floor routes nonsense to `box`. **All embed calls pass `keep_alive=0`** (server default is -1 = pin-forever; embeddinggemma is 1.11 GB resident, verified).

### 12-E. Gradio → installed 6.17.3; required components verified present
Generator handlers for streaming, `gr.Timer` footer, `AnnotatedImage` for the re-forge money shot, `gr.Code` for planner token streaming. `Model3D` lacks camera round-trip → slider-driven re-forge. Implementer verifies exact kwarg names against 6.17.3 at build time.

### 12.6 Resolved disputes (reviewer contradictions, decided)

1. **Director chat `keep_alive`: "5m" (R1) vs "2m" (R3) → "5m".** Any short value works because eviction is enforced by the `phase("diffusion")` barrier, not by expiry; 5m comfortably covers a slow plan+repair+style sequence without a mid-phase reload.
2. **`embed()` default `keep_alive`: pass-through "5m" (R1) vs 0 (R2) → 0.** The committed embed cache makes query-time embeds rare; 1.11 GB (measured) is the single line item that broke the v1.0 coresident budget. The parameter exists for callers that want otherwise.
3. **`enable_vae_tiling`: keep (R1) vs drop (R2) → drop both tiling and slicing from defaults.** At 768² with ~10 GiB sequential headroom they only add latency and contaminate M1's measurements; both re-enable together at degrade V2.
4. **gemma `num_ctx`: 4096-for-coresident (R1 fix 1) vs standardize-8192 (R1 fix 10) → standardize 4096 everywhere.** Planner prompts are ~1–2k tokens; 4096 saves ~0.6 GB KV (the margin coresident needs), and a single value avoids Ollama's silent 3 s reload on ctx change. If a prompt ever exceeds 4096 the client raises rather than silently truncating.
5. **`negative_prompt`: keep-but-document (R2) vs drop-from-LLM-schema (R3) → drop from the LLM-facing schema (R3), document deadness (R2).** It's a hardcoded constant on resolved `StyleSpec`, active only at L2/L3 (`guidance_scale > 1`); styler tokens are not spent on it; duplicate suppression is handled by the eval hallucination term instead.
6. **numpy rasterizer: full backend + cross-backend test in M2 (R2) vs contingency stub (R3) → stub (R3).** EGL+SEG re-verified working today and the Python rasterizer's realistic 5–12 s/frame would degrade the demo anyway. R2's substance is preserved: the projection equations are normative in `camera.py` docstrings, and the mask-IoU ≥ 0.98 / depth < 2 mm equivalence test is the mandatory activation checklist if the stub is ever implemented.
7. **Style category enforcement: first-clause requirement (R2) vs append-missing (R3) → deterministic REBUILD with category-first prefix.** Appending puts categories past the CLIP-77 truncation point, defeating R2's fix; the rebuild (`"a photo of {cats} on a table, " + environment tail`, then 77-token trim) satisfies both (§6.6).
8. **M1 harness: gate-eligible instances (R2: area ≥ 1000, ≥30 eligible) inside R3's self-contained harness → combined.** The 4 inline scenes use only medium/large primitives (box, can, mug, sphere/plate) so 12 renders × 3–4 objects yield ≥ 36 GT instances, ≥ 30 of them gate-eligible at area ≥ 1000.
9. **Hallucination quarantine: hard quarantine at count > 0 vs λ-penalty (both offered by R2) → λ-penalty** (`fidelity_adj = fidelity − 0.5·FP_rate`): OWLv2 itself false-positives, so a single spurious detection should dent the score, not nuke the image; `hallucination_rate` is still reported separately so drift is visible.

### 12.7 Minimal Lovable Demo + pre-committed cut order (review fix — no ad-hoc cutting)

**MLD (never ship less):** task box → streamed spec → Model3D + depth preview → 1–2 layouts × 4 styles streaming grid → overlay toggle → camera-slider re-forge with AnnotatedImage → fidelity number → COCO zip.

**Cut order (first to last):** 1) numpy rasterizer (already a stub); 2) GLB ingest hook; 3) qwen quality-mode checkbox (a 17 GB swap mid-demo is a liability); 4) 1024 px dropdown (fix 768); 5) nudge buttons (camera sliders + reseed already satisfy judge mandate #3); 6) overlay modes 4→2 (off/both); 7) quarantine checkbox (keep the fidelity number); 8) builders 15→10 (keep the kitchen set §5.1 #1–10); 9) `gr.Timer` VRAM footer → static per-forge stats line; 10) soak 20→10 tasks.

**Never cut:** overlay alignment quality, streaming grid fill, re-forge camera path, COCO export, fidelity number, Ollama-dead fallback.

---

## 13) Milestone plan summary (full checklist: `docs/TASKS.md`)

Schedule fix (was a blocker): **M1 is now fully self-contained throwaway code** — it shares NO package code except `compat.py`, `diffusion/depth_prep.py`, the `ForgePipeline` L0 recipe, and a minimal OWLv2 scorer (the latter three are written in M1 and promoted). It hardcodes 4 inline scenes (box, can-cylinder, composed mug, sphere/plate — medium/large only) × 3 seeds = 12 renders with an inline orbit camera + SEG pass + mask→bbox (~150 lines). It does NOT wait for M2's builders or M3's fallback planner.

| | scope | gate / acceptance (full criteria in TASKS.md) |
|---|---|---|
| **M1** Day 1 AM | P0 repairs (incl. HF snapshot check FIRST), self-contained go/no-go, depth-prep A/B, ladder walk, VRAM mode pick, quality-mode timing | **GO: `match_rate(IoU≥0.5) ≥ 0.70` AND `mean_matched_iou ≥ 0.65`** over GT instances with area ≥ 1000 px (≥ 30 eligible); hallucination_rate reported (informational); chosen level + mode + measured s/img frozen into `sceneforge.yaml` |
| **M2** Day 1 PM | 15 builders, AssetLibrary, placement, compose, EGL backend (manual Scene), masks | tests green; contact sheet; byte-identical seg_ids across 2 runs; RLE round-trip; tint-copy test; target-visibility guard works |
| **M3** Day 2 AM | director + clamp/repair + RAG + style post-validation + fallbacks | 10/10 tasks → valid SceneSpec via **LLM-or-fallback, zero exceptions**; ≥ 8/10 via LLM path with ≤ 2 repairs (transcripts logged); Ollama-stopped path completes; embed cache committed |
| **M4** Day 2 PM | diffusion at M1 level, batch loop, overlays, COCO, **fidelity wiring (moved up from M6)** | 2×4 end-to-end within `8 × s_img(M1) + 60 s`; COCO loads + annToMask round-trip; fidelity_lbl live |
| **M5** Day 3 | full UI, streaming, re-forge ≤ 6 s, hardening, **rough backup demo video recorded** | scripted 2-min happy path ×3 consecutive; kill-Ollama drill; 20-task soak (cut #10: 10) |
| **M6** Day 4 | quarantine UI polish, 100+ image export, README + requirements freeze, agent_log polish, final video via `demo_prep.sh` | all course deliverables in repo; polished video (after one throwaway warm-up forge) |

If even L3 fails the M1 gate, the thesis is falsified for this stack → pivot ruling: ship L3 with the fidelity meter as the headline honest-eval feature (demo survives, claim softened) — decision recorded in `docs/agent_log.md`.

**requirements.txt (frozen at M6 from the working env):** torch 2.6.0+cu124, `transformers>=4.49,<5` (4.51.3), diffusers 0.38.0, gradio 6.17.3, pyrender 0.1.45, trimesh 4.12.2, pydantic 2.13.4, numpy 2.4.4, opencv-python-headless, pycocotools, shapely, ollama, accelerate, safetensors, huggingface_hub.
