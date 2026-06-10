# Agent Collaboration Log — SceneForge

This log documents how the project was built end-to-end with an agentic workflow, as required by the assignment. It records the tools used, the key prompts, the multi-agent orchestration patterns, and the technical bottlenecks resolved with agent assistance.

## Toolchain

| Layer | Tool |
|---|---|
| Orchestrating agent | Claude Code CLI (model: Fable 5), "ultracode" multi-agent workflow mode |
| Sub-agent fan-out | Claude Code `Workflow` orchestration (deterministic JS scripts spawning parallel sub-agents with JSON-schema-constrained outputs) |
| App inference backend (LLM) | Ollama 0.20.3 — `gemma4:e4b` (scene director), `embeddinggemma` (RAG), `qwen3.5:27b` (optional quality mode) |
| App inference backend (diffusion) | `diffusers` — SDXL base 1.0 + `controlnet-depth-sdxl-1.0` + SDXL-Lightning 4-step, fp16 on one RTX 3090 |
| Hardware | Single RTX 3090 (24 GB), 125 GB RAM, Ubuntu |

---

## Phase 1 — Ideation and Planning (2026-06-10)

### Step 1: Environment scouting (agent-driven)

Before proposing anything, the agent inventoried the machine so proposals would be grounded in reality rather than wishful thinking:

- `nvidia-smi`, `free -h`, `df -h` → RTX 3090 24 GB (0.8 GB used by desktop), 125 GB RAM, 243 GB disk
- `ollama list` + API probe → 4 local models already pulled, OpenAI-compatible endpoint live
- `pip list` / `conda env list` → no ML stack yet; fresh `dgan` env created (Python 3.11, torch 2.6.0+cu124) **in the background while ideation ran**
- HF connectivity check → model downloads possible

**Bottleneck identified by the agent at this stage:** the GPU is shared — a 17 GB Ollama LLM and SDXL cannot co-reside in 24 GB. This constraint was injected into every proposal prompt, forcing each design to carry an explicit VRAM plan. This single constraint shaped the entire project architecture.

### Step 2: Multi-agent ideation (5 proposers × 3-lens judge panel = 20 sub-agents)

A workflow script spawned **5 proposal agents in parallel**, each locked to a different design angle (creative LLM+diffusion combo / ControlNet tool / RAG knowledge app / robotics-flavored / audio-multimodal). Key prompt skeleton given to each proposer:

> *"You are designing a capstone project proposal for a Deep Generative Models course. [HARDWARE/ENV FACTS…] [ASSIGNMENT RUBRIC…] YOUR ASSIGNED DESIGN ANGLE: […]. Be CONCRETE: name exact models (with HF repo ids), exact libraries, and a realistic VRAM strategy. Favor proposals that use BOTH an LLM and a diffusion model in a way that is integral, not decorative."*

Each proposal was returned as schema-validated JSON (title, concept, LLM component, diffusion component, VRAM plan, ASCII UI mockup, build plan, risks). Then **each proposal was attacked by 3 judge agents** with distinct lenses:

1. **Course-grading lens** — are both technologies integral, not decorative?
2. **Engineering-feasibility lens** — does the VRAM math hold on a real 3090? do the named HF models actually exist?
3. **Demo-impact lens** — is there a visceral wow moment in a 2-minute video?

### Step 3: Results and selection

| Rank | Proposal | Avg score |
|---|---|---|
| 1 (tie) | **SceneForge** — text-to-labeled-robot-vision-dataset (depth-ControlNet label transfer) | 8.3 |
| 1 (tie) | PanelForge — director-in-the-loop AI comic studio | 8.3 |
| 3 (tie) | DiffLab Tutor — RAG paper tutor that tool-calls a live SD pipeline | 8.0 |
| 3 (tie) | StoryReel — illustrated audiobook studio (image + audio diffusion) | 8.0 |
| 5 | ReRoom — ControlNet interior redesign studio | 7.8 |

Full judged proposals: [`phase1_ideation_results.json`](phase1_ideation_results.json).

**Human decision point:** the developer chose **SceneForge** (also the agent's recommendation): it is the only proposal that is simultaneously a strong DGM showcase (agentic LLM + RAG + structured outputs; ControlNet + distilled 4-step inference acceleration; custom diffusers pipeline) *and* a tool genuinely useful to the developer's own robotics research (synthetic data for π0/OpenVLA-style policies).

**Key insight the judges surfaced (carried into Phase 2):** the project's load-bearing assumption — *renderer labels stay valid on diffusion-generated variants because ControlNet-depth preserves geometry* — is unproven under 4-step Lightning sampling, where conditioning fidelity is known to degrade. Milestone M1 was therefore upgraded from a VRAM smoke test to a **quantitative go/no-go**: measure overlap between renderer ground-truth boxes and detections on generated images; if alignment fails, fall back to more steps / higher conditioning scale / stacked canny ControlNet.

---

## Phase 2 — Architecture Design and Task Decomposition (2026-06-10)

### Orchestration pattern: architect → 3 adversarial reviewers → finalizer

A second workflow spawned a **lead-architect agent** to produce an implementation-ready design doc (module decomposition, exact pydantic SceneSpec schema, coordinate conventions, data contracts for every stage, VRAM orchestration policy, Gradio event flow, milestone plan), which was then attacked by three parallel reviewers:

1. **VRAM & runtime reviewer** — re-derived every memory/latency claim with KV-cache and CUDA-context overheads included
2. **Data-contract & correctness reviewer** — walked every byte from SceneSpec to COCO zip; specifically tasked with verifying what depth representation `controlnet-depth-sdxl-1.0` was trained on, pyrender `RenderFlags.SEG` instance-mask mechanics, and COCO format validity
3. **Scope & demo reviewer** — hunted hidden time sinks and defined the minimal lovable demo path

A finalizer agent merged the critiques into [`ARCHITECTURE.md`](ARCHITECTURE.md) (v1.1, 701 lines) and the milestone checklist [`TASKS.md`](TASKS.md).

### Outcome: 38 issues found; the reviewers verified claims *live on the machine*

The adversarial reviewers did not just read the design — they ran commands against the real environment and falsified several assumptions that would each have cost hours at implementation time:

| Bottleneck found by agents | Resolution baked into the architecture |
|---|---|
| `transformers` 5.10.2 crashes on torch 2.6 (`torch.float8_e8m0fnu` needs torch≥2.7) — both OWLv2 and the SDXL ControlNet pipeline imports die | P0: pin `transformers==4.51.3` |
| Ollama 0.20.3 grammar-enforces JSON *structure/types* but **not numeric min/max** (verified: first live planner response emitted `elevation_deg: 0` against `ge=10`) | `clamp_to_bounds()` coerces numeric ranges deterministically *before* pydantic validation; the LLM repair loop handles only structural errors |
| Ollama server runs `OLLAMA_KEEP_ALIVE=-1` — any request omitting `keep_alive` pins its model in VRAM forever | every Ollama call passes explicit `keep_alive`; phase-barrier `ollama_unload()` sweeps `/api/ps` |
| `/api/generate` returns HTTP 400 for embedding models — the naive unload helper would eat a 15 s timeout every burst | unload embedding models via `/api/embed` with `keep_alive=0` (verified working) |
| pyrender `Scene.from_trimesh_scene` **drops node names** (verified in source), silently breaking `seg_node_map` instance masks | build the pyrender scene manually, retaining `Node` handles |
| Co-resident VRAM budget was 1–2 GB optimistic once measured (gemma4:e4b = 10.88 GB resident with KV cache, not 9.6 GB file size; free = 11.9 GiB < 12–13.5 GiB needed) | sequential mode is the expected default; co-resident demoted to an M1-verified bonus |
| `controlnet-depth-sdxl-1.0` was trained on MiDaS-style *inverse* depth (disparity), not a linear flip | `depth_prep.py` defaults to percentile-normalized disparity, with an M1 A/B test |
| pyrender 0.1.45 uses `np.infty`, removed in numpy 2.0 | `compat.py` shim (`np.infty = np.inf`) imported first everywhere |

Key architecture decisions (full rationale in ARCHITECTURE.md §12): procedural trimesh asset builders instead of downloaded mesh packs (depth silhouettes are all ControlNet sees); a two-layer LLM contract (strict LLM-facing schema → resolved `SceneSpec`); server-pinned camera `look_at`/`yfov` so the LLM can never empty every mask; M1 as a self-contained quantitative go/no-go (gate: `match_rate(IoU≥0.5) ≥ 0.70` and `mean_matched_iou ≥ 0.65`) with an L0→L4 conditioning ladder; every LLM-touching path has a deterministic fallback so the demo completes even with Ollama stopped.

---

## Phase 3 — Code Generation and Implementation (2026-06-10 →)

Environment prep (background, agent-driven): EGL headless render test passed (`EGL_OK`, valid depth buffer); ~13 GB of fp16 weights downloaded (SDXL base, depth ControlNet, VAE-fp16-fix, SDXL-Lightning 4-step UNet+LoRA, OWLv2); `transformers` pinned to 4.51.3 per P0.

### Orchestration pattern: parallel module agents against a frozen contract

Because ARCHITECTURE.md fixes every interface (exact file tree, function signatures, data contracts), six implementation agents were launched **in parallel**, each owning a disjoint set of files (spec/config/compat · asset builders · placement/compose · render backend · LLM director · labels/gpu), each writing and running its own unit tests. The M1 go/no-go harness ran as a separate GPU-bound agent in parallel, gated on model downloads.

### Result: all 7 agents green — 182 unit tests passing, zero module failures

### M1 go/no-go: the load-bearing assumption is TRUE, with a wide margin

The make-or-break experiment (12 synthetic scenes × OWLv2 detection vs renderer ground truth, full report in [`m1_report.md`](m1_report.md), evidence grids in `outputs/m1/`):

| Metric | Gate | Measured |
|---|---|---|
| match_rate (IoU ≥ 0.5) | ≥ 0.70 | **0.911** |
| mean matched IoU | ≥ 0.65 | **0.923** |
| hallucination rate | (report) | 0.067 |
| seconds / image @768² | est. 2–2.5 | **0.75** (3× faster than estimated) |

Decisions frozen into `sceneforge.yaml`: conditioning level **L0** (SDXL-Lightning 4-step UNet, cfg=0, cond_scale=0.85), depth mode **disparity** (tied with linear on IoU but 2× lower hallucination), VRAM mode **sequential** (co-resident measured at 0.92 GB min-free < 3.0 GB guard — correctly rejected by the rule the reviewers designed). 45/45 GT instances were gate-eligible; metrics reproduced exactly across two runs.

### Bottlenecks hit and resolved by agents during implementation

- **HF Xet downloads silently stalling** (dead connections, 0–5 MB/s): the M1 agent killed the stalled downloaders, deleted partial blobs, re-downloaded over plain HTTP with `HF_HUB_DISABLE_XET=1` at 35–99 MB/s, then **sha256-verified every large blob** against LFS etags.
- **ROS Jazzy contaminates pytest**: `/opt/ros/jazzy` site-packages on `PYTHONPATH` injects a broken `launch_testing` plugin (`ModuleNotFoundError: lark`) that crashes test collection. Standardized via `scripts/run_tests.sh` (`PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).
- **gemma4:e4b echoing the few-shot example**: live validation showed the planner repeats the example verbatim when the user task matches it — the director agent switched the few-shot to a workbench task so the demo's kitchen prompt can never collide.
- **Cross-agent test staleness**: M1 legitimately froze `s_per_img=0.75` into the yaml after the core agent had asserted it was null — the orchestrating agent reconciled the test to the post-M1 semantics. One agent's "done" is another agent's input; contracts in the architecture doc kept these collisions rare and shallow.

---

## Phase 4 — Interface Encapsulation and Finalization (2026-06-10)

### Orchestration pattern: integrator → adversarial verifier

A single **integrator agent** built the final layer against the frozen contracts (`orchestrator.py` demo-paced event pipeline, full Gradio Blocks UI with streamed planner tokens / Model3D viewer / live-filling gallery / re-forge sliders / COCO export, `app.py` with pre-warm), validated one real forge on GPU, then handed off to an **adversarial verifier agent** instructed to break the app the way a live demo would. Eight drills:

| Drill | Result |
|---|---|
| Full test suite | 189 tests green |
| Cold-start forge (new task, 2×3) | PASS — all artifacts, COCO round-trips, fidelity 0.81 |
| **Ollama-dead fallback** (LLM server unreachable) | PASS — forge completes via deterministic fallback, zero exceptions |
| Re-forge (camera+seed change, no LLM) | PASS — 1.7 s vs 6 s budget; labels track the moved camera; zero Ollama traffic |
| Gradio UI driven via API | PASS after a fix (below); full forge through the UI, streaming works |
| Edge prompts (unicorn/cloud-castle, 8-object stress) | PASS — RAG floor routes nonsense to `box`; collision resolver drops 1 object with a logged entry |
| Determinism | geometry/fallback/seed-law byte-identical; LLM plans not reproducible across fresh processes (Ollama server-side prefix-cache nondeterminism — documented caveat, not an app bug) |
| VRAM hygiene (consecutive forges) | PASS — free-VRAM delta +0.00 GB; GPU returns to desktop baseline on exit |

**Demo-breaking bug found and fixed by the verifier:** `on_forge` → `on_reforge` from the real UI crashed with `eglMakeCurrent EGL_BAD_ACCESS` — pyrender 0.1.45 leaves its EGL context current on the last render thread (`make_uncurrent` is a no-op `pass` upstream) and Gradio 6 hops worker threads per event. Fix: serialize renders behind a lock and explicitly unbind the EGL context after every render, plus a cross-thread regression test. This is exactly the class of bug that only appears under the real UI's threading model — unit tests and direct pipeline runs could never have caught it.

**Measured performance (RTX 3090):** warm 2-layout × 4-style forge **35.8 s** end-to-end (budget 66 s) · diffusion 0.75 s/img raw, 1.63 s/img wall · re-forge **1.7 s** · first visual 12.8 s (planner LLM dominates; tokens stream to the UI from ~1 s so it is never silent) · pipeline construction 17–22 s, hidden by `app.py` pre-warm · peak VRAM 11.6 GB reserved, min free 10.8 GB.

### Deliverables (agent-produced, human-verified)

Two final parallel agents produced the README (166 lines, with the measured numbers and an honest limitations section) and the demonstration materials — captured from a **real live session**: a Playwright-driven browser recorded a 74-second video of the actual demo flow (type task → FORGE → planner tokens stream → 3D layout + depth control appear → 8-image gallery fills live → mask overlay toggle → camera re-forge in 1.8 s → COCO export), plus three verified screenshots (`docs/img/`). Measured in the recorded session: 41 s click-to-done for 2 layouts × 4 styles, fidelity match_rate 97%, 8/8 images kept.

### Submission

- Public GitHub repository: https://github.com/kmliy901054/sceneforge (source, README.md, this workflow log, architecture doc, M1 experiment report, demo materials)
- Demo video: `docs/demo/demo_raw.mp4` (also submitted as `314831017_HW7.mp4`)

---

## Phase 5 — v2: video-first robot-data tools (2026-06-10)

Robot-learning datasets are mostly video episodes, so v2 extended SceneForge beyond single images: the `sceneforge.augment` module + `scripts/augment_frames.py` (appearance randomization of real robot frames/videos with bitwise-preserved workspace pixels and temporally smoothed masks, video in → video out), `scripts/augment_dataset.py` (**LeRobot v2.x dataset in → valid LeRobot datasets out**, one per style, with actions/states/timestamps copied bytewise — implemented as a minimal native pyarrow+ffmpeg reader/writer after the agent verified the on-disk layout against the official `lerobot/pusht` example, deliberately avoiding the heavy fast-churning `lerobot` dependency), `scripts/forge_viewsweep.py` (camera-grid viewpoint-randomized COCO export with per-image K/pose), RGB-D + camera-parameter export as the default, and a Gradio **Video Augment tab** alongside the Forge tab. Implemented and validated end-to-end by dedicated agents — including a real GPU run on a mini LeRobot dataset (0.56 s/img, outputs pass structural validation, near-pixels preserved within codec noise) and a gradio_client drive of the new tab. Test suite grew 189 → 229, all green.

### Final tally of the agentic build

| Phase | Agents | What they did |
|---|---|---|
| 1 Ideation | 20 | 5 grounded proposals × 3-lens judge panel; human picked SceneForge |
| 2 Architecture | 5 | architect → 3 adversarial reviewers (38 issues, many verified live on-machine) → finalizer |
| 3 Implementation | 7 | 6 parallel module agents + M1 go/no-go GPU experiment (gate passed 0.911/0.923) |
| 4 Integration | 2 | integrator → adversarial verifier (8 drills; found+fixed the EGL thread-hop demo-breaker) |
| Deliverables | 2 | README writer + Playwright demo capture |
| **Total** | **36 sub-agents** | orchestrated by one lead agent (Claude Code, Fable 5) across ~2M sub-agent tokens, in one day |
