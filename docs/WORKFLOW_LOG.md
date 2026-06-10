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

Because ARCHITECTURE.md fixes every interface (exact file tree, function signatures, data contracts), six implementation agents were launched **in parallel**, each owning a disjoint set of files (spec/config/compat · asset builders · placement/compose · render backend · LLM director · labels/gpu), each writing and running its own unit tests. The M1 go/no-go harness ran as a separate GPU-bound agent once model downloads completed.

*(Results recorded below as milestones complete.)*
