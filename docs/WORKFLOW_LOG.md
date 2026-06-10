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

A finalizer agent merged the critiques into [`ARCHITECTURE.md`](ARCHITECTURE.md) and the milestone checklist [`TASKS.md`](TASKS.md).

*(Results recorded below when the workflow completes.)*

In parallel, the environment-prep pipeline ran in the background: EGL headless-rendering smoke test + ~13 GB of fp16 model downloads (SDXL base, depth ControlNet, VAE-fp16-fix, SDXL-Lightning 4-step UNet+LoRA, OWLv2).
