# Appearance-Invariance Probe for OpenVLA-7B

**Date:** 2026-06-10 . **Hardware:** 1x RTX 3090 24GB . **Model:** `openvla/openvla-7b` (bfloat16, sdpa, deterministic `do_sample=False`, `unnorm_key="bridge_orig"`)

**Hypothesis under test:** VLA models lack 3D spatial understanding and largely memorize RGB appearance, so their predicted actions should change drastically when only the appearance (background, lighting, color) of a scene changes while geometry and task stay fixed.

## 1. Setup

- **Probe frames (n = 22):** real BridgeData V2 raw WidowX tabletop frames downloaded from
  `https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/` -- 11 scene/task
  combinations (toykitchen2/6/7, folding_table, tabletop_dark_wood, robot_desk x drawer_pnp,
  stack_blocks, many_skills, fold_cloth, sweep_granular), 2 trajectories each, one mid-episode frame
  per trajectory plus a temporal neighbor (t+3 steps). Instructions come from each trajectory's
  `lang.txt`. Full provenance (URLs, frame indices, instructions): `images/provenance.json`.
  This matches the `bridge_orig` training distribution of the checkpoint, so the original-frame
  predictions are in-distribution by construction.
- **Inference:** one model load, 440 deterministic predictions (22 originals + 22 temporal
  neighbors + 396 variants). Repeat-prediction on identical input gives bitwise-identical actions
  (max |diff| = 0.0), so all reported deviations are caused by the input image alone (instruction
  fixed per frame). Raw 7-DoF actions: `results/actions.json`.

### Variant arms (per frame)

| Arm | n/frame | What changes | What is fixed |
|---|---|---|---|
| C `jpeg95` | 1 | JPEG re-encode (quality 95) -- measurement floor | everything |
| P photometric | 10 | brightness +/-35%, contrast +/-35%, hue +/-20deg, gamma 0.6/1.6, warm white-balance shift, ISO noise (sigma=8) | geometry, semantics |
| S background restyle | 4 | SDXL-base + depth-ControlNet **inpainting restricted to the far-background mask** (4 style prompts: industrial / warm kitchen / white lab / night room). Mask = Depth-Anything-V2-Small depth, farthest 28th percentile region intersected with upper image (lower 38% always kept), keep-mask dilated 8 px. Original near pixels pasted back **exactly** (verified max diff = 0 inside keep mask) | robot, objects, workspace, geometry -- pixel-identical |
| G geometry baseline | 3 | 5% / 10% crop-shift, 10deg rotation -- genuine geometric change, calibrates "how big is a real change" | appearance statistics |
| T temporal baseline | 1 | the *same* episode 3 steps later -- natural scale of meaningful action change | scene, task |

Masks and audit overlays for every frame: `masks/` (`*_keep.png`, `*_audit.jpg`, `masks/audit_sheet.jpg`).
Background restyle covers 9-25% of pixels (mean ~19%) -- deliberately conservative so that task-relevant
content is untouched.

## 2. Results

Deviation of the predicted 7-DoF action w.r.t. the same frame's original prediction.
Mean +/- 95% CI computed over per-frame means (22 frames, avoids pseudo-replication).

| Arm | translation L2 (m) | translation cosine | rotation L2 (rad) | gripper flip rate |
|---|---|---|---|---|
| C: jpeg95 floor | 0.00072 +/- 0.00103 | 0.967 | 0.0066 +/- 0.0078 | 0.0% |
| P: photometric | 0.00245 +/- 0.00095 | 0.849 | 0.0145 +/- 0.0064 | 2.3% |
| S: bg restyle | 0.00607 +/- 0.00311 | 0.796 | 0.0263 +/- 0.0109 | 1.1% |
| G: geometry | 0.00891 +/- 0.00278 | 0.547 | 0.0367 +/- 0.0089 | 7.6% |
| T: temporal (t+3) | 0.01636 +/- 0.00821 | 0.108 | 0.0507 +/- 0.0148 | 22.7% |

Typical original action magnitude |dt|: mean 12.2 mm, median 9.3 mm.

**Key ratios (translation L2, per-frame paired):**

| comparison | ratio | frames where appearance > geometry | Wilcoxon p |
|---|---|---|---|
| P / G | **0.28** | 0% (0/22) | 0.0000 |
| S / G | **0.68** | 14% (3/22) | 0.0059 |
| P / T | 0.15 | -- | -- |
| S / T | 0.37 | -- | -- |

Robustness: excluding the two frames whose restyle mask grazed task-relevant objects
(frame16/17, sweep tray) leaves the conclusions unchanged (S/G = 0.62, P/G = 0.33).

Strongest individual perturbations (mean translation L2): S warm-kitchen (6.9 mm),
S white-lab (6.5 mm), S night-room (6.1 mm), contrast +35% (5.1 mm); weakest: white-balance
shift (0.8 mm, indistinguishable from the JPEG floor), brightness +/-35% (1.7 mm).

Plots: `results/per_arm_deviation.png` (violin/box per arm), `results/paired_scatter.png`
(per-frame P vs G and S vs G), `results/qualitative_grid.jpg` (4 frames x 6 conditions with
predicted action annotations). Numbers: `results/summary.json`, per-variant records:
`results/deviations.json`.

## 3. Interpretation (honest)

**The strong form of the hypothesis -- "actions collapse when only appearance changes" -- is NOT
supported by this probe.** The deviation ordering is exactly what a model with usable spatial
grounding would produce:

```
jpeg floor (0.7mm) < photometric (2.5mm) < bg restyle (6.1mm) < geometry (8.9mm) < temporal (16.4mm)
```

- **Photometric robustness is good.** Global lighting/color changes move the predicted translation
  by only 2.5 mm on average -- 28% of a genuine geometric change, and in 0/22 frames did photometric
  perturbations out-deviate geometric ones. Gripper decisions flip in only 2.3% of cases. Many P
  variants produce *bitwise-identical* actions, partly thanks to OpenVLA's 256-bin action
  discretization absorbing small logit shifts.
- **Background appearance does leak into the policy, though.** Restyling only the far background
  (robot/objects/workspace pixel-identical) shifts the commanded translation by 6.1 mm on average --
  half the magnitude of a typical commanded step (12.2 mm) and 68% of the geometry baseline, with
  translation cosine dropping to 0.80. A perfectly scene-grounded policy should be near the JPEG
  floor here; OpenVLA is ~8x above it. So the *weak* form of the hypothesis -- appearance is not
  fully disentangled from action -- **is** supported.
- **Geometry and time dominate, as they should.** Genuine geometric changes (G) and the natural
  temporal evolution of the episode (T) produce by far the largest deviations (cosine 0.55 / 0.11,
  gripper flips 7.6% / 22.7%), i.e. the model responds much more to *where things are* than to
  *what color the room is*.

## 4. Caveats

1. **Open-loop probe != closed-loop success.** A 6 mm action shift per step could either wash out or
   compound over a rollout; this probe cannot distinguish graceful degradation from task failure.
2. **Single checkpoint, single unnorm key** (`openvla-7b`, `bridge_orig`). No claim about pi0,
   OpenVLA-finetuned variants, or other action heads.
3. **Background-only restyle is conservative:** 9-25% of pixels, task surface and robot untouched,
   depth-consistent inpainting. Full-scene relighting, texture changes on the workspace itself, or
   camera-pose changes would very likely hurt more -- this probe lower-bounds appearance sensitivity.
4. **Mask imperfections:** in 2/22 frames (sweep-granular tray) the restyle region touched the rim
   of the task surface; excluding them changes S/G from 0.68 to 0.62 (conclusions unchanged).
5. **Instruction quality varies** (Bridge `lang.txt` crowd annotations, e.g. "it has opened the
   drawer"); instructions are held fixed within a frame, so they affect absolute actions, not
   deviations, but instruction-image interactions are not explored.
6. **Arm G includes a slight zoom** (crop-shift) and reflect-padding for rotation; it is a proxy
   for, not an exact model of, camera motion. Arm T uses t+3 raw steps (~0.6-1.5 s of motion).
7. **Discretization floor:** OpenVLA's tokenized action space quantizes outputs; tiny perturbations
   often map to the identical token sequence, which deflates P-arm deviations relative to a
   continuous-head VLA.

## 5. Repro

```
code/download_frames.py   # BridgeData V2 raw crawler (provenance JSON)
code/validate_harness.py  # step-1 harness check (norm_stats keys, determinism)
code/make_masks.py        # Depth-Anything-V2 -> near/far masks + audits   (env dgan)
code/make_variants_pg.py  # arms P, G, C                                  (env dgan)
code/make_variants_s.py   # arm S: SDXL depth-ControlNet bg inpainting    (env dgan)
code/run_sweep.py         # 440 deterministic predictions                 (env openvla)
code/analyze.py           # stats + plots                                 (env dgan)
code/make_grid.py         # qualitative grid
```
