# Model-Generation Comparison: OpenVLA-7B vs SpatialVLA-4B on the Appearance-Invariance Probe

**Date:** 2026-06-10 . **Hardware:** 1x RTX 3090 24GB . Extends `REPORT.md` (same 22 BridgeData V2
frames, same variants, same instructions, same metrics).

**Falsifiable prediction under test:** SpatialVLA-4B is a 2025 "3D-aware" VLA (explicit monocular
depth + Ego3D position encoding). If 3D-aware architectures disentangle appearance from action
better, SpatialVLA should show LOWER within-model appearance/geometry deviation ratios (S/G, P/G)
than OpenVLA-7B.

**Result in one line: the prediction is NOT supported.** SpatialVLA's aggregate ratios are
slightly *higher* (S/G 0.74 vs 0.68, P/G 0.40 vs 0.28) and its gripper flips under appearance
changes numerically more frequent; no per-frame paired difference between the models reaches
significance (all Wilcoxon p >= 0.22). Both models reproduce the same qualitative ordering
`floor < photometric < bg-restyle < geometry < temporal`.

## 1. Setup

- **Identical probe:** the exact (frame, variant, instruction) set of `REPORT.md` — 440
  deterministic predictions per model (22 originals + 22 temporal neighbors + 396 variants).
  Deviations are computed **within model** (each variant vs the *same model's* prediction on the
  same frame's original image), so the cross-model comparison of ratios is scale-free.
- **Model:** `IPEC-COMMUNITY/spatialvla-4b-224-pt` (bf16, ~8.1 GB GPU), env `dgan`
  (transformers 4.51.3, torch 2.6). `unnorm_key="bridge_orig/1.0.0"` — its q01/q99
  unnormalization bounds are identical to OpenVLA's `bridge_orig` key, and the output is the same
  7-DoF WidowX EE-delta convention (dx,dy,dz [m]; droll,dpitch,dyaw [rad]; gripper in [0,1]), so
  absolute mm are directly comparable. Verified on the harness-validation frames: plausible
  magnitudes (11-35 mm |dt| vs OpenVLA 12-13 mm), same gripper polarity.
- **Determinism:** unlike OpenVLA, greedy decoding alone was NOT bitwise-deterministic for
  SpatialVLA (near-tied logits + nondeterministic cuBLAS split-k reductions occasionally flipped an
  action token, max repeat |diff| 0.012 on frame00). Fixed with `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
  `torch.use_deterministic_algorithms(True)`: verified max repeat |diff| = 0.0 (3 runs x 6 frames,
  plus an end-of-sweep re-check). Raw actions: `results/actions_spatialvla.json`
  (sweep log `results/sweep_spatialvla.log`).

### Recipe differences (caveats for absolute-number comparisons)

| | OpenVLA-7B | SpatialVLA-4B |
|---|---|---|
| backbone | DINOv2+SigLIP -> Llama-2-7B | SigLIP-400M + **ZoeDepth Ego3D position encoding** -> Gemma2-2.6B (PaliGemma2) |
| input | 224x224 RGB | 224x224 RGB (+ internal monocular depth from the same RGB; Bridge camera intrinsics from its config) |
| action head | 7 tokens/step, 256 **uniform** bins per dimension | 3 tokens/step from 8194 **adaptive Gaussian** spatial bins (spherical translation 8x16x32, rotation 16^3, gripper 2) |
| horizon | 1 step | chunk of 4 steps — the **first** (current-step) action is used here |
| prompt | `In: What action should the robot take to {instr}?\nOut:` | `What action should the robot take to {instr}?` |
| training data | OXE (incl. bridge_orig) | OXE + RH20T (incl. bridge_orig) — Bridge in-distribution for both |

The coarser 3-token head quantizes more aggressively: fraction of variants mapping to the
**bitwise-identical** action — OpenVLA C 82% / P 49% / S 30% / G 5%; SpatialVLA C 91% / P 60% /
S 28% / **G 18%**. SpatialVLA absorbs even genuine geometric changes into the same action token
3.6x more often, which deflates its G baseline and pushes its S/G and P/G ratios *up*; this is a
real property of the deployed policy, but it means small ratio differences should not be
over-interpreted (hence the paired tests below).

## 2. Results

### Per-arm deviations (mean of per-frame means +/- 95% CI, 22 frames)

| Arm | OpenVLA trans L2 (mm) | SpatialVLA trans L2 (mm) | OpenVLA cos | SpatialVLA cos | OpenVLA grip flip | SpatialVLA grip flip |
|---|---|---|---|---|---|---|
| C: jpeg95 floor | 0.72 +/- 1.03 | 0.63 +/- 1.10 | 0.967 | 0.974 | 0.0% | 0.0% |
| P: photometric | 2.45 +/- 0.95 | 3.42 +/- 2.18 | 0.849 | 0.906 | 2.3% | 4.5% |
| S: bg restyle | 6.07 +/- 3.11 | 6.37 +/- 2.87 | 0.796 | 0.805 | 1.1% | 3.4% |
| G: geometry | 8.91 +/- 2.78 | 8.61 +/- 3.75 | 0.547 | 0.672 | 7.6% | 12.1% |
| T: temporal (t+3) | 16.36 +/- 8.21 | 23.78 +/- 8.57 | 0.108 | -0.346 | 22.7% | 22.7% |

Typical original |dt|: OpenVLA 12.2 mm mean / 9.3 mm median; SpatialVLA 15.3 / 11.5 mm.
Cross-model paired Wilcoxon on absolute per-frame trans L2: no arm differs significantly except T
(23.8 vs 16.4 mm, p = 0.003 — the chunked policy reacts more to a genuine 3-step temporal change;
its t+3 first-chunk action even reverses direction on average, cos = -0.35).

### Key within-model ratios (translation L2, per-frame paired)

| model | P/G | S/G | P/T | S/T | frames P>G | frames S>G | Wilcoxon P<G / S<G |
|---|---|---|---|---|---|---|---|
| OpenVLA-7B | **0.28** | **0.68** | 0.15 | 0.37 | 0% | 14% | p=0.0000 / p=0.0059 |
| SpatialVLA-4B | **0.40** | **0.74** | 0.14 | 0.27 | 23% | 23% | p=0.0012 / p=0.0355 |

Both models keep appearance significantly below geometry (the probe's main conclusion replicates
on a second, architecturally different VLA). Robustness: excluding the two imperfect-mask frames
(frame16/17) gives OpenVLA S/G 0.62 / P/G 0.33, SpatialVLA S/G 0.70 / P/G 0.42 — same picture.

### Cross-model paired tests (the comparison that matters)

Per-frame ratio is undefined when a model's G mean is exactly 0 (3/22 frames for SpatialVLA, 0/22
for OpenVLA — the discretization effect above), so two complementary paired tests:

| test | n frames | OpenVLA | SpatialVLA | frames SpatialVLA lower | Wilcoxon p |
|---|---|---|---|---|---|
| P/G ratio (both G>0) | 19 | mean 0.32* | mean 0.45* | 53% | 0.57 |
| S/G ratio (both G>0) | 19 | mean 0.54* | mean 0.81* | 42% | 0.42 |
| P idx (P-G)/(P+G), zero-safe | 21 | -0.57 | -0.39 | 48% | 0.36 |
| S idx (S-G)/(S+G), zero-safe | 21 | -0.36 | -0.15 | 38% | 0.22 |

\* mean of per-frame ratios (long-tailed; the aggregate ratios in the table above are
mean(arm)/mean(G), as in REPORT.md).

Gripper flips under appearance change: SpatialVLA flips numerically more (P 4.5% vs 2.3%,
S 3.4% vs 1.1%) but per-frame paired Wilcoxon is null (p = 0.24 / 0.32); flips under G
(12.1% vs 7.6%, p = 0.60) and T (22.7% = 22.7%) likewise indistinguishable.

Plots: `results/per_arm_deviation_comparison.png` (grouped violin/box per arm, both models),
`results/paired_ratio_scatter.png` (per-frame paired S/G and P/G, OpenVLA vs SpatialVLA).
Numbers: `results/summary_comparison.json`; per-variant records
`results/deviations_spatialvla.json`; console log `results/analyze_comparison.log`.

## 3. Verdict (honest)

**No evidence that the 3D-aware model disentangles appearance better.** Every point estimate of
the pre-declared key metrics goes the *wrong* way for the prediction — SpatialVLA's S/G (0.74 vs
0.68), P/G (0.40 vs 0.28), symmetric indices, fraction of frames where appearance out-deviates
geometry (23% vs 0-14%), and appearance-driven gripper flips are all equal or worse than
OpenVLA's — but none of the paired cross-model differences is statistically distinguishable from
zero at n = 22 (p >= 0.22). The defensible conclusion is a null: **a 2025 depth-conditioned
architecture shows no measurable improvement in appearance-action disentanglement over 2024
OpenVLA on this probe**, while the probe's own within-model conclusions (appearance < geometry,
significant; background leak ~8-10x above the jpeg floor) replicate cleanly on both models. One
metric nominally favors SpatialVLA — S/T = 0.27 vs 0.37 — but T is not a comparable yardstick
across models (the chunked policy's temporal reaction is significantly larger, p = 0.003), so we
do not lean on it.

A plausible mechanism for why explicit depth does not help here: SpatialVLA's Ego3D encoding is
*estimated from the same RGB pixels* (ZoeDepth), so a background restyle can perturb the depth
map itself — the "3D" channel is not independent of appearance.

## 4. pi0 feasibility (assessed, not run)

A fair pi0 arm is **not feasible on these probe frames**: the released openpi checkpoints that
support out-of-the-box inference are DROID- or ALOHA-conditioned policies that require
observations our Bridge frames do not contain — a wrist-camera image plus proprioceptive state
(joint positions + gripper) alongside the exterior view — and zero-filling those inputs would put
the policy far out of distribution, so any measured deviation would reflect OOD noise rather than
appearance (dis)entanglement, in an action space (Franka joint-space chunks) that is not
comparable to WidowX EE-deltas. There is no Bridge/WidowX input adapter or norm-stats asset in
the openpi release (`/home/pairlab/code_monitor_pi0/openpi/src/openpi/policies/` has only aloha,
droid, libero), and the pi0_base checkpoint is explicitly a fine-tuning artifact, not a deployable
Bridge policy. A fair pi0 probe needs DROID-style episodes — synchronized exterior + wrist camera
frames with recorded joint/gripper state and language annotations — through `pi0_fast_droid` or
`pi05_droid`, with the S-arm restyle applied to the exterior view only.

## 5. Caveats

1. **Confounded model pair.** OpenVLA-7B vs SpatialVLA-4B differ in size, VLM lineage, training
   mix (OXE vs OXE+RH20T), action discretization, and horizon — this is a comparison of two
   *model generations as deployed*, not a controlled ablation of 3D-awareness. A within-family
   ablation (SpatialVLA with Ego3D/depth disabled) would isolate the mechanism; the released
   checkpoint does not expose that switch.
2. **Discretization asymmetry inflates SpatialVLA's ratios.** Its 3-token adaptive-bin head maps
   18% of geometry variants to the identical action (OpenVLA: 5%), shrinking the G denominator.
   The zero-safe symmetric-index test addresses the undefined-ratio frames but not the underlying
   coarseness; conclusions rest on the (null) paired tests, not on the point estimates.
3. **Chunked policy, first step only.** Using the first action of SpatialVLA's 4-step chunk is the
   standard deployment choice, but chunk-averaged deviations could differ.
4. **n = 22 frames** gives limited power for cross-model differences: roughly, paired effects
   below ~0.5 SD of the per-frame differences are undetectable. "No significant difference" here
   is a power-limited null, not proof of equivalence.
5. **Determinism handling differs**: SpatialVLA required forced-deterministic kernels; the fact
   that single-token flips occur at kernel-noise level confirms near-tied logits at bin
   boundaries — deviations for SpatialVLA are "lumpier" (single bin flips are large).
6. All REPORT.md caveats (open-loop probe, conservative background-only restyle, mask
   imperfections on frame16/17, instruction quality) apply unchanged.

## 6. Repro

```
code/validate_spatialvla.py     # step-1 harness check (norm key, determinism, scale)   (env dgan)
code/run_sweep_spatialvla.py    # 440 deterministic predictions; needs CUBLAS_WORKSPACE_CONFIG=:4096:8  (env dgan)
code/analyze_comparison.py      # merged stats + comparison plots                        (env dgan)
```
