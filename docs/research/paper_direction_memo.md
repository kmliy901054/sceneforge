# Paper Direction Memo — SceneForge / VLA Appearance-Geometry Disentanglement

**Date:** 2026-06-10
**Status:** Final direction decision, synthesizing 4 candidate directions, novelty sweeps, feasibility audit, and 3 judge scorecards.

---

## 1. Recommendation

**RECOMMENDED: D1, narrowed and merged with the D4 seed — "GeoProbe: Counterfactual Sensitivity Metrology for VLAs."**
A measurement-methodology + diagnosis paper (explicitly *not* a leaderboard/benchmark paper), whose flagship result is the depth-provenance finding, and whose final arm — open-loop counterfactual sensitivity predicts closed-loop robustness — is the bridge that makes the D4 monitoring paper the natural follow-on.

**RUNNER-UP: D4 — counterfactual appearance probing as a runtime failure signal** (one-paragraph version in §3).

### Judges' aggregate reasoning

| Direction | J1 (novelty) | J2 (feasibility) | J3 (thesis/lab fit) | Sum /30 | Top picks |
|---|---|---|---|---|---|
| D1-benchmark | 5 | 8 | 8.5 | **21.5** | J2, J3 |
| D4-monitor | 6 | 6 | 7.5 | 19.5 | — |
| D2-distill | 8 | 3 | 4.5 | 15.5 | J1 |
| D3-counterfactual | 4 | 2 | 5 | 11 | — |

D1 wins on aggregate and is the top pick of two of three judges — but **only in narrowed form**. All three judges and the novelty sweep converge on the same prescription: the grand "multi-VLA leaderboard + perturbation benchmark" version is dead on arrival (LIBERO-Plus, VLATest, INT-ACT, COLOSSEUM jointly own it, and leaderboard numbers go stale within one review cycle), while the narrow core — counterfactual-*paired* measurement with bitwise-fixed geometry, a calibrated continuous S/G quotient, the depth-provenance causal chain, and predictive validity against closed-loop failure — is **unclaimed by anyone** and is exactly what the completed 880-prediction probe already seeds. J1's low D1 score punished the leaderboard pitch; his salvage note ("deploy those assets as the evaluation instrument of a thesis paper") describes precisely what J2 and J3 scored 8/8.5.

The merge with D4's seed is judge-endorsed: J3 notes D4 "is literally this instrument run at deployment... done second, it is the natural lab-agenda payoff chapter," and the D1 novelty verdict lists the open-loop→closed-loop correlation as one of its three unclaimed contributions. We therefore fold a *predictive-validity arm* into this paper and reserve the full runtime monitor for paper #2.

Why not the others: **D2** (J1's pick — genuinely the sharpest falsifiable claim) fails the hardware audit: 24GB sits at or below every documented fine-tuning floor (pi0 LoRA <1.5GB headroom; OpenVLA LoRA needs ~27GB), teacher-resident distillation doubles VRAM, and overnight-per-run iteration starvation is what kills distillation projects; GLaD (2512.09619) already owns the headline, leaving a single fragile contrast that dies if GT and estimated teachers tie. It is the right *chapter-3* paper after a GPU upgrade — and it will be far stronger if our instrument is already the published standard it gets evaluated on. **D3** combines D2's VRAM wall with research-grade 3DGS capture infrastructure (single-camera Bridge/DROID episodes cannot be faithfully re-rendered), a technically fragile equivariance framing, mandatory data-mixing and representation-loss baselines, and an acute scoop trajectory from Invariance Co-training (2512.05230) — three hard subsystems for one student is the canonical unfinishable project.

---

## 2. Recommended direction in full

**Working title:** *GeoProbe: Counterfactual Sensitivity Metrology for Vision-Language-Action Models — appearance leakage, depth provenance, and what open-loop sensitivity predicts about closed-loop failure.*

### 2.1 Thesis statement (falsifiable)

> VLA action heads are causally sensitive to action-irrelevant appearance, and this leakage (i) is measurable open-loop on real robot frames via counterfactual pairs in which workspace geometry is held **bitwise-fixed**, using a continuous action-space sensitivity quotient **S/G** (appearance-sensitivity over matched geometric-sensitivity) calibrated against a measurement floor; (ii) is **inherited, not corrected**, by "3D-aware" VLAs whose depth channel is estimated from the same RGB — the causal chain *appearance edit → estimated-depth shift → action shift* is demonstrable by intervening on the depth channel; and (iii) **predicts closed-loop robustness rankings** under appearance and viewpoint shift in SIMPLER/LIBERO.

Each clause is independently falsifiable: (i) fails if action deltas under certified-identical-geometry restyles do not exceed the calibrated floor across models (our existing data already refutes this null at ~8x floor); (ii) fails if freezing or ground-truthing the depth channel under restyle does not reduce action shift in depth-conditioned models; (iii) fails if the rank correlation between open-loop S/G and closed-loop perturbation robustness is indistinguishable from zero — and that calibrated null is itself a publishable finding about what binary benchmarks measure.

### 2.2 Contributions (as a reviewer would list them)

1. **The first counterfactual-paired disentanglement instrument for VLAs.** Matched evaluation suites where appearance varies while geometry is *certifiably bitwise-identical* (and vice versa, with geometric edits magnitude-matched to expected action deltas), built on real Bridge/DROID frames plus SceneForge synthetic scenes with exported ground-truth geometry and camera-grid viewpoint sweeps; a continuous, deterministic, mm-level action-space metric (the S/G quotient) calibrated against an explicit measurement floor. Released as a versioned artifact (pair suites + harness + metric code). No existing benchmark pairs its perturbations or reports below the binary-success granularity.
2. **The depth-provenance finding.** A controlled demonstration that 3D-aware VLAs whose depth is RGB-estimated (SpatialVLA's ZoeDepth Ego3D; DepthVLA/QDepth-VLA-style pseudo-depth) inherit appearance sensitivity *through* the depth channel — shown by intervention: depth re-estimated from restyled RGB vs. depth frozen from the original frame vs. ground-truth depth. This converts the motivation sentence of the entire 3D-injection literature (Spatial Forcing, GLaD, Evo-0...) from assertion into measurement, and explains our published SpatialVLA null result (p≥0.22).
3. **Cross-generation diagnosis with layer-wise localization.** S/G profiles across 6–8 VLAs spanning architectures (autoregressive discretized, flow/diffusion action heads, 3D-conditioned), plus the unclaimed mechanistic question: *at which layer does appearance contaminate the action readout*, answered with counterfactual-paired activations rather than generic probing.
4. **Predictive validity → a deployment-time reliability signal.** The first correlation study between cheap open-loop counterfactual sensitivity and closed-loop robustness rankings (sampled SIMPLER + LIBERO-Plus-style conditions), establishing the metric as a runtime-usable signal and reconciling LIBERO-Plus's "background-resilient" success rates with sub-threshold action shifts. This arm is the explicit bridge to the lab's Code-as-Monitor / LooCoT-VLA agenda.

### 2.3 The counterintuitive hook

Two-part, both already seeded by completed work: **(a) "3D-aware" VLAs are not appearance-robust, because their 3D is RGB in disguise.** The depth channel everyone adds as a grounding firewall is actually a leakage conduit when estimated from the same pixels — our pre-registered prediction that SpatialVLA disentangles better than OpenVLA *failed*. **(b) Binary success hides ungroundedness.** LIBERO-Plus reports VLAs are *resilient* to background changes at success-rate level; our instrument shows bitwise-identical-workspace restyles shift commanded actions ~6mm — 8x the measurement floor and ~70% of a genuine geometric change. Both findings say the field's standard evidence (success rates, depth inputs) systematically misleads about geometric grounding.

### 2.4 Positioning against the five closest works

**LIBERO-Plus** (arXiv:2510.13626) is the scale benchmark: 10,030 sim tasks, 7 perturbation axes, binary success — but its perturbations are unpaired (geometry is never certifiably held fixed while appearance varies), it has no continuous action-space metric, no floor calibration, and no real-robot frames; we are the *metrology* complement, and our Contribution 4 directly reconciles its background-resilience finding with sub-threshold leakage. **"VLAs Are More Generalizable Than You Think"** (arXiv:2512.02902) owns a mechanism-level diagnosis ("failure lives in spatial modeling") but reaches it via adaptation ablations on a single axis (viewpoint, LIBERO-only); we diagnose via controlled counterfactual probing with a factorized appearance-vs-geometry design and add depth provenance, which it never touches. **INT-ACT** (arXiv:2506.09930) is the closest probing-suite-in-spirit, but its factorization is perception-vs-execution in SIMPLER; ours is appearance-vs-geometry, open-loop on real frames, with a quotient metric and internals. **The 3D-injection family — Spatial Forcing** (arXiv:2510.12276), **GLaD** (arXiv:2512.09619), DepthVLA (2510.13375), QDepth-VLA (2510.14836), Evo-0 (2507.00416) — *asserts* that RGB-estimated depth is weak grounding as its motivation and validates fixes only by LIBERO success; none measures disentanglement; we supply both the measured leak path their motivation assumes and the instrument their fixes should be scored on (they become evaluation subjects, not competitors). **COLOSSEUM** (arXiv:2402.08191) **/ FactorWorld** (arXiv:2307.03659) established vary-factors-independently methodology pre-VLA; the reviewer question "how is this not COLOSSEUM for VLAs?" is answered: LIBERO-Plus already is — we contribute what neither can express: counterfactual *pairing*, floor-calibrated continuous sensitivity, causal depth-channel intervention, and open-loop→closed-loop predictive validity. (Adjacent: BYOVLA, arXiv:2410.01971, uses perturbation sensitivity as a runtime *patch*; we validate it as a *measurement*, which is what licenses the monitoring follow-on.)

### 2.5 Experimental plan

**Suites / datasets.**
- *Suite A — appearance counterfactuals (real):* scale the existing probe from 22 to ~200 Bridge frames + ~100 DROID frames, ~20 restyle variants each via the shipped episode background-restyler (workspace mask bitwise-identical; labels exact). Per-model measurement-floor recalibration (deterministic decoding, codec/resize controls).
- *Suite B — geometry counterfactuals (matched magnitude):* SceneForge synthetic scenes with exported GT geometry; object-displacement edits magnitude-matched to expected action deltas (the G denominator); camera-grid viewpoint sweeps with exported K/pose (shipped tooling).
- *Suite C — depth-provenance interventions:* for depth-conditioned models, three conditions under identical restyles: depth re-estimated from restyled RGB / depth frozen from original RGB / ground-truth depth (SceneForge sim + RGB-D export; real RGB-D where available).
- *Closed-loop:* SIMPLER (Visual Matching + Variant Aggregation; WidowX-Bridge + Google-Robot subsets) via the DelinQu/SimplerEnv-OpenVLA fork (the path SpatialVLA's own repo endorses); LIBERO + sampled LIBERO-Plus-style perturbation conditions via openpi's dockerized client-server harness. **Sampled design, not a sweep:** ~4 tasks × ~8 perturbation conditions × ~25 episodes per model.

**Models (6–8, inference-only, all documented to fit one 3090):** OpenVLA-7B (bf16 16.8GB; int4 7GB), OpenVLA-OFT, SpatialVLA-4B, pi0 and pi0-FAST (openpi inference >8GB), GR00T N1.5 (3090-benchmarked by NVIDIA), Octo-base, MiniVLA. Stretch: CogACT, or a Spatial-Forcing/GLaD checkpoint if released — scored *on* the instrument.

**Open-loop metrics:** per-frame commanded-action delta (translation mm, rotation, gripper, per-dimension); S = mean appearance-counterfactual delta; G = matched geometric delta; S/G quotient with bootstrap CIs over frames; floor-normalized effect sizes; per-axis decomposition. Layer-wise: counterfactual activation deltas and linear probes per block; "contamination depth" = earliest layer where appearance pairs separate in the action-relevant subspace.

**Closed-loop metrics:** success-rate deltas under matched perturbation conditions; Spearman rank correlation between open-loop S/G and closed-loop robustness across model×condition cells (pre-registered analysis, ≥8 cells for power); per-frame divergence on rollout frames vs. episode outcome (the D4 seed, collected for free during rollouts).

**Ablations:** restyle family (ControlNet restyle vs. chroma-key vs. color jitter at matched pixel budget); viewpoint-shift magnitude grid; depth-provenance conditions (Suite C); language paraphrase control (prompt held fixed vs. varied); action-head type (discretized AR vs. flow); floor controls (codec, resize, nondeterminism).

**Baselines / controls:** random-noise perturbation at matched pixel budget (isolates appearance *structure* from pixel delta); LIBERO-Plus-style binary protocol run on our identical conditions (quantifies exactly what binary success misses); published robustness numbers cross-checked per model; INT-ACT-style decomposition where comparable.

### 2.6 Resource budget (mapped to one RTX 3090 24GB)

- **Open-loop (Suites A–C): trivially fits.** Largest model is OpenVLA-7B at 16.8GB bf16 (7GB int4 with no documented performance loss); SpatialVLA-4B, pi0 inference (>8GB per openpi README), GR00T N1.5 (explicitly benchmarked on 3090) all fit singly. ~200–300k deterministic predictions ≈ a few GPU-days, batched overnight.
- **Closed-loop: wall-clock-bound, not VRAM-bound.** SIMPLER wants an RTX-class GPU; sim and policy contend on one card → hours-to-days per model per condition block. The sampled design (~800 episodes/model) budgets to ~4–6 GPU-weeks total, run as a continuous overnight queue from M4.
- **No fine-tuning anywhere = the 24GB ceiling never binds.** The real costs are (a) integration debt — separate uv/conda/docker environments per policy (the fork's transformers/JAX/lerobot matrix), and (b) calendar. If a second 24GB lab card frees up, run sim and policy on separate cards or two models concurrently — roughly halves the closed-loop calendar. No new hardware is required to complete the paper.

### 2.7 Milestones (June 2026 → March 2027)

| Milestone | Window | Deliverable |
|---|---|---|
| M1 | Jun–Jul 2026 | Suite A scaled to ~300 frames / 2 datasets; per-model floors recalibrated; OpenVLA-OFT, pi0, pi0-FAST, Octo onboarded open-loop; S/G metric + CI machinery frozen |
| M2 | Jul–Aug 2026 | Suite C depth-provenance interventions on SpatialVLA (+ any depth-conditioned model); layer-wise contamination analysis on OpenVLA + pi0; first internal writeup + arXiv preprint of the extended probe |
| M3 | Sep 2026 | **CoRL 2026 workshop submission** (instrument + depth-provenance result); SIMPLER fork + LIBERO openpi harness stood up; optional ICRA 2027 short version if the ~Sep 15 deadline is confirmed and M1–M2 are clean |
| M4 | Sep–Nov 2026 | Sampled closed-loop sweeps across 6–8 models; rollout-frame divergence logged (D4 seed data) |
| M5 | Nov–Dec 2026 | Correlation analysis; ablations; GR00T/MiniVLA onboarding; LIBERO-Plus reconciliation section |
| M6 | Dec 2026–Jan 2027 | Paper writing; artifact packaging (pair suites, harness, metric code); red-team pass against the five closest works |
| M7 | late Jan–Feb 2027 | **RSS 2027 submission** (primary target) |
| M8 | Feb–Mar 2027 | Begin D4 monitor paper from M4 rollout data; extended-artifact version toward NeurIPS 2027 E&D / CoRL 2027 if needed |

### 2.8 Top 3 risks and mitigations

1. **Scooped on the depth-provenance result or the counterfactual-metrology framing** (this subarea publishes monthly; Princeton IRoM is adjacent via BYOVLA/Predictive Red Teaming). *Mitigation:* flag-plant early — arXiv the extended probe at M2 and submit the CoRL 2026 workshop version in September; the bitwise-paired real-frame instrument plus 880 predictions already in hand is a ~6-month head start nobody else has; design suites and harness to be model-extensible so any newly published fix (Spatial Forcing/GLaD checkpoints) becomes an evaluation subject rather than a competitor.
2. **The open-loop→closed-loop correlation comes out weak or null.** *Mitigation:* pre-register the analysis and power it (≥8 model×condition cells, rank statistics with CIs); structure the paper so Contributions 1–3 stand alone; a calibrated null is itself a finding ("continuous sensitivity and binary success measure different things — and here is what each misses"), which still feeds the monitoring agenda by telling it which signal to trust.
3. **Integration debt and wall-clock on one GPU** (fork dependency matrix, per-policy environments, sim/policy contention). *Mitigation:* cap closed-loop at the sampled validation — the judges' explicit advice — and never compete on sweep scale; dockerized per-policy environments from day one; overnight queue discipline; under schedule pressure drop lowest-value models (Octo, MiniVLA) before dropping conditions; request a second lab card only for the M4 window.

### 2.9 Target venues (deadlines verified 2026-06-10)

- **CoRL 2026 workshops** — Nov 9, 2026, Austin TX (main conference Nov 10–12; main-track deadline May 29, 2026 has passed). Workshop CFP deadlines typically Aug–Sep 2026. *Flag-plant target (M3).*
- **ICRA 2027, Seoul** — deadlines TBA; expect ~Sep 15, 2026 by ICRA's standing pattern (ICRA 2026 deadline was Sep 15, 2025). *Optional early shot if M1–M2 land cleanly.*
- **RSS 2027** — deadline TBA; RSS pattern is late January/early February of the conference year (RSS 2026: late Jan 2026, conference Jul 13–17, Sydney). *Primary target (M7).*
- **NeurIPS 2027, Evaluations & Datasets track** (renamed from D&B, scope now explicitly "evaluation as a scientific object of study" — an unusually good fit for a metrology paper) — expect ~early May 2027 (2026: abstract May 4 / paper May 6, passed). *Home for the artifact-heavy extended version.*
- **CoRL 2027** — expect ~late May 2027 (2026: abstract May 26 / paper May 29). *Safety net.*

---

## 3. Runner-up: D4 — counterfactual appearance probing as a runtime failure signal (in case the advisor prefers it)

If the advisor prefers the monitoring-first route: the claim is that a VLA's action divergence under geometry-preserving appearance counterfactuals of its *current* observation is a calibrated, attributable, training-free, black-box per-timestep failure predictor, and that gating execution/recovery on it improves closed-loop success — integrated with Code-as-Monitor as the complementary *policy-pathology* signal to its world-constraint monitoring, demonstrated in SIMPLER/LIBERO and on the OpenArm. The niche is real but thin: BYOVLA (2410.01971) already built ~70% of the mechanism (appearance jitter → action divergence, on OpenVLA/Bridge) as a runtime patch, Sentinel/STAC (2410.04640) owns calibrated temporal action-divergence as a failure signal, and Predictive Red Teaming (2502.06575) does offline edit-observations-to-predict-failure — all from or near one fast-moving Princeton group whose obvious next paper is exactly this, so the honest plan is workshop-first (CoRL 2026) with head-to-head AUROC/lead-time wins over SAFE (2506.09937), STAC, and FIPER (2510.09459) as mandatory table stakes, plus a control-rate-cheap jitter family (the ControlNet restyler is too slow for the loop) whose geometry-preservation must be re-certified, and an explicit reconciliation of LIBERO-Plus's background-resilience with our 6mm sub-threshold shifts; the unclaimed probe-frequency-vs-detection-latency ("anytime monitor") analysis is a free bonus contribution. It fits the hardware (int4 policy + monitor on one 3090; comfortable on two) and fits the lab agenda best of all four directions — judges scored it a clear second (19.5/30, above D2 for two of three judges) — but its credibility rests on the divergence-to-failure calibration that the recommended paper's M4 arm produces anyway, which is why the right sequencing is instrument first, monitor second.
