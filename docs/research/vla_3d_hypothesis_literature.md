# Literature Verdict Memo: Do Current VLAs Lack 3D Spatial Understanding and Memorize RGB Appearance?

**Date:** 2026-06-10
**Hypothesis under test (user, robotics grad student, pi0/OpenVLA):** *Current VLA models lack comprehensive 3D spatial understanding; they largely memorize RGB appearance, so they collapse when the scene, background, or lighting changes.*
**Basis:** Three independent literature sweeps (benchmarks/perturbation studies; 3D-representation papers; generative-augmentation papers), 2022–2026. Numbers below were verified against full-text sources by the sweeps except where flagged.

---

## 1. Verdict

**Substantially correct, with two important corrections to the mechanism.**

1. **The collapse is real and severe.** Every systematic perturbation study agrees: policies and VLAs that score 76–97% in-distribution lose 30–90 points under visual/spatial perturbations *with task geometry and skill requirements held fixed* (COLOSSEUM, VLATest, LIBERO-Plus, Factor World, GreenAug, RoboEngine). The headline benchmark numbers of pi0 and OpenVLA are largely benchmark-specific.

2. **Correction A — the worst axis is geometric, not photometric.** The hypothesis as stated ("collapses when background or lighting changes") names the *mildest* factors. Factor-ordered studies consistently find: **camera viewpoint and spatial/initial-state shifts are catastrophic** (60–90 point drops: pi0 94.2%→15.8%, OpenVLA 76.5%→1.1% under viewpoint shift in LIBERO-Plus), **object color/texture/distractors are severe** (30–50% drops in COLOSSEUM; 82.6% real-world drop from object color alone), while **background and lighting alone are the most survivable** (~5–25 point drops). So the stronger, better-supported form of the hypothesis is: *VLAs lack viewpoint-invariant 3D grounding AND bind to local RGB appearance of task-relevant objects* — not primarily background/lighting memorization.

3. **Correction B — "lacks 3D understanding" is too strong as a perception claim.** INT-ACT shows pi0 retains 84.5% semantic *intention* correctness while task success falls to 30.4% under shift — the model often still "sees" and "wants" the right thing but cannot ground actions. And arXiv:2512.02902 shows a 4K-parameter affine adapter on vision tokens recovers pi0.5 from 48.5% to 87.1% under novel viewpoints, arguing much of the brittleness is recoverable representation misalignment rather than absent spatial capability. The defensible claim: **VLAs do not possess a shift-invariant 3D representation that their action heads can exploit; their perception-to-action grounding is appearance- and viewpoint-conditioned.**

4. **Backbone evidence supports the 3D-deficit premise.** The VLM components VLAs inherit are measurably weak at 3D primitives: GPT-4V scores 51.3% on BLINK (humans 95.7%), below chance on several multi-view/localization tasks; 58.1% on SpatialRGPT-Bench qualitative spatial relations vs 91.8% for a 3D-grounded 7B model.

---

## 2. Strongest Quantitative Evidence FOR the Hypothesis

| Perturbation factor | Measured success drop | Models | Source |
|---|---|---|---|
| Camera viewpoint shift | 76.5% → 1.1% (−75.4 pts); 94.2% → 15.8% (−78.4 pts); 95.2% → 4.3% (−90.9 pts) | OpenVLA; pi0; UniVLA | LIBERO-Plus (arXiv:2510.13626) |
| Camera pose, only ±5° / ±5 cm | Models retain on average **34.0%** of baseline (Octo: 9.1–9.6% retention) | 7 VLAs incl. OpenVLA, Octo, RT-1-X | VLATest (arXiv:2409.12894) |
| Robot initial state | 94.2% → 6.6% (−87.6 pts) | pi0 | LIBERO-Plus (arXiv:2510.13626) |
| Target-object color (real robot) | **−82.6%** | SOTA manipulation policies | COLOSSEUM (arXiv:2402.08191) |
| Any single perturbation factor (avg) | **30–50%** degradation per factor; ≥75% when factors combined | RVT, PerAct, R3M-MLP, MVP-MLP + 1 more | COLOSSEUM (arXiv:2402.08191), sim-to-real R²=0.614 |
| 4 distractor objects added | pick-up 17.3% → 8.3% (−52% rel.); move-near 8.3% → 1.1% (−87% rel.) | VLA average | VLATest (arXiv:2409.12894) |
| Distractors/backgrounds at test time | **~−40%** despite near-perfect nominal success | OpenVLA | BYOVLA (arXiv:2410.01971) |
| Novel scene, identical task geometry | 12% (no aug) vs 64% (background randomization); BC: 55% vs 91% | DQN / ACT policies | GreenAug (arXiv:2407.07868) |
| Entirely new scenes, single-scene training | normalized score 0.20 vs 0.62 augmented; Put-Mouse 0.0% unaugmented | Diffusion policies (DINOv2) | RoboEngine (arXiv:2503.18738) |
| Unseen environments (real robot) | 38% → 80% with appearance-only augmentation (actions untouched) | CLIPort-style BC | GenAug (arXiv:2302.06671) |
| Lighting mutation (2.5× brightness) | ~30% immediate drop; only 61.3% of passing cases still pass | 7 VLAs | VLATest (arXiv:2409.12894) |
| Unseen objects | −66 to −74% on most tasks | 7 VLAs | VLATest (arXiv:2409.12894) |
| Background texture | OpenVLA −25.3 pts; pi0 −15.7 pts | OpenVLA, pi0 | LIBERO-Plus (arXiv:2510.13626) |
| Language instruction blanked | performance "largely unchanged" (object suite) | OpenVLA-OFT | LIBERO-Plus (arXiv:2510.13626) |

The decisive *mechanistic* evidence that the failure is appearance-binding: **appearance-only interventions, with actions and geometry untouched, recover most of the loss.** GreenAug 55→91% (chroma-key random textures), ROSIE 33→71% on new backgrounds (diffusion inpainting), RoboEngine +210% relative, BYOVLA recovers Octo to roughly nominal *purely by editing pixels at inference time*. If the skill were missing, repainting pixels could not restore it. PointVLA's qualitative result is the cleanest illustration: RGB-only OpenVLA/DexVLA attempt to grasp a *printed photo* of objects; the point-cloud-injected variant refuses (arXiv:2503.07511).

---

## 3. Nuances and Counter-Evidence

1. **It is not "zero 3D understanding."** INT-ACT (arXiv:2506.09930): pi0 keeps 84.5% intention correctness vs 30.4% task success (54.1-pt gap); SpatialVLA 69.6% vs 21.5%. VLM-initialized VLAs reach 80–100% intention correctness across shift categories. The bottleneck is substantially in the **vision-to-action grounding**, not (only) scene recognition.
2. **Much brittleness may be recoverable feature misalignment.** arXiv:2512.02902: pi0.5 at 48.5% under novel LIBERO viewpoints is restored to 90.8% by a 4.7M-param LoRA on ViT linear layers, and to 87.1% by a **4K-parameter** global affine on vision tokens — matching full 467M-param LoRA. Hard to reconcile with "no spatial capability exists"; easy to reconcile with "the capability is not shift-invariantly *exposed*."
3. **Factor ordering cuts against naive "background memorization."** Factor World (arXiv:2307.03659), real robot: baseline 91.7%; new backgrounds 88.9%, new lighting 83.3%, new distractors 80.6% — but new table texture 52.8% and new camera position 45.8%. LIBERO-Plus: lighting costs only ~10–15 pts. Background/lighting are the *easiest* shifts for diverse-data-trained policies.
4. **Scale and co-training measurably help.** Factor World: the generalization gap shrinks from ~0.40 (5 training environments) to <0.10 (100 environments). CACTI: held-out-layout success 14.1%→47.2% when scaling 10→100 layout variants. OpenVLA beats RT-2-X by 16.5% absolute overall on an OOD-heavy evaluation and beats Octo/RT-1-X on visual generalization — "pure memorization" understates what scaled VLM-pretrained VLAs already do. Wrist cameras help too (OpenVLA-OFT loses only 37.4 pts under viewpoint shift vs 75–91 for others).
5. **3D representations are not a complete cure.** COLOSSEUM's 3D-voxel models (PerAct, RVT) were robust to camera shifts but still broke on color and distractors; BridgeVLA's SOTA on COLOSSEUM is still only 64.0% (~1/3 of perturbed trials fail); 3D Diffuser Actor's *unseen-environment* gain on CALVIN is a modest +7% relative. 3D structure buys geometric robustness, not photometric robustness.
6. **Caveats on evidence quality.** Most VLA perturbation numbers are simulation-based (COLOSSEUM's sim-to-real R²=0.614 is reassuring but imperfect); VLATest baselines were absolutely low (~12.8% avg pick-up), so relative retention is noisy; GeoVLA/PointVLA robustness figures are author-reported without third-party replication; SIMPLER per-variant tables and OpenVLA per-category numbers are figure-embedded and were not text-verified; GenAug's iGibson 1%→60% figure and RoVi-Aug's per-task rows came from secondary summaries.

---

## 4. The Two Established Fix Families

### Family A — 3D-aware inputs/representations (fixes the *geometric* axis)

| Method | Mechanism | Measured gain | Source |
|---|---|---|---|
| DP3 (3D Diffusion Policy) | point clouds in an otherwise-identical diffusion policy | appearance generalization 5/5 colors vs **0/5** for RGB and RGB+depth baselines; camera views 3/3 vs 0/3; unseen positions 4/5 vs 0/5; +24.2% rel. over Diffusion Policy across 72 sim tasks | arXiv:2403.03954 |
| SpatialVLA | Ego3D position encoding (monocular depth) + spatial action grids | SimplerEnv-Bridge zero-shot 34.4% vs OpenVLA **1.0%**; variant aggregation (backgrounds/lighting/textures) 68.8% vs 45.0%; drops only ~3 pts from visual-matching to variants | arXiv:2501.15830 |
| 3D Diffuser Actor | lift 2D features to 3D with sensed depth | +18.1 pts absolute RLBench multi-view; +7% relative on CALVIN ABC→D unseen environment | arXiv:2402.10885 |
| OBSBench finding | point cloud vs RGB vs RGB-D, controlled | point-cloud policies ~2.2× more robust to zero-shot camera shifts (0.40 vs 0.18 RGB); **RGB-D-as-channels is the WORST modality** | arXiv:2402.02500 |
| BridgeVLA | point-cloud→multi-view projection into a VLM-VLA | COLOSSEUM 56.7% → 64.0% (+7.3 pts, new SOTA) | arXiv:2506.07961 |
| GeoVLA / PointVLA | point-cloud experts injected into VLAs | 45° viewpoint shift: 70% vs **0%** (CogACT); refuses photo-of-object that fools OpenVLA (author-reported) | arXiv:2508.09071, 2503.07511 |

**Key design lesson:** the gains come from *structured* 3D (point clouds, 3D position encodings, virtual views) — naive depth-as-image-channel actively hurts (OBSBench worst modality; DP3's depth baseline scores 0; SpatialRGPT's depth ablation adds only ~2 pts vs 3D-grounded training data).

### Family B — appearance-randomized / generated data (fixes the *photometric* axis)

| Method | Mechanism | Measured gain | Source |
|---|---|---|---|
| GreenAug | chroma-key random-texture backgrounds | novel scenes 55% → **91%** (beats CV-aug 70% and diffusion-aug 75–77%) | arXiv:2407.07868 |
| ROSIE | diffusion inpainting of demos | new backgrounds 33% → 71%; novel objects 25% → 75%; sink placement 0% → 60% (but OOD distractors only 33→37%) | arXiv:2302.11550 |
| GenAug | text-to-image augmentation of 10 demos | unseen environments 38% → 80%; unseen pick objects 10% → 46% | arXiv:2302.06671 |
| RoboAgent / MT-ACT | segmentation+inpainting semantic augmentation | ~+100% relative at new-backgrounds level, ~+400% relative at novel skill-object combos | arXiv:2309.01918 |
| RoboEngine | robot segmentation + physics-aware background generation | 6 new scenes: 0.20 → 0.62 normalized (+210% rel.); Put-Mouse 0% → 75% grasping | arXiv:2503.18738 |
| BYOVLA | *test-time* inpainting of irrelevant regions | +20–40% under visual distraction, no fine-tuning; Octo back to ~nominal | arXiv:2410.01971 |
| RoVi-Aug | diffusion-synthesized robot + **viewpoint** augmentation | up to +30%; cross-robot transfer under 10 cm/20° camera shift: 80% vs 50–60% baselines | arXiv:2409.03403 |
| LIBERO-Plus retraining | perturbation-augmented fine-tuning | camera robustness 55.6% → 92.8% (+37.2 pts) | arXiv:2510.13626 |

**Key design lessons:** (i) appearance *diversity* matters more than realism — GreenAug's random textures (91%) beat semantically realistic diffusion backgrounds (75–77%); (ii) the family-A/family-B boundary is porous: RoVi-Aug and LIBERO-Plus show even the geometric (viewpoint) axis can be partially closed with 2D-generated, viewpoint-diverse data.

### Where SceneForge fits

SceneForge (this repo: LLM-planned 3D scene → deterministic depth/mask renderer → depth-ControlNet SDXL re-texturing into N photoreal styles, labels transferred for free) is a **Family-B engine with a Family-A skeleton**:

- **Directly implements the winning Family-B recipe:** geometry held fixed, appearance randomized across many style worlds — exactly the intervention that GreenAug/ROSIE/RoboEngine showed recovers 30–55 points, and at the diversity-over-realism operating point GreenAug validated. The depth-ControlNet constraint is the chroma-key idea generalized to the whole scene.
- **Geometry-grounded labels enable controlled factor isolation:** because the renderer owns ground-truth depth, masks, and poses, SceneForge can produce *matched pairs* (same geometry, different appearance; same appearance, different camera) — the controlled axes COLOSSEUM/LIBERO-Plus identify as the diagnostic and curative dimensions. Sampling camera poses in the renderer extends it to RoVi-Aug-style viewpoint augmentation, attacking the *catastrophic* axis (the one 2D background augmentation alone cannot fix).
- **Bridge to Family A:** the per-image depth maps and instance-level 3D layout SceneForge already emits are precisely the supervision SpatialRGPT-style 3D-grounded training and SpatialVLA-style position encodings consume — so the same outputs can serve both fix families. Caveat from the literature: ship the *structured* 3D (point clouds / poses / 3D-grounded QA), not depth-as-a-fourth-channel.
- **Honest scoping:** SceneForge data cannot by itself fix the intention-action gap (INT-ACT) or replace real-robot rollouts; its leverage is on the perception-binding failure modes, which the evidence says account for the large majority of the measured collapse.

---

## 5. Citations

1. Pumacay et al., *THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation*, RSS 2024. arXiv:2402.08191. https://robot-colosseum.github.io/
2. Xie, Lee, Xiao, Finn, *Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation* (Factor World), ICRA 2024. arXiv:2307.03659. https://github.com/RLAgent/factor-world
3. Wang et al., *VLATest: Testing and Evaluating Vision-Language-Action Models for Robotic Manipulation*, FSE 2025. arXiv:2409.12894. https://github.com/ma-labo/VLATest
4. OpenMOSS, *LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*, 2025. arXiv:2510.13626.
5. Teoh et al., *Green Screen Augmentation Enables Scene Generalisation in Robotic Manipulation* (GreenAug), 2024. arXiv:2407.07868. https://greenaug.github.io/
6. Li et al., *Evaluating Real-World Robot Manipulation Policies in Simulation* (SIMPLER), CoRL 2024. arXiv:2405.05941. https://simpler-env.github.io/
7. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024. arXiv:2406.09246.
8. Fang et al., *From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models* (INT-ACT), 2025. arXiv:2506.09930. https://ai4ce.github.io/INT-ACT/
9. *VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling*, 2025. arXiv:2512.02902.
10. Fu et al., *BLINK: Multimodal Large Language Models Can See but Not Perceive*, ECCV 2024. arXiv:2404.12390. https://zeyofu.github.io/blink/
11. Cheng et al., *SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models*, NeurIPS 2024. arXiv:2406.01584.
12. Qu et al., *SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model*, RSS 2025. arXiv:2501.15830.
13. Ke et al., *3D Diffuser Actor: Policy Diffusion with 3D Scene Representations*, CoRL 2024. arXiv:2402.10885. https://3d-diffuser-actor.github.io/
14. Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations* (DP3), RSS 2024. arXiv:2403.03954.
15. Zhu et al., *Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning* (OBSBench), NeurIPS 2024 D&B. arXiv:2402.02500.
16. Li et al., *BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models*, NeurIPS 2025. arXiv:2506.07961.
17. *GeoVLA: Empowering 3D Representations in Vision-Language-Action Models*, 2025. arXiv:2508.09071.
18. *PointVLA: Injecting the 3D World into Vision-Language-Action Models*, 2025. arXiv:2503.07511.
19. Zhen et al., *3D-VLA: A 3D Vision-Language-Action Generative World Model*, ICML 2024. arXiv:2403.09631.
20. Yu et al., *Scaling Robot Learning with Semantically Imagined Experience* (ROSIE), RSS 2023. arXiv:2302.11550. https://diffusion-rosie.github.io/
21. Chen, Kiami, Gupta, Kumar, *GenAug: Retargeting Behaviors to Unseen Situations via Generative Augmentation*, RSS 2023. arXiv:2302.06671. https://genaug.github.io/
22. Mandi et al., *CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning*, 2022. arXiv:2212.05711. https://cacti-framework.github.io/
23. Bharadhwaj et al., *RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking*, ICRA 2024. arXiv:2309.01918. https://robopen.github.io/
24. Yuan et al., *RoboEngine: Plug-and-Play Robot Data Augmentation with Semantic Robot Segmentation and Background Generation*, IROS 2025. arXiv:2503.18738. https://roboengine.github.io/
25. Hancock, Ren, Majumdar, *Run-time Observation Interventions Make Vision-Language-Action Models More Visually Robust* (BYOVLA), ICRA 2025. arXiv:2410.01971. https://aasherh.github.io/byovla
26. Chen, Xu et al., *RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning*, CoRL 2024 (oral). arXiv:2409.03403.

**Verification flags:** SIMPLER per-model per-variant tables and OpenVLA per-category OOD numbers are figure-embedded (not text-verified); CACTI real-robot ablations are figure-derived; GeoVLA/PointVLA robustness numbers are author-reported; GenAug iGibson 1%→60% and RoVi-Aug per-task rows came from secondary summaries — re-check before quoting in a paper.
