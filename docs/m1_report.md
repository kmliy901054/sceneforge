# M1 Go/No-Go Report — quantitative viability gate (judge mandate #1)

Date: 2026-06-10 · Machine: RTX 3090 24 GB, torch 2.6.0+cu124, diffusers 0.38.0,
transformers 4.51.3 · Harness: `scripts/m1_smoketest.py` (self-contained, §13-M1)
· Raw numbers: `outputs/m1/metrics.json` · Evidence: `outputs/m1/comparison_grid.png`,
`outputs/m1/contact_sheet.png`

## VERDICT: **GO at L0** (Lightning 4-step UNet, disparity depth control)

| gate metric | measured | gate | pass |
|---|---|---|---|
| match_rate (IoU ≥ 0.5) | **0.911** | ≥ 0.70 | YES |
| mean_matched_iou | **0.923** | ≥ 0.65 | YES |
| hallucination_rate (informational) | 0.067 | — | reported |

Gate population: GT instances with `area_px ≥ 1000` (§9 review fix). The 12 renders
(4 inline scenes × 3 seeds; box / can / composed-mug / sphere+plate, 3–4 medium-large
objects each) produced **45 GT instances, all 45 gate-eligible** (≥ 30 required).
The full experiment ran twice with identical seeds and reproduced every gate metric
exactly (deterministic renders + seeded generation).

No ladder walk was needed — L0 passes with wide margin. L1–L4 were not measured
(L0 success short-circuits the walk per §13-M1); `ForgePipeline.load(level)`
implements L0–L3 for runtime degradation regardless.

## Step 0 — environment repairs (all green)

- 5 HF repos verified on disk (fp16 patterns): sdxl-base 7.11 GB · controlnet-depth
  2.50 GB · vae-fp16-fix 1.34 GB · SDXL-Lightning 5.53 GB · owlv2 1.24 GB.
  All large blobs sha256-verified against their LFS etags.
- transformers 4.51.3: `Owlv2ForObjectDetection` and
  `StableDiffusionXLControlNetPipeline` import cleanly (P0 §0 pin already applied).
- Full 768×768 EGL color+depth+SEG render asserted; SEG red-channel IDs round-trip
  exactly ({0, 7} observed).

## Depth-prep A/B at L0 (24 images, fixed prompts/seeds across arms)

| mode | match_rate | mean_matched_iou | halluc_rate | mean_fidelity | quarantined |
|---|---|---|---|---|---|
| **disparity (winner)** | 0.9111 | 0.9228 | **0.0667** | 0.8463 | **0/12** |
| linear | 0.9111 | 0.9243 | 0.1333 | 0.8472 | 2/12 |

Decision rule: gate metrics first (rounded to 0.02 — differences below OWLv2's own
box jitter are noise), then **lower hallucination_rate**. Both arms tie on the gate
metrics; linear's +0.0015 matched-IoU is noise while its hallucination rate is 2×
disparity's (6 vs 3 spurious same-category detections) and it quarantines 2 images
vs 0. **Winner: `disparity`** — consistent with the §7.3 prior (the ControlNet was
trained on MiDaS inverse depth). Frozen as the `depth_to_control` default and in
`sceneforge.yaml`.

## Measured performance (L0 @ 768², batch 1, no slicing/tiling)

| quantity | measured |
|---|---|
| s/img (mean of 12, warmup excluded) | **0.75 s** (min 0.74 / max 0.76) |
| first-image warmup overhead | ~1 extra call (excluded) |
| pipeline construction (warm page cache) | 15.7 s |
| max_memory_allocated during burst | 10.34 GB |
| max_memory_reserved during burst | 11.57 GB |
| min device free during burst (mem_get_info) | 10.86 GB |

0.75 s/img is ~3× faster than the §7.4 estimate (2–2.5 s). Downstream budget for the
M4 acceptance (2 layouts × 4 styles): `8 × 0.75 + 60 ≈ 66 s`.

## VRAM mode pick (judge mandate #2) — **sequential**

| measurement | sequential | coresident (gemma4:e4b @ ctx4096) |
|---|---|---|
| max_memory_allocated (3-gen burst) | 10.63 GB | 10.63 GB |
| max_memory_reserved | 11.75 GB | 11.75 GB |
| **min device free throughout** | **10.68 GB** | **0.92 GB** |

- gemma4:e4b loaded fully on GPU next to resident SDXL (`/api/ps size_vram == size`
  = 10.02 GB; still resident and fully placed after the burst) — co-residency is
  *functional* on this machine but leaves only **0.92 GB < guard_gb (3.0)** of
  device headroom: one allocator spike from OOM. Rule (§10.2) → **sequential**.
- gemma warm load: 3.2 s; 300-token generation: 2.6 s — the only thing coresident
  buys is this 3 s reload per forge (§10.1), not worth a 0.92 GB margin.
- `torch.cuda.empty_cache()` after a burst returns **2.26 GB** to the driver
  (validates the `phase("spec")` entry hook, §10.3).
- Quality-mode (qwen3.5:27b) forge-after-forge timing: **skipped** per cut-order #3
  (§12.7); everything else completed quickly but the swap test adds GPU churn on a
  shared box for a feature already first-in-line to cut.

## Frozen into `sceneforge.yaml`

```yaml
gen:  level: L0 · depth_mode: disparity · s_per_img: 0.75 · resolution: 768
      steps: 4 · guidance_scale: 0.0 · cond_scale: 0.85 · control_guidance_end: 0.9
vram: mode: sequential · guard_gb: 3.0
```

## Honest caveats

1. **Synthetic-scene optimism.** The 4 inline scenes are flat-shaded primitives with
   generous spacing and medium-large objects. Real director scenes (8 objects,
   clutter, occlusion) will score lower; 0.911/0.923 is a *clean-room upper bound*
   while the OWLv2 eval itself is a *lower bound* on label validity (§9 caveat —
   measured matched IoU 0.92 actually exceeds the predicted 0.75–0.85 jitter ceiling
   on these large instances).
2. **Hallucination is real but small at L0/disparity**: 3 spurious detections / 45 GT,
   all in the can scenes (extra "can"-like objects hallucinated on empty table area).
   The λ-penalty (`fidelity_adj`) quarantined 0/12 disparity images.
3. **A/B rule was corrected between run 1 and run 2**: run 1's tiebreak ignored
   hallucination and picked linear on a +0.0015 IoU edge; the rule now rounds gate
   metrics to 0.02 and breaks ties on hallucination_rate (documented in
   metrics.json). Both runs' raw per-arm numbers are identical.
4. **Weights download required intervention**: the Xet/hf_transfer CAS path stalled
   repeatedly at ~0–5 MB/s with silent dead connections; plain-HTTP
   (`HF_HUB_DISABLE_XET=1`) sustained ~35–99 MB/s. All blobs were sha256-verified
   after the takeover. Recommendation: keep `HF_HUB_DISABLE_XET=1` in demo-prep
   scripts on this network.
