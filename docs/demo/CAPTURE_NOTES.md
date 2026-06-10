# Demo capture notes — SceneForge

Captured 2026-06-10 on the project machine (RTX 3090, conda env `dgan`),
driven by a Playwright (chromium, 1440×900) script against the real app
(`python app.py`, port 7860, with the §10.5 pre-warm; `scripts/demo_prep.sh`
run beforehand so all caches were hot). No mocked data — the video is one
continuous live session, forge run `20260610_183609_42`.

## Files

| file | what it is |
| --- | --- |
| `demo_raw.webm` | raw Playwright recording, 73.7 s, 1440×900 |
| `demo_raw.mp4` | same recording transcoded with ffmpeg (h264, `-crf 20`, faststart) for easy playback |
| `../img/ui_forged.png` | hero screenshot (full page) right after the forge finished |
| `../img/ui_overlay_masks.png` | full page with the overlay radio switched to **masks** |
| `../img/ui_reforge.png` | full page after a Re-Forge ⚡ with moved camera sliders |

## What the video shows (timestamps)

| time | beat |
| --- | --- |
| 0:00 | page loads — header, task box, FORGE button, VRAM footer ticking |
| 0:03–0:05 | task typed live: “pick the red mug from a cluttered kitchen table” |
| 0:05–0:06 | layouts slider set to **2**, styles slider set to **4** (seed 42) |
| 0:07 | **FORGE** clicked — status flips to *planning…* |
| 0:09–0:20 | gemma planner tokens stream live into the *planner stream* Code panel |
| 0:21 | **Model3D viewer** (layout 0 `viewer.glb`) appears — camera pans to it |
| 0:23 | **depth control** (ControlNet input) image appears |
| 0:31 | first SDXL image lands in the gallery; gallery fills layout-major |
| 0:43 | gallery complete: **8 images** (2 layouts × 4 styles) with bbox overlays |
| 0:48 | status → *done · run 20260610_183609_42 · total 37.8 s · first visual 13.4 s*; fidelity label populates (match_rate 97 %, fidelity_adj 79 %, hallucination 19 %, kept 8/8) |
| 0:49–0:52 | slow pan top→gallery; **ui_forged.png** captured here |
| 0:53–0:55 | overlay radio → **masks** (instant cached-path swap); **ui_overlay_masks.png** |
| 0:55 | Re-Forge ⚡ accordion opened |
| 0:56–0:58 | camera sliders dragged: azimuth 35°→**−60°**, elevation 30°→**55°** |
| 0:59 | **Re-Forge ⚡** clicked (no LLM call) |
| 1:01 | re-forged in **1.8 s** — annotated image with legend (mug (target), cutting_board, screwdriver, bowl), labels track the new camera; Model3D updates |
| 1:04 | **ui_reforge.png** captured |
| 1:05 | overlay radio → **both** (boxes + masks) |
| 1:10 | **Export COCO** clicked → `dataset.zip` download button populated (<0.5 s) |
| 1:14 | recording ends |

## Measured numbers (this exact run)

- Forge wall time, FORGE click → done status rendered: **41.1 s**
  (pipeline-internal `total_s` 37.75 s: plan 9.6 s · ground 3.5 s ·
  style 8.5 s · diffusion 13.8 s = 1.73 s/img · eval 1.9 s;
  first visual at 13.4 s).
- Re-Forge slider-to-image: **1.8 s** (≤ 6 s budget).
- Fidelity (OWLv2 spot-check): fidelity_adj 0.785, match_rate 0.969,
  mean matched IoU 0.891, hallucination_rate 0.188, 8/8 images kept,
  0 quarantined.
- VRAM footer steady around 10.5/24 GB during the session
  (diffusion peak min_free 10.7 GB).

## Capture quirks worth knowing

- Gradio 6 sliders: the visible block label is linked to the
  `input[type=range]` via `label[for="range_id_N"]` — locating sliders by
  a generic `div.block:has(label:…)` matches the accordion ancestor too
  (first attempt silently dragged the azimuth slider twice). Resolving the
  range input through the label's `for` attribute is the stable handle.
- The overlay Radio's caption is not a `<label>` element; scrolling to it by
  `label:has-text("overlay")` times out. Scroll to the radio option itself
  (`getByRole('radio', { name: 'masks' })`).
- Everything else behaved: no errors in the app log, overlay swaps and the
  COCO export were effectively instant, the gallery streamed strictly
  layout-major as asserted in `handlers.on_forge`.
