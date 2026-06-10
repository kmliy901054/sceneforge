# Demo capture notes — SceneForge (v2, both tabs)

Captured 2026-06-11 on the project machine (RTX 3090, conda env `dgan`),
driven by a Playwright (chromium, 1440×900) script against the real app
(`python app.py`, port 7860, with the §10.5 pre-warm; `scripts/demo_prep.sh`
run beforehand so all caches were hot). No mocked data — the video is one
continuous live session covering BOTH tabs: classic forge run
`20260611_042439_42`, IP-Adapter styled forge run `20260611_042536_42`,
video-augment run `video_augment_20260611_042610`.

## Files

| file | what it is |
| --- | --- |
| `demo_raw.webm` | raw Playwright recording, 114.1 s, 1440×900 |
| `demo_raw.mp4` | same recording transcoded with ffmpeg (h264, `-crf 20`, faststart) for easy playback |
| `demo_v1.mp4` | the previous (2026-06-10) 73.7 s recording — Forge tab only, predates the Video Augment tab |
| `../img/ui_forged.png` | hero screenshot (full page) right after the classic forge finished |
| `../img/ui_overlay_masks.png` | full page with the overlay radio switched to **masks** |
| `../img/ui_reforge.png` | full page after a Re-Forge ⚡ with the azimuth slider moved |
| `../img/ui_video_tab.png` | full page of the Video Augment tab after AUGMENT finished |

## What the video shows (timestamps)

| time | beat |
| --- | --- |
| 0:00 | page loads — header, **Forge / Video Augment** tabs, task box, VRAM footer ticking |
| 0:02–0:04 | task typed live: “pick the red mug from a cluttered kitchen table” |
| 0:05–0:06 | layouts slider set to **2**, styles slider set to **4** (seed 42) |
| 0:06 | **FORGE** clicked — status flips to *planning…* |
| 0:08–0:20 | gemma planner tokens stream live into the *planner stream* Code panel |
| 0:21 | **Model3D viewer** (layout 0 `viewer.glb`) appears — camera pans to it |
| 0:23 | **depth control** (ControlNet input) image appears |
| 0:31 | first SDXL image lands in the gallery; gallery fills layout-major |
| 0:43 | gallery complete: **8 images** (2 layouts × 4 styles) with bbox overlays |
| 0:48 | status → *done · run 20260611_042439_42 · total 38.6 s · first visual 14.1 s*; fidelity label populates (match_rate 97 %, fidelity_adj 79 %, hallucination 19 %, kept 8/8) |
| 0:49–0:51 | slow pan top→gallery; **ui_forged.png** captured here |
| 0:52–0:54 | overlay radio → **masks** (instant cached-path swap); **ui_overlay_masks.png** |
| 0:54 | Re-Forge ⚡ accordion opened |
| 0:55 | azimuth slider dragged 35°→**−60°** |
| 0:56 | **Re-Forge ⚡** clicked (no LLM call) — re-forged in **1.7 s**; annotated image with legend (mug (target), cutting_board, screwdriver, bowl), labels track the new camera; Model3D updates; **ui_reforge.png** |
| 1:02 | **Style reference 🎨 (IP-Adapter)** accordion opened, sunset-stripe reference image uploaded (`outputs/ip_adapter_validate/reference.png`, scale 0.6) |
| 1:03 | quick styled forge: layouts **1** × styles **2**, **FORGE** clicked |
| 1:29 | styled forge done (run `20260611_042536_42`, 25.8 s) — gallery images visibly inherit the reference’s magenta/orange palette while keeping the depth-controlled table geometry |
| 1:32 | switch to the **Video Augment** tab |
| 1:36 | episode video uploaded (`episode_traj0.mp4`, 6 frames @ 5 fps) |
| 1:37 | styles slider → **2**; two style prompts typed (*industrial workshop*, *bright white laboratory*) |
| 1:38 | **AUGMENT** clicked — status streams *restyling k/12 frames…* live |
| 1:47 | done in **9.3 s** — mask audit sheet (red = restyled region) + two **side-by-side original/restyled players** + zip download populate |
| 1:50–1:53 | pan over audit sheet and players; **ui_video_tab.png** captured |
| 1:54 | recording ends (114.1 s total) |

## Measured numbers (this exact session)

- Classic forge wall time, FORGE click → done status rendered: **41.3 s**
  (pipeline-internal `total_s` 38.64 s: plan 9.6 s · ground 3.8 s ·
  style 8.5 s · diffusion 14.0 s = 1.75 s/img · eval 1.9 s;
  first visual at 14.1 s).
- Fidelity (OWLv2 spot-check): fidelity_adj 0.785, match_rate 0.969,
  mean matched IoU 0.891, hallucination_rate 0.188, 8/8 images kept,
  0 quarantined.
- Re-Forge slider-to-image: **1.7 s** (≤ 6 s budget).
- Styled forge (IP-Adapter h94 sdxl vit-h, scale 0.6): wall **25.8 s** for
  1 layout × 2 styles; diffusion 2.53 s/img — the first styled image includes
  the one-time IP-Adapter + CLIP-ViT-H encoder load; peak VRAM
  max_allocated **12.17 GB** (vs 10.34 GB for the classic forge), still
  > 8.8 GB min_free on the 3090.
- Video Augment: 6 frames × 2 styles in **9.3 s** UI wall (internal 8.55 s:
  depth 0.9 s · diffusion 7.2 s = 0.58 s/img); restyled fraction per frame
  0.08–0.11 (robot/workspace pixels bitwise kept).
- VRAM footer steady around 10.5/24 GB during the session.

## Capture quirks worth knowing

- Both tabs have a slider labeled **styles** — `label[for^=range_id]` must be
  scoped to the *visible* tabpanel (`div[role="tabpanel"]:visible`) or the
  Video Augment drag lands on the hidden Forge slider.
- `gr.Image` / `gr.Video` uploads: locate the labeled block
  (`div.block:has(label:has-text("…"))`) and `setInputFiles` on its
  `input[type=file]`; the preview (img/video element) appears ~1–2 s later.
- A wait for `text=/re-forged/` matches the AnnotatedImage block label
  (“re-forged — labels track the camera”) which exists before the result —
  the script’s timing print shows 0.0 s, but the actual re-forge (status line
  `re-forged in 1.7 s`) lands inside the scripted 2.8 s linger, so the video
  and screenshot still show the finished state. Use `text=/re-forged in/`
  next time.
- Second FORGE click in one session: the old *done* status is still in the
  DOM, so wait for `text=/planning/` first, then for the new
  `total … s · first visual` status.
- Gradio 6 sliders: the visible block label is linked to the
  `input[type=range]` via `label[for="range_id_N"]` — resolve the range input
  through the label's `for` attribute (a generic `div.block:has(label:…)`
  matches accordion ancestors too).
- The overlay Radio's caption is not a `<label>` element; scroll to the radio
  option itself (`getByRole('radio', { name: 'masks' })`).
- Everything else behaved: no errors in the app log, the augment status line
  streamed 1 s ticks during the SDXL burst, and the gallery streamed strictly
  layout-major as asserted in `handlers.on_forge`.
