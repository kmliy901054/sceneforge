"""sceneforge/ui/blocks.py — build_app() → gr.Blocks (ARCHITECTURE.md §11).

Gradio 6.17.3 (kwargs verified against the installed version). The app is a
two-tab layout: tab 1 ("Forge") is the original forge UI, UNCHANGED — same
components, same handler wiring, same queue discipline; tab 2 ("Video
Augment") drives ``sceneforge.augment.restyle_frames`` on an uploaded episode
video or a frames directory (background restyle, near pixels bitwise kept,
side-by-side original/restyled players per style + mask audit + zip download).
The VRAM footer + its timer sit OUTSIDE the tabs so both tabs see them.

Forge tab layout map:

    Row: task box · FORGE
    Row: layouts/styles/seed controls
    Row: stage status line (updates on EVERY event — the UI is never silent)
    Row: [ Model3D viewer + depth control + Re-Forge accordion ]
         [ planner token stream (Code) + SceneSpec JSON + source line ]
    Row: gallery (rows=layout / cols=style — REQUIRES layout-major generation,
         pinned in the orchestrator and asserted in on_forge) · overlay radio
         (pure cached-path swap)
    Row: fidelity Label · include-quarantined checkbox · Export COCO · Download
    Row: VRAM footer fed by gr.Timer(2.0)

Queue concurrency is 1 (one 3090 — a second concurrent forge would OOM).
Cut orders applied per §12.7: #3 quality checkbox, #4 1024 px dropdown and
#5 nudge buttons are omitted (camera sliders + reseed satisfy the re-forge
mandate; ``ForgeRun.reforge`` still accepts ``nudge=`` programmatically).
"""
from __future__ import annotations

import sceneforge.compat  # noqa: F401  — first import (§0)

import gradio as gr

from sceneforge.config import get_config
from sceneforge.labels.overlay import OVERLAY_MODES
from sceneforge.ui import handlers


def _build_forge_tab(run_state: gr.State) -> None:
    """Tab 1 — the original forge UI, verbatim (components + event wiring)."""
    with gr.Row():
        task_tb = gr.Textbox(
            label="manipulation task",
            value=handlers.DEFAULT_TASK,
            placeholder="e.g. pick the red mug from a cluttered kitchen table",
            scale=4,
        )
        forge_btn = gr.Button("FORGE", variant="primary", scale=1)

    with gr.Row():
        n_layouts = gr.Slider(1, 4, value=2, step=1, label="layouts")
        n_styles = gr.Slider(2, 6, value=4, step=1, label="styles")
        seed = gr.Number(value=42, precision=0, label="seed")

    status_md = gr.Markdown("**ready** — enter a task and hit FORGE")

    with gr.Row():
        with gr.Column():
            layout_3d = gr.Model3D(label="layout 0 — viewer.glb", height=340)
            control_img = gr.Image(label="depth control (ControlNet input)",
                                   type="filepath", height=260)
            with gr.Accordion("Re-Forge ⚡ (camera + seed, no LLM)", open=False):
                az_sl = gr.Slider(-180, 180, value=35, step=1,
                                  label="azimuth °")
                el_sl = gr.Slider(10, 80, value=30, step=1,
                                  label="elevation °")
                dist_sl = gr.Slider(0.5, 2.5, value=1.1, step=0.05,
                                    label="distance m")
                with gr.Row():
                    style_dd = gr.Dropdown(choices=[], label="style",
                                           interactive=True)
                    reforge_seed = gr.Number(
                        value=-1, precision=0,
                        label="gen seed (-1 = seed law)")
                    reseed_btn = gr.Button("🎲", scale=0)
                reforge_btn = gr.Button("Re-Forge ⚡", variant="primary")
                reforge_img = gr.AnnotatedImage(
                    label="re-forged — labels track the camera", height=340)
        with gr.Column():
            plan_stream = gr.Code(label="planner stream (live tokens)",
                                  language="json", lines=14, max_lines=18)
            spec_json = gr.JSON(label="validated SceneSpec (incl. grounding_log)",
                                max_height=320)
            source_md = gr.Markdown("")

    with gr.Row():
        gallery = gr.Gallery(
            label="forged dataset — rows = layout, cols = style",
            columns=4, height=520, object_fit="contain", preview=False,
        )
    with gr.Row():
        overlay_radio = gr.Radio(
            choices=list(OVERLAY_MODES), value="boxes",
            label="overlay (cached-path swap)",
        )

    with gr.Row():
        fidelity_lbl = gr.Label(label="fidelity (OWLv2 spot-check)",
                                num_top_classes=3)
        include_q_cb = gr.Checkbox(value=False,
                                   label="include quarantined in export")
        export_btn = gr.Button("Export COCO")
        download = gr.DownloadButton("dataset.zip", visible=True)

    # ------------------------------------------------------------ events
    forge_btn.click(
        handlers.on_forge,
        inputs=[task_tb, n_layouts, n_styles, seed, overlay_radio],
        outputs=[status_md, plan_stream, spec_json, source_md, layout_3d,
                 control_img, gallery, fidelity_lbl, style_dd, run_state,
                 download],
        concurrency_limit=1,
    )
    reforge_btn.click(
        handlers.on_reforge,
        inputs=[run_state, az_sl, el_sl, dist_sl, style_dd, reforge_seed],
        outputs=[reforge_img, layout_3d, control_img, status_md],
        concurrency_limit=1,
    )
    reseed_btn.click(handlers.on_reseed, outputs=[reforge_seed]).then(
        handlers.on_reforge,
        inputs=[run_state, az_sl, el_sl, dist_sl, style_dd, reforge_seed],
        outputs=[reforge_img, layout_3d, control_img, status_md],
        concurrency_limit=1,
    )
    overlay_radio.change(
        handlers.on_toggle,
        inputs=[run_state, overlay_radio],
        outputs=[gallery],
    )
    export_btn.click(
        handlers.on_export,
        inputs=[run_state, include_q_cb],
        outputs=[download, status_md],
        concurrency_limit=1,
    )


def _build_video_augment_tab() -> None:
    """Tab 2 — episode background restyler (sceneforge.augment, feature A).

    Upload a video OR point at a frames directory → n styles × (original |
    restyled) side-by-side players, mask audit sheet, zip download. The
    handler streams status while the depth→mask→SDXL pipeline runs; GPU work
    happens inside ``gpu.phase("diffusion")`` on the shared ForgeRun pipeline.
    """
    gr.Markdown(
        "Restyle the **far background** of a real robot episode into N "
        "appearance worlds — robot/workspace pixels stay bitwise identical "
        "(pre-encode). Upload a video **or** give a frames directory."
    )
    with gr.Row():
        with gr.Column(scale=1):
            va_video = gr.Video(label="episode video (upload)",
                                sources=["upload"], height=240)
            va_frames = gr.Textbox(
                label="…or absolute path to a directory of frames",
                placeholder="/path/to/episode/frames")
            with gr.Row():
                va_nstyles = gr.Slider(
                    1, handlers.VIDEO_AUGMENT_MAX_STYLES, value=2, step=1,
                    label="styles")
                va_keep = gr.Slider(40, 95, value=72, step=1,
                                    label="keep percentile (near %)")
                va_window = gr.Slider(1, 11, value=5, step=2,
                                      label="smoothing window (frames)")
            va_styles_tb = gr.Textbox(
                label="style prompts (optional, one per line)", lines=3,
                placeholder=("an industrial workshop, photo\n"
                             "a bright white laboratory, photo"))
            va_btn = gr.Button("AUGMENT", variant="primary")
            va_status = gr.Markdown(
                "**ready** — upload a video or set a frames dir, then AUGMENT")
            va_download = gr.DownloadButton("augment bundle (.zip)")
        with gr.Column(scale=2):
            va_audit = gr.Image(label="mask audit (red = restyled region)",
                                type="filepath", height=300)
            va_slots: list[gr.Video] = []
            for i in range(handlers.VIDEO_AUGMENT_MAX_STYLES):
                with gr.Row():
                    orig = gr.Video(label=f"original · style {i + 1}",
                                    visible=False, interactive=False)
                    sty = gr.Video(label=f"restyled · style {i + 1}",
                                   visible=False, interactive=False)
                va_slots += [orig, sty]

    va_btn.click(
        handlers.on_video_augment,
        inputs=[va_video, va_frames, va_nstyles, va_styles_tb, va_keep,
                va_window],
        outputs=[va_status, va_audit, va_download, *va_slots],
        concurrency_limit=1,
    )


def build_app() -> gr.Blocks:
    """Construct the SceneForge Gradio app (§11). Pure construction — no GPU
    work happens here; heavy singletons build lazily / in the pre-warm thread."""
    cfg = get_config()

    with gr.Blocks(title="SceneForge") as demo:
        run_state = gr.State()  # JSON-able dict of paths only (§4.8)

        gr.Markdown("# SceneForge — task → labeled photoreal COCO dataset")

        with gr.Tabs():
            with gr.Tab("Forge"):
                _build_forge_tab(run_state)
            with gr.Tab("Video Augment"):
                _build_video_augment_tab()

        footer_md = gr.Markdown(
            f"VRAM —/— GB · mode {cfg.vram.mode} · waiting for first snapshot…")
        timer = gr.Timer(2.0)
        timer.tick(handlers.on_timer, outputs=[footer_md])

    # One 3090: a second concurrent forge would OOM — queue everything at 1.
    demo.queue(default_concurrency_limit=1)
    return demo
