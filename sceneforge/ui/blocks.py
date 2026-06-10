"""sceneforge/ui/blocks.py — build_app() → gr.Blocks (ARCHITECTURE.md §11).

Gradio 6.17.3 (kwargs verified against the installed version). Layout map:

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


def build_app() -> gr.Blocks:
    """Construct the SceneForge Gradio app (§11). Pure construction — no GPU
    work happens here; heavy singletons build lazily / in the pre-warm thread."""
    cfg = get_config()

    with gr.Blocks(title="SceneForge") as demo:
        run_state = gr.State()  # JSON-able dict of paths only (§4.8)

        gr.Markdown("# SceneForge — task → labeled photoreal COCO dataset")

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

        footer_md = gr.Markdown(
            f"VRAM —/— GB · mode {cfg.vram.mode} · waiting for first snapshot…")
        timer = gr.Timer(2.0)

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
        timer.tick(handlers.on_timer, outputs=[footer_md])

    # One 3090: a second concurrent forge would OOM — queue everything at 1.
    demo.queue(default_concurrency_limit=1)
    return demo
