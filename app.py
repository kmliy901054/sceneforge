"""SceneForge entry point: python app.py → Gradio on :7860 (ARCHITECTURE.md §2, §10.5).

Launch sequence (§10.5 pre-warm, in a background thread so the UI is up
immediately; the footer Timer shows a "warming up" banner until done):

  1. load + immediately unload gemma4:e4b (populates the page cache so
     subsequent reloads take the measured 3.0 s instead of 33.4 s cold);
  2. eagerly construct the ForgePipeline on cuda (30–90 s first build);
  3. instantiate OWLv2 processor+model once to CPU (moved to cuda only inside
     gpu.phase("eval")).

``--no-prewarm`` skips all three for fast dev iteration.
"""
import sceneforge.compat  # noqa: F401  — MUST be line 1 (§0 import rule)

import argparse
import logging
import os
import threading


def main() -> None:
    parser = argparse.ArgumentParser(description="SceneForge Gradio app")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="skip the §10.5 pre-warm (fast dev iteration)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    parser.add_argument("--server-name",
                        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    from sceneforge.ui import handlers
    from sceneforge.ui.blocks import build_app

    demo = build_app()

    if not args.no_prewarm:
        threading.Thread(target=handlers.prewarm, name="prewarm",
                         daemon=True).start()

    demo.launch(server_name=args.server_name, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
