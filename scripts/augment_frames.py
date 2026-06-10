"""scripts/augment_frames.py — ROSIE-style appearance randomizer CLI (v2 feature A).

Restyle the BACKGROUND of a real robot episode (directory of frames or a video)
into N appearance worlds while keeping robot/objects/workspace pixels bitwise
identical — the productized arm-S pipeline of the VLA probe
(experiments/vla_probe/REPORT.md: background restyle moves VLA actions ~8-10×
above the JPEG floor, so this is the targeted augmentation).

Usage:
    python scripts/augment_frames.py INPUT --out OUT_DIR [--n-styles 4]
        [--keep-percentile 72] [--keep-below 0.62] [--dilate 8] [--window 5]
        [--seed 42] [--styles-json styles.json | --style-prompt "..." ...]
        [--llm-styles] [--task "..."]

Style precedence: --style-prompt/--styles-json > --llm-styles (director, with
deterministic fallback) > DEFAULT_STYLES. styles.json: a list of prompt strings
or of {"name": ..., "prompt": ...} objects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `sceneforge`

import sceneforge.compat  # noqa: F401,E402  — FIRST sceneforge import (§0)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Background-appearance randomizer for real robot episodes "
                    "(near pixels stay bitwise identical)."
    )
    parser.add_argument("input", help="directory of frames OR a video file")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-styles", type=int, default=4)
    parser.add_argument("--keep-percentile", type=float, default=72.0,
                        help="keep the nearest X%% of pixels (restyle the rest)")
    parser.add_argument("--keep-below", type=float, default=0.62,
                        help="always keep rows below this height fraction "
                             "(negative disables)")
    parser.add_argument("--dilate", type=int, default=8,
                        help="keep-mask dilation px (protects robot borders)")
    parser.add_argument("--window", type=int, default=5,
                        help="odd temporal majority-vote window over frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--styles-json", type=Path, default=None,
                        help="JSON file: list of prompts or {name, prompt} objects")
    parser.add_argument("--style-prompt", action="append", default=None,
                        help="explicit style prompt (repeatable)")
    parser.add_argument("--llm-styles", action="store_true",
                        help="synthesize style prompts via the director "
                             "(deterministic fallback if Ollama is down)")
    parser.add_argument("--task", default="a robot arm manipulating objects on a cluttered table",
                        help="task hint for --llm-styles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    style_prompts = None
    if args.styles_json is not None:
        style_prompts = json.loads(args.styles_json.read_text(encoding="utf-8"))
    elif args.style_prompt:
        style_prompts = list(args.style_prompt)

    from sceneforge.augment.restyle import restyle_frames

    provenance = restyle_frames(
        args.input,
        args.out,
        n_styles=args.n_styles,
        keep_percentile=args.keep_percentile,
        style_prompts=style_prompts,
        seed=args.seed,
        keep_below_frac=None if args.keep_below < 0 else args.keep_below,
        dilate_px=args.dilate,
        smooth_window=args.window,
        use_llm_styles=args.llm_styles,
        task=args.task,
    )

    print(json.dumps({
        "n_frames": provenance["n_frames"],
        "styles": [s["name"] for s in provenance["styles"]],
        "style_source": provenance["style_source"],
        "restyle_frac_mean": round(
            sum(provenance["restyle_frac_per_frame"]) /
            max(1, len(provenance["restyle_frac_per_frame"])), 4),
        "timings_s": provenance["timings_s"],
        "outputs": provenance["outputs"],
        "provenance": str(Path(args.out) / "provenance.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
