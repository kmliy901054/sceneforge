"""scripts/augment_dataset.py — LeRobot-dataset background restyler CLI.

Restyle the chosen camera streams of a LeRobot v2.x dataset into N appearance
worlds, writing ONE valid output dataset per style (``<name>_<style_slug>``):
parquet/actions/states/timestamps copied bitwise, only the selected cameras'
mp4 frames replaced (near-workspace pixels bitwise identical per frame).
Implemented natively (pyarrow + ffmpeg) — see sceneforge/augment/lerobot_io.py
for the layout contract and the no-lerobot-dependency rationale.

Usage:
    python scripts/augment_dataset.py --dataset DIR --out DIR \
        --cameras observation.images.top[,observation.images.wrist] \
        [--n-styles 2] [--styles "prompt one; prompt two" | --styles-json f.json]
        [--llm-styles] [--episodes 0:10] [--keep-percentile 72]
        [--keep-below 0.62] [--dilate 8] [--window 5] [--seed 42] [--crf 18]
        [--no-audit] [--task "..."]

Style precedence: --styles/--styles-json > --llm-styles (director, with
deterministic fallback) > DEFAULT_STYLES. ``--styles`` is a ';'-separated list
of prompts; --styles-json is a JSON list of prompts or {name, prompt} objects.
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
        description="LeRobot v2.x dataset background-appearance randomizer "
                    "(near pixels stay bitwise identical; parquet untouched).")
    parser.add_argument("--dataset", required=True,
                        help="source LeRobot v2.x dataset directory")
    parser.add_argument("--out", required=True,
                        help="output directory (one dataset per style inside)")
    parser.add_argument("--cameras", required=True,
                        help="comma-separated video keys to restyle "
                             "(full key or unique suffix, e.g. 'top')")
    parser.add_argument("--n-styles", type=int, default=2)
    parser.add_argument("--styles", default=None,
                        help="';'-separated explicit style prompts")
    parser.add_argument("--styles-json", type=Path, default=None,
                        help="JSON file: list of prompts or {name, prompt} objects")
    parser.add_argument("--llm-styles", action="store_true",
                        help="synthesize style prompts via the director "
                             "(deterministic fallback if Ollama is down)")
    parser.add_argument("--episodes", default=None,
                        help="half-open episode slice 'start:stop', e.g. 0:10")
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
    parser.add_argument("--crf", type=int, default=18,
                        help="libx264 CRF for the restyled output videos")
    parser.add_argument("--no-audit", action="store_true",
                        help="skip the per-episode mask audit sheets")
    parser.add_argument("--task",
                        default="a robot arm manipulating objects on a cluttered table",
                        help="task hint for --llm-styles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    style_prompts = None
    if args.styles_json is not None:
        style_prompts = json.loads(args.styles_json.read_text(encoding="utf-8"))
    elif args.styles:
        style_prompts = [s.strip() for s in args.styles.split(";") if s.strip()]

    from sceneforge.augment.lerobot_io import (
        augment_dataset, parse_episode_range, validate_dataset,
    )

    summary = augment_dataset(
        args.dataset,
        args.out,
        cameras=[c.strip() for c in args.cameras.split(",") if c.strip()],
        n_styles=args.n_styles,
        style_prompts=style_prompts,
        use_llm_styles=args.llm_styles,
        episodes=parse_episode_range(args.episodes),
        keep_percentile=args.keep_percentile,
        keep_below_frac=None if args.keep_below < 0 else args.keep_below,
        dilate_px=args.dilate,
        smooth_window=args.window,
        seed=args.seed,
        crf=args.crf,
        write_audit=not args.no_audit,
        task=args.task,
    )

    validations = {slug: validate_dataset(root)
                   for slug, root in summary["outputs"].items()}

    print(json.dumps({
        "outputs": summary["outputs"],
        "validated": {slug: v["ok"] for slug, v in validations.items()},
        "episodes": summary["episodes"],
        "cameras_restyled": summary["cameras_restyled"],
        "styles": [s["name"] for s in summary["styles"]],
        "style_source": summary["style_source"],
        "restyle_frac_mean": summary["restyle_frac_mean"],
        "n_frames_restyled": summary["n_frames_restyled"],
        "timings_s": summary["timings_s"],
        "audit_dir": summary["audit_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
