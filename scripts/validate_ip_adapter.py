"""GPU validation for the IP-Adapter style-reference feature (DO NOT run while
the GPU is busy — the orchestrator schedules this).

Loads the real ForgePipeline (SDXL + depth ControlNet + Lightning at the
configured level), builds one synthetic depth control, then generates the SAME
(prompt, seed) twice:

    A. baseline            — text prompt only
    B. style reference on  — enable_style_reference(ref, scale) so the
                             IP-Adapter (h94/IP-Adapter, sdxl vit-h +
                             laion CLIP-ViT-H encoder) ADDS the reference's
                             visual style on top of the text prompt

and saves ``ref | baseline | styled`` side by side plus the individual PNGs
under outputs/ip_adapter_validate/. Pass criteria (eyeballed from the sheet):
B clearly inherits the reference's palette/texture while keeping the depth
layout of A, and disable_style_reference() restores baseline behavior
(image C, generated after disabling, must match A pixel-for-pixel).

Run: cd /home/pairlab/DGAN/sceneforge && python scripts/validate_ip_adapter.py \
        [--ref path/to/reference.png] [--scale 0.6] [--prompt "..."] [--seed 42]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `sceneforge`

import sceneforge.compat  # noqa: F401,E402  — FIRST sceneforge import (§0)

import argparse
import time

import numpy as np
from PIL import Image, ImageDraw

from sceneforge.config import get_config
from sceneforge.diffusion.pipeline import ForgePipeline

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ip_adapter_validate"


def synthetic_control(res: int) -> Image.Image:
    """Depth-style control: bright floor gradient + two object blobs."""
    y = np.linspace(1.0, 0.25, res, dtype=np.float32)[:, None]
    ctrl = np.repeat(y, res, axis=1)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / res
    for cx, cy, r, v in ((0.38, 0.62, 0.16, 0.95), (0.66, 0.55, 0.10, 0.85)):
        ctrl[(xx - cx) ** 2 + (yy - cy) ** 2 < r**2] = v
    arr = (ctrl * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def synthetic_reference(res: int = 512) -> Image.Image:
    """Strongly-styled fallback reference: warm sunset palette + stripes —
    distinctive enough that style transfer is obvious at scale 0.6."""
    top, bottom = np.array([255, 94, 19]), np.array([64, 8, 84])
    t = np.linspace(0, 1, res)[:, None, None]
    arr = (top * (1 - t) + bottom * t).astype(np.uint8)
    img = Image.fromarray(np.broadcast_to(arr, (res, res, 3)).copy())
    draw = ImageDraw.Draw(img)
    for x in range(0, res, 64):
        draw.line([(x, 0), (x + res // 3, res)], fill=(255, 214, 90), width=10)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=Path, default=None,
                    help="style reference image (default: synthetic sunset)")
    ap.add_argument("--scale", type=float, default=None,
                    help="IP-Adapter scale (default: cfg.gen.ip_adapter_scale)")
    ap.add_argument("--prompt", default="a tidy workshop table with a mug and "
                                        "a small box, photo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_config()
    res = cfg.gen.resolution
    scale = args.scale if args.scale is not None else cfg.gen.ip_adapter_scale
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = (Image.open(args.ref).convert("RGB") if args.ref
           else synthetic_reference())
    ref.save(OUT_DIR / "reference.png")
    control = synthetic_control(res)
    control.save(OUT_DIR / "control.png")

    fp = ForgePipeline(resolution=res)
    level = int(cfg.gen.level[1])
    print(f"[1/4] loading ForgePipeline (level L{level}) …", flush=True)
    fp.load(level)

    print(f"[2/4] baseline generate (seed {args.seed}) …", flush=True)
    t0 = time.monotonic()
    base = fp.generate(control, args.prompt, seed=args.seed,
                       cond_scale=cfg.gen.cond_scale, steps=cfg.gen.steps)
    print(f"      {time.monotonic() - t0:.1f} s", flush=True)
    base.save(OUT_DIR / "baseline.png")

    print(f"[3/4] styled generate (IP-Adapter scale {scale}) …", flush=True)
    fp.enable_style_reference(ref, scale=scale)  # AFTER load() — see pipeline.py
    assert fp.style_reference_enabled
    t0 = time.monotonic()
    styled = fp.generate(control, args.prompt, seed=args.seed,
                         cond_scale=cfg.gen.cond_scale, steps=cfg.gen.steps)
    print(f"      {time.monotonic() - t0:.1f} s "
          f"(first styled call includes the IP-Adapter load)", flush=True)
    styled.save(OUT_DIR / "styled.png")

    print("[4/4] disable + regression generate …", flush=True)
    fp.disable_style_reference()
    again = fp.generate(control, args.prompt, seed=args.seed,
                        cond_scale=cfg.gen.cond_scale, steps=cfg.gen.steps)
    again.save(OUT_DIR / "baseline_after_disable.png")
    identical = np.array_equal(np.asarray(base), np.asarray(again))
    print(f"      disable restores baseline pixel-exact: {identical}")

    # ref | baseline | styled comparison sheet
    tiles = [ref.resize((res, res)), base, styled]
    sheet = Image.new("RGB", (res * 3 + 20, res + 30), (24, 24, 24))
    for i, (tile, name) in enumerate(zip(tiles, ("reference", "baseline",
                                                 f"styled (scale {scale})"))):
        sheet.paste(tile, (i * (res + 10), 24))
        ImageDraw.Draw(sheet).text((i * (res + 10) + 4, 6), name,
                                   fill=(240, 240, 240))
    sheet_path = OUT_DIR / "comparison.png"
    sheet.save(sheet_path)
    print(f"\nwrote {sheet_path}")
    print(f"peak vram: {fp.peak_vram()}")
    if not identical:
        print("FAIL: disable_style_reference() did not restore the baseline")
        return 1
    print("PASS (visual check: styled must inherit the reference palette)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
