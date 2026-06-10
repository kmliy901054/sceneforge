"""Qualitative grid: rows = example frames, cols = original + selected variants.
Each tile is annotated with the predicted translation (dx, dy, dz in mm) and
gripper state, plus an arrow showing the (dx, dy) robot-frame translation
(arrow drawn in image plane for magnitude/direction comparison only — the
robot base frame is NOT camera-aligned).
"""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"
with open(f"{ROOT}/results/actions.json") as f:
    rows = json.load(f)
A = {(r["frame"], r["arm"], r["variant"]): np.array(r["action"]) for r in rows}

SHOW_FRAMES = ["frame00", "frame08", "frame12", "frame18"]
COLS = [
    ("orig", "orig", "original"),
    ("P", "bright_dn", "P: brightness -35%"),
    ("P", "hue_p20", "P: hue +20deg"),
    ("S", "industrial", "S: bg industrial"),
    ("S", "nightroom", "S: bg night room"),
    ("G", "shift10", "G: 10% crop-shift"),
]
TW, TH = 320, 240
PAD_TOP = 26

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    font = fsm = ImageFont.load_default()

sheet = Image.new("RGB", (len(COLS) * TW, len(SHOW_FRAMES) * (TH + PAD_TOP)), "white")
draw_s = ImageDraw.Draw(sheet)
for ri, fr in enumerate(SHOW_FRAMES):
    for ci, (arm, var, label) in enumerate(COLS):
        if arm == "orig":
            p = f"{ROOT}/images/originals/{fr}.jpg"
        elif var == "jpeg95":
            p = f"{ROOT}/images/variants/{fr}__C_jpeg95.jpg"
        else:
            p = f"{ROOT}/images/variants/{fr}__{arm}_{var}.png"
        im = Image.open(p).convert("RGB").resize((TW, TH))
        a = A[(fr, arm, var)]
        d = ImageDraw.Draw(im)
        # arrow: robot-frame (dx, dy) scaled; origin bottom-center
        ox, oy = TW // 2, TH - 28
        sc = 4000.0  # m -> px
        ex, ey = ox + a[0] * sc, oy - a[1] * sc
        d.line([(ox, oy), (ex, ey)], fill="lime", width=3)
        d.ellipse([ox - 3, oy - 3, ox + 3, oy + 3], fill="lime")
        grip = "open" if a[6] > 0.5 else "closed"
        txt = f"d=({a[0]*1000:+.1f},{a[1]*1000:+.1f},{a[2]*1000:+.1f})mm grip:{grip}"
        d.rectangle([0, TH - 18, TW, TH], fill=(0, 0, 0, 180))
        d.text((4, TH - 16), txt, fill="yellow", font=fsm)
        y0 = ri * (TH + PAD_TOP)
        draw_s.text((ci * TW + 6, y0 + 5), label if ri == 0 else "", fill="black", font=font)
        sheet.paste(im, (ci * TW, y0 + PAD_TOP))
    inst = [r["instruction"] for r in rows if r["frame"] == fr][0]
    draw_s.text((6, ri * (TH + PAD_TOP) + PAD_TOP - 21 if ri else 5), "", fill="black")
for ri, fr in enumerate(SHOW_FRAMES):
    inst = [r["instruction"] for r in rows if r["frame"] == fr][0]
    draw_s.text(
        (len(COLS) * TW - 640, ri * (TH + PAD_TOP) + 6),
        f'{fr}: "{inst}"',
        fill="black",
        font=fsm,
    )
sheet.save(f"{ROOT}/results/qualitative_grid.jpg", quality=90)
print("grid saved")
