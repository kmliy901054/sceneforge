"""Arm S step 1: Depth-Anything-V2-Small depth -> percentile split near(keep)/far(restyle).

Outputs per frame in masks/: depth16 PNG, keep-mask PNG (255=near/keep),
and an audit overlay (red = region to be restyled).
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"
ORIG = sorted(glob.glob(f"{ROOT}/images/originals/frame??.jpg"))
MASKS = f"{ROOT}/masks"
os.makedirs(MASKS, exist_ok=True)

# restyle the farthest PCT percent of pixels, but never below KEEP_BELOW*H
# (workspace spatial prior: Bridge tabletop occupies the lower image)
PCT = 28
KEEP_BELOW = 0.62
DILATE = 8  # dilate keep-mask by this many px to protect robot/object borders

depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0,
    torch_dtype=torch.float32,
)

import cv2

for f in ORIG:
    name = os.path.basename(f).replace(".jpg", "")
    img = Image.open(f).convert("RGB")
    out = depth_pipe(img)
    depth = np.array(out["predicted_depth"].squeeze().cpu(), dtype=np.float32)
    depth = cv2.resize(depth, img.size, interpolation=cv2.INTER_CUBIC)
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # 1 = near
    thr = np.percentile(d, PCT)
    keep = (d >= thr).astype(np.uint8) * 255  # near workspace
    keep[int(KEEP_BELOW * d.shape[0]) :, :] = 255  # lower image always kept
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
    keep = cv2.dilate(keep, k)
    # save artifacts
    cv2.imwrite(f"{MASKS}/{name}_depth.png", (d * 65535).astype(np.uint16))
    cv2.imwrite(f"{MASKS}/{name}_keep.png", keep)
    ov = np.array(img).copy()
    far = keep == 0
    ov[far] = (0.45 * ov[far] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    Image.fromarray(ov).save(f"{MASKS}/{name}_audit.jpg", quality=90)
    print(name, f"restyle_frac={far.mean():.3f}")
print("done")
