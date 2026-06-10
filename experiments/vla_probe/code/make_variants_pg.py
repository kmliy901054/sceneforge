"""Arm P (photometric, 10/frame), Arm G (geometry baseline, 3/frame),
Arm C (jpeg95 re-encode floor control, 1/frame).
Pure pixel ops; variants saved as PNG (lossless). Manifest in images/manifest_pg.json.
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"
OUT = f"{ROOT}/images/variants"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def hue_shift(img, deg):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(round(deg / 2.0))) % 180  # OpenCV hue: 0..179
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def gamma(img, g):
    a = np.array(img).astype(np.float32) / 255.0
    return Image.fromarray((np.power(a, g) * 255).clip(0, 255).astype(np.uint8))


def wb_shift(img):  # warm shift ~ +600K-equivalent crude gains
    a = np.array(img).astype(np.float32)
    a[..., 0] *= 1.18
    a[..., 2] *= 0.82
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def iso_noise(img, seed):
    r = np.random.default_rng(seed)
    a = np.array(img).astype(np.float32)
    a += r.normal(0, 8.0, a.shape)  # sensor read noise
    a += r.normal(0, 4.0, (a.shape[0], a.shape[1], 1))  # luma component
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def crop_shift(img, frac):
    w, h = img.size
    dx, dy = int(w * frac), int(h * frac)
    return img.crop((dx, dy, w, h)).resize((w, h), Image.BICUBIC)


def rotate(img, deg):
    a = np.array(img)
    h, w = a.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


P_OPS = {
    "bright_up": lambda im, i: ImageEnhance.Brightness(im).enhance(1.35),
    "bright_dn": lambda im, i: ImageEnhance.Brightness(im).enhance(0.65),
    "contrast_up": lambda im, i: ImageEnhance.Contrast(im).enhance(1.35),
    "contrast_dn": lambda im, i: ImageEnhance.Contrast(im).enhance(0.65),
    "hue_p20": lambda im, i: hue_shift(im, +20),
    "hue_m20": lambda im, i: hue_shift(im, -20),
    "gamma_06": lambda im, i: gamma(im, 0.6),
    "gamma_16": lambda im, i: gamma(im, 1.6),
    "wb_warm": lambda im, i: wb_shift(im),
    "iso_noise": lambda im, i: iso_noise(im, 1000 + i),
}
G_OPS = {
    "shift05": lambda im, i: crop_shift(im, 0.05),
    "shift10": lambda im, i: crop_shift(im, 0.10),
    "rot10": lambda im, i: rotate(im, 10.0),
}

manifest = []
frames = sorted(glob.glob(f"{ROOT}/images/originals/frame??.jpg"))
for i, f in enumerate(frames):
    name = os.path.basename(f).replace(".jpg", "")
    img = Image.open(f).convert("RGB")
    for arm, ops in (("P", P_OPS), ("G", G_OPS)):
        for vname, fn in ops.items():
            out = fn(img, i)
            path = f"{OUT}/{name}__{arm}_{vname}.png"
            out.save(path)
            manifest.append({"frame": name, "arm": arm, "variant": vname, "path": path})
    # floor control: jpeg re-encode q95
    pj = f"{OUT}/{name}__C_jpeg95.jpg"
    img.save(pj, quality=95)
    manifest.append({"frame": name, "arm": "C", "variant": "jpeg95", "path": pj})
    print(name, "ok")

with open(f"{ROOT}/images/manifest_pg.json", "w") as fp:
    json.dump(manifest, fp, indent=2)
print("variants:", len(manifest))
