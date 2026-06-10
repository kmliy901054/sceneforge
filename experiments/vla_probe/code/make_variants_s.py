"""Arm S: SceneForge-style background restyle (4 styles/frame).

SDXL-base + diffusers/controlnet-depth-sdxl-1.0 inpainting restricted to the
far-background mask (white = repaint). Depth conditioning from the saved
Depth-Anything-V2 maps keeps the room layout; after generation the ORIGINAL
near pixels are pasted back exactly (hard mask), so robot+objects+workspace
geometry are pixel-identical and only background appearance changes.
"""
import glob
import json
import os

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from PIL import Image

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"
OUT = f"{ROOT}/images/variants"
GEN_W, GEN_H = 1024, 768  # 4:3 like the 640x480 source

STYLES = {
    "industrial": "an industrial workshop interior, bare concrete walls, metal shelving, cold blue fluorescent lighting, photo",
    "warmkitchen": "a cozy home kitchen, warm wooden cabinets, soft warm evening sunlight, photo",
    "whitelab": "a bright white laboratory room, clean white wall panels, bright clinical lighting, photo",
    "nightroom": "a dim room at night, dark painted walls, moody orange lamp light, photo",
}
NEG = "robot, robotic arm, gripper, person, hands, text, watermark, blurry, distorted"

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)

manifest = []
frames = sorted(glob.glob(f"{ROOT}/images/originals/frame??.jpg"))
for fi, f in enumerate(frames):
    name = os.path.basename(f).replace(".jpg", "")
    img = Image.open(f).convert("RGB")
    w, h = img.size
    keep = cv2.imread(f"{ROOT}/masks/{name}_keep.png", cv2.IMREAD_GRAYSCALE)
    depth16 = cv2.imread(f"{ROOT}/masks/{name}_depth.png", cv2.IMREAD_UNCHANGED)
    depth8 = (depth16.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    inpaint_mask = 255 - keep  # white = repaint (far background)

    img_g = img.resize((GEN_W, GEN_H), Image.BICUBIC)
    mask_g = Image.fromarray(
        cv2.resize(inpaint_mask, (GEN_W, GEN_H), interpolation=cv2.INTER_NEAREST)
    )
    depth_g = Image.fromarray(
        cv2.cvtColor(cv2.resize(depth8, (GEN_W, GEN_H), cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
    )

    keep_f = (keep.astype(np.float32) / 255.0)[..., None]
    orig_np = np.array(img).astype(np.float32)

    for si, (sname, prompt) in enumerate(STYLES.items()):
        gen = torch.Generator("cuda").manual_seed(10_000 + 100 * fi + si)
        out = pipe(
            prompt=prompt,
            negative_prompt=NEG,
            image=img_g,
            mask_image=mask_g,
            control_image=depth_g,
            num_inference_steps=25,
            guidance_scale=7.0,
            controlnet_conditioning_scale=0.5,
            strength=0.99,
            generator=gen,
            width=GEN_W,
            height=GEN_H,
        ).images[0]
        gen_np = np.array(out.resize((w, h), Image.BICUBIC)).astype(np.float32)
        comp = orig_np * keep_f + gen_np * (1.0 - keep_f)  # hard composite, near exact
        comp_img = Image.fromarray(comp.clip(0, 255).astype(np.uint8))
        path = f"{OUT}/{name}__S_{sname}.png"
        comp_img.save(path)
        manifest.append({"frame": name, "arm": "S", "variant": sname, "path": path})
    print(name, "ok", flush=True)

with open(f"{ROOT}/images/manifest_s.json", "w") as fp:
    json.dump(manifest, fp, indent=2)
print("variants:", len(manifest))
