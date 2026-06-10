"""Step 4: OpenVLA inference sweep — one model load, deterministic predictions
for originals, temporal neighbors (Arm T), and every variant. Raw actions JSON.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"
DEV = "cuda:0"

with open(f"{ROOT}/images/provenance.json") as f:
    prov = {p["frame"]: p for p in json.load(f)}
manifest = []
for mf in ("manifest_pg.json", "manifest_s.json"):
    with open(f"{ROOT}/images/{mf}") as f:
        manifest += json.load(f)
for name in sorted(prov):
    manifest.append(
        {"frame": name, "arm": "orig", "variant": "orig",
         "path": f"{ROOT}/images/originals/{name}.jpg"}
    )
    manifest.append(
        {"frame": name, "arm": "T", "variant": "tplus",
         "path": f"{ROOT}/images/originals/{name}_tplus.jpg"}
    )
print(f"{len(manifest)} predictions to run")

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(DEV)
vla.eval()

results = []
t0 = time.time()
for i, m in enumerate(manifest):
    img = Image.open(m["path"]).convert("RGB")
    instruction = prov[m["frame"]]["instruction"]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEV, dtype=torch.bfloat16)
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    results.append({**{k: m[k] for k in ("frame", "arm", "variant")},
                    "instruction": instruction,
                    "action": np.asarray(action).tolist()})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(manifest)}  {el:.0f}s  ({el/(i+1):.2f}s/pred)", flush=True)

with open(f"{ROOT}/results/actions.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"saved {len(results)} actions  total {time.time()-t0:.0f}s")
