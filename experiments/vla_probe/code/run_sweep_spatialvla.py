"""Comparison arm, step 2: SpatialVLA-4B inference sweep (env dgan) — one model
load, deterministic predictions for the IDENTICAL (frame, variant, instruction)
set as the OpenVLA sweep (440 predictions). Raw actions JSON.

Determinism: greedy decoding alone is NOT bitwise-deterministic for this model
(near-tied logits + nondeterministic cuBLAS split-k reductions flip an action
token occasionally). Run with CUBLAS_WORKSPACE_CONFIG=:4096:8 and the
torch.use_deterministic_algorithms(True) below; this was verified to give
max |repeat diff| = 0.0 over 3 runs x 6 frames.

SpatialVLA emits an action chunk of 4 steps; "action" holds the FIRST
(current-step) action — same 7-DoF Bridge EE-delta convention and the same
q01/q99 unnormalization bounds as OpenVLA's bridge_orig key. Full chunk kept
in "action_chunk".
"""
import json
import os
import time

assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", (
    "run with CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic decoding"
)

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- shim: transformers 4.51 removed make_batched_images from paligemma
# processing; make_flat_list_of_images is its direct replacement.
import transformers.models.paligemma.processing_paligemma as _ppg

if not hasattr(_ppg, "make_batched_images"):
    from transformers.image_utils import make_flat_list_of_images

    _ppg.make_batched_images = make_flat_list_of_images

from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "IPEC-COMMUNITY/spatialvla-4b-224-pt"
UNNORM = "bridge_orig/1.0.0"
ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"

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

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
model = (
    AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
    )
    .eval()
    .cuda()
)

results = []
t0 = time.time()
for i, m in enumerate(manifest):
    img = Image.open(m["path"]).convert("RGB")
    instruction = prov[m["frame"]]["instruction"]
    prompt = f"What action should the robot take to {instruction.lower()}?"
    inputs = processor(images=[img], text=prompt, unnorm_key=UNNORM, return_tensors="pt")
    with torch.inference_mode():
        gen = model.predict_action(inputs)
    out = processor.decode_actions(gen, unnorm_key=UNNORM)
    chunk = np.asarray(out["actions"])
    results.append({**{k: m[k] for k in ("frame", "arm", "variant")},
                    "instruction": instruction,
                    "action": chunk[0].tolist(),
                    "action_chunk": chunk.tolist()})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(manifest)}  {el:.0f}s  ({el/(i+1):.2f}s/pred)", flush=True)

# end-of-sweep determinism re-check on the first manifest entry
m = manifest[0]
img = Image.open(m["path"]).convert("RGB")
prompt = f"What action should the robot take to {prov[m['frame']]['instruction'].lower()}?"
inputs = processor(images=[img], text=prompt, unnorm_key=UNNORM, return_tensors="pt")
with torch.inference_mode():
    gen = model.predict_action(inputs)
rep = np.asarray(processor.decode_actions(gen, unnorm_key=UNNORM)["actions"])[0]
print("determinism re-check max |diff|:", np.max(np.abs(rep - np.array(results[0]["action"]))))

with open(f"{ROOT}/results/actions_spatialvla.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"saved {len(results)} actions  total {time.time()-t0:.0f}s")
