"""Comparison arm, step 1: validate SpatialVLA-4B harness (env dgan).

Checks: norm-stats key, deterministic decoding, action shape/scale/order vs the
OpenVLA bridge_orig convention (7-DoF EE delta: dx,dy,dz [m], droll,dpitch,dyaw
[rad], gripper in [0,1]). SpatialVLA emits an action chunk (size 4); the FIRST
action is the current-step action and is what we compare.
"""
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# --- shim: transformers 4.51 removed make_batched_images from paligemma
# processing; make_flat_list_of_images is its direct replacement. The
# SpatialVLA remote code imports it from the old location.
import transformers.models.paligemma.processing_paligemma as _ppg

if not hasattr(_ppg, "make_batched_images"):
    from transformers.image_utils import make_flat_list_of_images

    _ppg.make_batched_images = make_flat_list_of_images

MODEL = "IPEC-COMMUNITY/spatialvla-4b-224-pt"
UNNORM = "bridge_orig/1.0.0"
ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
keys = sorted(processor.statistics.keys())
bridge_keys = [k for k in keys if "bridge" in k]
print(f"statistics keys: {len(keys)} total; bridge: {bridge_keys}")
assert UNNORM in keys, f"{UNNORM} missing!"
print("intrinsic keys:", sorted(processor.dataset_intrinsics.keys())[:5], "...")
print("action_chunk_size:", processor.action_chunk_size)

model = (
    AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
    )
    .eval()
    .cuda()
)
print(f"loaded; cuda mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

with open(f"{ROOT}/images/provenance.json") as f:
    prov = {p["frame"]: p for p in json.load(f)}
with open(f"{ROOT}/results/actions.json") as f:
    ov = {r["frame"]: np.array(r["action"]) for r in json.load(f) if r["arm"] == "orig"}

for fr in ("frame00", "frame04"):
    img = Image.open(f"{ROOT}/images/originals/{fr}.jpg").convert("RGB")
    instruction = prov[fr]["instruction"]
    prompt = f"What action should the robot take to {instruction.lower()}?"
    inputs = processor(images=[img], text=prompt, unnorm_key=UNNORM, return_tensors="pt")
    with torch.inference_mode():
        gen = model.predict_action(inputs)
        gen2 = model.predict_action(inputs)
    out = processor.decode_actions(gen, unnorm_key=UNNORM)
    out2 = processor.decode_actions(gen2, unnorm_key=UNNORM)
    acts = out["actions"]
    print(f"\n{fr}: '{instruction}'")
    print("  chunk shape:", acts.shape, " token ids:", out["action_ids"].tolist())
    print("  action[0]   :", np.round(acts[0], 5).tolist())
    print("  repeat diff :", np.max(np.abs(acts - out2["actions"])))
    print("  openvla orig:", np.round(ov[fr], 5).tolist())
    print(f"  |dt| spatialvla {np.linalg.norm(acts[0][:3])*1000:.2f} mm  "
          f"openvla {np.linalg.norm(ov[fr][:3])*1000:.2f} mm  "
          f"grip sv={acts[0][6]:.2f} ov={ov[fr][6]:.2f}")
