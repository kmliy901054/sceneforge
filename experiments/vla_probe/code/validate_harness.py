"""Step 1: load openvla-7b, check norm_stats keys, predict one action."""
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = "openvla/openvla-7b"
DEV = "cuda:0"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
try:
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEV)
    print("attn=sdpa OK")
except Exception as e:
    print(f"sdpa failed ({e}); falling back to eager")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEV)
vla.eval()

keys = sorted(vla.norm_stats.keys())
print(f"norm_stats keys ({len(keys)}):")
for k in keys:
    print("  ", k)
assert "bridge_orig" in keys, "bridge_orig missing!"

img = Image.open(
    "/home/pairlab/DGAN/sceneforge/experiments/vla_probe/images/originals/frame00.jpg"
).convert("RGB")
instruction = "put the red block in front of the strawberry"
prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
inputs = processor(prompt, img).to(DEV, dtype=torch.bfloat16)
with torch.inference_mode():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("7-DoF action:", action)
# determinism check
with torch.inference_mode():
    action2 = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("repeat      :", action2)
import numpy as np

print("max |diff| on repeat:", np.max(np.abs(action - action2)))
