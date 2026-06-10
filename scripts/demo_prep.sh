#!/usr/bin/env bash
# scripts/demo_prep.sh — pre-warm gemma + SDXL + one throwaway forge before
# recording (ARCHITECTURE.md §10.5 / M6 demo checklist).
#
# Runs the §10.5 pre-warm (gemma page-cache prime + eager ForgePipeline cuda
# build + OWLv2 to CPU), then one throwaway forge so every cache, allocator
# pool and CUDA graph is hot before the demo recording starts.
set -euo pipefail
cd "$(dirname "$0")/.."

source /home/pairlab/miniconda3/etc/profile.d/conda.sh
conda activate dgan

python - <<'EOF'
import sceneforge.compat  # noqa: F401  — first import (§0)

import logging
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from sceneforge.ui import handlers

t0 = time.monotonic()
print("demo_prep: pre-warm (gemma prime + SDXL cuda build + OWLv2 to CPU)…",
      flush=True)
handlers.prewarm()
print(f"demo_prep: pre-warm done in {time.monotonic() - t0:.1f} s", flush=True)

runner = handlers.get_runner()
t0 = time.monotonic()
print('demo_prep: throwaway forge ("pick the red mug from a cluttered kitchen '
      'table", 1 layout x 2 styles)…', flush=True)
last = None
for event in runner.run("pick the red mug from a cluttered kitchen table",
                        n_layouts=1, n_styles=2, seed=7):
    if event.kind != "plan_token":
        print(f"  event: {event.kind}", flush=True)
    last = event
assert last is not None and last.kind == "done", \
    f"throwaway forge did not finish cleanly (last event: {last and last.kind})"
print(f"demo_prep: throwaway forge done in {time.monotonic() - t0:.1f} s "
      f"(run {last.payload['run_id']}) — ready to record", flush=True)
EOF
