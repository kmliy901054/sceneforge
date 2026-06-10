"""Step 5: analysis of appearance-invariance probe results.

For each variant vs the SAME frame's original prediction:
  - translation L2 (m, unnormalized bridge_orig scale) and cosine similarity
  - rotation L2 (rad)
  - gripper flip (binarize at 0.5)
Arm T = prediction(orig frame) vs prediction(t+3 frame) — natural temporal scale.
Per-arm stats use per-frame means first, then a 95% CI over the 22 frames
(avoids pseudo-replication). Key ratio: P,S deviation relative to G.
"""
import json

import numpy as np
from scipy import stats

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"

with open(f"{ROOT}/results/actions.json") as f:
    rows = json.load(f)

orig = {r["frame"]: np.array(r["action"]) for r in rows if r["arm"] == "orig"}
frames = sorted(orig)

records = []  # one per (frame, arm, variant)
for r in rows:
    if r["arm"] == "orig":
        continue
    a = np.array(r["action"])
    o = orig[r["frame"]]
    dt = a[:3] - o[:3]
    cos = float(
        np.dot(a[:3], o[:3]) / (np.linalg.norm(a[:3]) * np.linalg.norm(o[:3]) + 1e-12)
    )
    records.append(
        dict(
            frame=r["frame"],
            arm=r["arm"],
            variant=r["variant"],
            trans_l2=float(np.linalg.norm(dt)),
            trans_cos=cos,
            rot_l2=float(np.linalg.norm(a[3:6] - o[3:6])),
            grip_flip=int((a[6] > 0.5) != (o[6] > 0.5)),
            grip_abs=float(abs(a[6] - o[6])),
        )
    )

with open(f"{ROOT}/results/deviations.json", "w") as f:
    json.dump(records, f, indent=1)

ARMS = ["C", "P", "S", "G", "T"]
ARM_LABEL = {
    "C": "C: jpeg95 floor",
    "P": "P: photometric",
    "S": "S: bg restyle",
    "G": "G: geometry",
    "T": "T: temporal (t+3)",
}
METRICS = ["trans_l2", "trans_cos", "rot_l2", "grip_flip"]


def per_frame_means(arm, metric):
    out = []
    for fr in frames:
        v = [r[metric] for r in records if r["arm"] == arm and r["frame"] == fr]
        if v:
            out.append(np.mean(v))
    return np.array(out)


def ci95(x):
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


summary = {}
print(f"{'arm':<20}{'trans L2 (m)':<22}{'trans cos':<14}{'rot L2 (rad)':<22}{'grip flip'}")
for arm in ARMS:
    s = {}
    for metric in METRICS:
        x = per_frame_means(arm, metric)
        s[metric] = {"mean": float(x.mean()), "ci95": ci95(x), "n_frames": len(x)}
    summary[arm] = s
    print(
        f"{ARM_LABEL[arm]:<20}"
        f"{s['trans_l2']['mean']:.5f} ±{s['trans_l2']['ci95']:.5f}     "
        f"{s['trans_cos']['mean']:.3f}     "
        f"{s['rot_l2']['mean']:.5f} ±{s['rot_l2']['ci95']:.5f}     "
        f"{s['grip_flip']['mean']*100:.1f}%"
    )

# key ratios + paired tests (per-frame pairing, Wilcoxon signed-rank)
print("\nKey ratios (translation L2, per-frame paired):")
ratios = {}
for arm in ("P", "S"):
    a = per_frame_means(arm, "trans_l2")
    g = per_frame_means("G", "trans_l2")
    t = per_frame_means("T", "trans_l2")
    w_g = stats.wilcoxon(a, g)
    ratios[arm] = {
        "ratio_vs_G": float(a.mean() / g.mean()),
        "ratio_vs_T": float(a.mean() / t.mean()),
        "frac_frames_gt_G": float((a > g).mean()),
        "wilcoxon_vs_G_p": float(w_g.pvalue),
    }
    print(
        f"  {arm}/G = {ratios[arm]['ratio_vs_G']:.2f}   {arm}/T = {ratios[arm]['ratio_vs_T']:.2f}   "
        f"frames {arm}>G: {ratios[arm]['frac_frames_gt_G']*100:.0f}%   wilcoxon p={w_g.pvalue:.4f}"
    )

# per-variant breakdown (translation L2 mean over frames)
print("\nPer-variant mean translation L2 (m):")
pv = {}
for arm in ARMS:
    vars_ = sorted({r["variant"] for r in records if r["arm"] == arm})
    for v in vars_:
        x = np.array([r["trans_l2"] for r in records if r["arm"] == arm and r["variant"] == v])
        pv[f"{arm}/{v}"] = {"mean": float(x.mean()), "std": float(x.std()), "n": len(x)}
        print(f"  {arm:<3}{v:<14}{x.mean():.5f} ± {x.std():.5f}")

# typical action magnitude for context
mags = np.array([np.linalg.norm(orig[f][:3]) for f in frames])
print(f"\n|t| of original predictions: mean {mags.mean():.5f} m, median {np.median(mags):.5f}")
summary_out = {
    "per_arm": summary,
    "ratios": ratios,
    "per_variant": pv,
    "orig_trans_magnitude_mean": float(mags.mean()),
    "n_frames": len(frames),
    "n_predictions": len(rows),
}
with open(f"{ROOT}/results/summary.json", "w") as f:
    json.dump(summary_out, f, indent=1)

# ------------------------------------------------------------------- plots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"C": "#999999", "P": "#1f77b4", "S": "#d62728", "G": "#2ca02c", "T": "#9467bd"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, metric, title, unit in (
    (axes[0], "trans_l2", "Translation deviation", "L2 (m)"),
    (axes[1], "rot_l2", "Rotation deviation", "L2 (rad)"),
    (axes[2], "grip_flip", "Gripper flip rate", "fraction"),
):
    if metric == "grip_flip":
        means = [summary[a]["grip_flip"]["mean"] for a in ARMS]
        cis = [summary[a]["grip_flip"]["ci95"] for a in ARMS]
        ax.bar(range(len(ARMS)), means, yerr=cis, capsize=4,
               color=[colors[a] for a in ARMS])
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=20, ha="right", fontsize=8)
    else:
        data = [[r[metric] for r in records if r["arm"] == a] for a in ARMS]
        vp = ax.violinplot(data, showmeans=False, showextrema=False)
        for body, a in zip(vp["bodies"], ARMS):
            body.set_facecolor(colors[a])
            body.set_alpha(0.5)
        bp = ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=True)
        for patch, a in zip(bp["boxes"], ARMS):
            patch.set_facecolor(colors[a])
        ax.set_xticks(range(1, len(ARMS) + 1))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=20, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("OpenVLA-7B action deviation vs original frame (22 BridgeData V2 frames)")
fig.tight_layout()
fig.savefig(f"{ROOT}/results/per_arm_deviation.png", dpi=150)

# per-frame paired scatter: P vs G and S vs G
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
g = per_frame_means("G", "trans_l2")
for ax, arm in zip(axes, ("P", "S")):
    a = per_frame_means(arm, "trans_l2")
    lim = max(a.max(), g.max()) * 1.1
    ax.scatter(g, a, c=colors[arm])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("G: geometry baseline trans L2 (m)")
    ax.set_ylabel(f"{ARM_LABEL[arm]} trans L2 (m)")
    ax.set_title(f"{arm} vs G per frame (above line = appearance > geometry)")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{ROOT}/results/paired_scatter.png", dpi=150)
print("\nplots saved")
