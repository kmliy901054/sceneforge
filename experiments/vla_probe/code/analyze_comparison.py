"""Comparison arm, step 3: merged OpenVLA-7B vs SpatialVLA-4B analysis.

Same per-(frame, arm, variant) metrics as analyze.py, computed independently
inside each model (deviation w.r.t. the SAME model's prediction on the same
frame's original image). The KEY comparison is within-model ratios across
models: per-frame S/G and P/G translation-L2 ratios, paired by frame, Wilcoxon
signed-rank on the per-frame ratio differences (OpenVLA - SpatialVLA).
Falsifiable prediction: the 3D-aware model (SpatialVLA) shows LOWER S/G and
P/G (better appearance disentanglement).
"""
import json

import numpy as np
from scipy import stats

ROOT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe"

MODELS = {
    "openvla": f"{ROOT}/results/actions.json",
    "spatialvla": f"{ROOT}/results/actions_spatialvla.json",
}
MODEL_LABEL = {"openvla": "OpenVLA-7B", "spatialvla": "SpatialVLA-4B"}
ARMS = ["C", "P", "S", "G", "T"]
ARM_LABEL = {
    "C": "C: jpeg95 floor",
    "P": "P: photometric",
    "S": "S: bg restyle",
    "G": "G: geometry",
    "T": "T: temporal (t+3)",
}
METRICS = ["trans_l2", "trans_cos", "rot_l2", "grip_flip"]


def build_records(rows):
    orig = {r["frame"]: np.array(r["action"]) for r in rows if r["arm"] == "orig"}
    records = []
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
            )
        )
    return orig, records


def per_frame_means(records, frames, arm, metric):
    out = []
    for fr in frames:
        v = [r[metric] for r in records if r["arm"] == arm and r["frame"] == fr]
        out.append(np.mean(v))
    return np.array(out)


def ci95(x):
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


data, recs, frames = {}, {}, None
for mdl, path in MODELS.items():
    with open(path) as f:
        rows = json.load(f)
    orig, records = build_records(rows)
    data[mdl] = orig
    recs[mdl] = records
    f_set = sorted(orig)
    assert frames is None or f_set == frames, "frame sets differ between models!"
    frames = f_set
print(f"{len(frames)} frames per model\n")

if "spatialvla" in recs:
    with open(f"{ROOT}/results/deviations_spatialvla.json", "w") as f:
        json.dump(recs["spatialvla"], f, indent=1)

# ----------------------------------------------------------- per-arm tables
summary = {m: {} for m in MODELS}
for mdl in MODELS:
    print(f"=== {MODEL_LABEL[mdl]} ===")
    print(f"{'arm':<20}{'trans L2 (m)':<24}{'trans cos':<12}{'rot L2 (rad)':<24}{'grip flip'}")
    for arm in ARMS:
        s = {}
        for metric in METRICS:
            x = per_frame_means(recs[mdl], frames, arm, metric)
            s[metric] = {"mean": float(x.mean()), "ci95": ci95(x), "n_frames": len(x)}
        summary[mdl][arm] = s
        print(
            f"{ARM_LABEL[arm]:<20}"
            f"{s['trans_l2']['mean']:.5f} ±{s['trans_l2']['ci95']:.5f}     "
            f"{s['trans_cos']['mean']:.3f}     "
            f"{s['rot_l2']['mean']:.5f} ±{s['rot_l2']['ci95']:.5f}     "
            f"{s['grip_flip']['mean']*100:.1f}%"
        )
    mags = np.array([np.linalg.norm(data[mdl][f][:3]) for f in frames])
    summary[mdl]["orig_trans_magnitude"] = {
        "mean": float(mags.mean()), "median": float(np.median(mags))
    }
    print(f"|t| of original predictions: mean {mags.mean():.5f} m, "
          f"median {np.median(mags):.5f} m\n")

# ------------------------------------- within-model ratios, paired per frame
# NOTE on zeros: coarse action discretization can map a variant to the
# bitwise-identical action, so a per-frame G mean can be exactly 0 (3/22
# frames for SpatialVLA). Per-frame ratios are only defined where G > 0;
# cross-model paired tests therefore use (a) ratios on the subset of frames
# where BOTH models have G > 0 and (b) a zero-safe symmetric index
# (A - G)/(A + G) in [-1, 1] on all frames (negative = appearance < geometry).
print("=== Key within-model ratios (translation L2, per-frame paired) ===")
ratio_cmp = {}
pf = {}  # pf[mdl][arm] = per-frame mean trans_l2 array
for mdl in MODELS:
    pf[mdl] = {arm: per_frame_means(recs[mdl], frames, arm, "trans_l2")
               for arm in ("P", "S", "G", "T")}
    g, t = pf[mdl]["G"], pf[mdl]["T"]
    ratio_cmp[mdl] = {}
    for arm in ("P", "S"):
        a = pf[mdl][arm]
        valid = g > 0
        w = stats.wilcoxon(a, g)
        ratio_cmp[mdl][arm] = {
            "agg_ratio_vs_G": float(a.mean() / g.mean()),
            "median_per_frame_ratio": float(np.median(a[valid] / g[valid])),
            "n_frames_ratio_defined": int(valid.sum()),
            "agg_ratio_vs_T": float(a.mean() / t.mean()),
            "frac_frames_gt_G": float((a > g).mean()),
            "wilcoxon_arm_vs_G_p": float(w.pvalue),
        }
        r = ratio_cmp[mdl][arm]
        print(
            f"{MODEL_LABEL[mdl]:<14}{arm}/G = {r['agg_ratio_vs_G']:.2f} "
            f"(median per-frame {r['median_per_frame_ratio']:.2f}, "
            f"n={r['n_frames_ratio_defined']})   "
            f"{arm}/T = {r['agg_ratio_vs_T']:.2f}   frames {arm}>G: "
            f"{r['frac_frames_gt_G']*100:.0f}%   wilcoxon p={r['wilcoxon_arm_vs_G_p']:.4f}"
        )

# ------------------------- cross-model test on per-frame ratio differences
print("\n=== Cross-model: per-frame paired differences (OpenVLA - SpatialVLA) ===")
cross = {}
for arm in ("P", "S"):
    a_ov, g_ov = pf["openvla"][arm], pf["openvla"]["G"]
    a_sv, g_sv = pf["spatialvla"][arm], pf["spatialvla"]["G"]
    # (a) ratios where both models' G > 0
    m = (g_ov > 0) & (g_sv > 0)
    r_ov, r_sv = a_ov[m] / g_ov[m], a_sv[m] / g_sv[m]
    d = r_ov - r_sv
    try:
        p_ratio = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_ratio = float("nan")
    # (b) zero-safe symmetric index on all frames with A+G > 0 in both models
    m2 = ((a_ov + g_ov) > 0) & ((a_sv + g_sv) > 0)
    i_ov = (a_ov[m2] - g_ov[m2]) / (a_ov[m2] + g_ov[m2])
    i_sv = (a_sv[m2] - g_sv[m2]) / (a_sv[m2] + g_sv[m2])
    d2 = i_ov - i_sv
    try:
        p_idx = float(stats.wilcoxon(d2).pvalue)
    except ValueError:
        p_idx = float("nan")
    cross[arm] = {
        "ratio": {
            "n_frames": int(m.sum()),
            "openvla_mean": float(r_ov.mean()),
            "spatialvla_mean": float(r_sv.mean()),
            "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)),
            "frac_frames_spatialvla_lower": float((d > 0).mean()),
            "wilcoxon_p": p_ratio,
        },
        "symmetric_index": {
            "n_frames": int(m2.sum()),
            "openvla_mean": float(i_ov.mean()),
            "spatialvla_mean": float(i_sv.mean()),
            "mean_diff": float(d2.mean()),
            "frac_frames_spatialvla_lower": float((d2 > 0).mean()),
            "wilcoxon_p": p_idx,
        },
    }
    c = cross[arm]
    print(
        f"{arm}/G ratio (n={c['ratio']['n_frames']}):  "
        f"mean diff {c['ratio']['mean_diff']:+.3f}  "
        f"median {c['ratio']['median_diff']:+.3f}  "
        f"SpatialVLA lower in {c['ratio']['frac_frames_spatialvla_lower']*100:.0f}% of frames  "
        f"wilcoxon p={c['ratio']['wilcoxon_p']:.4f}"
    )
    print(
        f"{arm} idx (A-G)/(A+G) (n={c['symmetric_index']['n_frames']}):  "
        f"OV {c['symmetric_index']['openvla_mean']:+.3f}  "
        f"SV {c['symmetric_index']['spatialvla_mean']:+.3f}  "
        f"SpatialVLA lower in {c['symmetric_index']['frac_frames_spatialvla_lower']*100:.0f}%  "
        f"wilcoxon p={c['symmetric_index']['wilcoxon_p']:.4f}"
    )

# cross-model gripper-flip comparison (paired per frame, per arm)
print("\n=== Cross-model gripper flip rates ===")
grip = {}
for arm in ARMS:
    a = per_frame_means(recs["openvla"], frames, arm, "grip_flip")
    b = per_frame_means(recs["spatialvla"], frames, arm, "grip_flip")
    d = a - b
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
    except ValueError:
        p = float("nan")
    grip[arm] = {"openvla": float(a.mean()), "spatialvla": float(b.mean()),
                 "wilcoxon_p": p}
    print(f"{ARM_LABEL[arm]:<20} openvla {a.mean()*100:5.1f}%   "
          f"spatialvla {b.mean()*100:5.1f}%   wilcoxon p={p:.4f}")

# absolute cross-model deviation difference per arm (translation L2)
print("\n=== Cross-model absolute trans L2 (paired per frame) ===")
abs_cmp = {}
for arm in ARMS:
    a = per_frame_means(recs["openvla"], frames, arm, "trans_l2")
    b = per_frame_means(recs["spatialvla"], frames, arm, "trans_l2")
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    abs_cmp[arm] = {"openvla_mm": float(a.mean() * 1000),
                    "spatialvla_mm": float(b.mean() * 1000), "wilcoxon_p": p}
    print(f"{ARM_LABEL[arm]:<20} openvla {a.mean()*1000:6.2f} mm   "
          f"spatialvla {b.mean()*1000:6.2f} mm   wilcoxon p={p:.4f}")

with open(f"{ROOT}/results/summary_comparison.json", "w") as f:
    json.dump(
        {
            "per_arm": summary,
            "within_model_ratios": ratio_cmp,
            "cross_model_ratio_diffs": cross,
            "cross_model_gripper": grip,
            "cross_model_abs_trans_l2": abs_cmp,
            "n_frames": len(frames),
        },
        f,
        indent=1,
    )

# ------------------------------------------------------------------- plots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MCOLOR = {"openvla": "#1f77b4", "spatialvla": "#d62728"}

# grouped violin/box: trans L2, rot L2, gripper flip — two models side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, metric, title, unit in (
    (axes[0], "trans_l2", "Translation deviation", "L2 (m)"),
    (axes[1], "rot_l2", "Rotation deviation", "L2 (rad)"),
    (axes[2], "grip_flip", "Gripper flip rate", "fraction"),
):
    if metric == "grip_flip":
        w = 0.38
        for j, mdl in enumerate(MODELS):
            means = [summary[mdl][a]["grip_flip"]["mean"] for a in ARMS]
            cis = [summary[mdl][a]["grip_flip"]["ci95"] for a in ARMS]
            ax.bar(np.arange(len(ARMS)) + (j - 0.5) * w, means, w, yerr=cis,
                   capsize=3, color=MCOLOR[mdl], label=MODEL_LABEL[mdl])
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=20,
                           ha="right", fontsize=8)
    else:
        for j, mdl in enumerate(MODELS):
            pos = np.arange(1, len(ARMS) + 1) + (j - 0.5) * 0.36
            dat = [[r[metric] for r in recs[mdl] if r["arm"] == a] for a in ARMS]
            vp = ax.violinplot(dat, positions=pos, widths=0.34,
                               showmeans=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(MCOLOR[mdl])
                body.set_alpha(0.45)
            bp = ax.boxplot(dat, positions=pos, widths=0.12, showfliers=False,
                            patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(MCOLOR[mdl])
        ax.set_xticks(range(1, len(ARMS) + 1))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=20,
                           ha="right", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], alpha=0.6)
                   for m in MODELS]
        ax.legend(handles, [MODEL_LABEL[m] for m in MODELS], fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("OpenVLA-7B vs SpatialVLA-4B — action deviation per arm "
             "(22 BridgeData V2 frames, within-model deviations)")
fig.tight_layout()
fig.savefig(f"{ROOT}/results/per_arm_deviation_comparison.png", dpi=150)

# per-frame paired ratio scatter: OpenVLA vs SpatialVLA (S/G and P/G),
# frames where both models' G > 0 (ratio defined)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
for ax, arm, c in zip(axes, ("S", "P"), ("#d62728", "#1f77b4")):
    m = (pf["openvla"]["G"] > 0) & (pf["spatialvla"]["G"] > 0)
    x = pf["openvla"][arm][m] / pf["openvla"]["G"][m]
    y = pf["spatialvla"][arm][m] / pf["spatialvla"]["G"][m]
    lim = max(x.max(), y.max()) * 1.08
    ax.scatter(x, y, c=c, alpha=0.8)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.axvline(1, color="gray", lw=0.6, ls=":")
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(f"OpenVLA per-frame {arm}/G ratio")
    ax.set_ylabel(f"SpatialVLA per-frame {arm}/G ratio")
    ax.set_title(f"{arm}/G per frame, n={m.sum()} with G>0 in both\n"
                 f"(below line = SpatialVLA more appearance-robust)")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{ROOT}/results/paired_ratio_scatter.png", dpi=150)
print("\nplots saved")
