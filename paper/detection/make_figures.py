"""Figures for the detection paper. Run from the repo root:
       python paper/detection/make_figures.py
All values are measured results recorded in CHANGELOG.md and
MyDrive/evb/runs/summary.json."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pathlib
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": .3, "grid.linestyle": ":", "axes.axisbelow": True})
OUT = pathlib.Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
BLUE, RED, GREEN, ORANGE, GREY = "#2471a3", "#c0392b", "#1e8449", "#d68910", "#7f8c8d"

# ---- Fig. 1: per-source screening, model-based, with the robust fence ---------
SRC   = ["battery_comp", "other", "final_mobilenet", "edfw3", "bmw_i3",
         "gqljq", "automated", "mtech"]
YOLO  = [1.000, 0.999, 0.985, 0.982, 0.967, 0.941, 0.641, 0.097]
FENCE = 0.864
fig, ax = plt.subplots(figsize=(3.4, 2.25))
ax.barh(SRC[::-1], YOLO[::-1],
        color=[RED if v < FENCE else BLUE for v in YOLO][::-1])
ax.axvline(FENCE, color="k", ls="--", lw=.9)
ax.text(FENCE - 0.02, 0.1, f"fence {FENCE:.3f}", fontsize=5.5, ha="right")
for i, v in enumerate(YOLO[::-1]):
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=6)
ax.set_xlabel("module mAP@50, merged-corpus detector"); ax.set_xlim(0, 1.14)
ax.tick_params(axis="y", labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_screen.pdf"); plt.close(fig)

# ---- Fig. 2: positive control, both screens ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axes[0]
before = [1.000, 0.999, 0.985, 0.982, 0.967, 0.941, 0.641, 0.097]
after  = [1.000, 0.989, 0.985, 0.973, 0.950, 0.667, 0.591, 0.097]
x = np.arange(len(SRC)); w = 0.38
cols = [RED if s == "gqljq" else BLUE for s in SRC]
ax.bar(x - w/2, before, w, color=cols, alpha=.45, label="original corpus")
ax.bar(x + w/2, after,  w, color=cols, label="manipulated corpus")
ax.axhline(0.864, color="k", ls="--", lw=.8)
ax.axhline(0.815, color="k", ls=":",  lw=.8)
ax.set_xticks(x); ax.set_xticklabels(SRC, rotation=45, ha="right", fontsize=5.5)
ax.set_ylabel("module mAP@50"); ax.set_ylim(0, 1.15)
ax.legend(fontsize=5.5, loc="lower left")
ax.set_title("model-based screen", fontsize=7)

ax = axes[1]
gsrc = ["mtech", "automated", "other", "gqljq", "edfw3", "final_mobilenet",
        "battery_comp", "bmw_i3", "ue_rav4", "ybmvt"]
gbef = [40.7, 15.0, 11.2, 11.1, 9.9, 8.1, 5.8, 3.2, 2.8, 1.2]
gaft = [40.7, 15.0, 11.2, 185.8, 9.9, 8.1, 5.8, 3.2, 2.8, 1.2]
xg = np.arange(len(gsrc))
cg = [RED if s == "gqljq" else BLUE for s in gsrc]
ax.bar(xg - w/2, gbef, w, color=cg, alpha=.45)
ax.bar(xg + w/2, gaft, w, color=cg)
ax.axhline(29.02, color="k", ls="--", lw=.8)
ax.axhline(35.24, color="k", ls=":",  lw=.8)
ax.set_xticks(xg); ax.set_xticklabels(gsrc, rotation=45, ha="right", fontsize=5.5)
ax.set_ylabel("granularity index $g_i$"); ax.set_yscale("log")
ax.set_title("training-free screen", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_control.pdf"); plt.close(fig)

# ---- Fig. 3: architecture comparison under a matched budget ------------------
fig, ax = plt.subplots(figsize=(3.4, 2.15))
labels = ["consensus\nsources", "mild outlier\n(automated)", "divergent\n(mtech)"]
y_ = [0.979, 0.641, 0.097]
r_ = [0.955, 0.779, 0.337]
x = np.arange(3); w = 0.34
ax.bar(x - w/2, y_, w, label="YOLO11n", color=GREY)
ax.bar(x + w/2, r_, w, label="RF-DETR", color=BLUE)
for i, v in enumerate(y_): ax.text(i - w/2, v + .015, f"{v:.3f}", ha="center", fontsize=6)
for i, v in enumerate(r_): ax.text(i + w/2, v + .015, f"{v:.3f}", ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("module mAP@50"); ax.set_ylim(0, 1.14)
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_arch.pdf"); plt.close(fig)

# ---- Fig. 4: model-selection inversion, with bootstrap intervals -------------
fig, ax = plt.subplots(figsize=(3.4, 2.0))
pts = [0.552, 0.735]
lo  = [0.471, 0.659]
hi  = [0.626, 0.808]
names = ["val-selected\ncheckpoint", "fixed-budget\nfinal checkpoint"]
ax.errorbar(pts, [0, 1],
            xerr=[[p - l for p, l in zip(pts, lo)], [h - p for p, h in zip(pts, hi)]],
            fmt="o", color=RED, ecolor=RED, capsize=4, ms=6, lw=1.4)
ax.errorbar([pts[1]], [1], xerr=[[pts[1]-lo[1]], [hi[1]-pts[1]]],
            fmt="o", color=BLUE, ecolor=BLUE, capsize=4, ms=6, lw=1.4)
for p, l, h, y in zip(pts, lo, hi, [0, 1]):
    ax.text(p, y + 0.16, f"{p:.3f}  [{l:.3f}, {h:.3f}]", ha="center", fontsize=6)
ax.set_yticks([0, 1]); ax.set_yticklabels(names, fontsize=6.5)
ax.set_ylim(-0.5, 1.6); ax.set_xlim(0.40, 0.88)
ax.set_xlabel("module mAP@50, 95\\% bootstrap interval")
fig.tight_layout(); fig.savefig(OUT / "fig_inversion.pdf"); plt.close(fig)

print("figures ->", OUT)
