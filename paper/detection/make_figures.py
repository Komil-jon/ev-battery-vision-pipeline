"""Figures for the detection-only paper. Run from repo root:
       python paper/detection/make_figures.py
All values are the measured results recorded in CHANGELOG.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pathlib
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": .3, "grid.linestyle": ":", "axes.axisbelow": True})
OUT = pathlib.Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
BLUE, RED, GREEN, ORANGE, GREY = "#2471a3", "#c0392b", "#1e8449", "#d68910", "#7f8c8d"

# ---- Fig. 1: the single-facility evaluation gap -------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.15)); x = np.arange(2); w = 0.26
spec, yolo, rfd = [0.818, 0.277], [0.195, 0.410], [0.541, 0.502]
ax.bar(x - w, spec, w, label="Single-source specialist", color=RED)
ax.bar(x,     yolo, w, label="Multi-source YOLO11n",     color=GREY)
ax.bar(x + w, rfd,  w, label="Multi-source RF-DETR",     color=BLUE)
for off, vals in ((-w, spec), (0, yolo), (w, rfd)):
    for i, v in enumerate(vals): ax.text(i + off, v + .015, f"{v:.3f}", ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(["In-domain\n(single source)", "Cross-facility\n(13 sources)"])
ax.set_ylabel("mAP@50"); ax.set_ylim(0, 1.0); ax.legend(fontsize=6, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_gap.pdf"); plt.close(fig)

# ---- Fig. 2: per-source screening with the MAD criterion ---------------------
src = ["ue_rav4", "bmw_i3", "gqljq", "edfw3", "MTech"]
val = [0.995, 0.910, 0.873, 0.749, 0.043]
med = float(np.median(val)); mad = float(np.median([abs(v - med) for v in val]))
thr = med - 3 * 1.4826 * mad
fig, ax = plt.subplots(figsize=(3.4, 2.15))
cols = [RED if v < thr else BLUE for v in val]
ax.barh(src[::-1], val[::-1], color=cols[::-1])
ax.axvline(thr, color="k", ls="--", lw=.9)
ax.text(thr + .015, 3.55, f"screen threshold\n{thr:.3f}", fontsize=6, va="center")
for i, v in enumerate(val[::-1]): ax.text(v + .015, i, f"{v:.3f}", va="center", fontsize=6.5)
ax.set_xlabel("module mAP@50 (merged-corpus detector)"); ax.set_xlim(0, 1.14)
fig.tight_layout(); fig.savefig(OUT / "fig_screen.pdf"); plt.close(fig)

# ---- Fig. 3: per-class scores, full vs screened benchmark --------------------
fig, ax = plt.subplots(figsize=(3.4, 2.15)); x = np.arange(2); w = 0.19
full_m, scr_m = [0.547, 0.680], [0.774, 0.771]
full_b, scr_b = [0.304, 0.353], [0.344, 0.366]
ax.bar(x - 1.5*w, full_m, w, label="module, full", color=GREEN)
ax.bar(x - 0.5*w, scr_m,  w, label="module, screened", color=GREEN, alpha=.45)
ax.bar(x + 0.5*w, full_b, w, label="busbar, full", color=ORANGE)
ax.bar(x + 1.5*w, scr_b,  w, label="busbar, screened", color=ORANGE, alpha=.45)
for off, vals in ((-1.5*w, full_m), (-0.5*w, scr_m), (0.5*w, full_b), (1.5*w, scr_b)):
    for i, v in enumerate(vals): ax.text(i + off, v + .012, f"{v:.2f}", ha="center", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(["YOLO11n", "RF-DETR (DINOv2)"])
ax.set_ylabel("mAP@50"); ax.set_ylim(0, 0.95); ax.legend(fontsize=5.5, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_perclass.pdf"); plt.close(fig)

# ---- Fig. 4: zero-shot detection rate on unseen pack families ---------------
names = ["BMW i4", "Hyundai Ioniq", "Ford Mondeo", "Mercedes GLE", "Volvo truck"]
rate  = [1.00, 1.00, 0.95, 0.40, 0.27]
fig, ax = plt.subplots(figsize=(3.4, 1.95))
ax.barh(names[::-1], rate[::-1], color=[RED if r < .5 else BLUE for r in rate][::-1])
ax.axvline(0.76, color="k", ls="--", lw=.9)
ax.text(0.77, 0.15, "mean 0.76", fontsize=6)
for i, v in enumerate(rate[::-1]): ax.text(v + .015, i, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_xlabel("detection rate, 17 unseen pack families"); ax.set_xlim(0, 1.14)
fig.tight_layout(); fig.savefig(OUT / "fig_zeroshot.pdf"); plt.close(fig)
print("threshold =", round(thr, 4), "| figures ->", OUT)

# ---- Fig. 5: training-free convention screen from annotation geometry --------
# Values produced by scripts/eval/annotation_statistics.py over the 4,760 label files.
src   = ["mtech","automated","gqljq","edfw3","final_mobilenet","battery_comp",
         "bmw_i3","ue_rav4","ybmvt"]
scale = [0.1423, 0.2800, 0.1874, 0.1692, 0.3101, 0.2092, 0.4191, 0.3513, 0.5437]
dens  = [5.78,   4.20,   2.09,   1.67,   2.50,   1.20,   1.32,   1.00,   0.68]
gran  = [d/s for d, s in zip(dens, scale)]
FENCE = 29.02

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
for n, s, d in zip(src, scale, dens):
    c = RED if n == "mtech" else BLUE
    ax.scatter(s, d, s=26, color=c, zorder=3)
    ax.annotate(n, (s, d), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
xs = np.linspace(0.10, 0.60, 100)
ax.plot(xs, FENCE * xs, "k--", lw=.9, zorder=2)
ax.text(0.44, FENCE * 0.46, "screen fence", fontsize=5.5, rotation=18)
ax.set_xlabel("median module scale (normalised)"); ax.set_ylabel("modules per image")
ax.set_xlim(0.08, 0.60); ax.set_ylim(0, 6.6)

ax = axes[1]
order = np.argsort(gran)[::-1]
names = [src[i] for i in order]; vals = [gran[i] for i in order]
ax.barh(names[::-1], vals[::-1],
        color=[RED if n == "mtech" else BLUE for n in names][::-1])
ax.axvline(FENCE, color="k", ls="--", lw=.9)
ax.text(FENCE + 0.6, 0.2, f"fence {FENCE:.1f}", fontsize=5.5)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=5.5)
ax.set_xlabel("granularity index"); ax.set_xlim(0, 48)
ax.tick_params(axis="y", labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_convention.pdf"); plt.close(fig)
print("convention figure ->", OUT / "fig_convention.pdf")

# ---- Fig. 5b: compact single-panel variant for the 2-column IEEE layout ------
fig, ax = plt.subplots(figsize=(3.4, 1.85))
order = np.argsort(gran)[::-1]
names = [src[i] for i in order]; vals = [gran[i] for i in order]
ax.barh(names[::-1], vals[::-1],
        color=[RED if n == "mtech" else BLUE for n in names][::-1])
ax.axvline(FENCE, color="k", ls="--", lw=.9)
ax.text(FENCE + 0.7, 0.3, f"fence {FENCE:.1f}", fontsize=6)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.7, i, f"{v:.1f}", va="center", fontsize=6)
ax.set_xlabel("granularity index $g_i = n_i / s_i$"); ax.set_xlim(0, 50)
ax.tick_params(axis="y", labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_convention_ieee.pdf"); plt.close(fig)
print("compact convention figure ->", OUT / "fig_convention_ieee.pdf")
