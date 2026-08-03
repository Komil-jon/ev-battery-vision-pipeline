"""Regenerate the ICMRA paper figures. Run from the repo root:
       python paper/make_figures.py
Numbers are the measured results recorded in CHANGELOG.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pathlib
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.grid":True,
                     "grid.alpha":.3,"grid.linestyle":":","axes.axisbelow":True})
OUT=pathlib.Path(__file__).parent/"figures"; OUT.mkdir(exist_ok=True)

fig,ax=plt.subplots(figsize=(3.4,2.2)); x=np.arange(2); w=0.35
ax.bar(x-w/2,[0.818,0.277],w,label="Single-facility specialist",color="#c0392b")
ax.bar(x+w/2,[0.541,0.502],w,label="Multi-source + DINOv2",color="#2471a3")
ax.set_xticks(x); ax.set_xticklabels(["In-domain\n(single facility)","Cross-facility\n(10 sources)"])
ax.set_ylabel("mAP@50"); ax.set_ylim(0,1.0)
for i,v in enumerate([0.818,0.277]): ax.text(i-w/2,v+.02,f"{v:.3f}",ha="center",fontsize=7)
for i,v in enumerate([0.541,0.502]): ax.text(i+w/2,v+.02,f"{v:.3f}",ha="center",fontsize=7)
ax.legend(fontsize=6.5,loc="upper right"); fig.tight_layout()
fig.savefig(OUT/"fig_gap.pdf"); fig.savefig(OUT/"fig_gap.png",dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(3.4,2.2))
src=["ue_rav4","bmw_i3","gqljq","edfw3","MTech"]; val=[0.995,0.910,0.873,0.749,0.043]
cols=["#2471a3"]*4+["#c0392b"]
ax.barh(src[::-1],val[::-1],color=cols[::-1]); ax.set_xlabel("module mAP@50"); ax.set_xlim(0,1.05)
for i,v in enumerate(val[::-1]): ax.text(v+.02,i,f"{v:.3f}",va="center",fontsize=7)
ax.text(0.42,0.15,"annotation-convention\noutlier",fontsize=6.5,color="#c0392b")
fig.tight_layout(); fig.savefig(OUT/"fig_persource.pdf"); fig.savefig(OUT/"fig_persource.png",dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(3.4,2.2)); x=np.arange(3); w=0.35
mod=[0.231,0.547,0.680]; bus=[0.0,0.304,0.353]
ax.bar(x-w/2,mod,w,label="module",color="#1e8449")
ax.bar(x+w/2,bus,w,label="busbar",color="#d68910")
ax.set_xticks(x); ax.set_xticklabels(["Specialist","YOLO11n\n(multi-source)","RF-DETR\n(DINOv2)"])
ax.set_ylabel("mAP@50 (cross-facility)"); ax.set_ylim(0,0.8)
for i,v in enumerate(mod): ax.text(i-w/2,v+.015,f"{v:.2f}",ha="center",fontsize=6.5)
for i,v in enumerate(bus):
    if v>0: ax.text(i+w/2,v+.015,f"{v:.2f}",ha="center",fontsize=6.5)
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(OUT/"fig_perclass.pdf"); fig.savefig(OUT/"fig_perclass.png",dpi=300); plt.close(fig)
print("figures ->",OUT)
