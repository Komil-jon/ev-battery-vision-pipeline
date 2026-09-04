"""Render the annotation-convention figure from real frames plus their labels.

Two panels: a consensus-convention frame, in which whole pack-level modules are
annotated, and a divergent-convention frame from the flagged source, in which
small internal sub-components carry the same class label. This is the visual
counterpart of the granularity index in the paper.

Images are not tracked in this repository (~4 GB). Point --image-dirs at any
directories holding them; filenames must match the label files.

    python paper/detection/render_convention_figure.py \
        --image-dirs ~/EV-battery-detection/data/detector
"""
import argparse, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from PIL import Image

LAB = pathlib.Path("data/labels_release/detector")
COLOR = {0: "#e74c3c", 1: "#f1c40f"}   # module, busbar
NAME = {0: "module", 1: "busbar"}

MTECH = ("aug_", "tesla_model", "chevrolet-bolt", "nissan-leaf")
CONSENSUS = ("roboflow_ev-battery-component-detection-gqljq",
             "roboflow_ev-battery-components-edfw3", "bmw_i", "ue_rav4_module",
             "battery_comp", "final_mobilenet_results", "ev-battery-sample-ybmvt")

def is_mtech(n): return any(n.lower().startswith(p) for p in MTECH)
def is_cons(n):  return any(n.lower().startswith(p) for p in CONSENSUS)

def parse(path):
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5: continue
        c = int(float(f[0])); v = [float(x) for x in f[1:]]
        if len(v) == 4:
            xc, yc, w, h = v; out.append((c, xc - w/2, yc - h/2, w, h))
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            out.append((c, min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)))
    return out

def draw(ax, img, boxes, title):
    ax.imshow(img); W, H = img.size
    for c, x, y, w, h in boxes:
        ax.add_patch(mp.Rectangle((x*W, y*H), w*W, h*H, fill=False,
                                  edgecolor=COLOR.get(c, "w"), lw=1.4))
    mods = [b for b in boxes if b[0] == 0]
    scale = st.median([(b[3]*b[4])**0.5 for b in mods]) if mods else 0
    ax.set_title(f"{title}\n{len(mods)} module boxes, median scale {scale:.3f}",
                 fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="paper/detection/figures/fig_convention_examples.pdf")
    a = ap.parse_args()

    imgs = {}
    for d in a.image_dirs:
        for p in pathlib.Path(d).expanduser().rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.setdefault(p.stem, p)

    labs = {}
    for sp in ("train", "val", "test"):
        if (LAB/sp).exists():
            for p in (LAB/sp).glob("*.txt"):
                labs[p.stem] = p

    avail = sorted(set(imgs) & set(labs))
    pick = lambda test: sorted(
        ((len([b for b in parse(labs[k]) if b[0] == 0]), k) for k in avail if test(k)),
        key=lambda t: -t[0])
    m, c = pick(is_mtech), pick(is_cons)
    print(f"{len(avail)} frames with both image and label "
          f"({len(m)} divergent-convention, {len(c)} consensus-convention)")
    if not m:
        raise SystemExit("no frames from the flagged source found in --image-dirs")

    panels = []
    if c: panels.append((c[0][1], "Consensus convention: pack-level modules"))
    for _, k in m[:2 if c else 3]:
        panels.append((k, "Divergent convention: internal sub-components"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0*len(panels), 2.7))
    if len(panels) == 1: axes = [axes]
    for ax, (k, t) in zip(axes, panels):
        draw(ax, Image.open(imgs[k]).convert("RGB"), parse(labs[k]), t)
    handles = [mp.Patch(color=COLOR[i], label=NAME[i]) for i in (0, 1)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("->", out, f"({len(panels)} panels)")
    if not c:
        print("NOTE: no consensus-convention frame available; rerun once those "
              "images are present to produce the full side-by-side.")

if __name__ == "__main__":
    main()
