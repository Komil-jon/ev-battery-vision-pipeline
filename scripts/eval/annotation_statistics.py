"""Per-source annotation statistics computed from label files alone.

No images and no trained model are required. The purpose is to test whether the
annotation-convention divergence identified by the merged-corpus screening
procedure is also visible directly in the geometry of the annotations, which
would give a training-free screen.

Run from the repo root:
    python scripts/eval/annotation_statistics.py
"""
import pathlib, statistics as st, json, sys

ROOT = pathlib.Path("data/labels_release/detector")
CLASSES = {0: "module", 1: "busbar"}

# Filename prefix -> source. Derived from the merge provenance in docs/DATASETS.md.
def source_of(name: str) -> str:
    n = name.lower()
    for pre, src in (
        ("roboflow_ev-battery-component-detection-gqljq", "gqljq"),
        ("roboflow_ev-battery-components-edfw3", "edfw3"),
        ("ev-battery-sample-ybmvt", "ybmvt"),
        ("final_mobilenet_results", "final_mobilenet"),
        ("automated-disassembly", "automated"),
        ("ue_rav4_module", "ue_rav4"),
        ("battery_comp", "battery_comp"),
        ("last_exp", "last_exp4"),
        ("bmw_i", "bmw_i3"),
        ("aug_", "mtech"),
        ("tesla_model", "mtech"),
        ("chevrolet-bolt", "mtech"),
        ("nissan-leaf", "mtech"),
    ):
        if n.startswith(pre):
            return src
    return "other"

def boxes(path: pathlib.Path):
    """Yield (cls, w, h, area) in normalised units; polygons reduced to their
    enclosing axis-aligned box, which is what the detector consumes."""
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        c = int(float(f[0]))
        v = [float(x) for x in f[1:]]
        if len(v) == 4:                      # xc yc w h
            w, h = v[2], v[3]
        elif len(v) >= 6 and len(v) % 2 == 0:  # polygon
            xs, ys = v[0::2], v[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        if w > 0 and h > 0:
            yield c, w, h, w * h

def main():
    if not ROOT.exists():
        sys.exit(f"missing {ROOT}")
    per = {}
    for split in ("train", "val", "test"):
        d = ROOT / split
        if not d.exists():
            continue
        for p in d.glob("*.txt"):
            s = per.setdefault(source_of(p.name),
                               {"imgs": 0, "areas": [], "scales": [], "cls": [],
                                "per_img": [], "mod_scales": []})
            s["imgs"] += 1
            n = 0
            for c, w, h, a in boxes(p):
                s["areas"].append(a); s["scales"].append(a ** 0.5)
                s["cls"].append(c)
                if c == 0:
                    s["mod_scales"].append(a ** 0.5); n += 1
            s["per_img"].append(n)

    rows = []
    for name, s in per.items():
        if s["imgs"] < 30 or len(s["mod_scales"]) < 40:
            continue
        scale = st.median(s["mod_scales"])
        density = len(s["mod_scales"]) / s["imgs"]
        rows.append({
            "source": name,
            "images": s["imgs"],
            "modules": len(s["mod_scales"]),
            "mod_per_img": round(density, 2),
            "median_mod_scale": round(scale, 4),
            # granularity index: many small module boxes per image indicates that the
            # source annotates sub-components rather than pack-level modules
            "granularity": round(density / scale, 1),
        })
    rows.sort(key=lambda r: -r["granularity"])

    hdr = f'{"source":<16}{"imgs":>7}{"modules":>9}{"mod/img":>9}{"med scale":>11}{"granularity":>13}'
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f'{r["source"]:<16}{r["images"]:>7}{r["modules"]:>9}{r["mod_per_img"]:>9}'
              f'{r["median_mod_scale"]:>11}{r["granularity"]:>13}')

    g = [r["granularity"] for r in rows]
    med = st.median(g)
    mad = st.median([abs(x - med) for x in g])
    fence = med + 3 * 1.4826 * mad
    print(f"\ngranularity index = modules per image / median module scale")
    print(f"median {med:.2f}  MAD {mad:.2f}  upper fence (k=3) {fence:.2f}")
    flagged = [r["source"] for r in rows if r["granularity"] > fence]
    print("flagged by the training-free screen:", flagged or "none")

    pathlib.Path("outputs").mkdir(exist_ok=True)
    pathlib.Path("outputs/annotation_statistics.json").write_text(
        json.dumps({"rows": rows, "median": med, "mad": mad, "fence": fence,
                    "flagged": flagged}, indent=2))

if __name__ == "__main__":
    main()
