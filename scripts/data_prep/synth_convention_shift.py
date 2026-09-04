"""Synthetic convention shift: a positive control for the screening procedures.

A consensus source is relabelled to a deliberately different annotation
convention, in which each pack-level module box is replaced by a grid of
sub-component boxes covering the same region. This imitates the divergence
observed in the real outlier source without changing a single image, so any
screen that flags the manipulated source is responding to annotation convention
and to nothing else.

Two uses:
  1. Training-free control (runs here, seconds): recompute the granularity index
     over the manipulated corpus and check that the screen flags the manipulated
     source and only it.
  2. Model-based control (needs a GPU retrain): train the detector on the
     manipulated corpus and rerun the per-source screen.

Run from the repo root:
    python scripts/data_prep/synth_convention_shift.py --source gqljq --grid 3 2
"""
import argparse, pathlib, shutil, statistics as st, sys

SRC_ROOT = pathlib.Path("data/labels_release/detector")
MODULE = 0

PREFIX = {
    "gqljq": "roboflow_ev-battery-component-detection-gqljq",
    "edfw3": "roboflow_ev-battery-components-edfw3",
    "ybmvt": "ev-battery-sample-ybmvt",
    "final_mobilenet": "final_mobilenet_results",
    "automated": "automated-disassembly",
    "ue_rav4": "ue_rav4_module",
    "battery_comp": "battery_comp",
    "last_exp4": "last_exp",
    "bmw_i3": "bmw_i",
}
MTECH = ("aug_", "tesla_model", "chevrolet-bolt", "nissan-leaf")


def source_of(name: str) -> str:
    n = name.lower()
    for src, pre in PREFIX.items():
        if n.startswith(pre):
            return src
    if any(n.startswith(p) for p in MTECH):
        return "mtech"
    return "other"


def parse(line: str):
    """-> (cls, x0, y0, x1, y1) in normalised corner form, or None."""
    f = line.split()
    if len(f) < 5:
        return None
    c = int(float(f[0])); v = [float(x) for x in f[1:]]
    if len(v) == 4:
        xc, yc, w, h = v
        return c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    if len(v) >= 6 and len(v) % 2 == 0:
        xs, ys = v[0::2], v[1::2]
        return c, min(xs), min(ys), max(xs), max(ys)
    return None


def subdivide(box, gx, gy, inset=0.06):
    """Replace one module box with a gx x gy grid of sub-boxes, each inset
    slightly so adjacent cells do not touch, imitating sub-component labelling."""
    _, x0, y0, x1, y1 = box
    w, h = (x1 - x0) / gx, (y1 - y0) / gy
    out = []
    for i in range(gx):
        for j in range(gy):
            a, b = x0 + i * w, y0 + j * h
            ix, iy = w * inset, h * inset
            cx, cy = a + w / 2, b + h / 2
            sw, sh = max(w - 2 * ix, 1e-4), max(h - 2 * iy, 1e-4)
            out.append((MODULE, cx, cy, sw, sh))
    return out


def granularity_table(root: pathlib.Path):
    per = {}
    for split in ("train", "val", "test"):
        d = root / split
        if not d.exists():
            continue
        for p in d.glob("*.txt"):
            s = per.setdefault(source_of(p.name), {"imgs": 0, "scales": []})
            s["imgs"] += 1
            for line in p.read_text().splitlines():
                b = parse(line)
                if b and b[0] == MODULE:
                    s["scales"].append(((b[3] - b[1]) * (b[4] - b[2])) ** 0.5)
    rows = []
    for name, s in per.items():
        if s["imgs"] < 30 or len(s["scales"]) < 40:
            continue
        scale = st.median(s["scales"]); dens = len(s["scales"]) / s["imgs"]
        rows.append((name, s["imgs"], len(s["scales"]), round(dens, 2),
                     round(scale, 4), round(dens / scale, 1)))
    rows.sort(key=lambda r: -r[5])
    g = [r[5] for r in rows]
    med = st.median(g); mad = st.median([abs(x - med) for x in g])
    return rows, med, mad, med + 3 * 1.4826 * mad


def report(title, rows, med, mad, fence):
    print(f"\n{title}")
    print(f'{"source":<18}{"imgs":>7}{"modules":>9}{"mod/img":>9}{"scale":>9}{"granularity":>13}')
    for r in rows:
        mark = "  <-- FLAGGED" if r[5] > fence else ""
        print(f'{r[0]:<18}{r[1]:>7}{r[2]:>9}{r[3]:>9}{r[4]:>9}{r[5]:>13}{mark}')
    print(f'median {med:.2f}  MAD {mad:.2f}  fence(k=3) {fence:.2f}')
    return [r[0] for r in rows if r[5] > fence]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gqljq", choices=sorted(PREFIX))
    ap.add_argument("--grid", nargs=2, type=int, default=[3, 2], metavar=("GX", "GY"))
    ap.add_argument("--out", default="data/detector_convshift")
    args = ap.parse_args()
    if not SRC_ROOT.exists():
        sys.exit(f"missing {SRC_ROOT}")

    out = pathlib.Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    gx, gy = args.grid
    touched = added = 0
    for split in ("train", "val", "test"):
        d = SRC_ROOT / split
        if not d.exists():
            continue
        (out / split).mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.txt"):
            if source_of(p.name) != args.source:
                shutil.copy(p, out / split / p.name)
                continue
            lines = []
            for line in p.read_text().splitlines():
                b = parse(line)
                if b is None:
                    continue
                if b[0] != MODULE:
                    _, x0, y0, x1, y1 = b
                    lines.append(f"{b[0]} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} "
                                 f"{x1-x0:.6f} {y1-y0:.6f}")
                    continue
                for c, xc, yc, w, h in subdivide(b, gx, gy):
                    lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    added += 1
            (out / split / p.name).write_text("\n".join(lines) + "\n")
            touched += 1

    print(f"manipulated source '{args.source}' with a {gx}x{gy} grid: "
          f"{touched} files rewritten, {added} sub-component boxes emitted")
    print(f"corpus written to {out}")

    r0, m0, d0, f0 = granularity_table(SRC_ROOT)
    flagged0 = report("BEFORE (original corpus)", r0, m0, d0, f0)
    r1, m1, d1, f1 = granularity_table(out)
    flagged1 = report(f"AFTER ('{args.source}' relabelled to a sub-component convention)",
                      r1, m1, d1, f1)
    print(f"\nflagged before: {flagged0}\nflagged after : {flagged1}")
    print("positive control PASSED" if args.source in flagged1 else
          "positive control FAILED: the manipulated source was not flagged")


if __name__ == "__main__":
    main()
