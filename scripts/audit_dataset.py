"""
audit_dataset.py
================
Quality audit for a YOLO-format detection dataset. Flags the problems that
actually hurt training, and writes contact sheets so you can eyeball each class
of problem instead of trusting the numbers alone.

Checks:
  * unlabelled images -- separated into "probably true background" (nothing to
    detect: closed pack lid, a person, an empty bench) and "suspect
    under-labelled" (an image from a source that normally has labels). Under-
    labelled images are the harmful case: they teach the detector that visible
    modules are background.
  * blur (Laplacian variance). NOTE a low score also means "smooth surface", so
    always look at the contact sheet before deleting anything.
  * greyscale images, very dark / very blown-out images.
  * degenerate boxes (zero area, out of bounds) and per-source label density.

Nothing is deleted. The script writes a list of flagged files so you can review
them and decide.

Usage:
    python scripts/audit_dataset.py --images data/detector/images/train \
                                    --labels data/detector/labels/train
    python scripts/audit_dataset.py --images DIR --labels DIR --out outputs/audit
"""

import argparse
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

BLUR_T = 100.0     # Laplacian variance below this = smooth or out of focus
GRAY_T = 3.0       # mean channel difference below this = effectively greyscale
DARK_T = 50.0
BRIGHT_T = 200.0
IMG_EXT = (".jpg", ".jpeg", ".png")


def source_of(stem: str) -> str:
    return re.split(r"[_\-]", stem)[0][:14]


def read_labels(label_path: Path):
    """Return (n_boxes, n_degenerate). Handles both box and polygon lines."""
    if not label_path.exists():
        return 0, 0
    n = bad = 0
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            _, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:])
            if bw <= 0 or bh <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
                bad += 1
            n += 1
        elif len(p) > 5 and len(p) % 2 == 1:
            xs = list(map(float, p[1::2])); ys = list(map(float, p[2::2]))
            if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
                bad += 1
            n += 1
    return n, bad


def contact_sheet(paths, out_path, title, cols=5, thumb=(300, 220)):
    import cv2
    import numpy as np
    if not paths:
        return
    paths = paths[:25]
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * thumb[1] + 34, cols * thumb[0], 3), 25, np.uint8)
    cv2.putText(sheet, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, thumb)
        cv2.putText(im, p.name[:34], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        r, c = divmod(i, cols)
        sheet[34 + r * thumb[1]:34 + (r + 1) * thumb[1], c * thumb[0]:(c + 1) * thumb[0]] = im
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Audit a YOLO detection dataset for quality problems")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="outputs/audit", help="Where to write sheets and the flag list")
    ap.add_argument("--no-sheets", action="store_true", help="Numbers only, skip contact sheets")
    args = ap.parse_args()

    import cv2

    img_dir, lab_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"No images in {img_dir}")
    print(f"Auditing {len(imgs)} images in {img_dir}\n")

    rows = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            print(f"  UNREADABLE: {p.name}")
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(g, cv2.CV_64F).var()
        b, gr, r = cv2.split(im.astype(int))
        sat = (abs(b - gr).mean() + abs(gr - r).mean() + abs(b - r).mean()) / 3
        n, degen = read_labels(lab_dir / (p.stem + ".txt"))
        rows.append(dict(path=p, src=source_of(p.stem), blur=blur, sat=sat,
                         mean=g.mean(), n=n, degen=degen))

    total = len(rows)
    unlabelled = [r for r in rows if r["n"] == 0]
    blurry = [r for r in rows if r["blur"] < BLUR_T]
    gray = [r for r in rows if r["sat"] < GRAY_T]
    dark = [r for r in rows if r["mean"] < DARK_T]
    bright = [r for r in rows if r["mean"] > BRIGHT_T]
    degen = [r for r in rows if r["degen"] > 0]

    # An unlabelled image is only a statistical anomaly when nearly every other
    # image from the same source IS labelled. Sources with a large unlabelled
    # share are ambiguous (they may legitimately contain many background frames)
    # and can only be resolved by looking at the contact sheet.
    labelled_share = defaultdict(lambda: [0, 0])
    for r in rows:
        labelled_share[r["src"]][0] += 1
        if r["n"] > 0:
            labelled_share[r["src"]][1] += 1
    ANOMALY_T = 0.9
    suspect, ambiguous = [], []
    for r in unlabelled:
        tot, lab = labelled_share[r["src"]]
        (suspect if lab / max(tot, 1) >= ANOMALY_T else ambiguous).append(r)

    print("=" * 62)
    print(f"{'images':<34}{total:>8}")
    print(f"{'unlabelled':<34}{len(unlabelled):>8}  {100*len(unlabelled)/total:>5.1f}%")
    print(f"{'  ANOMALY (source is >=90% labelled)':<34}{len(suspect):>8}  {100*len(suspect)/total:>5.1f}%")
    print(f"{'  ambiguous (needs visual review)':<34}{len(ambiguous):>8}  {100*len(ambiguous)/total:>5.1f}%")
    print(f"{'low-detail / blurry':<34}{len(blurry):>8}  {100*len(blurry)/total:>5.1f}%")
    print(f"{'greyscale':<34}{len(gray):>8}  {100*len(gray)/total:>5.1f}%")
    print(f"{'very dark':<34}{len(dark):>8}")
    print(f"{'very bright':<34}{len(bright):>8}")
    print(f"{'images with degenerate boxes':<34}{len(degen):>8}")
    print("=" * 62)

    counts = [r["n"] for r in rows if r["n"]]
    if counts:
        print(f"labels per labelled image: median {statistics.median(counts):.0f}, max {max(counts)}\n")

    print(f"{'source':<16}{'imgs':>6}{'unlab':>7}{'%':>6}{'blurry':>8}{'grey':>6}{'med lab':>9}")
    per_src = defaultdict(list)
    for r in rows:
        per_src[r["src"]].append(r)
    for src, rs in sorted(per_src.items(), key=lambda kv: -len(kv[1])):
        u = sum(1 for r in rs if r["n"] == 0)
        bl = sum(1 for r in rs if r["blur"] < BLUR_T)
        gy = sum(1 for r in rs if r["sat"] < GRAY_T)
        med = statistics.median([r["n"] for r in rs if r["n"]]) if any(r["n"] for r in rs) else 0
        print(f"{src:<16}{len(rs):>6}{u:>7}{100*u/len(rs):>5.0f}%{bl:>8}{gy:>6}{med:>9.0f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    flag_file = out_dir / "flagged.txt"
    with open(flag_file, "w") as f:
        f.write("# Files flagged by audit_dataset.py -- REVIEW before removing anything.\n")
        f.write("# 'suspect_underlabelled' is the harmful class: visible objects, no labels.\n\n")
        for tag, group in [("anomaly_unlabelled", suspect), ("ambiguous_unlabelled", ambiguous),
                           ("blurry", blurry), ("greyscale", gray),
                           ("very_dark", dark), ("degenerate_boxes", degen)]:
            for r in group:
                f.write(f"{tag}\t{r['path']}\n")
    print(f"\nFlag list -> {flag_file}")

    if not args.no_sheets:
        print("Contact sheets:")
        contact_sheet([r["path"] for r in sorted(suspect, key=lambda r: -r["blur"])],
                      out_dir / "anomaly_unlabelled.jpg",
                      "ANOMALY: unlabelled, but this source is >=90% labelled")
        contact_sheet([r["path"] for r in sorted(ambiguous, key=lambda r: -r["blur"])],
                      out_dir / "ambiguous_unlabelled.jpg",
                      "AMBIGUOUS unlabelled: true background, or under-labelled? (review)")
        contact_sheet([r["path"] for r in sorted(blurry, key=lambda r: r["blur"])],
                      out_dir / "blurry.jpg", "Lowest-detail images (smooth OR out of focus)")
        contact_sheet([r["path"] for r in gray], out_dir / "greyscale.jpg", "Greyscale images")
        contact_sheet([r["path"] for r in dark], out_dir / "dark.jpg", "Very dark images")

    print("\nHow to read this:")
    print("  * Greyscale and blur are NOT automatically bad -- they add robustness, and a low")
    print("    blur score often just means a smooth metal surface. Keep them unless the")
    print("    contact sheet shows the objects are genuinely impossible to see.")
    print("  * True background images (closed pack, a person, empty bench) are USEFUL; ~10%")
    print("    is a healthy amount and reduces false positives.")
    print("  * Unlabelled images with VISIBLE modules/busbars are the harmful case: they")
    print("    teach the detector those objects are background. No heuristic can tell that")
    print("    apart from a genuine background frame, so open the two unlabelled contact")
    print("    sheets and decide by eye, then relabel or drop what is actually wrong.")


if __name__ == "__main__":
    main()
