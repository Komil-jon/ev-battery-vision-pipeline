"""
boxes_to_masks_sam.py
=====================
Upgrade box-only YOLO labels to polygon (segmentation) labels using SAM 2, so the
whole dataset can train an instance-segmentation model. Each existing box is fed to
SAM as a prompt; SAM returns a mask; the mask contour is written back as a YOLO
polygon line (`cls x1 y1 x2 y2 ...`, normalised).

**These are proposals, not ground truth.** SAM cannot fix a bad box -- if the box
covers an arbitrary region, the mask will too. Always review with --preview before
converting a whole split, and correct the output in a labelling tool afterwards.

Model choice matters a lot on this data: `mobile_sam.pt` produced fragmented,
unusable blobs on EV module tops, while `sam2.1_b.pt` produced coherent masks. Use
the default unless you have a reason not to.

Lines that are already polygons are copied through untouched.

Usage:
    # look at 12 converted images before committing to anything
    python scripts/boxes_to_masks_sam.py --images DIR --labels DIR --preview --limit 12

    # convert a split into a NEW label dir (never overwrites the originals)
    python scripts/boxes_to_masks_sam.py --images data/detector/images/train \
        --labels data/detector/labels/train --out data/detector/labels_seg/train
"""

import argparse
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_label(line):
    """-> ('box', cls, [cx,cy,bw,bh]) | ('poly', cls, [x1,y1,...]) | None"""
    p = line.split()
    if len(p) == 5:
        return "box", int(float(p[0])), [float(v) for v in p[1:]]
    if len(p) > 5 and len(p) % 2 == 1:
        return "poly", int(float(p[0])), [float(v) for v in p[1:]]
    return None


def simplify(points, epsilon_frac=0.002, max_pts=64):
    """Douglas-Peucker simplify so labels stay a sane size. points: Nx2 pixel array."""
    import cv2
    import numpy as np
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, epsilon_frac * peri, True).reshape(-1, 2)
    if len(approx) > max_pts:  # keep every k-th point if still huge
        step = len(approx) // max_pts + 1
        approx = approx[::step]
    return approx


def main():
    ap = argparse.ArgumentParser(description="Convert box labels to SAM-generated polygon labels")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None, help="Output label dir (required unless --preview)")
    ap.add_argument("--model", default="sam2.1_b.pt",
                    help="SAM weights. sam2.1_b.pt (default, good) / mobile_sam.pt (fast, poor here)")
    ap.add_argument("--preview", action="store_true",
                    help="Draw results to outputs/sam_preview/ instead of writing labels")
    ap.add_argument("--limit", type=int, default=0, help="Cap images processed (0 = all)")
    ap.add_argument("--min-area-frac", type=float, default=0.2,
                    help="Reject a mask smaller than this fraction of its prompt box area")
    args = ap.parse_args()

    if not args.preview and not args.out:
        ap.error("--out is required unless --preview")

    import cv2
    import numpy as np
    from ultralytics import SAM

    img_dir, lab_dir = Path(args.images), Path(args.labels)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)

    # only images that actually contain box lines are worth processing
    todo = []
    for p in imgs:
        lp = lab_dir / (p.stem + ".txt")
        if not lp.exists():
            continue
        if any((parse_label(l) or ("", 0, []))[0] == "box" for l in lp.read_text().splitlines() if l.strip()):
            todo.append(p)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(imgs)} images, {len(todo)} contain box labels to convert")
    if not todo:
        print("Nothing to do -- every label is already a polygon.")
        return

    print(f"Loading {args.model} ...")
    sam = SAM(args.model)

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = Path("outputs/sam_preview")
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    n_box = n_mask = n_reject = 0
    for i, p in enumerate(todo, 1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        lp = lab_dir / (p.stem + ".txt")

        keep, boxes, box_cls = [], [], []
        for line in lp.read_text().splitlines():
            parsed = parse_label(line)
            if not parsed:
                continue
            kind, cls, vals = parsed
            if kind == "poly":
                keep.append(line)                      # already a mask, pass through
            else:
                cx, cy, bw, bh = vals
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
                box_cls.append(cls)
        if not boxes:
            continue
        n_box += len(boxes)

        res = sam(str(p), bboxes=boxes, verbose=False)
        masks = res[0].masks.xy if (res and res[0].masks is not None) else []

        vis = im.copy()
        ov = vis.copy()
        for idx, (cls, box) in enumerate(zip(box_cls, boxes)):
            box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
            poly = masks[idx] if idx < len(masks) else None
            ok = poly is not None and len(poly) >= 3 and cv2.contourArea(
                np.asarray(poly, np.float32)) >= args.min_area_frac * box_area
            if not ok:
                # SAM failed on this one -- keep the original box so nothing is lost
                cx = (box[0] + box[2]) / 2 / w; cy = (box[1] + box[3]) / 2 / h
                bw = (box[2] - box[0]) / w; bh = (box[3] - box[1]) / h
                keep.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_reject += 1
                cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                continue
            pts = simplify(poly)
            flat = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in pts)
            keep.append(f"{cls} {flat}")
            n_mask += 1
            col = (0, 220, 0) if cls == 0 else (255, 200, 0)
            ipts = pts.astype(np.int32)
            cv2.fillPoly(ov, [ipts], col)
            cv2.polylines(vis, [ipts], True, col, 2)

        if args.preview:
            vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
            cv2.imwrite(str(prev_dir / p.name), vis)
        else:
            (out_dir / (p.stem + ".txt")).write_text("\n".join(keep) + "\n")

        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")

    print(f"\nboxes seen: {n_box} | converted to masks: {n_mask} | "
          f"SAM rejected, box kept: {n_reject}")
    if args.preview:
        print(f"Preview images -> {prev_dir}/  -- CHECK THESE before a full run.")
    else:
        print(f"New labels -> {out_dir}")
        print("Originals untouched. Review and correct these before training.")


if __name__ == "__main__":
    main()
