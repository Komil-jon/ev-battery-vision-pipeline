"""
compare_detectors.py
====================
Metric-matched comparison of any detectors in the model zoo (YOLO and RF-DETR
alike) on the SAME test images with the SAME mAP implementation (supervision) and
the SAME confidence threshold. Ultralytics' .val() and supervision's mAP are NOT
directly comparable, so cross-architecture claims must come from this script.

It also reports per-class mAP50 and can exclude the MTech-convention sources,
which use an idiosyncratic annotation style (tiny sub-module boxes) and dominate
the aggregate otherwise. See CHANGELOG.md 2026-07-24.

Requires: pip install supervision  (plus rfdetr if comparing an RF-DETR model)

Usage:
    python scripts/compare_detectors.py --images DIR --labels DIR
    python scripts/compare_detectors.py --images DIR --labels DIR --exclude-mtech
    python scripts/compare_detectors.py --images DIR --labels DIR \
        --models specialist generalist_yolo generalist_rfdetr
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

MTECH_PREFIXES = ("mtech", "automated-disassembly")


def yolo_labels_to_coco(images_dir: Path, labels_dir: Path, out_dir: Path, exclude_mtech=False):
    """Build a COCO json from YOLO .txt labels. Polygon-format lines are converted
    to bounding boxes (NOT dropped -- dropping them silently discarded 80% of the
    labels in an earlier run and produced incomparable metrics)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png")
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if exclude_mtech:
        imgs = [p for p in imgs if not p.name.lower().startswith(MTECH_PREFIXES)]

    images, anns, aid = [], [], 1
    for iid, ip in enumerate(imgs, 1):
        w, h = Image.open(ip).size
        if not (out_dir / ip.name).exists():
            shutil.copy(ip, out_dir / ip.name)
        images.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
        lp = labels_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) == 5:                        # cls cx cy bw bh
                c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:])
                x, y, bw, bh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            elif len(p) > 5 and len(p) % 2 == 1:   # cls x1 y1 x2 y2 ... polygon
                c = int(float(p[0]))
                xs = list(map(float, p[1::2])); ys = list(map(float, p[2::2]))
                x0, x1n, y0, y1n = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
                x, y, bw, bh = x0, y0, x1n - x0, y1n - y0
            else:
                continue
            if bw <= 0 or bh <= 0:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1

    cats = [{"id": 0, "name": "module", "supercategory": "none"},
            {"id": 1, "name": "busbar", "supercategory": "none"}]
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(out_dir / "_annotations.coco.json", "w"))
    return len(images), len(anns)


def main():
    ap = argparse.ArgumentParser(description="Metric-matched detector comparison")
    ap.add_argument("--images", required=True, help="Test images dir")
    ap.add_argument("--labels", required=True, help="YOLO labels dir for those images")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Model names from the zoo (default: all available)")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Confidence threshold; keep low so mAP integrates the full PR curve")
    ap.add_argument("--exclude-mtech", action="store_true",
                    help="Drop MTech-convention images (tiny sub-module annotation style)")
    args = ap.parse_args()

    import numpy as np
    import supervision as sv
    from model_zoo import MODEL_REGISTRY, load_detector

    names = args.models or [n for n, i in MODEL_REGISTRY.items() if i.weights.exists()]
    if not names:
        raise SystemExit("No model weights found. See models/detector/*/README.md")

    tmp = Path(tempfile.mkdtemp(prefix="cmp_coco_"))
    n_img, n_ann = yolo_labels_to_coco(Path(args.images), Path(args.labels), tmp,
                                       exclude_mtech=args.exclude_mtech)
    print(f"Test set: {n_img} images, {n_ann} ground-truth boxes"
          f"{' (MTech excluded)' if args.exclude_mtech else ''}\n")

    ds = sv.DetectionDataset.from_coco(str(tmp), str(tmp / "_annotations.coco.json"))
    paths, targets = [], []
    for path, _image, ann in ds:
        paths.append(path); targets.append(ann)

    rows = []
    for name in names:
        info = MODEL_REGISTRY[name]
        if not info.weights.exists():
            print(f"[skip] {name}: weights missing ({info.weights})"); continue
        print(f"Evaluating {name} ...")
        det = load_detector(name=name)
        preds = []
        for p in paths:
            dets = det.predict(p, conf=args.conf)
            preds.append(sv.Detections(
                xyxy=np.array([d.xyxy for d in dets], dtype=float).reshape(-1, 4),
                class_id=np.array([d.cls_id for d in dets], dtype=int),
                confidence=np.array([d.conf for d in dets], dtype=float),
            ))
        overall = sv.MeanAveragePrecision.from_detections(preds, targets)
        per_class = {}
        for cid, cname in [(0, "module"), (1, "busbar")]:
            pc = sv.MeanAveragePrecision.from_detections(
                [p[p.class_id == cid] for p in preds],
                [t[t.class_id == cid] for t in targets])
            per_class[cname] = pc.map50
        rows.append((name, overall.map50, overall.map50_95, per_class["module"], per_class["busbar"]))

    print(f"\n{'model':<22}{'mAP50':>9}{'mAP50-95':>11}{'module':>9}{'busbar':>9}")
    print("-" * 60)
    for name, m50, m5095, mod, bus in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<22}{m50:>9.3f}{m5095:>11.3f}{mod:>9.3f}{bus:>9.3f}")
    print(f"\nProtocol: supervision mAP, conf={args.conf}, identical images for every model.")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
