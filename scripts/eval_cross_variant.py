"""
eval_cross_variant.py
=====================
Zero-shot cross-variant generalization probe for the detector, using the Zenodo
"Battery Image Dataset for EV Circularity" (19 vehicle types, CC BY 4.0). The
Zenodo images have NO bounding-box labels, so this cannot compute mAP; instead it
measures, per battery variant, the detector's zero-shot behaviour on pack types
it never saw in training:

  - detection rate: fraction of images with >=1 module OR busbar detection
  - mean detections / image, mean confidence
  - module vs busbar detection counts

This is a generalization *proxy* (recall-oriented: an undamaged detector should
at least fire on a battery pack it has never seen), NOT an accuracy metric. It
answers the reviewers' single-facility criticism qualitatively and flags which
pack geometries the current model transfers to vs fails on.

Variants whose names indicate overlap with the MTech training source
(Tesla Model 3, Tesla Model S black) are marked and excluded from the headline
"unseen" average so the number is honest.

Usage:
    python scripts/eval_cross_variant.py                 # all variants
    python scripts/eval_cross_variant.py --conf 0.25 --max_per_variant 40
    python scripts/eval_cross_variant.py --save_annotated   # write example overlays
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

ROOT      = Path(__file__).resolve().parent.parent
WEIGHTS   = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
ZENODO    = (ROOT / "data" / "external" / "zenodo_ev_circularity" /
             "Battery Image Dataset" / "images")
OUT_DIR   = ROOT / "outputs" / "cross_variant"

# Variants sourced from the MTech Roboflow set that this project already trains on.
TRAIN_OVERLAP = {"Tesla Model 3", "Tesla Model S black"}


def main():
    ap = argparse.ArgumentParser(description="Zero-shot cross-variant detector probe (Zenodo)")
    ap.add_argument("--conf", type=float, default=0.21, help="Detection confidence (paper optimum)")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--max_per_variant", type=int, default=1000,
                    help="Cap images scored per variant (default: all)")
    ap.add_argument("--save_annotated", action="store_true",
                    help="Save one annotated example per variant to outputs/cross_variant/")
    args = ap.parse_args()

    if not ZENODO.exists():
        raise SystemExit(f"Zenodo images not found at {ZENODO}\n"
                         "Download + unzip first: "
                         "python scripts/download_external_datasets.py --zenodo")

    from ultralytics import YOLO
    det = YOLO(str(WEIGHTS))
    names = det.names   # {0: 'module', 1: 'busbar'}

    if args.save_annotated:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant_dir in sorted(ZENODO.iterdir()):
        if not variant_dir.is_dir():
            continue
        imgs = sorted(p for p in variant_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.max_per_variant]
        if not imgs:
            continue

        n_img = len(imgs)
        n_with_det = 0
        total_det = 0
        conf_sum = 0.0
        cls_counts = defaultdict(int)
        saved_example = False

        for img_path in imgs:
            res = det(str(img_path), conf=args.conf, imgsz=args.imgsz,
                      device="cpu", verbose=False)[0]
            nb = len(res.boxes)
            if nb > 0:
                n_with_det += 1
                total_det += nb
                for box in res.boxes:
                    conf_sum += float(box.conf)
                    cls_counts[int(box.cls)] += 1
                if args.save_annotated and not saved_example:
                    im = cv2.imread(str(img_path))
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c = int(box.cls)
                        col = (0, 200, 0) if c == 0 else (255, 200, 0)
                        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(im, f"{names[c]} {float(box.conf):.2f}",
                                    (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, col, 1, cv2.LINE_AA)
                    safe = variant_dir.name.replace(" ", "_").replace("/", "_")
                    cv2.imwrite(str(OUT_DIR / f"{safe}.jpg"), im)
                    saved_example = True

        rows.append({
            "variant": variant_dir.name,
            "overlap": variant_dir.name in TRAIN_OVERLAP,
            "n_img": n_img,
            "det_rate": n_with_det / n_img,
            "mean_det": total_det / n_img,
            "mean_conf": conf_sum / total_det if total_det else 0.0,
            "module": cls_counts[0],
            "busbar": cls_counts[1],
        })

    rows.sort(key=lambda r: r["det_rate"], reverse=True)

    print(f"\n## Zero-shot cross-variant detector probe "
          f"(conf={args.conf}, imgsz={args.imgsz})\n")
    print("| Variant | Source | Images | Detection rate | Mean det/img | "
          "Mean conf | Module | Busbar |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tag = "train-overlap" if r["overlap"] else "unseen"
        print(f"| {r['variant']} | {tag} | {r['n_img']} | {r['det_rate']:.2f} "
              f"| {r['mean_det']:.1f} | {r['mean_conf']:.2f} "
              f"| {r['module']} | {r['busbar']} |")

    unseen = [r for r in rows if not r["overlap"]]
    if unseen:
        avg_rate = sum(r["det_rate"] for r in unseen) / len(unseen)
        n_bus_variants = sum(1 for r in unseen if r["busbar"] > 0)
        print(f"\nUnseen variants ({len(unseen)}): mean detection rate "
              f"{avg_rate:.2f}; busbar detected in {n_bus_variants}/{len(unseen)}.")
        print("Interpretation: high module detection rate on unseen packs = geometry "
              "transfers; low busbar counts = busbar generalisation is the weak axis "
              "(consistent with the in-domain busbar recall gap).")
    print("\nNOTE: detection-rate proxy only (Zenodo images are unlabelled); "
          "not an mAP measurement.")


if __name__ == "__main__":
    main()
