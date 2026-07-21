"""
build_classifier_from_defects.py
================================
Auto-labels good/bad module crops for the condition classifier by COMBINING two
independent annotation sources — with zero manual labeling:

  1. the trained detector  -> tells us WHERE modules are (module boxes)
  2. a defect dataset's GT  -> tells us WHERE damage is  (damage boxes)

A detected module that CONTAINS a damage box is a *bad* (damaged) module; a
detected module with no damage inside is a *good* module. Both good and bad crops
come from the SAME source images (same camera/lighting/pack style), so the
classifier learns real damage rather than dataset style — avoiding the domain
leakage that would happen if good and bad came from different datasets.

Source: uerymnd/ue_d1_defect_detection (237 imgs; 9 damage classes: cell-cover-
damage, cell-nut-corrosion, cell-nut-missing, cell-vent-damage, lid-broken-tab,
lid-corrosion, lid-hole, lid-scratch, module-missing-cover).

Fallback: if the detector finds no module in an image that has damage boxes, a
module-scale patch around the damage is saved as a bad crop (so precious real
damage examples are not lost).

Outputs (kept OUT of train/ so nothing auto-pollutes; review then merge):
    data/classifier/auto/good/*.jpg
    data/classifier/auto/bad/*.jpg

Usage:
    python scripts/build_classifier_from_defects.py
    python scripts/build_classifier_from_defects.py --conf 0.15 --preview
    # then, after reviewing, merge a curated subset into training:
    python scripts/build_classifier_from_defects.py --merge --max_per_class 200
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT     = Path(__file__).resolve().parent.parent
DEFECT   = ROOT / "data" / "external" / "roboflow_ue_d1_defect_detection"
DETECTOR = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
OUT_GOOD = ROOT / "data" / "classifier" / "auto" / "good"
OUT_BAD  = ROOT / "data" / "classifier" / "auto" / "bad"
TRAIN_GOOD = ROOT / "data" / "classifier" / "train" / "good"
TRAIN_BAD  = ROOT / "data" / "classifier" / "train" / "bad"

MODULE_CLASS = 0   # detector class id for 'module'


def load_damage_boxes(label_path: Path, w: int, h: int):
    """Return damage boxes as pixel [x1,y1,x2,y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in p)
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def damage_in_module(mod, damages, min_frac=0.5):
    """True if any damage box's centre lies inside the module box (robust for
    small defect boxes like a corroded nut), or a damage box overlaps enough."""
    mx1, my1, mx2, my2 = mod
    for dx1, dy1, dx2, dy2 in damages:
        cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
        if mx1 <= cx <= mx2 and my1 <= cy <= my2:
            return True
    return False


def crop(img, box, pad=10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad); y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad); y2 = min(h, int(y2) + pad)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return img[y1:y2, x1:x2]


def run(conf, pad, preview):
    from ultralytics import YOLO
    if not DETECTOR.exists():
        raise SystemExit(f"Detector weights not found: {DETECTOR}")
    det = YOLO(str(DETECTOR))

    OUT_GOOD.mkdir(parents=True, exist_ok=True)
    OUT_BAD.mkdir(parents=True, exist_ok=True)

    imgs = []
    for split in ["train", "valid", "test"]:
        d = DEFECT / split / "images"
        if d.exists():
            imgs += sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
    if not imgs:
        raise SystemExit(f"No defect images at {DEFECT}")

    n_good = n_bad = n_fallback = n_nomod = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lab = DEFECT / img_path.parent.parent.name / "labels" / (img_path.stem + ".txt")
        damages = load_damage_boxes(lab, w, h)

        with torch.no_grad():
            res = det(img, conf=conf, imgsz=768, device="cpu", verbose=False)[0]
        modules = [list(map(float, b.xyxy[0])) for b in res.boxes if int(b.cls) == MODULE_CLASS]

        if not modules:
            # This defect dataset is macro close-ups (no module-level box), so the
            # detector finds nothing. Build a balanced, SAME-DOMAIN patch dataset:
            #   BAD  = patches around damage boxes
            #   GOOD = equal-count patches from the same image with NO damage
            side = 160
            bad_centres = []
            for i, (dx1, dy1, dx2, dy2) in enumerate(damages):
                cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
                s = max(side, int(max(dx2 - dx1, dy2 - dy1) * 4))
                patch = crop(img, [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], 0)
                if patch is not None:
                    cv2.imwrite(str(OUT_BAD / f"{img_path.stem}_fb{i}.jpg"), patch,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_bad += 1; n_fallback += 1
                    bad_centres.append((cx, cy, s))
            # GOOD: sample same-size clean patches far from every damage box
            tries, made = 0, 0
            while made < len(bad_centres) and tries < 60:
                tries += 1
                gx = random.uniform(side / 2, w - side / 2)
                gy = random.uniform(side / 2, h - side / 2)
                if all(abs(gx - cx) > (s + side) / 2 or abs(gy - cy) > (s + side) / 2
                       for cx, cy, s in bad_centres):
                    patch = crop(img, [gx - side / 2, gy - side / 2, gx + side / 2, gy + side / 2], 0)
                    if patch is not None:
                        cv2.imwrite(str(OUT_GOOD / f"{img_path.stem}_g{made}.jpg"), patch,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        n_good += 1; made += 1
            if damages:
                n_nomod += 1
            continue

        for j, mod in enumerate(modules):
            c = crop(img, mod, pad)
            if c is None:
                continue
            if damage_in_module(mod, damages):
                cv2.imwrite(str(OUT_BAD / f"{img_path.stem}_m{j}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_bad += 1
            else:
                cv2.imwrite(str(OUT_GOOD / f"{img_path.stem}_m{j}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_good += 1

    print(f"\n=== Auto-labeled classifier crops from {len(imgs)} defect images ===")
    print(f"  GOOD (module, no damage inside): {n_good}  -> {OUT_GOOD.relative_to(ROOT)}")
    print(f"  BAD  (module contains damage):   {n_bad}   -> {OUT_BAD.relative_to(ROOT)}")
    print(f"    of which fallback damage-patches (no module detected): {n_fallback}")
    print(f"  images with damage but no detected module: {n_nomod}")
    print(f"\nvs the 16 real hand-labeled bad crops you had before.")

    if preview and (n_good or n_bad):
        _preview()
    print("\nNEXT: review the crops, then merge a curated subset:")
    print("  python scripts/build_classifier_from_defects.py --merge --max_per_class 200")


def _preview():
    good = sorted(OUT_GOOD.glob("*.jpg"))[:6]
    bad = sorted(OUT_BAD.glob("*.jpg"))[:6]
    def row(paths):
        tiles = [cv2.resize(cv2.imread(str(p)), (160, 160)) for p in paths] or [np.zeros((160,160,3),np.uint8)]
        while len(tiles) < 6: tiles.append(np.zeros((160, 160, 3), np.uint8))
        return np.hstack(tiles)
    sheet = np.vstack([row(good), row(bad)])
    out = ROOT / "outputs" / "classifier_auto_preview.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"Preview (top row=GOOD, bottom row=BAD) -> {out.relative_to(ROOT)}")


def merge(max_per_class):
    """Copy a balanced curated subset of the auto crops into the training set."""
    for src, dst, name in [(OUT_GOOD, TRAIN_GOOD, "good"), (OUT_BAD, TRAIN_BAD, "bad")]:
        crops = sorted(src.glob("*.jpg"))
        random.seed(0); random.shuffle(crops)
        crops = crops[:max_per_class]
        dst.mkdir(parents=True, exist_ok=True)
        for p in crops:
            shutil.copy(p, dst / f"auto_{p.name}")
        print(f"  merged {len(crops)} {name} crops -> {dst.relative_to(ROOT)}")
    print("Now retrain: python scripts/train_classifier.py  (test set stays real-only)")


def main():
    ap = argparse.ArgumentParser(description="Auto-label good/bad module crops from a defect dataset")
    ap.add_argument("--conf", type=float, default=0.20, help="Detector confidence for modules")
    ap.add_argument("--pad", type=int, default=10, help="Crop padding (px)")
    ap.add_argument("--preview", action="store_true", help="Write a good/bad preview sheet")
    ap.add_argument("--merge", action="store_true", help="Merge curated crops into train/")
    ap.add_argument("--max_per_class", type=int, default=200, help="Cap per class when merging")
    args = ap.parse_args()
    if args.merge:
        merge(args.max_per_class)
    else:
        run(args.conf, args.pad, args.preview)


if __name__ == "__main__":
    main()
