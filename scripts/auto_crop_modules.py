"""
auto_crop_modules.py
====================
Uses the trained Stage 1 detector to automatically crop module ROIs
from training images and saves them ready for manual good/bad sorting.

Instead of manually finding crops, this script:
1. Runs the detector on all training images
2. Crops every detected module with 10px padding
3. Saves them all to data/classifier/to_sort/
4. You then just drag them into good/ or bad/ folders

Usage:
    python scripts/auto_crop_modules.py
"""

import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT             = Path(__file__).resolve().parent.parent
DETECTOR_WEIGHTS = ROOT / "models" / "detector" / "stage1" / "weights" / "best.pt"
IMAGES_DIR       = ROOT / "data" / "detector" / "images" / "train"
TO_SORT_DIR      = ROOT / "data" / "classifier" / "to_sort"
GOOD_TRAIN       = ROOT / "data" / "classifier" / "train" / "good"
BAD_TRAIN        = ROOT / "data" / "classifier" / "train" / "bad"
GOOD_TEST        = ROOT / "data" / "classifier" / "test" / "good"
BAD_TEST         = ROOT / "data" / "classifier" / "test" / "bad"

CROP_PADDING = 10


def main():
    # Create all needed folders
    for d in [TO_SORT_DIR, GOOD_TRAIN, BAD_TRAIN, GOOD_TEST, BAD_TEST]:
        d.mkdir(parents=True, exist_ok=True)

    if not DETECTOR_WEIGHTS.exists():
        print(f"Detector weights not found: {DETECTOR_WEIGHTS}")
        return

    print(f"Loading detector from {DETECTOR_WEIGHTS}")
    detector = YOLO(str(DETECTOR_WEIGHTS))

    img_paths = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    print(f"Found {len(img_paths)} training images")
    print(f"Cropping module detections...")

    count = 0
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = detector(str(img_path), conf=0.21, imgsz=768, device="cpu", verbose=False)
        for r in results:
            for box in r.boxes:
                if detector.names[int(box.cls)] != "module":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add padding
                x1p = max(0, x1 - CROP_PADDING)
                y1p = max(0, y1 - CROP_PADDING)
                x2p = min(w, x2 + CROP_PADDING)
                y2p = min(h, y2 + CROP_PADDING)
                crop = img[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue
                out_name = f"{img_path.stem}_crop{count:04d}.jpg"
                cv2.imwrite(str(TO_SORT_DIR / out_name), crop)
                count += 1

    print(f"\nDone! Saved {count} module crops to:")
    print(f"  {TO_SORT_DIR}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Open this folder in Finder:")
    print(f"     open {TO_SORT_DIR}")
    print(f"  2. Drag GOOD (undamaged) crops → {GOOD_TRAIN}")
    print(f"     Drag BAD  (damaged)   crops → {BAD_TRAIN}")
    print(f"  3. Keep ~10 of each for test:")
    print(f"     good test → {GOOD_TEST}")
    print(f"     bad test  → {BAD_TEST}")
    print(f"  4. You only need ~30-50 total to get the classifier working")
    print(f"\nTip: Look for corrosion, burn marks, dents, discolouration → BAD")
    print(f"     Clean, uniform surface → GOOD")


if __name__ == "__main__":
    main()