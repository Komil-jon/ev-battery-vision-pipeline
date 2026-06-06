"""
augment_busbars.py
==================
Generates busbar-targeted augmented training images for the Stage 2 recall-boost.

Paper rationale (Section 3.3):
  - Busbar recall was the primary bottleneck after Stage 1 (recall 0.793)
  - Augmentation applied ONLY to busbar-containing images
  - ~600+ additional busbar-focused images generated
  - Validation and test sets remain real and non-augmented

Augmentation strategy matches paper:
  - Horizontal flip (p=0.5)
  - Brightness/contrast variation (hsv_v=0.30, hsv_s=0.25)
  - Rotation limited to ±2° (fixed orientations on disassembly fixtures)
  - Translation limited to 6%
  - NO hue jitter (would corrupt corrosion/burn mark cues)
  - NO destructive transforms

Usage:
    python scripts/augment_busbars.py
    python scripts/augment_busbars.py --n_augments 8 --output_dir data/detector/images/train_augmented
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
TRAIN_IMGS = ROOT / "data" / "detector" / "images" / "train"
TRAIN_LABS = ROOT / "data" / "detector" / "labels" / "train"
AUG_IMGS   = ROOT / "data" / "detector" / "images" / "train_busbar_aug"
AUG_LABS   = ROOT / "data" / "detector" / "labels" / "train_busbar_aug"

BUSBAR_CLASS_ID = 1   # matches dataset.yaml: names: ['module', 'busbar']


def has_busbar(label_path: Path) -> bool:
    """Return True if label file contains at least one busbar annotation."""
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == BUSBAR_CLASS_ID:
                return True
    return False


def random_brightness_contrast(img: np.ndarray, v_range=0.30, s_range=0.25) -> np.ndarray:
    """Apply HSV brightness/saturation variation (paper hsv_v=0.30, hsv_s=0.25)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Saturation
    s_factor = 1.0 + random.uniform(-s_range, s_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
    # Value (brightness)
    v_factor = 1.0 + random.uniform(-v_range, v_range)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_flip(img: np.ndarray, labels: list, p: float = 0.5):
    """Horizontal flip with YOLO label adjustment."""
    if random.random() < p:
        img = cv2.flip(img, 1)
        flipped = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, bw, bh = parts
                cx_new = str(round(1.0 - float(cx), 6))
                flipped.append(f"{cls} {cx_new} {cy} {bw} {bh}")
            else:
                flipped.append(line.strip())
        labels = flipped
    return img, labels


def random_rotate(img: np.ndarray, labels: list, max_deg: float = 2.0):
    """
    Small rotation (±2°) — paper limits rotation because modules present
    at near-fixed orientations on disassembly fixtures.
    Note: bounding boxes are kept as-is (small angle, negligible error).
    """
    angle = random.uniform(-max_deg, max_deg)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, labels


def random_translate(img: np.ndarray, labels: list, max_frac: float = 0.06):
    """
    Random translation up to 6% of image dimension.
    Adjusts YOLO bounding-box centre coordinates accordingly.
    """
    h, w = img.shape[:2]
    tx = random.uniform(-max_frac, max_frac) * w
    ty = random.uniform(-max_frac, max_frac) * h
    M  = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    shifted = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, bw, bh = parts
            cx_new = float(cx) + tx / w
            cy_new = float(cy) + ty / h
            # Clamp to valid range
            cx_new = max(0.0, min(1.0, cx_new))
            cy_new = max(0.0, min(1.0, cy_new))
            shifted.append(f"{cls} {round(cx_new, 6)} {round(cy_new, 6)} {bw} {bh}")
        else:
            shifted.append(line.strip())
    labels = shifted
    return img, labels


def augment_image(img: np.ndarray, labels: list) -> tuple:
    """Apply full augmentation chain to one image/label pair."""
    img, labels = random_flip(img, labels, p=0.5)
    img, labels = random_rotate(img, labels, max_deg=2.0)
    img, labels = random_translate(img, labels, max_frac=0.06)
    img         = random_brightness_contrast(img, v_range=0.30, s_range=0.25)
    return img, labels


def run(n_augments: int, output_img_dir: Path, output_lab_dir: Path):
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lab_dir.mkdir(parents=True, exist_ok=True)

    # Find all busbar-containing training images
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg")) + sorted(TRAIN_IMGS.glob("*.png"))
    busbar_imgs = []
    for img_p in img_paths:
        lab_p = TRAIN_LABS / (img_p.stem + ".txt")
        if has_busbar(lab_p):
            busbar_imgs.append((img_p, lab_p))

    if not busbar_imgs:
        print(
            f"No busbar-containing images found in {TRAIN_IMGS}\n"
            "Add training images with busbar annotations first."
        )
        return

    print(f"Found {len(busbar_imgs)} busbar-containing training images.")
    print(f"Generating {n_augments} augments per image → ~{len(busbar_imgs) * n_augments} new images")

    count = 0
    for img_path, lab_path in busbar_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        with open(lab_path) as f:
            labels = f.readlines()

        # Copy original into augmented folder too
        shutil.copy(img_path, output_img_dir / img_path.name)
        shutil.copy(lab_path, output_lab_dir / lab_path.name)

        for i in range(n_augments):
            aug_img, aug_labels = augment_image(img.copy(), labels.copy())
            stem     = f"{img_path.stem}_aug{i:03d}"
            out_img  = output_img_dir / f"{stem}.jpg"
            out_lab  = output_lab_dir / f"{stem}.txt"
            cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lab, "w") as f:
                f.write("\n".join(aug_labels))
            count += 1

    print(f"\nDone. Generated {count} augmented images.")
    print(f"Images → {output_img_dir}")
    print(f"Labels → {output_lab_dir}")
    print(
        "\nNEXT STEP: Update dataset.yaml to include the augmented folder in your training set,\n"
        "or copy these files into data/detector/images/train/ before running Stage 2."
    )


def main():
    parser = argparse.ArgumentParser(description="Busbar-targeted augmentation for recall boost")
    parser.add_argument("--n_augments",  type=int, default=6,
                        help="Number of augmented copies per busbar image (default: 6)")
    parser.add_argument("--output_dir",  type=str,
                        default=str(AUG_IMGS),
                        help="Output directory for augmented images")
    args = parser.parse_args()

    out_img = Path(args.output_dir)
    out_lab = Path(str(out_img).replace("images", "labels"))
    run(args.n_augments, out_img, out_lab)


if __name__ == "__main__":
    main()
