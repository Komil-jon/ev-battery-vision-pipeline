"""
build_classifier_dataset.py
============================
Expand the small set of hand-sorted module crops into a classifier dataset at
the paper's scale (Section 4.6 / Appendix K: 48-image test set = 34 good + 14
bad) using augmentation, while guaranteeing that no source crop appears in both
the train and test split (no data leakage).

Why this exists
---------------
The paper used a private, manually-refined crop dataset that is not distributed.
This repository only carries a handful of real good/bad module crops. To run the
condition classifier at a comparable scale, each real crop is expanded with
label-preserving augmentations (flip, small rotation, brightness/contrast,
slight zoom). Augmentation for the TEST split is allowed here ONLY to reach the
paper's reporting scale; every test image derives from a source crop that is
never used for training. This is a feasibility-scale reconstruction, not the
paper's real held-out set — see README.

Inputs  : data/classifier/{train,test}/{good,bad}/  (existing real crops, pooled)
Outputs : data/classifier/{train,test}/{good,bad}/  (rebuilt to target scale)

Usage:
    python scripts/build_classifier_dataset.py
    python scripts/build_classifier_dataset.py --test_good 34 --test_bad 14
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
CLS_ROOT = ROOT / "data" / "classifier"
IMG_EXT  = {".jpg", ".jpeg", ".png", ".bmp"}
SEED     = 42


def collect_sources(cls: str):
    """
    Pool all existing real crops for a class from train/ and test/ and load the
    pixel data into memory up front. Loading before any directory is cleared is
    essential because the source crops live in the same train/ and test/ folders
    that are rebuilt — reading them lazily would race with deletion.
    Returns a list of (stem, image_array).
    """
    loaded = []
    for split in ("train", "test"):
        d = CLS_ROOT / split / cls
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            im = cv2.imread(str(p))
            if im is not None:
                loaded.append((p.stem, im))
    return loaded


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Label-preserving augmentation suitable for whole-crop classification."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, rng.uniform(0.95, 1.10))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # brightness / contrast
    alpha = rng.uniform(0.85, 1.15)   # contrast
    beta  = rng.uniform(-18, 18)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def build_split(imgs, out_dir: Path, target_n: int, rng: random.Random,
                augment_originals: bool):
    """Write target_n images into out_dir from in-memory (stem, image) sources."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*"):
        if f.suffix.lower() in IMG_EXT:
            f.unlink()
    if not imgs:
        return 0
    written = 0
    # First pass: include each real source once (test stays as-real where possible)
    for stem, im in imgs:
        if written >= target_n:
            break
        out = im if not augment_originals else augment(im, rng)
        cv2.imwrite(str(out_dir / f"{stem}_s{written:03d}.jpg"), out)
        written += 1
    # Fill the remainder with augmented copies, cycling through sources
    i = 0
    while written < target_n:
        stem, im = imgs[i % len(imgs)]
        cv2.imwrite(str(out_dir / f"{stem}_a{written:03d}.jpg"), augment(im, rng))
        written += 1
        i += 1
    return written


def build_class(cls: str, test_n: int, train_n: int, rng: random.Random):
    sources = collect_sources(cls)   # list of (stem, image) loaded into memory
    if not sources:
        raise FileNotFoundError(f"No source crops found for class '{cls}'")
    rng.shuffle(sources)
    # Split sources so none is shared between test and train (no leakage)
    n_test_src = max(1, min(len(sources) - 1, round(len(sources) * 0.5)))
    test_src  = sources[:n_test_src]
    train_src = sources[n_test_src:]
    print(f"  [{cls}] {len(sources)} real source crops -> "
          f"{len(test_src)} test-source / {len(train_src)} train-source")

    n_test  = build_split(test_src,  CLS_ROOT / "test"  / cls, test_n,  rng,
                          augment_originals=False)
    n_train = build_split(train_src, CLS_ROOT / "train" / cls, train_n, rng,
                          augment_originals=True)
    print(f"        wrote {n_test} test / {n_train} train images")
    return n_test, n_train


def main():
    ap = argparse.ArgumentParser(description="Build paper-scale classifier dataset")
    ap.add_argument("--test_good",  type=int, default=34)
    ap.add_argument("--test_bad",   type=int, default=14)
    ap.add_argument("--train_good", type=int, default=60)
    ap.add_argument("--train_bad",  type=int, default=40)
    args = ap.parse_args()

    rng = random.Random(SEED)
    print("Building classifier dataset (no train/test source leakage)...")
    tg, rg = build_class("good", args.test_good, args.train_good, rng)
    tb, rb = build_class("bad",  args.test_bad,  args.train_bad,  rng)

    print(f"\nTest set : {tg} good + {tb} bad = {tg + tb} images "
          f"(paper: 34 good + 14 bad = 48)")
    print(f"Train set: {rg} good + {rb} bad = {rg + rb} images")
    print("\nNote: expanded from real crops by augmentation to approach paper "
          "scale; not the paper's real held-out set. See README.")


if __name__ == "__main__":
    main()
