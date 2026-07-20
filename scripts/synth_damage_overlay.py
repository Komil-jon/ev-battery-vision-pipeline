"""
synth_damage_overlay.py
=======================
Procedural "bad" crop generator for the condition classifier. Composites three
damage archetypes onto good module crops with OpenCV only (CPU, no models):

  corrosion  — white/green powdery blotches built from blurred noise
  burn       — darkened soft-edged patches with charred texture
  scratch    — thin bright polylines with slight metallic sheen

This is the zero-setup fallback to the diffusion-inpainting notebook
(notebooks/colab_defect_inpainting.ipynb) — less realistic, but instant, and the
CutPaste literature shows even crude synthetic damage teaches useful decision
boundaries. See docs/IMPROVING_ACCURACY.md section B.3.

Outputs go to data/classifier/bad_synth/ — a SEPARATE folder so the real
and synthetic bad pools never mix silently. To train with them, copy into
data/classifier/train/bad/ (keep synthetic ≤50% of the class) and always
evaluate on the real-only test set.

Usage:
    python scripts/synth_damage_overlay.py                 # 4 variants per good crop
    python scripts/synth_damage_overlay.py --n_per_image 8
    python scripts/synth_damage_overlay.py --preview       # write a 3x3 contact sheet only
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
GOOD_DIR = ROOT / "data" / "classifier" / "train" / "good"
# Deliberately OUTSIDE train/ — ImageFolder(train/) must only ever see good/ and
# bad/, otherwise a third folder becomes a third class.
OUT_DIR  = ROOT / "data" / "classifier" / "bad_synth"

CORROSION_COLORS = [  # BGR: white oxide, verdigris green, pale blue sulfate
    (230, 235, 235), (140, 200, 150), (200, 180, 140),
]


def noise_blob_mask(h: int, w: int, coverage: float) -> np.ndarray:
    """Organic blob mask in [0,1]: threshold heavily blurred uniform noise."""
    noise = np.random.rand(h // 4 + 1, w // 4 + 1).astype(np.float32)
    noise = cv2.resize(noise, (w, h))
    noise = cv2.GaussianBlur(noise, (0, 0), min(h, w) / 12)
    thresh = np.quantile(noise, 1.0 - coverage)
    mask = np.clip((noise - thresh) / max(1e-6, noise.max() - thresh), 0, 1)
    return cv2.GaussianBlur(mask, (0, 0), 3)


def apply_corrosion(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = noise_blob_mask(h, w, coverage=random.uniform(0.06, 0.20))[..., None]
    color = np.array(random.choice(CORROSION_COLORS), np.float32)
    # Powdery texture: modulate the colour with fine noise
    grain = 0.75 + 0.25 * np.random.rand(h, w, 1).astype(np.float32)
    overlay = color[None, None, :] * grain
    out = img.astype(np.float32) * (1 - 0.85 * mask) + overlay * 0.85 * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_burn(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(random.randint(1, 2)):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        ax = random.randint(w // 8, w // 3)
        ay = random.randint(h // 8, h // 3)
        cv2.ellipse(mask, (cx, cy), (ax, ay), random.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), min(h, w) / 16)[..., None]
    # Charred texture: darken strongly, add sooty noise
    soot = np.random.rand(h, w, 1).astype(np.float32) * 30
    out = img.astype(np.float32) * (1 - 0.75 * mask) + soot * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_scratch(img: np.ndarray) -> np.ndarray:
    """Short, roughly-straight gouges: a main direction with small jitter."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(random.randint(1, 3)):
        length = random.uniform(0.2, 0.5) * min(h, w)
        angle  = random.uniform(0, 2 * np.pi)
        x, y   = random.uniform(0.15, 0.85) * w, random.uniform(0.15, 0.85) * h
        n_seg  = random.randint(3, 6)
        step   = length / n_seg
        pts = [(x, y)]
        for _ in range(n_seg):
            angle += random.uniform(-0.25, 0.25)  # slight waviness, keeps direction
            x += step * np.cos(angle)
            y += step * np.sin(angle)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        brightness = random.randint(180, 250)  # bright exposed metal
        # Faint dark shadow beside the scratch for depth, then the bright gouge
        cv2.polylines(out, [pts + 1], False, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.polylines(out, [pts], False,
                      (brightness, brightness, brightness),
                      thickness=random.randint(1, 2), lineType=cv2.LINE_AA)
    return out


DAMAGE_FNS = {
    "corrosion": apply_corrosion,
    "burn":      apply_burn,
    "scratch":   apply_scratch,
}


def synth_bad(img: np.ndarray) -> tuple:
    """Apply 1-2 random damage types; returns (image, tag)."""
    kinds = random.sample(list(DAMAGE_FNS), k=random.randint(1, 2))
    out = img.copy()
    for k in kinds:
        out = DAMAGE_FNS[k](out)
    return out, "-".join(kinds)


def run(n_per_image: int, preview: bool, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    good_imgs = sorted(GOOD_DIR.glob("*.jpg")) + sorted(GOOD_DIR.glob("*.png"))
    if not good_imgs:
        print(f"No good crops found in {GOOD_DIR}.\n"
              "Build the classifier dataset first (scripts/build_classifier_dataset.py).")
        return

    if preview:
        # 3x3 contact sheet from the first good crop for quick visual QA
        img = cv2.imread(str(good_imgs[0]))
        img = cv2.resize(img, (224, 224))
        tiles = [img] + [cv2.resize(synth_bad(img)[0], (224, 224)) for _ in range(8)]
        sheet = np.vstack([np.hstack(tiles[r * 3:r * 3 + 3]) for r in range(3)])
        out = ROOT / "outputs" / "synth_damage_preview.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), sheet)
        print(f"Preview contact sheet (top-left = original) → {out.relative_to(ROOT)}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(good_imgs)} good crops; generating {n_per_image} damaged "
          f"variants each → ~{len(good_imgs) * n_per_image} images")

    count = 0
    for img_p in good_imgs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        for i in range(n_per_image):
            bad, tag = synth_bad(img)
            out = OUT_DIR / f"{img_p.stem}_synth{i:02d}_{tag}.jpg"
            cv2.imwrite(str(out), bad, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

    print(f"\nDone. {count} synthetic bad crops → {OUT_DIR.relative_to(ROOT)}")
    print("\nNEXT STEPS:")
    print("  1. Visually review; delete implausible crops.")
    print("  2. Copy a curated subset into data/classifier/train/bad/ "
          "(keep synthetic ≤50% of the class).")
    print("  3. Retrain: python scripts/train_classifier.py")
    print("  4. Evaluate on the REAL-only test set: python scripts/evaluate.py --skip_detector")


def main():
    ap = argparse.ArgumentParser(description="Procedural damage synthesis for classifier bad class")
    ap.add_argument("--n_per_image", type=int, default=4,
                    help="Damaged variants per good crop (default: 4)")
    ap.add_argument("--preview", action="store_true",
                    help="Write a 3x3 contact sheet to outputs/ and exit")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()
    run(args.n_per_image, args.preview, args.seed)


if __name__ == "__main__":
    main()
