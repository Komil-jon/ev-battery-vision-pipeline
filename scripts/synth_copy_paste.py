"""
synth_copy_paste.py
===================
Copy-paste compositing for the detector: cuts labeled module/busbar instances out
of training images and pastes them onto other training images with scale and
brightness jitter, writing new images + YOLO labels. Optionally adds a synthetic
glare pass targeting the measured bright-light weakness (mAP50 0.816 -> 0.672).

Published copy-paste variants report +1-5% mAP, with the largest gains on
minority classes (our busbar). See docs/IMPROVING_ACCURACY.md section B.1/B.4.

Outputs (train-only; never touches val/test):
    data/detector/images/train_copy_paste/*.jpg
    data/detector/labels/train_copy_paste/*.txt

Usage:
    python scripts/synth_copy_paste.py --n_images 300
    python scripts/synth_copy_paste.py --n_images 300 --busbar_only   # minority-class boost
    python scripts/synth_copy_paste.py --glare --n_images 200         # glare-robustness pass
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_IMGS = ROOT / "data" / "detector" / "images" / "train"
TRAIN_LABS = ROOT / "data" / "detector" / "labels" / "train"
OUT_IMGS   = ROOT / "data" / "detector" / "images" / "train_copy_paste"
OUT_LABS   = ROOT / "data" / "detector" / "labels" / "train_copy_paste"

BUSBAR_CLASS_ID = 1   # dataset.yaml: names: ['module', 'busbar']


def load_pairs():
    """Return [(img_path, [label lines])] for every labelled training image."""
    pairs = []
    for img_p in sorted(TRAIN_IMGS.glob("*.jpg")) + sorted(TRAIN_IMGS.glob("*.png")):
        lab_p = TRAIN_LABS / (img_p.stem + ".txt")
        if not lab_p.exists():
            continue
        lines = [ln.split() for ln in lab_p.read_text().splitlines()
                 if len(ln.split()) == 5]
        if lines:
            pairs.append((img_p, lines))
    return pairs


def extract_instances(pairs, busbar_only: bool):
    """
    Build an instance bank: (class_id, crop_bgr) tuples cut from training images.
    Boxes touching the image border or too small are skipped.
    """
    bank = []
    for img_p, lines in pairs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh in lines:
            cls = int(cls)
            if busbar_only and cls != BUSBAR_CLASS_ID:
                continue
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            x0 = int((cx - bw / 2) * w); x1 = int((cx + bw / 2) * w)
            y0 = int((cy - bh / 2) * h); y1 = int((cy + bh / 2) * h)
            if x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1:
                continue  # border-clipped instance: likely truncated
            if (x1 - x0) < 24 or (y1 - y0) < 24:
                continue
            bank.append((cls, img[y0:y1, x0:x1].copy()))
    return bank


def paste_instance(canvas: np.ndarray, crop: np.ndarray):
    """
    Paste a jittered crop at a random location with soft-edge alpha blending.
    Returns (canvas, yolo_box) where yolo_box = (cx, cy, bw, bh) normalised,
    or None if no valid placement was found.
    """
    H, W = canvas.shape[:2]
    scale = random.uniform(0.7, 1.3)
    ch, cw = max(24, int(crop.shape[0] * scale)), max(24, int(crop.shape[1] * scale))
    if ch >= H - 4 or cw >= W - 4:
        return canvas, None
    crop = cv2.resize(crop, (cw, ch))

    # Brightness jitter so the paste matches varied target lighting
    crop = np.clip(crop.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

    x0 = random.randint(2, W - cw - 2)
    y0 = random.randint(2, H - ch - 2)

    # Soft-edge mask hides the rectangular seam
    mask = np.ones((ch, cw), np.float32)
    edge = max(2, min(ch, cw) // 12)
    mask[:edge, :]  *= np.linspace(0, 1, edge)[:, None]
    mask[-edge:, :] *= np.linspace(1, 0, edge)[:, None]
    mask[:, :edge]  *= np.linspace(0, 1, edge)[None, :]
    mask[:, -edge:] *= np.linspace(1, 0, edge)[None, :]
    mask = mask[..., None]

    roi = canvas[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    canvas[y0:y0 + ch, x0:x0 + cw] = (crop * mask + roi * (1 - mask)).astype(np.uint8)

    box = ((x0 + cw / 2) / W, (y0 + ch / 2) / H, cw / W, ch / H)
    return canvas, box


def add_glare(img: np.ndarray) -> np.ndarray:
    """
    Composite 1-3 specular highlight blobs + a mild overexposure gradient,
    simulating the bright/glare condition that degrades detection the most.
    Labels are unchanged — geometry is preserved.
    """
    H, W = img.shape[:2]
    out = img.astype(np.float32)

    for _ in range(random.randint(1, 3)):
        cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
        radius  = random.randint(min(H, W) // 10, min(H, W) // 3)
        blob    = np.zeros((H, W), np.float32)
        cv2.circle(blob, (cx, cy), radius, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), radius / 2)
        out += blob[..., None] * random.uniform(80, 160)

    # Directional overexposure gradient (window / lamp side-light)
    axis = random.choice([0, 1])
    ramp = np.linspace(0, random.uniform(20, 60), W if axis else H, dtype=np.float32)
    out += ramp[None, :, None] if axis else ramp[:, None, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def run(n_images: int, pastes_per_image: int, busbar_only: bool, glare: bool, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    pairs = load_pairs()
    if not pairs:
        print(f"No labelled training images found in {TRAIN_IMGS}.\n"
              "Download and prepare the dataset first (see README Quickstart).")
        return

    # Guard: labels must already be in the 2-class scheme (0=module, 1=busbar).
    # Compositing from un-remapped 7-class labels would bake wrong classes into
    # the synthetic set.
    bad_classes = {int(ln[0]) for _, lines in pairs for ln in lines} - {0, 1}
    if bad_classes:
        print(f"Found label classes {sorted(bad_classes)} — labels are still in the "
              "7-class Roboflow scheme.\nRun the remap first:  "
              "python scripts/remap_labels.py\nAborting; nothing was written.")
        return

    bank = extract_instances(pairs, busbar_only)
    if not bank:
        print("No usable instances extracted (all border-clipped or too small).")
        return

    cls_counts = {}
    for cls, _ in bank:
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
    print(f"Instance bank: {len(bank)} crops "
          f"({', '.join(f'class {c}: {n}' for c, n in sorted(cls_counts.items()))})")
    print(f"Generating {n_images} composite images → {OUT_IMGS.relative_to(ROOT)}")

    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LABS.mkdir(parents=True, exist_ok=True)

    made = 0
    for i in range(n_images):
        bg_path, bg_labels = random.choice(pairs)
        canvas = cv2.imread(str(bg_path))
        if canvas is None:
            continue

        # Existing annotations stay valid — start from them
        new_labels = [" ".join(ln) for ln in bg_labels]

        for _ in range(pastes_per_image):
            cls, crop = random.choice(bank)
            canvas, box = paste_instance(canvas, crop)
            if box is not None:
                cx, cy, bw, bh = (round(v, 6) for v in box)
                new_labels.append(f"{cls} {cx} {cy} {bw} {bh}")

        if glare:
            canvas = add_glare(canvas)

        stem = f"cp_{'glare_' if glare else ''}{i:05d}"
        cv2.imwrite(str(OUT_IMGS / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT_LABS / f"{stem}.txt").write_text("\n".join(new_labels) + "\n")
        made += 1

    print(f"\nDone. {made} images written.")
    print(f"Images → {OUT_IMGS.relative_to(ROOT)}")
    print(f"Labels → {OUT_LABS.relative_to(ROOT)}")
    print("\nNEXT STEP: copy/symlink into data/detector/images/train/ (and labels/train/)\n"
          "before retraining, exactly like the train_busbar_aug workflow.")


def main():
    ap = argparse.ArgumentParser(description="Copy-paste + glare synthesis for the detector")
    ap.add_argument("--n_images",         type=int, default=300,
                    help="Number of composite images to generate (default: 300)")
    ap.add_argument("--pastes_per_image", type=int, default=3,
                    help="Instances pasted per image (default: 3)")
    ap.add_argument("--busbar_only",      action="store_true",
                    help="Paste only busbar instances (minority-class boost)")
    ap.add_argument("--glare",            action="store_true",
                    help="Add synthetic specular glare (lighting robustness)")
    ap.add_argument("--seed",             type=int, default=0, help="Random seed")
    args = ap.parse_args()
    run(args.n_images, args.pastes_per_image, args.busbar_only, args.glare, args.seed)


if __name__ == "__main__":
    main()
