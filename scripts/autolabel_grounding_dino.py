"""
autolabel_grounding_dino.py
===========================
Auto-labels EV battery images to a SINGLE consistent standard using Grounding DINO
(open-vocabulary detection). This is the modern fix for the annotation-consistency
problem that broke multi-source merging: instead of trusting each dataset's own
(inconsistent) labels, re-label everything with one model + one prompt, then a
human verifies. It also labels images that had NO labels (Zenodo, YouTube frames).

Why Grounding DINO and not YOLO-World: on this niche domain YOLO-World zero-shot
scored 0.004 mAP (useless), while Grounding DINO grounds modules/busbars/cells at
0.26-0.41 confidence. Open-vocab is viable for LABELING here, not for detection.

Pipeline: image -> Grounding DINO boxes for the text prompt -> map each detection
to module(0)/busbar(1) by keyword -> write YOLO label. Review before training
(these are proposals, not ground truth), then train a small fast model on them.

Refs: Grounding DINO (arXiv 2303.05499), Autodistill/Grounded-SAM-2, Roboflow
auto-labeling guides. See docs/IMPROVING_ACCURACY.md.

Usage:
    pip install transformers torch pillow
    python scripts/autolabel_grounding_dino.py --images data/external/zenodo_ev_circularity/... --out data/autolabeled
    python scripts/autolabel_grounding_dino.py --images FOLDER --conf 0.3 --prompt "battery module. busbar."
    python scripts/autolabel_grounding_dino.py --images FOLDER --preview   # draw boxes for a visual check
"""

import argparse
from pathlib import Path

MODULE_KEYS = ("module", "cell")          # -> class 0
BUSBAR_KEYS  = ("bus", "bar")             # -> class 1


def label_to_class(label: str):
    l = label.lower()
    if any(k in l for k in BUSBAR_KEYS):
        return 1
    if any(k in l for k in MODULE_KEYS):
        return 0
    return None


def main():
    ap = argparse.ArgumentParser(description="Auto-label EV battery images with Grounding DINO")
    ap.add_argument("--images", required=True, help="Folder of images to label")
    ap.add_argument("--out", default="data/autolabeled", help="Output dataset dir")
    ap.add_argument("--prompt", default="battery module. busbar. battery cell.",
                    help="Grounding DINO text prompt (period-separated phrases)")
    ap.add_argument("--conf", type=float, default=0.30, help="Box confidence threshold")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny",
                    help="grounding-dino-tiny (fast) or grounding-dino-base (better)")
    ap.add_argument("--preview", action="store_true", help="Draw boxes to outputs/ instead of writing labels")
    ap.add_argument("--limit", type=int, default=0, help="Cap images (0 = all)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    img_dir = Path(args.images)
    imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        raise SystemExit(f"No images in {img_dir}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (Path("outputs") / "autolabel_preview").mkdir(parents=True, exist_ok=True)
        import cv2
        import numpy as np

    n_mod = n_bus = n_skipped = 0
    for i, p in enumerate(imgs):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        inp = proc(images=im, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=args.conf, target_sizes=[(H, W)])[0]

        lines = []
        boxes_for_preview = []
        for box, label in zip(res["boxes"], res["labels"]):
            cls = label_to_class(label)
            if cls is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy, bw, bh = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_mod += cls == 0
            n_bus += cls == 1
            boxes_for_preview.append((cls, x1, y1, x2, y2))

        if args.preview:
            imcv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            for cls, x1, y1, x2, y2 in boxes_for_preview:
                col = (0, 200, 0) if cls == 0 else (255, 200, 0)
                cv2.rectangle(imcv, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.imwrite(str(Path("outputs") / "autolabel_preview" / p.name), imcv)
        else:
            import shutil
            shutil.copy(p, out / "images" / p.name)
            (out / "labels" / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            if not lines:
                n_skipped += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)} processed")

    print(f"\nDone. module boxes={n_mod}, busbar boxes={n_bus}, images with no detection={n_skipped}")
    if args.preview:
        print("Preview overlays -> outputs/autolabel_preview/  (review quality before a full run)")
    else:
        print(f"Auto-labeled dataset -> {out}")
        print("NEXT: review/correct labels (they are proposals), then train a small model on them.")
        print("Consistency: every image is labeled by the SAME model+prompt, fixing cross-source drift.")


if __name__ == "__main__":
    main()
