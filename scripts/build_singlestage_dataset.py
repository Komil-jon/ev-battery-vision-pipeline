"""
build_singlestage_dataset.py
============================
Builds the dataset for the single-stage ablation reviewers asked for: a unified
3-class YOLO scheme (0 = module-good, 1 = module-bad, 2 = busbar) so a single
detector can be trained end-to-end and compared against the two-stage pipeline
on identical data.

The blocker for this ablation is per-box condition labels, which the 2-class
detection annotations do not carry. Workflow:

  1. Bootstrap a condition CSV using the trained classifier on every GT module
     box (NOT detector output — the ablation must not inherit detector errors):
         python scripts/build_singlestage_dataset.py --bootstrap
     → data/detector_singlestage/condition_labels.csv
       (columns: split, image_stem, box_index, p_bad, condition)

  2. HUMAN PASS (required): open the CSV, review each row — especially rows
     with 0.3 < p_bad < 0.7 — and correct the `condition` column. The CSV is
     a pre-annotation to speed you up, not ground truth.

  3. Emit the 3-class dataset:
         python scripts/build_singlestage_dataset.py --build
     → data/detector_singlestage/{images,labels}/{train,val,test}/
       + dataset_singlestage.yaml

  4. Train the single-stage baseline (Colab recommended; CPU works):
         yolo detect train data=dataset_singlestage.yaml model=yolov8n.pt \\
             epochs=100 imgsz=640
     and compare module-bad recall against the two-stage pipeline's
     end-to-end propagation numbers (paper Tables 5-6).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DET_IMGS = ROOT / "data" / "detector" / "images"
DET_LABS = ROOT / "data" / "detector" / "labels"
OUT_ROOT = ROOT / "data" / "detector_singlestage"
CSV_PATH = OUT_ROOT / "condition_labels.csv"
YAML_OUT = ROOT / "dataset_singlestage.yaml"

SPLITS = ["train", "val", "test"]
MODULE, BUSBAR = 0, 1                       # current 2-class scheme
SS_GOOD, SS_BAD, SS_BUSBAR = 0, 1, 2        # single-stage 3-class scheme


def iter_labelled(split):
    lab_dir = DET_LABS / split
    img_dir = DET_IMGS / split
    for lab in sorted(lab_dir.glob("*.txt")):
        img = next((img_dir / f"{lab.stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
                    if (img_dir / f"{lab.stem}{ext}").exists()), None)
        if img is None:
            continue
        lines = [ln.split() for ln in lab.read_text().splitlines() if len(ln.split()) == 5]
        yield img, lab, lines


def bootstrap():
    """Run the trained classifier on every GT module box → pre-annotation CSV."""
    import cv2
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms

    weights   = ROOT / "models" / "classifier" / "resnet18_binary.pth"
    class_map = ROOT / "models" / "classifier" / "class_map.json"
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(str(weights), map_location="cpu"))
    model.eval()
    bad_idx = json.load(open(class_map)).get("bad", 0) if class_map.exists() else 0

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, n_boxes = [], 0
    for split in SPLITS:
        for img_path, lab_path, lines in iter_labelled(split):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for bi, (cls, cx, cy, bw, bh) in enumerate(lines):
                if int(cls) != MODULE:
                    continue
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
                y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
                crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                if crop.size == 0:
                    continue
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    p_bad = torch.softmax(model(tf(pil).unsqueeze(0)), dim=1)[0][bad_idx].item()
                rows.append({"split": split, "image_stem": img_path.stem,
                             "box_index": bi, "p_bad": round(p_bad, 3),
                             "condition": "bad" if p_bad >= 0.5 else "good"})
                n_boxes += 1

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image_stem", "box_index",
                                          "p_bad", "condition"])
        w.writeheader()
        w.writerows(rows)

    uncertain = sum(1 for r in rows if 0.3 < r["p_bad"] < 0.7)
    print(f"Bootstrapped {n_boxes} module boxes → {CSV_PATH.relative_to(ROOT)}")
    print(f"{uncertain} boxes have 0.3 < p_bad < 0.7 — review those first.")
    print("\nNEXT: manually review/correct the `condition` column, then run --build.")


def build():
    """Emit 3-class labels using the (reviewed) condition CSV."""
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found — run --bootstrap first, then review it.")

    cond = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            cond[(r["split"], r["image_stem"], int(r["box_index"]))] = r["condition"]

    n_good = n_bad = n_bus = 0
    for split in SPLITS:
        out_img = OUT_ROOT / "images" / split
        out_lab = OUT_ROOT / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)

        for img_path, lab_path, lines in iter_labelled(split):
            new_lines = []
            for bi, (cls, cx, cy, bw, bh) in enumerate(lines):
                if int(cls) == BUSBAR:
                    new_lines.append(f"{SS_BUSBAR} {cx} {cy} {bw} {bh}")
                    n_bus += 1
                else:
                    c = cond.get((split, img_path.stem, bi), "good")
                    ss = SS_BAD if c == "bad" else SS_GOOD
                    new_lines.append(f"{ss} {cx} {cy} {bw} {bh}")
                    n_bad += ss == SS_BAD
                    n_good += ss == SS_GOOD
            shutil.copy(img_path, out_img / img_path.name)
            (out_lab / lab_path.name).write_text("\n".join(new_lines) + "\n")

    YAML_OUT.write_text(
        f"path: {OUT_ROOT}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: module-good\n  1: module-bad\n  2: busbar\n"
    )
    print(f"3-class dataset → {OUT_ROOT.relative_to(ROOT)}")
    print(f"Instances: module-good={n_good}, module-bad={n_bad}, busbar={n_bus}")
    print(f"YOLO config → {YAML_OUT.name}")
    print("\nTrain the ablation baseline:")
    print(f"  yolo detect train data={YAML_OUT.name} model=yolov8n.pt epochs=100 imgsz=640")


def main():
    ap = argparse.ArgumentParser(description="Single-stage (3-class) ablation dataset builder")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Pre-annotate module conditions with the trained classifier")
    ap.add_argument("--build", action="store_true",
                    help="Emit 3-class dataset from the reviewed condition CSV")
    args = ap.parse_args()
    if args.bootstrap:
        bootstrap()
    elif args.build:
        build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
