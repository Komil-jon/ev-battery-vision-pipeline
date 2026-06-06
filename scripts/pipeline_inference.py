"""
pipeline_inference.py
=====================
Full two-stage inference pipeline replicating the paper system:
  Stage 1 — YOLOv8n detects module and busbar instances
  Stage 2 — ResNet18 classifies each module crop as good/bad
             Bad-class probability → Grade A / B / C triage output

Grade thresholds (paper Section 3.4 & Appendix F, Table F2):
  Grade A : p_bad < 0.30  → likely reusable
  Grade B : 0.30 ≤ p_bad < 0.70  → manual review required
  Grade C : p_bad ≥ 0.70  → likely damaged

Confidence threshold:
  Paper F1-confidence curve peaks at ~0.21 (Appendix C, Figure C2).
  Default here is 0.21 — adjust per facility cost-benefit analysis.

Usage:
    # Single image
    python scripts/pipeline_inference.py --input path/to/image.jpg

    # Folder of images
    python scripts/pipeline_inference.py --input path/to/folder/

    # Custom confidence threshold
    python scripts/pipeline_inference.py --input image.jpg --conf 0.30
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
DETECTOR_WEIGHTS = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = ROOT / "models" / "classifier" / "resnet18_binary.pth"
CLASS_MAP_PATH   = ROOT / "models" / "classifier" / "class_map.json"
OUTPUT_DIR       = ROOT / "outputs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── grade thresholds (paper Section 3.4) ──────────────────────────────────────
GRADE_A_THRESHOLD = 0.30
GRADE_C_THRESHOLD = 0.70

GRADE_COLOURS = {
    "A": (0, 200, 0),     # green  – likely reusable
    "B": (0, 165, 255),   # orange – manual review
    "C": (0, 0, 220),     # red    – likely damaged
}


# ── classifier transforms ──────────────────────────────────────────────────────
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_detector(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights_path}\n"
            "Run train_detector.py first."
        )
    return YOLO(str(weights_path))


def load_classifier(weights_path: Path, class_map_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}\n"
            "Run train_classifier.py first."
        )

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    bad_idx = 0  # default
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
        bad_idx = class_map.get("bad", 0)

    return model, bad_idx


def get_grade(p_bad: float) -> tuple[str, tuple]:
    if p_bad < GRADE_A_THRESHOLD:
        return "A", GRADE_COLOURS["A"]
    elif p_bad < GRADE_C_THRESHOLD:
        return "B", GRADE_COLOURS["B"]
    else:
        return "C", GRADE_COLOURS["C"]


def run_pipeline(
    image_path: Path,
    detector: YOLO,
    classifier,
    bad_idx: int,
    conf_threshold: float = 0.21,
    crop_padding: int = 10,
) -> dict:
    """
    Run the full two-stage pipeline on a single image.
    Returns annotated image (BGR) and timing/result metadata.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()
    results_meta = {
        "image":    image_path.name,
        "modules":  [],
        "busbars":  [],
        "det_ms":   0.0,
        "total_ms": 0.0,
    }

    t0 = time.perf_counter()

    # ── Stage 1: Detection ────────────────────────────────────────────────────
    det_results = detector(str(image_path), conf=conf_threshold, imgsz=768, device="cpu", verbose=False)
    det_ms = (time.perf_counter() - t0) * 1000
    results_meta["det_ms"] = round(det_ms, 1)

    for r in det_results:
        for box in r.boxes:
            cls_id  = int(box.cls)
            label   = detector.names[cls_id]
            det_conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "module":
                # ── Stage 2: Classification ───────────────────────────────────
                px1 = max(0, x1 - crop_padding)
                py1 = max(0, y1 - crop_padding)
                px2 = min(w, x2 + crop_padding)
                py2 = min(h, y2 + crop_padding)
                crop_bgr = img_bgr[py1:py2, px1:px2]

                if crop_bgr.size == 0:
                    continue

                pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                tensor   = CLASSIFIER_TRANSFORM(pil_crop).unsqueeze(0)

                with torch.no_grad():
                    probs = torch.softmax(classifier(tensor), dim=1)[0]
                    p_bad = probs[bad_idx].item()

                grade, colour = get_grade(p_bad)
                label_str = f"module | Grade {grade} | p_bad={p_bad:.2f} | d={det_conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    annotated, label_str,
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA
                )

                results_meta["modules"].append({
                    "bbox":   [x1, y1, x2, y2],
                    "det_conf": round(det_conf, 3),
                    "p_bad":  round(p_bad, 3),
                    "grade":  grade,
                })

            elif label == "busbar":
                colour = (255, 200, 0)  # cyan-ish
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    annotated, f"busbar {det_conf:.2f}",
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA
                )
                results_meta["busbars"].append({
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": round(det_conf, 3),
                })

    total_ms = (time.perf_counter() - t0) * 1000
    results_meta["total_ms"] = round(total_ms, 1)
    results_meta["fps"]      = round(1000 / total_ms, 1) if total_ms > 0 else 0

    return annotated, results_meta


def process_path(input_path: Path, detector, classifier, bad_idx, conf: float):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in image_extensions]
        print(f"Found {len(paths)} image(s) in {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    all_meta = []
    for img_path in paths:
        print(f"\nProcessing: {img_path.name}")
        try:
            annotated, meta = run_pipeline(img_path, detector, classifier, bad_idx, conf_threshold=conf)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Save output
        out_path = OUTPUT_DIR / f"pipeline_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        n_mod = len(meta["modules"])
        n_bus = len(meta["busbars"])
        grades = [m["grade"] for m in meta["modules"]]
        print(
            f"  Modules: {n_mod} | Busbars: {n_bus} | "
            f"Grades: {grades} | "
            f"Det: {meta['det_ms']}ms | Total: {meta['total_ms']}ms ({meta['fps']} FPS)"
        )
        print(f"  Saved → {out_path}")
        all_meta.append(meta)

    # Summary
    if len(all_meta) > 1:
        avg_total = sum(m["total_ms"] for m in all_meta) / len(all_meta)
        print(f"\n=== Summary: {len(all_meta)} images | avg pipeline: {avg_total:.1f}ms ===")


def main():
    parser = argparse.ArgumentParser(description="EV Battery two-stage inspection pipeline")
    parser.add_argument("--input",  type=str,   required=True, help="Image file or folder")
    parser.add_argument("--conf",   type=float, default=0.21,  help="Detection confidence threshold (paper optimum ~0.21)")
    parser.add_argument(
        "--detector",   type=str,
        default=str(DETECTOR_WEIGHTS),
        help="Path to detector .pt weights"
    )
    parser.add_argument(
        "--classifier", type=str,
        default=str(CLASSIFIER_WEIGHTS),
        help="Path to classifier .pth weights"
    )
    args = parser.parse_args()

    print("Loading models...")
    detector              = load_detector(Path(args.detector))
    classifier, bad_idx   = load_classifier(Path(args.classifier), CLASS_MAP_PATH)
    print(f"  Detector:   {args.detector}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Bad-class index: {bad_idx}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Grade thresholds: A<{GRADE_A_THRESHOLD} | B<{GRADE_C_THRESHOLD} | C≥{GRADE_C_THRESHOLD}")

    process_path(Path(args.input), detector, classifier, bad_idx, args.conf)


if __name__ == "__main__":
    main()
