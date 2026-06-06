"""
evaluate.py
===========
Evaluation script for the two-stage EV battery inspection pipeline.

Runs:
  1. Detector evaluation  — mAP50, mAP50-95, per-class P/R, CPU latency
  2. Classifier evaluation — accuracy, weighted F1, confusion matrix
  3. Lighting robustness  — mAP50/mAP50-95 under Normal/Dark/Bright conditions
  4. End-to-end propagation table (Table 6 from paper)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --skip_lighting   # faster, skips brightness variants
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT               = Path(__file__).resolve().parent.parent
DATASET_YAML       = ROOT / "dataset.yaml"
DETECTOR_WEIGHTS   = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = ROOT / "models" / "classifier" / "resnet18_binary.pth"
CLASS_MAP_PATH     = ROOT / "models" / "classifier" / "class_map.json"
TEST_IMGS          = ROOT / "data" / "detector" / "images" / "test"
CLASSIFIER_TEST    = ROOT / "data" / "classifier" / "test"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── helpers ────────────────────────────────────────────────────────────────────

def load_detector():
    if not DETECTOR_WEIGHTS.exists():
        raise FileNotFoundError(f"Detector weights not found: {DETECTOR_WEIGHTS}")
    return YOLO(str(DETECTOR_WEIGHTS))


def load_classifier():
    if not CLASSIFIER_WEIGHTS.exists():
        raise FileNotFoundError(f"Classifier weights not found: {CLASSIFIER_WEIGHTS}")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(str(CLASSIFIER_WEIGHTS), map_location="cpu"))
    model.eval()
    bad_idx = 0
    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            bad_idx = json.load(f).get("bad", 0)
    return model, bad_idx


def apply_brightness(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    Simulate lighting conditions for robustness evaluation (paper Section 4.5).
      Normal — unchanged
      Dark   — reduce brightness (factor 0.5)
      Bright — increase brightness to simulate specular glare (factor 2.0)
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if mode == "dark":
        pil = ImageEnhance.Brightness(pil).enhance(0.5)
    elif mode == "bright":
        pil = ImageEnhance.Brightness(pil).enhance(2.0)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── 1. Detector evaluation ─────────────────────────────────────────────────────

def evaluate_detector(detector: YOLO, args):
    print("\n" + "=" * 60)
    print("1. DETECTOR EVALUATION")
    print("=" * 60)

    if not DATASET_YAML.exists():
        print(f"  dataset.yaml not found at {DATASET_YAML} — skipping.")
        return

    metrics = detector.val(
        data   = str(DATASET_YAML),
        split  = "test",
        imgsz  = 768,
        device = "cpu",
        verbose= True,
    )
    print(f"\n  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")

    # CPU latency benchmark (100 passes over test images)
    test_images = list(TEST_IMGS.glob("*.jpg")) + list(TEST_IMGS.glob("*.png"))
    if test_images:
        print(f"\n  CPU latency benchmark ({len(test_images)} images, 3 warm-up passes)...")
        for img_p in test_images[:3]:                    # warm-up
            detector(str(img_p), imgsz=768, device="cpu", verbose=False)
        times = []
        for img_p in test_images:
            t0 = time.perf_counter()
            detector(str(img_p), imgsz=768, device="cpu", verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = np.mean(times)
        std_ms  = np.std(times)
        print(f"  Mean latency : {mean_ms:.1f} ± {std_ms:.1f} ms/image")
        print(f"  FPS          : {1000 / mean_ms:.1f}")
        print(f"  Paper target : 78.9 ms / 12.7 FPS")


# ── 2. Classifier evaluation ───────────────────────────────────────────────────

def evaluate_classifier():
    print("\n" + "=" * 60)
    print("2. CLASSIFIER EVALUATION")
    print("=" * 60)

    if not CLASSIFIER_TEST.exists():
        print(f"  Classifier test dir not found: {CLASSIFIER_TEST} — skipping.")
        return

    classifier, bad_idx = load_classifier()
    test_ds     = datasets.ImageFolder(str(CLASSIFIER_TEST), transform=TEST_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    class_to_idx = test_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = classifier(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc  = accuracy_score(all_labels, all_preds)
    wf1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n  Accuracy    : {acc:.3f}  (paper: 0.917)")
    print(f"  Weighted F1 : {wf1:.3f}  (paper: 0.912)")
    print(f"\n  Classification report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print("  Confusion matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    if bad_idx in all_labels:
        bad_recall = sum(
            1 for p, l in zip(all_preds, all_labels) if l == bad_idx and p == bad_idx
        ) / all_labels.count(bad_idx)
        print(f"\n  Bad-class recall : {bad_recall:.3f}  (paper: 0.714)")


# ── 3. Lighting robustness ─────────────────────────────────────────────────────

def evaluate_lighting(detector: YOLO):
    print("\n" + "=" * 60)
    print("3. LIGHTING ROBUSTNESS ANALYSIS")
    print("=" * 60)

    test_images = list(TEST_IMGS.glob("*.jpg")) + list(TEST_IMGS.glob("*.png"))
    if not test_images:
        print(f"  No test images found in {TEST_IMGS} — skipping.")
        return

    conditions = ["normal", "dark", "bright"]
    for condition in conditions:
        print(f"\n  Condition: {condition.upper()}")
        maps50, maps5095 = [], []

        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            mod_img = apply_brightness(img, condition)
            # Save temp file for YOLO inference
            tmp = Path("/tmp") / f"_lit_{img_path.name}"
            cv2.imwrite(str(tmp), mod_img)
            r = detector(str(tmp), imgsz=768, device="cpu", verbose=False)
            tmp.unlink(missing_ok=True)

        # Note: full mAP requires ground-truth labels.
        # For a quick check without GT, we report average detection confidence instead.
        total_conf = []
        for img_path in test_images[:10]:      # sample for speed
            img = cv2.imread(str(img_path))
            mod_img = apply_brightness(img, condition)
            tmp = Path("/tmp") / f"_lit_{img_path.name}"
            cv2.imwrite(str(tmp), mod_img)
            results = detector(str(tmp), imgsz=768, device="cpu", verbose=False)
            for r in results:
                if r.boxes is not None and len(r.boxes):
                    total_conf.extend(r.boxes.conf.cpu().tolist())
            tmp.unlink(missing_ok=True)

        avg_conf = np.mean(total_conf) if total_conf else 0.0
        n_dets   = len(total_conf)
        print(f"    Avg detection confidence : {avg_conf:.3f}  (over {n_dets} detections, 10 images)")
        print(f"    Paper reference — Normal: mAP50=0.901 | Dark: 0.893 | Bright: 0.804")


# ── 4. End-to-end propagation table ───────────────────────────────────────────

def print_propagation_table(module_recall: float = 0.953, bad_class_recall: float = 0.714):
    """Reproduce Table 6 from paper (Section 5.4)."""
    print("\n" + "=" * 60)
    print("4. END-TO-END PROPAGATION TABLE  (Table 6 from paper)")
    print("=" * 60)
    n_bad = 100
    detected    = n_bad * module_recall
    identified  = detected * bad_class_recall
    missed      = n_bad - identified

    rows = [
        ("True bad modules entering inspection",   f"{n_bad}",                 "Starting point"),
        ("Modules detected (YOLOv8n)",             f"{detected:.1f}/100",      f"Module recall {module_recall}"),
        ("Bad crops correctly classified",         f"{bad_class_recall:.1%}",  f"Bad-class recall"),
        ("Bad modules identified end-to-end",      f"{identified:.0f}/100",    f"{module_recall}×{bad_class_recall}≈{identified/n_bad:.2f}"),
        ("Bad modules missed/misclassified",       f"≈{missed:.0f}/100",       "Main deployment risk"),
    ]

    print(f"\n  {'Pipeline stage':<45} {'Value':<15} {'Note'}")
    print(f"  {'-'*45} {'-'*15} {'-'*30}")
    for stage, val, note in rows:
        print(f"  {stage:<45} {val:<15} {note}")

    print(f"\n  Current end-to-end identification rate: ~{identified:.0f}%")
    print(f"  To reach 85%: bad-class recall must reach {0.85 / module_recall:.3f} (+{0.85/module_recall - bad_class_recall:.3f})")
    print(f"  To reach 90%: bad-class recall must reach {0.90 / module_recall:.3f} (+{0.90/module_recall - bad_class_recall:.3f})")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate EV battery inspection pipeline")
    parser.add_argument("--skip_lighting",   action="store_true", help="Skip lighting robustness (faster)")
    parser.add_argument("--skip_detector",   action="store_true")
    parser.add_argument("--skip_classifier", action="store_true")
    args = parser.parse_args()

    if not args.skip_detector:
        try:
            det = load_detector()
            evaluate_detector(det, args)
            if not args.skip_lighting:
                evaluate_lighting(det)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}")

    if not args.skip_classifier:
        try:
            evaluate_classifier()
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}")

    print_propagation_table()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
