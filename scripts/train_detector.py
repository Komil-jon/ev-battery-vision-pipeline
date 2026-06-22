"""
train_detector.py
=================
Two-stage YOLOv8n training pipeline replicating the paper:
  - Stage 1: Train from COCO weights on multi-class dataset (100 epochs, 640px)
  - Stage 2: Fine-tune with busbar recall-boost augmentation (30 epochs, 768px, AdamW)

Usage:
    python scripts/train_detector.py --stage 1
    python scripts/train_detector.py --stage 2
    python scripts/train_detector.py --stage all
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "dataset.yaml"
MODELS_DIR   = ROOT / "models" / "detector"
STAGE1_NAME  = "stage1"
STAGE2_NAME  = "stage2_recall_boost"


def train_stage1():
    """
    Stage 1 — cold-start training from COCO-pretrained YOLOv8n weights.
    Matches paper Table 2: 640px, 100 epochs, SGD-like defaults.
    """
    print("\n=== STAGE 1: Multi-class detector training ===")
    model = YOLO("yolov8n.pt")          # downloads on first run

    model.train(
        data        = str(DATASET_YAML),
        epochs      = 100,
        imgsz       = 640,
        batch       = 16,
        optimizer   = "SGD",
        lr0         = 0.01,
        weight_decay= 0.0005,
        # ── safe augmentation (paper Section 3.3) ──
        hsv_v       = 0.30,             # brightness variation
        hsv_s       = 0.25,             # saturation variation
        mosaic      = 0.5,              # reduced mosaic
        degrees     = 2.0,              # ±2° rotation
        translate   = 0.06,             # 6% translation
        fliplr      = 0.5,              # horizontal flip
        # ── colour jitter excluded (would corrupt condition cues) ──
        hsv_h       = 0.0,
        device      = "cpu",            # Apple M1 CPU – matches paper
        project     = str(MODELS_DIR),
        name        = STAGE1_NAME,
        exist_ok    = True,
        verbose     = True,
    )
    print(f"\nStage 1 complete. Weights at: {MODELS_DIR}/{STAGE1_NAME}/weights/best.pt")


def train_stage2():
    """
    Stage 2 — fine-tune converged Stage 1 checkpoint.
    Matches paper Table 2: 768px, 30 epochs, AdamW, busbar recall-boost dataset.
    Requires Stage 1 to have run first.
    """
    checkpoint = MODELS_DIR / STAGE1_NAME / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {checkpoint}.\n"
            "Run Stage 1 first:  python scripts/train_detector.py --stage 1"
        )

    print("\n=== STAGE 2: Busbar recall-boost fine-tuning ===")
    print(f"Loading checkpoint: {checkpoint}")
    model = YOLO(str(checkpoint))

    model.train(
        data         = str(DATASET_YAML),   # now includes busbar-augmented training set
        epochs       = 30,
        imgsz        = 768,                 # increased for thin busbars
        batch        = 8,                   # smaller batch at higher resolution
        optimizer    = "AdamW",             # preserves Stage 1 representations
        lr0          = 0.001,
        lrf          = 0.00001,             # cosine decay to low floor
        weight_decay = 0.0005,
        mosaic       = 0.5,
        patience     = 8,                    # early stopping (paper Table 2 / Appendix E3)
        # ── same safe augments, no hue jitter ──
        hsv_v        = 0.30,
        hsv_s        = 0.25,
        degrees      = 2.0,
        translate    = 0.06,
        fliplr       = 0.5,
        hsv_h        = 0.0,
        device       = "cpu",
        project      = str(MODELS_DIR),
        name         = STAGE2_NAME,
        exist_ok     = True,
        verbose      = True,
    )
    print(f"\nStage 2 complete. Weights at: {MODELS_DIR}/{STAGE2_NAME}/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train EV battery YOLOv8n detector")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Which training stage to run (default: all)"
    )
    args = parser.parse_args()

    if args.stage in ("1", "all"):
        train_stage1()
    if args.stage in ("2", "all"):
        train_stage2()


if __name__ == "__main__":
    main()
