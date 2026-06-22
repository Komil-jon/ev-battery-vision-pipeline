"""
train_classifier.py
===================
Trains a ResNet18 binary condition classifier (good / bad) for EV battery module crops.

Architecture decisions from paper (Section 3.4):
  - ImageNet-pretrained backbone, FROZEN during training
  - Only the final FC layer is trained (prevents overfitting on small dataset)
  - Binary output: class 0 = bad, class 1 = good  (folder order from ImageFolder)
  - Grade A/B/C triage via bad-class probability thresholds (not a separate model)

Dataset structure expected:
    data/classifier/
        train/
            good/   ← visually good module crops
            bad/    ← visibly damaged module crops
        test/
            good/
            bad/

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 30 --batch 8
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
TRAIN_DIR      = ROOT / "data" / "classifier" / "train"
TEST_DIR       = ROOT / "data" / "classifier" / "test"
SAVE_DIR       = ROOT / "models" / "classifier"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH   = SAVE_DIR / "resnet18_binary.pth"
CLASS_MAP_PATH = SAVE_DIR / "class_map.json"


# ── transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # safe for classifier
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes: int = 2) -> nn.Module:
    """
    ResNet18 with frozen backbone and trainable classification head.
    Paper rationale: insufficient data to fine-tune convolutions without overfitting.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False                 # freeze backbone
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # new trainable head
    return model


def train(args):
    # ── data ──────────────────────────────────────────────────────────────────
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            "Create data/classifier/train/good/ and data/classifier/train/bad/"
        )

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TRANSFORMS)
    test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=TEST_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Save class index mapping so inference knows which index = bad
    class_to_idx = train_ds.class_to_idx   # e.g. {'bad': 0, 'good': 1}
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class map: {class_to_idx}")
    bad_idx = class_to_idx.get("bad", 0)
    print(f"Bad-class index: {bad_idx}")

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    model  = build_model(num_classes=2).to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Class-weighted loss: the bad (damaged) class is the minority and the most
    # costly to miss (a missed bad module is routed to reuse). Weight inversely
    # by class frequency so bad-class recall is not sacrificed to majority good.
    counts = [0, 0]
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    weights = torch.tensor(
        [len(train_ds) / (2 * c) if c > 0 else 0.0 for c in counts],
        dtype=torch.float32,
    )
    print(f"Class counts {counts} -> loss weights {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_f1    = 0.0
    best_epoch = 0

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(train_ds)

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc    = accuracy_score(all_labels, all_preds)
        wf1    = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        bad_r  = 0.0
        if bad_idx in all_labels:
            bad_r = sum(
                1 for p, l in zip(all_preds, all_labels) if l == bad_idx and p == bad_idx
            ) / all_labels.count(bad_idx)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | acc={acc:.3f} | "
            f"wF1={wf1:.3f} | bad-recall={bad_r:.3f}"
        )

        if wf1 > best_f1:
            best_f1    = wf1
            best_epoch = epoch
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  ✓ Saved best model (wF1={best_f1:.3f})")

    print(f"\nTraining complete. Best wF1={best_f1:.3f} at epoch {best_epoch}")
    print(f"Weights saved to: {WEIGHTS_PATH}")

    # ── final evaluation ──────────────────────────────────────────────────────
    print("\n=== Final evaluation on test set ===")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 binary condition classifier")
    parser.add_argument("--epochs", type=int,   default=20,    help="Number of epochs")
    parser.add_argument("--batch",  type=int,   default=16,    help="Batch size")
    parser.add_argument("--lr",     type=float, default=1e-3,  help="Learning rate")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
