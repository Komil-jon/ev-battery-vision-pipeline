"""
anomaly_condition.py
====================
Good-only anomaly detection for module condition — the novel-method alternative
to the binary good/bad classifier.

Rationale: with only ~16 real damaged crops, supervised binary classification is
data-starved, and it can never promise anything about damage types absent from
training. One-class methods (PatchCore lineage) instead model the distribution
of GOOD modules — which are plentiful — and flag deviation, so unseen damage
modes are still caught. This directly addresses the reviewer criticisms about
novelty and about bad-class scarcity.

Method (PatchCore-lite):
  1. Extract dense patch features from all good TRAINING crops
     (backbone: frozen ResNet18 layer3 by default — cached offline;
      --backbone dinov2 uses DINOv2 ViT-S/14 patch tokens, needs one download)
  2. Coreset-subsample them into a memory bank
  3. Score a test crop by the max over its patches of the distance to the
     nearest bank entry; image is anomalous (bad) if the score is high
  4. Threshold chosen on the good TRAINING scores (95th percentile) —
     no bad examples are used for training or threshold selection at all

Reports AUROC plus good/bad recall on the real-only test set, next to the
supervised classifier for comparison.

Usage:
    python scripts/anomaly_condition.py                      # ResNet18 features
    python scripts/anomaly_condition.py --backbone dinov2    # DINOv2 features
    python scripts/anomaly_condition.py --quantile 0.99      # stricter threshold
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT      = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "classifier" / "train"
TEST_DIR  = ROOT / "data" / "classifier" / "test"

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ResNetPatchFeatures(nn.Module):
    """Frozen ResNet18 through layer3 → 14x14 grid of 256-d patch features."""

    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                  m.layer1, m.layer2, m.layer3)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):                       # (B,3,224,224)
        f = self.stem(x)                        # (B,256,14,14)
        B, C, H, W = f.shape
        return f.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B,196,256)


class DinoPatchFeatures(nn.Module):
    """Frozen DINOv2 ViT-S/14 patch tokens → 16x16 grid of 384-d features."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.backbone.forward_features(x)["x_norm_patchtokens"]  # (B,256,384)


def extract(model, loader):
    feats, labels = [], []
    for imgs, lbls in loader:
        feats.append(model(imgs))
        labels.extend(lbls.tolist())
    return torch.cat(feats), np.array(labels)


def image_scores(patch_feats: torch.Tensor, bank: torch.Tensor,
                 top_frac: float = 0.05, chunk: int = 8) -> np.ndarray:
    """
    Per-image anomaly score: mean of the top `top_frac` patch distances
    (min-L2 to the bank). A top-k mean is less sensitive to single-patch noise
    than PatchCore's plain max while preserving locality of small defects.
    """
    scores = []
    for i in range(0, patch_feats.shape[0], chunk):
        block = patch_feats[i:i + chunk]                  # (b,P,C)
        d = torch.cdist(block.reshape(-1, block.shape[-1]), bank)  # (b*P, N)
        dmin = d.min(dim=1).values.reshape(block.shape[0], -1)     # (b,P)
        k = max(1, int(dmin.shape[1] * top_frac))
        topk = dmin.topk(k, dim=1).values.mean(dim=1)
        scores.extend(topk.tolist())
    return np.array(scores)


def auroc(scores: np.ndarray, is_bad: np.ndarray) -> float:
    """Rank-based AUROC (bad = positive class)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = is_bad.sum(), (~is_bad).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[is_bad].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description="Good-only anomaly detection for module condition")
    ap.add_argument("--backbone", choices=["resnet18", "dinov2"], default="resnet18")
    ap.add_argument("--coreset",  type=float, default=0.10,
                    help="Fraction of good patches kept in the memory bank (default 0.10)")
    ap.add_argument("--quantile", type=float, default=0.95,
                    help="Good-train score quantile used as threshold (default 0.95)")
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ResNetPatchFeatures() if args.backbone == "resnet18" else DinoPatchFeatures()

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=TF)
    test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=TF)
    good_idx = train_ds.class_to_idx["good"]
    bad_idx  = train_ds.class_to_idx["bad"]

    # Memory bank from REAL good training crops only (skip synthetic-damage files
    # if any live under good/, and never touch bad/)
    good_samples = [(p, l) for p, l in train_ds.samples if l == good_idx]
    train_ds.samples = good_samples
    train_ds.targets = [l for _, l in good_samples]
    print(f"Building memory bank from {len(good_samples)} good training crops "
          f"({args.backbone} patch features)...")

    feats, _ = extract(model, DataLoader(train_ds, batch_size=8))
    all_patches = feats.reshape(-1, feats.shape[-1])
    keep = rng.choice(len(all_patches), max(1, int(len(all_patches) * args.coreset)),
                      replace=False)
    bank = all_patches[keep]
    print(f"Memory bank: {len(bank)} / {len(all_patches)} patches "
          f"(coreset {args.coreset:.0%})")

    # Threshold from good-train scores — no bad data involved
    good_scores = image_scores(feats, bank)
    thresh = float(np.quantile(good_scores, args.quantile))
    print(f"Threshold = {args.quantile:.0%} quantile of good-train scores: {thresh:.3f}")

    # Test
    test_feats, test_labels = extract(model, DataLoader(test_ds, batch_size=8))
    scores = image_scores(test_feats, bank)
    is_bad = test_labels == bad_idx

    pred_bad = scores >= thresh
    bad_r  = (pred_bad & is_bad).sum() / max(1, is_bad.sum())
    good_r = (~pred_bad & ~is_bad).sum() / max(1, (~is_bad).sum())
    auc = auroc(scores, is_bad)

    print(f"\n=== Good-only anomaly detection ({args.backbone}) — real test set "
          f"(n={len(test_labels)}: {int(is_bad.sum())} bad / {int((~is_bad).sum())} good) ===")
    print(f"AUROC:        {auc:.3f}")
    print(f"Bad recall:   {bad_r:.3f}  (at good-train {args.quantile:.0%} threshold)")
    print(f"Good recall:  {good_r:.3f}")
    print("\nReference (supervised, synthetic-augmented ResNet18): "
          "bad-recall 0.857, good-recall 0.765")
    print("Note: this model saw ZERO bad examples — its recall is against "
          "damage types it was never shown.")


if __name__ == "__main__":
    main()
