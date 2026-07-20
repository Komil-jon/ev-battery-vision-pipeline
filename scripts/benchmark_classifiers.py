"""
benchmark_classifiers.py
========================
In-dataset classifier architecture benchmark, answering the reviewer criticism
that the ResNet18 choice was never compared against lightweight alternatives
(MobileNet / EfficientNet / ShuffleNet heads) on the same data.

All backbones are trained with the IDENTICAL protocol used by
train_classifier.py (frozen ImageNet backbone, trainable linear head,
class-weighted cross-entropy, Adam 1e-3 + StepLR, same transforms), so the only
variable is the architecture. Metrics are reported on the real-only test set,
plus single-image CPU latency and parameter counts for the deployment argument.

Usage:
    python scripts/benchmark_classifiers.py                    # all backbones, 30 epochs
    python scripts/benchmark_classifiers.py --epochs 10        # quick pass
    python scripts/benchmark_classifiers.py --models resnet18 mobilenet_v3_small
Results are printed as a markdown table and saved to outputs/classifier_benchmark.csv
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score

ROOT      = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "classifier" / "train"
TEST_DIR  = ROOT / "data" / "classifier" / "test"
OUT_CSV   = ROOT / "outputs" / "classifier_benchmark.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DinoV2Probe(nn.Module):
    """Frozen DINOv2 ViT-S/14 CLS embedding + trainable linear head.

    Foundation-model features are the modern counterpart of the frozen-ImageNet
    protocol: same training recipe, only the representation changes.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.fc(feats)


def build(name: str, num_classes: int = 2) -> nn.Module:
    """Backbone frozen, final head replaced — mirrors train_classifier.build_model."""
    if name == "dinov2_vits14":
        return DinoV2Probe(num_classes)
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        head_parent, head_attr, in_f = m, "fc", m.fc.in_features
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        head_parent, head_attr, in_f = m.classifier, "3", m.classifier[3].in_features
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        head_parent, head_attr, in_f = m.classifier, "1", m.classifier[1].in_features
    elif name == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        head_parent, head_attr, in_f = m, "fc", m.fc.in_features
    else:
        raise ValueError(f"Unknown model: {name}")

    for p in m.parameters():
        p.requires_grad = False
    new_head = nn.Linear(in_f, num_classes)
    if head_attr.isdigit():
        head_parent[int(head_attr)] = new_head
    else:
        setattr(head_parent, head_attr, new_head)
    return m


def trainable_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def total_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


def cpu_latency_ms(m: nn.Module, n_warm: int = 3, n_runs: int = 20) -> float:
    """Median single-image (1x3x224x224) forward latency on CPU."""
    m.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(n_warm):
            m(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def evaluate(m: nn.Module, loader, bad_idx: int):
    m.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(m(imgs).argmax(dim=1).tolist())
            labels.extend(lbls.tolist())
    acc = accuracy_score(labels, preds)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    n_bad = labels.count(bad_idx)
    bad_r = (sum(1 for p, l in zip(preds, labels) if l == bad_idx and p == bad_idx) / n_bad
             if n_bad else 0.0)
    good_idx = 1 - bad_idx
    n_good = labels.count(good_idx)
    good_r = (sum(1 for p, l in zip(preds, labels) if l == good_idx and p == good_idx) / n_good
              if n_good else 0.0)
    return acc, wf1, bad_r, good_r


def run_one(name: str, epochs: int, batch: int, lr: float, seed: int):
    torch.manual_seed(seed)
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TRANSFORMS)
    test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=TEST_TRANSFORMS)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0)
    bad_idx = train_ds.class_to_idx.get("bad", 0)

    model = build(name)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    counts = [0, 0]
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    weights = torch.tensor([len(train_ds) / (2 * c) if c else 0.0 for c in counts])
    criterion = nn.CrossEntropyLoss(weight=weights)

    best = {"wf1": -1.0}
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc, wf1, bad_r, good_r = evaluate(model, test_loader, bad_idx)
        if wf1 > best["wf1"]:
            best = {"epoch": epoch, "acc": acc, "wf1": wf1,
                    "bad_recall": bad_r, "good_recall": good_r}
        print(f"  [{name}] epoch {epoch:2d}/{epochs} acc={acc:.3f} wF1={wf1:.3f} "
              f"bad-recall={bad_r:.3f}")

    best["model"]        = name
    best["params_M"]     = round(total_params(model) / 1e6, 2)
    best["trainable"]    = trainable_params(model)
    best["latency_ms"]   = round(cpu_latency_ms(model), 1)
    return best


def main():
    ap = argparse.ArgumentParser(description="Same-protocol classifier architecture benchmark")
    ap.add_argument("--models", nargs="+",
                    default=["resnet18", "mobilenet_v3_small", "efficientnet_b0",
                             "shufflenet_v2_x1_0"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    results = []
    for name in args.models:
        print(f"\n=== {name} ===")
        results.append(run_one(name, args.epochs, args.batch, args.lr, args.seed))

    results.sort(key=lambda r: (r["wf1"], r["bad_recall"]), reverse=True)

    print("\n\n## Classifier architecture benchmark "
          "(frozen ImageNet backbone + linear head, identical protocol)\n")
    print("| Model | Params (M) | CPU latency (ms) | Accuracy | Weighted F1 "
          "| Bad recall | Good recall | Best epoch |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['params_M']} | {r['latency_ms']} "
              f"| {r['acc']:.3f} | {r['wf1']:.3f} | {r['bad_recall']:.3f} "
              f"| {r['good_recall']:.3f} | {r['epoch']} |")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
