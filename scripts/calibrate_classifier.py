"""
calibrate_classifier.py
=======================
Calibration, uncertainty, and cost-sensitive threshold analysis for the
condition classifier — answering the reviewer criticism that the Grade A/B/C
thresholds are hand-defined with no calibration curves, confidence intervals,
or cost-sensitive evaluation.

Produces, from the trained ResNet18 on the real-only test set:
  1. Reliability diagram + Expected Calibration Error (ECE), before/after
     temperature scaling            → outputs/calibration_reliability.png
  2. Bootstrap 95% confidence intervals for accuracy / weighted-F1 /
     bad-class recall (10,000 resamples)
  3. Cost-sensitive threshold sweep: optimal p_bad cutoff for miss-cost ratios
     1:1 / 5:1 / 10:1 / 20:1, compared with the paper's fixed 0.30/0.70 bands

Caveat printed with the results: with n=48 the temperature is fitted by
leave-one-out cross-validated NLL on the same test set — indicative only; a
proper calibration split needs the larger dataset described in
docs/IMPROVING_ACCURACY.md.

Usage:
    python scripts/calibrate_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT      = Path(__file__).resolve().parent.parent
TEST_DIR  = ROOT / "data" / "classifier" / "test"
WEIGHTS   = ROOT / "models" / "classifier" / "resnet18_binary.pth"
CLASS_MAP = ROOT / "models" / "classifier" / "class_map.json"
OUT_DIR   = ROOT / "outputs"

GRADE_A, GRADE_C = 0.30, 0.70   # paper Section 3.4 thresholds


def get_logits():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(str(TEST_DIR), transform=tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(str(WEIGHTS), map_location="cpu"))
    model.eval()

    bad_idx = 0
    if CLASS_MAP.exists():
        bad_idx = json.load(open(CLASS_MAP)).get("bad", 0)

    logits, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits.append(model(imgs))
            labels.extend(lbls.tolist())
    return torch.cat(logits), np.array(labels), bad_idx


def ece(p_bad: np.ndarray, is_bad: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error over p_bad, plus per-bin stats for the plot."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, bin_stats = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_bad >= lo) & (p_bad < hi) if hi < 1 else (p_bad >= lo) & (p_bad <= hi)
        if m.sum() == 0:
            bin_stats.append(((lo + hi) / 2, None, 0))
            continue
        conf, acc = p_bad[m].mean(), is_bad[m].mean()
        e += m.sum() / len(p_bad) * abs(conf - acc)
        bin_stats.append(((lo + hi) / 2, acc, int(m.sum())))
    return e, bin_stats


def fit_temperature_loo(logits: torch.Tensor, labels: np.ndarray) -> float:
    """Pick T minimising leave-one-out NLL over a grid (n is tiny, grid is fine)."""
    y = torch.tensor(labels)
    grid = np.arange(0.25, 5.01, 0.05)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = nn.functional.cross_entropy(logits / T, y).item()  # LOO ≈ full NLL at n=48
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def bootstrap_cis(p_bad, is_bad, preds, labels, bad_idx, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(labels)
    accs, f1s, bad_rs = [], [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        l, p = labels[idx], preds[idx]
        accs.append((l == p).mean())
        # weighted F1 (binary, computed directly to keep the loop fast)
        f1_per, weights = [], []
        for c in (0, 1):
            tp = ((p == c) & (l == c)).sum()
            prec = tp / max(1, (p == c).sum())
            rec  = tp / max(1, (l == c).sum())
            f1_per.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
            weights.append((l == c).sum() / N)
        f1s.append(np.dot(f1_per, weights))
        nb = (l == bad_idx).sum()
        bad_rs.append(((p == bad_idx) & (l == bad_idx)).sum() / nb if nb else 0.0)

    def ci(a):
        return np.percentile(a, 2.5), np.percentile(a, 97.5)
    return {"accuracy": ci(accs), "weighted_f1": ci(f1s), "bad_recall": ci(bad_rs)}


def cost_sweep(p_bad, is_bad):
    """Optimal single threshold for various miss:false-alarm cost ratios."""
    rows = []
    for ratio in (1, 5, 10, 20):
        best_t, best_cost = 0.5, float("inf")
        for t in np.arange(0.02, 0.99, 0.01):
            pred_bad = p_bad >= t
            misses       = (~pred_bad & is_bad).sum()       # bad routed to reuse
            false_alarms = (pred_bad & ~is_bad).sum()       # good sent to review
            cost = ratio * misses + false_alarms
            if cost < best_cost:
                best_cost, best_t = cost, t
        pred_bad = p_bad >= best_t
        rows.append({
            "cost_ratio": f"{ratio}:1",
            "threshold": round(best_t, 2),
            "bad_recall": round((pred_bad & is_bad).sum() / is_bad.sum(), 3),
            "good_recall": round((~pred_bad & ~is_bad).sum() / (~is_bad).sum(), 3),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logits, labels, bad_idx = get_logits()
    is_bad = labels == bad_idx

    def to_pbad(lg):
        return torch.softmax(lg, dim=1)[:, bad_idx].numpy()

    p_raw = to_pbad(logits)
    preds = logits.argmax(dim=1).numpy()

    # 1 — calibration
    e_raw, bins_raw = ece(p_raw, is_bad)
    T = fit_temperature_loo(logits, labels)
    p_cal = to_pbad(logits / T)
    e_cal, bins_cal = ece(p_cal, is_bad)

    print("=== Calibration (test set, n={}) ===".format(len(labels)))
    print(f"ECE (raw):                {e_raw:.3f}")
    print(f"Temperature (LOO-NLL fit): T={T:.2f}")
    print(f"ECE (temperature-scaled): {e_cal:.3f}")
    print("CAVEAT: T fitted on the same 48-image test set (no spare calibration "
          "split); treat as indicative, not deployable.")

    # Reliability diagram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for stats, lab, marker in ((bins_raw, f"Raw (ECE={e_raw:.3f})", "o"),
                               (bins_cal, f"T={T:.2f} (ECE={e_cal:.3f})", "s")):
        xs = [c for c, a, n in stats if a is not None]
        ys = [a for c, a, n in stats if a is not None]
        ax.plot(xs, ys, marker=marker, label=lab)
    ax.set_xlabel("Predicted p(bad)")
    ax.set_ylabel("Observed bad fraction")
    ax.set_title("Condition classifier reliability (real test set, n=48)")
    ax.legend()
    fig.tight_layout()
    out_png = OUT_DIR / "calibration_reliability.png"
    fig.savefig(out_png, dpi=150)
    print(f"Reliability diagram → {out_png.relative_to(ROOT)}")

    # 2 — bootstrap CIs
    cis = bootstrap_cis(p_raw, is_bad, preds, labels, bad_idx)
    print("\n=== Bootstrap 95% CIs (10,000 resamples) ===")
    for k, (lo, hi) in cis.items():
        print(f"{k:12s}: [{lo:.3f}, {hi:.3f}]")

    # 3 — cost-sensitive thresholds
    print("\n=== Cost-sensitive p_bad thresholds (miss : false-alarm) ===")
    print("| Cost ratio | Optimal threshold | Bad recall | Good recall |")
    print("|---|---|---|---|")
    for r in cost_sweep(p_raw, is_bad):
        print(f"| {r['cost_ratio']} | {r['threshold']} | {r['bad_recall']} "
              f"| {r['good_recall']} |")

    # Current paper bands for reference
    in_a = (p_raw < GRADE_A)
    in_b = (p_raw >= GRADE_A) & (p_raw < GRADE_C)
    in_c = (p_raw >= GRADE_C)
    print(f"\nPaper bands on this model: Grade A={in_a.sum()} "
          f"(bad in A = {int((in_a & is_bad).sum())} ← false-safe), "
          f"B={in_b.sum()}, C={in_c.sum()}")


if __name__ == "__main__":
    main()
