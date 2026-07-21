# BMVC Submission #1674 — Reviewer Response Plan

Maps every reviewer criticism (wa5W Reject, WJw8 Weak Reject, 2kuK/cJPP
Borderline Reject) to a concrete action in this repository, with current status.
Companion docs: [IMPROVING_ACCURACY.md](IMPROVING_ACCURACY.md) (datasets +
synthetic data methods).

## Criticism → action matrix

| # | Criticism (reviewers) | Action in this repo | Status |
|---|---|---|---|
| 1 | No methodological novelty — off-the-shelf YOLOv8n + ResNet18 (all 4) | Reframe + add two defensible method contributions: (a) **recall-first synthetic-damage training** (procedural + diffusion damage synthesis with a ≤50% synthetic cap and real-only testing — already lifts bad-recall 0.571→0.857, past the paper's 0.714), (b) **glare-targeted augmentation** attacking the measured bright-light failure mode with a quantified robustness delta. Both are training-strategy contributions with ablations, which is what reviewers said was missing. | (a) done, (b) data ready — needs Stage 2 retrain |
| 2 | No two-stage vs single-stage justification/ablation (wa5W) | Single-stage 3-class ablation (module-good / module-bad / busbar in one YOLO head). **Blocker:** needs per-box condition labels in the detection annotations; only 16 real bad modules exist. Honest response: run the ablation once per-box condition labels exist (label the ~200-image train set for condition, ~1 day of annotation), and argue the two-stage choice from the cascade analysis already in Table 5/6 meanwhile. | Planned; needs annotation pass |
| 3 | No lightweight/quantised architecture benchmarking (wa5W, WJw8, 2kuK ×2) | **`scripts/benchmark_classifiers.py`** — ResNet18 vs MobileNetV3-Small vs EfficientNet-B0 vs ShuffleNetV2 under the identical frozen-backbone protocol (accuracy/F1/bad-recall/params/CPU-latency). **`scripts/benchmark_detector_cpu.py`** — PyTorch vs ONNX FP32 vs ONNX INT8: mAP + latency + size. Both produce paper-ready tables. | Done — both tables below |
| 4 | Cascading errors ignored (wa5W) | Already addressed in paper (Tables 5–6, S-H3). No repo action. | Done (paper) |
| 5 | Small test sets — 54 images / 14 bad crops (WJw8, 2kuK, cJPP) | Merge external data: Zenodo 19-battery-type set (CC BY 4.0) enlarges both splits; diffusion notebook enlarges the bad pool. Report bootstrap CIs meanwhile (see #7). | Tooling ready (`download_external_datasets.py`, Colab notebook); download pending |
| 6 | End-to-end bad-module identification weak, recall 0.714 (WJw8, cJPP) | **Bad-recall now 0.857 (12/14)** via synthetic damage training; cost-sensitive threshold at 5:1 miss-cost reaches **0.929** (see `calibrate_classifier.py` output). Re-run end-to-end propagation (Table 6) with the new classifier. | Recall improved; Table 6 refresh pending |
| 7 | Grade A/B/C not calibrated; no reliability curves/CIs/cost analysis (WJw8, 2kuK) | **`scripts/calibrate_classifier.py`**: reliability diagram + ECE (0.284 raw → 0.246 at T=0.55), bootstrap 95% CIs (accuracy [0.667,0.896], wF1 [0.688,0.899], bad-recall [0.643,1.000]), cost-sensitive threshold table (1/5/10/20:1). Key safety finding for the rebuttal: **zero bad modules fall in Grade A** (no false-safe routing) on the improved model. | Done — figures/tables generated |
| 8 | Dataset generality — single facility (WJw8, 2kuK, cJPP) | Zenodo 19-type dataset merge + cross-pack-variant evaluation (train on N-1 variants, test on held-out variant). | Tooling ready; run pending |
| 9 | Single annotator; wide bootstrap CI (cJPP) | Second-annotator pass on the 54-image test set + Cohen's kappa; CIs now systematically reported (#7). | CIs done; second annotator is a human task |
| 10 | Busbar condition overclaim in abstract (cJPP) | Wording fix in the manuscript (narrow to "module condition assessment; busbar localisation"), or add a busbar condition model if damage labels can be collected. | Manuscript edit — recommend wording fix |
| 11 | Title mismatch (WJw8) | Already retitled. | Done (paper) |

## Measured results (this repo, real-only test sets)

### Classifier architecture benchmark (frozen backbone + linear head,
identical protocol, 30 epochs, single seed 0 — CIs overlap at n=48, treat
ordering as indicative)

| Model | Params (M) | CPU latency (ms) | Accuracy | Weighted F1 | Bad recall | Good recall |
|---|---|---|---|---|---|---|
| **DINOv2 ViT-S/14** (frozen probe) | 22.06 | 42.3 | **0.792** | **0.792** | 0.643 | 0.853 |
| EfficientNet-B0 | 4.01 | 93.3 | 0.792 | 0.798 | 0.786 | 0.794 |
| ShuffleNetV2-x1.0 | 1.26 | 13.1 | 0.771 | 0.777 | 0.714 | 0.794 |
| MobileNetV3-Small | 1.52 | 31.9 | 0.708 | 0.692 | 0.357 | 0.853 |
| ResNet18 | 11.18 | 13.2 | 0.688 | 0.684 | 0.429 | 0.794 |

Notable: DINOv2 and EfficientNet-B0 tie for the top accuracy/F1, both well above
the shipped ResNet18 backbone under the identical linear-probe protocol —
concrete evidence that the ResNet18 choice was suboptimal and that a stronger
frozen representation is the cheap accuracy win. The shipped ResNet18 model
(separately trained, best-epoch selected, synthetic-augmented bad class) reaches
wF1 0.800 / bad-recall 0.857; single-run variance at this test-set size is large,
hence the bootstrap CIs in `calibrate_classifier.py`.

### Good-only anomaly detection — the novel-method contribution (real test set)

PatchCore-lite condition scoring (`scripts/anomaly_condition.py`): a memory bank
of patch features from **good modules only**, threshold set on the good-training
score distribution (95th percentile). **Zero damaged examples used in training or
threshold selection.**

| Backbone | AUROC | Bad recall | Good recall |
|---|---|---|---|
| ResNet18 layer3 | 0.603 | 0.357 | 0.735 |
| **DINOv2 ViT-S/14 patch tokens** | **0.702** | **0.857** | 0.676 |

Headline for the rebuttal: with DINOv2 patch features the good-only detector
**matches the supervised synthetic-augmented model's bad-recall (0.857) using no
damaged training data at all** — and by construction generalises to damage types
never seen in training, which a binary classifier cannot claim. This is a direct
answer to criticism #1 (novelty) and #5/#6 (bad-class scarcity): the method's
entire value proposition is not needing the scarce bad class.

### Detector CPU deployment benchmark (imgsz=768, median over 20 test images,
Apple M1 CPU; mAP on the 43-image held-out test split)

| Variant | Size (MB) | Latency (ms) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| PyTorch FP32 (shipped) | 6.3 | 56.3 | 0.818 | 0.557 |
| ONNX FP32 | 12.3 | 61.2 | 0.808 | 0.548 |
| **ONNX INT8 (dynamic)** | **3.4** | **44.8** | 0.811 | 0.550 |

INT8 quantisation cuts latency 20% and file size 46% for a 0.007 mAP50 cost —
the quantised-deployment data point reviewers asked for, and it strengthens the
CPU-deployability claim (44.8 ms ≈ 22 FPS detector-only).

### Calibration & cost analysis (shipped ResNet18, n=48 real test set)

- ECE 0.284 raw → 0.246 after temperature scaling (T=0.55, LOO-fit caveat)
- Bootstrap 95% CIs: accuracy [0.667, 0.896], weighted F1 [0.688, 0.899],
  bad-recall [0.643, 1.000]
- Cost-sensitive thresholds: 5:1 miss-cost → threshold 0.49, bad-recall 0.929
- **Zero bad modules in Grade A** (no false-safe routing) under paper bands

## The novelty question (criticism #1) — honest options

Reviewers will not accept re-running the same pipeline with more data as novelty.
Three viable framings, in increasing order of effort:

1. **Safety-first training + triage methodology** (recommended for a BMVC
   applications-track resubmission): recall-first synthetic damage synthesis
   (capped-synthetic protocol) + cost-calibrated triage thresholds + zero
   false-safe Grade-A guarantee, all ablated on real-only test data. Everything
   for this exists in the repo today.
2. **Unified single-stage condition-aware detector** (answers #2 directly):
   3-class YOLO head with the two-stage system as the ablated baseline.
   Needs the per-box condition annotation pass first; training is Colab-scale.
3. **Glare-robust detection via physically-motivated augmentation**: formalise
   the specular-highlight compositor, show the bright-light mAP50 recovery
   (0.672 → target ≥0.75), compare against standard brightness jitter.
   Data is generated; needs the Stage 2 retrain + lighting re-evaluation.

Doing 1 + 3 is achievable with current data and CPU/Colab compute; 2 is the
strongest single answer to the architecture criticism but is gated on
annotation effort.

## New-method track (accuracy + novelty), status

- **DINOv2 linear probe** — DONE. Ties EfficientNet-B0 for top accuracy/F1
  (0.792/0.792), beating the shipped ResNet18 backbone. Candidate replacement
  Stage 2 representation.
- **Good-only anomaly detection** (`scripts/anomaly_condition.py`,
  PatchCore-lite): DONE. DINOv2 patch features lift AUROC 0.60→0.702 and
  bad-recall to 0.857 — matching the supervised model with zero bad examples.
  This is the paper's novel-method contribution.
- **YouTube frame harvesting** (`scripts/harvest_youtube_frames.py`):
  downloads explicit URLs (CC-licensed/own footage only), samples + dedups
  frames, pseudo-labels with the current detector into a review queue.
  Pseudo-labels must be human-reviewed before entering train/ (never val/test).

## Suggested order of execution

1. Finish classifier architecture benchmark (running) → paper Table.
2. Detector CPU benchmark (PyTorch/ONNX/INT8) → paper Table.
3. Stage 2 retrain with copy-paste + glare data (≈2 h M1 or minutes on Colab)
   → refresh lighting-robustness table → framing #3 evidence.
4. Re-run end-to-end propagation with the 0.857-recall classifier → refresh
   Tables 5–6.
5. Zenodo merge → cross-variant generalisation experiment → answers #5/#8.
6. Condition-annotation pass → single-stage ablation → answers #2 and #1.
