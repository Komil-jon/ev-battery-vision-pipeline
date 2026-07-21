# Post-Submission Changelog

Running log of every change made to this project **after** the BMVC 2026
submission (#1674). Newest entries at the top. Each entry: what changed, why,
and the measured result where relevant. Maintained across work sessions.

> This file is updated continuously. When new work is done, add a dated entry
> here as part of the same change — do not let the log fall behind the code.

---

## 2026-07-21

### Detector Stage 2 retrain with copy-paste + glare augmentation
- Retrained the Stage 2 detector on the training split enlarged with 300
  copy-paste composites + 200 synthetic-glare images (all derived from existing
  training data; test split untouched).
- **Result (real held-out test set, 43 images / 324 instances):**
  mAP50 **0.818 → 0.882**, mAP50-95 **0.557 → 0.634**.
- Note: gain is on the same-distribution test set (in-distribution improvement);
  cross-facility generalization is measured separately by the cross-variant probe.

### Zero-shot cross-variant generalization probe (`scripts/eval_cross_variant.py`)
- Runs the detector on each of the 19 Zenodo pack variants and reports detection
  rate / mean confidence / module-vs-busbar counts. A recall-oriented
  generalization *proxy* (Zenodo images are unlabelled → no mAP).
- Purpose: answer the reviewers' single-facility criticism with out-of-distribution
  evidence. Runs against the retrained detector once training completes.

### Zenodo dataset downloaded + inspected (generalization data)
- Downloaded "Battery Image Dataset for EV Circularity" (DOI 10.5281/zenodo.19818270,
  CC BY 4.0), 712 images / 19 vehicle types, 5.35 GB.
- **Key finding:** no bounding-box labels (classification dataset), and ~158
  images (Tesla Model 3 / Model S black) are re-used from the MTech training
  source — using them as "unseen" would be leakage. ~554 images across 17
  genuinely-new pack types are the usable part.
- gitignored `data/external/`, `data/youtube_harvest/`, `data/detector_singlestage/`,
  `data/classifier/bad_synth/` so large/generated data never enters the repo.

### DINOv2 + good-only anomaly detection (novel-method track)
- **DINOv2 ViT-S/14 frozen linear probe** added to the classifier benchmark:
  accuracy 0.792 / wF1 0.792, beating the ResNet18 backbone (0.688 / 0.684)
  under the identical protocol.
- **Good-only anomaly detector** (`scripts/anomaly_condition.py`, PatchCore-lite):
  with DINOv2 patch features, AUROC 0.702 and **bad-recall 0.857 using ZERO
  damaged training examples** — matches the supervised model, and by construction
  generalizes to unseen damage types. This is the candidate novel-method
  contribution answering the "no novelty" criticism.
- `scripts/harvest_youtube_frames.py`: sample + pseudo-label frames from
  CC-licensed EV battery videos into a human-review queue (new pack variants).

## 2026-07-20

### Reviewer-response tooling
- `docs/REVIEWER_RESPONSE_PLAN.md`: maps all four BMVC reviews' criticisms to
  concrete repo actions with status, plus measured-results tables.
- `scripts/benchmark_classifiers.py`: same-protocol comparison of ResNet18 /
  MobileNetV3-Small / EfficientNet-B0 / ShuffleNetV2 (later + DINOv2).
  EfficientNet-B0 and DINOv2 lead; answers "why ResNet18?".
- `scripts/benchmark_detector_cpu.py`: PyTorch vs ONNX FP32 vs ONNX INT8.
  **INT8 cuts latency 20% and size 46% for -0.007 mAP50** (44.8 ms, 3.4 MB).
- `scripts/calibrate_classifier.py`: reliability diagram + ECE (0.284 → 0.246
  via temperature scaling), bootstrap 95% CIs, cost-sensitive thresholds.
  Key finding: **zero bad modules land in Grade A** (no false-safe routing).
- `scripts/build_singlestage_dataset.py`: 3-class (module-good / module-bad /
  busbar) ablation dataset builder with classifier-bootstrapped pre-annotation,
  for the single-stage vs two-stage comparison reviewers asked for.

### Classifier bad-recall improvement (synthetic damaged crops)
- Expanded the classifier bad class 40 → 80 crops with procedural damage
  synthesis (`scripts/synth_damage_overlay.py`; synthetic ≤50% of the class;
  test set 100% real).
- **Result:** bad-class recall **0.571 → 0.857** (exceeds the paper's 0.714),
  weighted F1 0.768 → 0.800, accuracy 0.771 → 0.792. Retrained weights shipped.

### Accuracy/generalization tooling + guide
- `docs/IMPROVING_ACCURACY.md`: curated external-dataset catalogue + synthetic
  data method guide.
- `scripts/download_external_datasets.py`: Zenodo / Roboflow download + class-
  remap merge.
- `scripts/synth_copy_paste.py`: copy-paste compositing (+ synthetic glare pass)
  for the detector, guarded against un-remapped 7-class labels.
- `scripts/synth_damage_overlay.py`: procedural corrosion/burn/scratch synthesis.
- `notebooks/colab_defect_inpainting.ipynb`: few-shot SD-inpainting LoRA pipeline
  for diffusion-generated damaged crops (free Colab GPU).

## 2026-07-14

### Repository cleanup + public release
- Removed the tracked raw dataset (`EV-Battery-pack--1/`, `data/`) and the
  `.claude/` agent config; kept trained weights, scripts, and README.
- Repo reduced from 4,786 tracked files to 69; pushed to GitHub and renamed to
  **ev-battery-vision-pipeline** for public discoverability.
