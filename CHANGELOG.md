# Post-Submission Changelog

Running log of every change made to this project **after** the BMVC 2026
submission (#1674). Newest entries at the top. Each entry: what changed, why,
and the measured result where relevant. Maintained across work sessions.

> This file is updated continuously. When new work is done, add a dated entry
> here as part of the same change — do not let the log fall behind the code.

---

## 2026-07-21

### Major data expansion via Shiv's link doc — train 1,759 → 4,440 (deduped)
- Enumerated 10 Roboflow workspaces (~50 projects) from
  Roboflow_EV_Battery_Related_Links.docx via the API; identified all projects
  with module/busbar classes, excluding known duplicates (gqljq, edfw3, MTech)
  and the ~10 near-identical ca-2kt9o experiment forks (took one).
- Downloaded 10 new projects; merged the module/busbar ones with a **perceptual-
  hash (dHash) dedup + test-set leakage guard**: auto-detected each dataset's
  module/busbar class indices from its data.yaml (verified correct), remapped to
  2-class, skipped 131 duplicate/leaking images.
- **Result: train 3,091 → 4,440 images** (1,349 new deduped). Module instances
  10,367 → 14,590; busbar 3,621 → **5,040 (+39% on the weak class)**. Sources
  now ~9 independent datasets. 43-image real TEST split untouched.
- Also downloaded **uerymnd/ue_d1_defect_detection** (115 imgs, real damage
  classes: corrosion/scratch/dent/missing-cover) — a candidate source of REAL
  damaged crops for the classifier bad class (currently only 16 real). Not yet
  processed.
- Licence caveat: multi-source; licences vary (CC BY / BY-NC-SA / etc.) — must be
  audited and disclosed before any non-academic use.

### First REAL new labeled data since the paper (+1,332 train images)
- Downloaded 3 user-provided Roboflow datasets via API. Two are genuinely new
  with named module/busbar classes and independent images (0 exact duplicates
  vs our test set):
  - **ev-battery-component-detection-gqljq** (1,045 imgs, "Arrival-Van" frames;
    battery_module + Busbars; licence BY-NC-SA 4.0 — non-commercial).
  - **ev-battery-components-edfw3** (680 imgs; battery_module + busbar; CC BY 4.0).
  - (ev-battery-pack-62ig0 = duplicate of the MTech source, skipped.)
- Remapped to the 2-class scheme and merged into training:
  **train 1,759 → 3,091 images**; busbar instances 3,621 → 4,325 (+704 real),
  module 10,367 → 12,848 (+2,481). Val 43 → 292 (external valid splits); the
  43-image real TEST split is untouched.
- Licence note: gqljq is BY-NC-SA 4.0, so the combined training set carries a
  non-commercial restriction — fine for academic use, must be disclosed.
- Next: retrain detector on the enriched real data (unlike the failed synthetic
  augmentation, this is genuine added diversity).

### Two dataset-request emails sent (user-reviewed)
- Komiljon sent (after review) polite academic data-request emails to Anselmo
  Parnada (Birmingham disassembly group) and Ville Pitkäkangas (Zenodo/RECIRCULATE,
  Centria). A third (Sci. Reports lithium-defect dataset, sundaozong@scau.edu.cn)
  is prepared. Awaiting replies.

### Cross-variant generalization result (positive)
- Ran `scripts/eval_cross_variant.py` on the Zenodo 17 unseen pack types with the
  baseline detector. **Mean detection rate 0.76; busbar found in 16/17 variants.**
- Strong on BMW i4 (1.00), Hyundai Ioniq (1.00), Ford Mondeo (0.95); weak on
  Volvo truck (0.27) and Mercedes GLE (0.40). Module geometry transfers to unseen
  packs; busbar + a few unusual geometries are the weak axis.
- Honest generalization evidence answering the single-facility criticism
  (detection-rate proxy — Zenodo is unlabelled, so not mAP).

### Roboflow ev-battery-iceh6 — not usable
- Downloaded (machine-iimx4/ev-battery-iceh6, 1,302 train images) but its 9
  classes are named '0'-'8' with no semantics → cannot safely remap to
  module/busbar. Discarded. Getting clean labelled module/busbar data from
  Roboflow is harder than hoped (datasets either overlap MTech or are unnamed).

### YOLOv8n vs YOLO11n architecture comparison (running)
- `scripts/benchmark_detector_arch.py`: trains both from COCO weights with an
  identical reduced 40-epoch budget on the clean 1,759-image set, compares
  test mAP / latency / params. YOLO11n is smaller (2.62M vs 3.16M params).
  Answers the "no architecture comparison" criticism. Result pending.

### Programmable inference API (`scripts/inference_api.py`)
- `BatteryInspector` class: image in (file path / numpy array / raw bytes) →
  structured dict out. Each detection gives class, confidence, `box_xyxy`,
  `box_xywh`, `center`, normalized `box_norm`, and (modules) `grade` + `p_bad`.
- Three entry points: Python library import, JSON-returning CLI, and a minimal
  stdlib HTTP service (`--serve`, POST an image to `/infer`).
- Verified live: e.g. an image returning 6 modules + 12 busbars with pixel boxes
  and per-module Grade A/B/C. Confirms the pipeline outputs 2D image-plane pixel
  boxes (not real-world 3D coords — that needs camera calibration + depth).

### NEGATIVE RESULT: copy-paste + glare augmentation hurts the detector
- Reran Stage 2 correctly on the true 2,260-image augmented set (via
  `dataset_aug.yaml` → worktree data). Validation mAP50 **collapsed** and kept
  declining: epochs 1-6 gave 0.503 → 0.404 → 0.275 → 0.34 → 0.313 → 0.27, far
  below the 0.818 baseline and trending down, not recovering.
- **Decision: stopped at epoch 7 and restored the committed baseline weights.**
  The synthetic copy-paste + glare augmentation, as implemented, degrades the
  detector rather than improving it — likely label noise from pasted objects
  and over-aggressive glare making training signal unnatural.
- Baseline detector (mAP50 0.818) remains the shipped model. `dataset.yaml`
  restored; temporary `dataset_aug.yaml` removed.
- Takeaway for the paper: report this as an honest ablation — naive
  copy-paste/glare synthesis does not help this detector. If revisited, try
  (a) much smaller synthetic fraction, (b) blending/quality filtering of pastes,
  (c) glare intensity tuned to match the real bright-condition statistics.

### CORRECTION: first augmentation retrain was invalid (config path bug)
- The initial Stage 2 retrain read `dataset.yaml`, which hardcodes an absolute
  `path:` to the **main repo** (`/Users/komiljon/research/data/detector`), while
  the 500 copy-paste + glare images were added in the **worktree**. Training
  therefore used the 1,760 unaugmented main-repo images; the augmentation was
  never applied.
- The reported 0.882/0.899 were the **val split** during training (normal
  val/test gap). The real held-out **test** mAP50 was **0.816 — unchanged from
  the 0.818 baseline** (byte-identical lighting numbers confirmed it).
- Fix: added `dataset_aug.yaml` pointing at the worktree data and relaunched a
  correct Stage 2 retrain on the true 2,260-image augmented set. Result pending.
- Lesson: `dataset.yaml`'s absolute main-repo path is a footgun for worktree runs
  (and machine-specific for public users) — should become relative.

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
