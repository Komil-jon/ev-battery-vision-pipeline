# EV Battery CV Pipeline

Computer vision for EV battery disassembly: module and busbar detection, condition
assessment, and monocular 3D localisation for robotic handling.

**Stage 1 — Detection:** YOLOv8n / YOLO11n / RF-DETR localise battery modules and busbars
**Stage 2 — Classification:** ResNet18 binary classifier (good/bad) on each module crop
**Output:** Grade A / B / C triage per detected module via confidence thresholds

---

## Publications

**[1] Vision Model for Detection and Condition Assessment of EV Battery Components
for Circular Manufacturing** — *published*
S. Katiyar, P. Venigalla, K. Kosimov, S. Nefti-Meziani.
Proc. 12th World Congress on Electrical Engineering and Computer Systems and
Sciences (EECSS 2026), London, UK, Aug. 2026, Paper MVML 125.
DOI: [10.11159/mvml26.125](https://doi.org/10.11159/mvml26.125)

> The single-facility two-stage pipeline. Most of this repository's Stage 1 / Stage 2
> code and the "Key results" table below correspond to this paper.

**[2] From Images to Robot Coordinates: Dataset Design, Detector Selection and
Monocular 3D Localisation for Automated EV Battery Disassembly** — *under review*
Y. Wang, K. Kosimov, S. A. Katiyar, Q. Nguyen. Source: [`paper/`](paper/)

> Extends [1] to cross-facility evaluation, compares detector architectures on a
> 13-source dataset, and adds metric 3D localisation validated on a UR5e.
> Localisation code: [D405-ArUco-UR5e-Validation](https://github.com/xcdgdj/D405-ArUco-UR5e-Validation)

Headline results from [2]:

| Finding | Result |
|---|---|
| Cross-facility generalisation gap | 0.818 → **0.277** mAP@50 (66% loss) |
| Best cross-facility detector | RF-DETR **0.502** vs YOLO11n 0.410 mAP@50 |
| Annotation convention spread | 0.995 → **0.043** per-source module mAP@50 |
| Localisation (UR5e, RGB only) | **2.245 mm** height MAE, **9/9** target reaches |

Released annotations: **16,945 polygon masks** in [`data/labels_release/`](data/labels_release/).

> **Replication note.** The paper's results were produced on a private in-house
> dataset that is not distributed. This repository replicates the *methodology*
> on the closest public proxy — the Roboflow "EV Battery pack" dataset (CC BY 4.0).
> Reproduced metrics are therefore legitimate but will not exactly equal the
> paper's headline figures. The public dataset ships with a **7-class** label
> scheme; it **must be remapped to the paper's 2-class scheme before training**
> (see *Dataset preparation* below). Skipping this step makes YOLO silently treat
> every multi-class label file as a corrupt/background image and learns the wrong
> classes.

---

## Key results from [1]

These are the headline figures from publication [1] (private dataset). Run
`python scripts/eval/evaluate.py` after training to print your reproduced numbers on
the public Roboflow proxy alongside these targets.

| Metric | Value in [1] |
|---|---|
| Detector mAP50 | 0.901 |
| Detector mAP50-95 | 0.715 |
| Detector latency (CPU) | 78.9 ms / 12.7 FPS |
| Classifier accuracy | 91.7% |
| Classifier weighted F1 | 0.912 |
| Full pipeline latency | 149.4 ms / 6.7 FPS |

---

## Reproduced results (this repo, public Roboflow data, Apple M1 CPU)

Produced by `python scripts/data_prep/remap_labels.py` → `train_detector.py --stage all`
→ `train_classifier.py` → `evaluate.py`. These are legitimate end-to-end results
on the public proxy dataset; they are **not** expected to equal the paper's
private-dataset figures.

**Detector — held-out test set (43 images, 323 instances):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Module | 0.761 | 0.838 | 0.787 | 0.581 |
| Busbar | 0.872 | 0.781 | 0.849 | 0.533 |
| **Overall** | **0.817** | **0.809** | **0.818** | **0.557** |

CPU latency: **70.8 ms/image (14.1 FPS)** — comfortably beating the paper's
78.9 ms / 12.7 FPS target on the same class of hardware.

**Lighting robustness (real mAP, not proxy confidence):**

| Condition | mAP50 | mAP50-95 |
|---|---|---|
| Normal | 0.816 | 0.555 |
| Dark | 0.788 | 0.537 |
| Bright / glare | 0.672 | 0.424 |

This **reproduces the paper's central optical finding**: bright/glare degrades
detection far more than darkness (specular reflection on metallic casings),
whereas dark conditions barely move mAP50.

**Classifier — 48-image test set (34 good + 14 bad), class-weighted ResNet18:**

| Metric | Reproduced | + synthetic bad crops | Paper |
|---|---|---|---|
| Accuracy | 0.771 | 0.792 | 0.917 |
| Weighted F1 | 0.768 | 0.800 | 0.912 |
| Bad-class recall | 0.571 (8/14) | **0.857 (12/14)** | 0.714 (10/14) |
| Good-class recall | 0.853 (29/34) | 0.765 (26/34) | 1.000 (34/34) |

The "+ synthetic bad crops" column is the current shipped model: the bad class was
expanded from 40 to 80 training crops with procedurally damaged good crops
(`scripts/data_prep/synth_damage_overlay.py`, synthetic kept ≤50% of the class; test set
remains 100% real). Bad-class recall — the safety-critical metric for triage —
now **exceeds the paper's 0.714**, at the cost of some good-class recall (more
Grade B/C flags sent to manual review, which is the conservative direction for
this application).

The remaining classifier gap is a **data-availability limit, not a code limit**:
this project contains only **16 real damaged-module crops** in total. The paper
itself relied on AI-generated synthetic damaged images for the same reason
(Appendix B); the synthetic-crop expansion above applies the same idea. Reaching
paper-level accuracy requires collecting more real bad-condition crops (the
paper recommends ≥200 across ≥3 pack variants) — see
[docs/IMPROVING_ACCURACY.md](docs/IMPROVING_ACCURACY.md) for datasets and the
diffusion-based generation pipeline.

---

## Project structure

```
ev-battery-vision-pipeline/
├── paper/                      ← LaTeX source + figures for publication [2]
├── data/
│   ├── sources/                ← raw datasets, one folder per origin (gitignored)
│   ├── detector/               ← merged train/val/test splits
│   ├── classifier/             ← good/bad module crops
│   └── labels_release/         ← 16,945 polygon masks, published here
├── models/
│   ├── detector/
│   │   ├── baseline_yolov8n_stage1/
│   │   ├── specialist_yolov8n/       ← --model specialist
│   │   ├── generalist_yolo11n/       ← --model generalist_yolo
│   │   └── generalist_rfdetr/        ← --model generalist_rfdetr (weights fetched separately)
│   └── classifier_resnet18/
├── scripts/
│   ├── common/     model_zoo.py  ← model registry; one interface over YOLO + RF-DETR
│   ├── data_prep/  download, remap, audit, auto-label, synthesise (14 scripts)
│   ├── train/      train_detector.py, train_classifier.py
│   ├── eval/       evaluate, compare_detectors, benchmarks, calibration (7)
│   └── inference/  pipeline_inference, webcam_demo, inference_api, anomaly_condition
├── notebooks/                  ← Colab training notebooks
├── docs/                       ← plans, dataset provenance, prior-art surveys
└── requirements.txt
```

Full layout and the model registry: [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md).

---

## Setup

### 1. Create conda environment

```bash
conda create -n ev-battery-cv python=3.13 -y
conda activate ev-battery-cv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify setup

```bash
python -c "
from ultralytics import YOLO
import torch
from torchvision import models
print('PyTorch:', torch.__version__)
model = YOLO('yolov8n.pt')
print('YOLOv8n: OK')
resnet = models.resnet18(weights='IMAGENET1K_V1')
print('ResNet18: OK')
print('All clear.')
"
```

---

## Quickstart with public dataset

The fastest way to get started — downloads a labelled EV Battery Pack dataset
(Battery Module + Bus-bar classes, CC BY 4.0) from Roboflow:

```bash
# Get a free API key at https://roboflow.com
python scripts/data_prep/download_dataset.py --api_key YOUR_KEY

# Verify class distribution
python scripts/data_prep/download_dataset.py --check_classes
```

---

## Dataset preparation (REQUIRED before training)

The public Roboflow dataset uses a 7-class scheme
(`Aluminum-frame, Battery Module, Bolt, Bus-bar, Cable, Nut, Screw`). This
project detects only **module** and **busbar**, so the labels must be remapped
to the 2-class scheme in `dataset.yaml` (`0=module`, `1=busbar`):

```bash
# Preview the remap (changes nothing)
python scripts/data_prep/remap_labels.py --dry_run

# Apply: Battery Module(1)->module(0), Bus-bar(3)->busbar(1); drop the rest
python scripts/data_prep/remap_labels.py

# Clear stale YOLO caches so the new labels take effect
find data/detector/labels -name '*.cache' -delete
```

`remap_labels.py` is idempotent — it skips any split already in the 2-class
scheme. Validation and test labels are remapped too so evaluation is valid.

---

## Training

### Step 1 — Generate busbar-targeted augmentations

```bash
python scripts/data_prep/augment_busbars.py --n_augments 6
# Copies augmented images to data/detector/images/train_busbar_aug/
# Copy or symlink into train/ before Stage 2
```

### Step 2 — Train detector (both stages)

```bash
# Run both stages
python scripts/train/train_detector.py --stage all

# Or individually
python scripts/train/train_detector.py --stage 1   # 100 epochs, 640px, SGD
python scripts/train/train_detector.py --stage 2   # 30 epochs, 768px, AdamW recall-boost
```

### Step 3 — Train condition classifier

```bash
python scripts/train/train_classifier.py
python scripts/train/train_classifier.py --epochs 30 --batch 8
```

Expects images in `data/classifier/train/good/` and `data/classifier/train/bad/`.

---

## Choosing a detector

Three trained detectors ship with this project. Every inference and evaluation
script takes `--model NAME`, so you can switch without editing code:

| `--model` | What it is | Diverse test mAP50 | Best for |
|---|---|---|---|
| `specialist` | Paper baseline, YOLOv8n trained on one facility (MTech) | 0.277 | That facility's imagery (0.818 in-domain) |
| `generalist_yolo` *(default)* | YOLO11n trained on 10 diverse pack sources | 0.410 | General use, CPU / real time |
| `generalist_rfdetr` | RF-DETR-Nano, frozen DINOv2 backbone | **0.502** | Best accuracy and robustness; needs a GPU to be fast |

```bash
# See all models, their metrics and whether the weights are present
python scripts/common/model_zoo.py

# Same, from any script
python scripts/inference/pipeline_inference.py --list-models
```

The specialist scores highest on the facility it was trained on but collapses on
other pack types; the generalists trade in-domain peak accuracy for cross-facility
robustness. On the 7 consensus sources (excluding the MTech annotation outlier)
`generalist_yolo` and `generalist_rfdetr` are effectively tied on modules
(0.774 vs 0.771) — RF-DETR's overall lead comes from degrading gracefully on
out-of-convention data. Full history in [CHANGELOG.md](CHANGELOG.md).

**Confidence thresholds are not comparable across models.** Each model carries a
suggested `--conf` (specialist 0.21, generalist_yolo 0.10, generalist_rfdetr 0.30)
which is applied automatically unless you pass `--conf` yourself.

`generalist_rfdetr` needs `pip install rfdetr`, and its checkpoint is downloaded
separately — see [models/detector/generalist_rfdetr/README.md](models/detector/generalist_rfdetr/README.md).

---

## Inference

```bash
# Single image (uses the default model, generalist_yolo)
python scripts/inference/pipeline_inference.py --input path/to/image.jpg

# Pick a specific model
python scripts/inference/pipeline_inference.py --input image.jpg --model specialist
python scripts/inference/pipeline_inference.py --input image.jpg --model generalist_rfdetr

# Folder of images
python scripts/inference/pipeline_inference.py --input data/detector/images/test/

# Override the model's default confidence threshold
python scripts/inference/pipeline_inference.py --input image.jpg --conf 0.21
```

Output images saved to `outputs/results/`.

### Live webcam / video demo

```bash
python scripts/inference/webcam_demo.py                        # default camera + default model
python scripts/inference/webcam_demo.py --model specialist     # pick a model
python scripts/inference/webcam_demo.py --imgsz 480            # faster / smoother on CPU
python scripts/inference/webcam_demo.py --input clip.mp4       # run on a video file instead
```

> Use a YOLO model for live video. `generalist_rfdetr` runs at roughly 0.2 FPS on
> a CPU, so it is for batch use unless you have a GPU.

Press `q` to quit, `s` to save the current annotated frame to `outputs/results/`.

**macOS camera permission:** the first run needs the terminal app authorised
under *System Settings → Privacy & Security → Camera*. Enable it, then fully
quit and reopen the terminal and rerun.

> The detector only knows EV **battery modules and busbars** — pointing the
> webcam at a room/face detects nothing. To see it work, point the camera at a
> photo of an EV battery pack on another screen, or use `--input` with a clip.

---

## Evaluation

```bash
# Full evaluation (detector + classifier + lighting robustness)
python scripts/eval/evaluate.py

# Evaluate a specific detector (YOLO models only -- uses Ultralytics .val())
python scripts/eval/evaluate.py --model specialist

# Skip lighting robustness (faster)
python scripts/eval/evaluate.py --skip_lighting

# Classifier only
python scripts/eval/evaluate.py --skip_detector
```

### Comparing detectors fairly

Ultralytics `.val()` and supervision's mAP are different implementations, so
numbers from `evaluate.py` must not be compared against RF-DETR results. Use
`compare_detectors.py`, which scores every model on the same images with the same
metric and threshold:

```bash
# Compare all available models
python scripts/eval/compare_detectors.py \
    --images data/detector/images/test --labels data/detector/labels/test

# Specific models, excluding the MTech annotation outlier
python scripts/eval/compare_detectors.py --images DIR --labels DIR \
    --models generalist_yolo generalist_rfdetr --exclude-mtech
```

Example output on the 43-image MTech test set — the specialist wins on its home
turf, `generalist_yolo`'s module score collapses on this annotation convention,
and RF-DETR degrades gracefully:

```
model                     mAP50   mAP50-95   module   busbar
specialist                0.840      0.581    0.871    0.860
generalist_rfdetr         0.541      0.288    0.543    0.558
generalist_yolo           0.195      0.112    0.047    0.347
```

---

## Grade triage thresholds

From paper Section 3.4 and Appendix K, Table K.2:

| Grade | p_bad rule | Interpretation |
|---|---|---|
| A | p_bad < 0.30 | Likely reusable |
| B | 0.30 ≤ p_bad < 0.70 | Manual review required |
| C | p_bad ≥ 0.70 | Likely damaged |

These thresholds use a symmetric 0.40-wide uncertainty band centred at 0.50.
Recalibrate using facility-specific cost-weighted analysis before deployment.

---

## Annotation guidelines

Following the paper (Section 3.2):

- Draw bounding boxes to **tightest enclosing rectangle** around the visible component
- Use **5-pixel tolerance** to avoid clipping edge features
- **Do not use auto-labelling** — early trials produced inconsistent boundaries
- Class IDs: `0 = module`, `1 = busbar` (matches `dataset.yaml`)
- YOLO format: `class_id cx cy w h` (normalised 0–1)
- Recommended tool: [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow Annotate](https://roboflow.com/annotate)

---

## Augmentation strategy

Safe transforms (paper Section 3.3):

| Transform | Value | Rationale |
|---|---|---|
| Horizontal flip | p=0.5 | Valid for any orientation |
| Brightness (hsv_v) | 0.30 | Simulates lighting variation |
| Saturation (hsv_s) | 0.25 | Simulates illumination change |
| Rotation | ±2° | Modules at near-fixed angles on fixtures |
| Translation | 6% | Small position variance |
| Mosaic | 0.5 | Reduced to preserve full-image context |
| **Hue jitter** | **EXCLUDED** | Would corrupt corrosion/burn mark cues |

Busbar-targeted augmentation applied only to busbar-containing images (Stage 2).

---

## Deployment notes

- Optimal confidence threshold: **~0.21** (F1-confidence curve peak, Appendix I, Figure I.2)
- Use **diffuse lighting** — bright/glare degrades mAP50 by 0.097 via specular reflection
- Standardise camera pose, working distance, and background colour
- Pipeline best suited to **static, indexed, or slow-moving** inspection stations
- At 149.4 ms/image: feasible up to ~0.50 m/s conveyor with triggered capture
- Recalibrate Grade A/B/C thresholds against facility-specific condition-labelled data

---

## Improving accuracy & generalization

The two known limits (classifier bad-class scarcity, single-dataset detector) can
be attacked with external datasets and synthetic data. See
**[docs/IMPROVING_ACCURACY.md](docs/IMPROVING_ACCURACY.md)** for the curated
dataset catalogue (incl. a 19-battery-type CC BY 4.0 set) and method guide.
Starter tooling:

```bash
# External datasets (Zenodo 19-type set, any Roboflow set) + class-remap merge
python scripts/data_prep/download_external_datasets.py --dry_run

# Copy-paste compositing (+ optional synthetic glare) for the detector
python scripts/data_prep/synth_copy_paste.py --n_images 300
python scripts/data_prep/synth_copy_paste.py --glare --n_images 200

# Procedural damaged-crop synthesis for the classifier (CPU-only fallback)
python scripts/data_prep/synth_damage_overlay.py --preview

# Diffusion-inpainting damaged-crop synthesis (free Colab GPU)
# -> notebooks/colab_defect_inpainting.ipynb
```

---

## References

Paper methodology (unpublished dissertation, 2025/2026):
- Architecture: Jocher et al., Ultralytics YOLOv8
- Classifier backbone: He et al., Deep Residual Learning (ResNet), CVPR 2016
- Framework context: RESCu-M2 circular manufacturing
- EU Battery Regulation: 2023/1542
