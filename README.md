# EV Battery CV Pipeline

CPU-deployable two-stage computer vision pipeline for EV battery module and busbar
localisation and condition assessment, replicating the paper methodology on Apple M1.

**Stage 1 — Detection:** YOLOv8n localises battery modules and busbars  
**Stage 2 — Classification:** ResNet18 binary classifier (good/bad) on each module crop  
**Output:** Grade A / B / C triage per detected module via confidence thresholds

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

## Key results from paper

These are the paper's headline figures (private dataset). Run
`python scripts/evaluate.py` after training to print your reproduced numbers on
the public Roboflow proxy alongside these targets.

| Metric | Paper value |
|---|---|
| Detector mAP50 | 0.901 |
| Detector mAP50-95 | 0.715 |
| Detector latency (CPU) | 78.9 ms / 12.7 FPS |
| Classifier accuracy | 91.7% |
| Classifier weighted F1 | 0.912 |
| Full pipeline latency | 149.4 ms / 6.7 FPS |

---

## Reproduced results (this repo, public Roboflow data, Apple M1 CPU)

Produced by `python scripts/remap_labels.py` → `train_detector.py --stage all`
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

| Metric | Reproduced | Paper |
|---|---|---|
| Accuracy | 0.771 | 0.917 |
| Weighted F1 | 0.768 | 0.912 |
| Bad-class recall | 0.571 (8/14) | 0.714 (10/14) |
| Good-class recall | 0.853 (29/34) | 1.000 (34/34) |

The classifier gap is a **data-availability limit, not a code limit**: this
project contains only **16 real damaged-module crops** in total. The classifier
dataset was expanded from them by augmentation (`build_classifier_dataset.py`)
to reach the paper's reporting scale, but augmented copies of 16 real crops
cannot reproduce a 48-image manually-refined real test set. The paper itself
relied on AI-generated synthetic damaged images for the same reason (Appendix B).
Reaching paper-level classifier accuracy requires collecting more real
bad-condition crops (the paper recommends ≥200 across ≥3 pack variants).

---

## Project structure

```
ev-battery-cv/
├── data/
│   ├── detector/
│   │   ├── images/{train, val, test}/
│   │   └── labels/{train, val, test}/
│   └── classifier/
│       ├── train/{good, bad}/
│       └── test/{good, bad}/
├── models/
│   ├── detector/
│   │   ├── stage1/weights/best.pt
│   │   └── stage2_recall_boost/weights/best.pt
│   └── classifier/
│       ├── resnet18_binary.pth
│       └── class_map.json
├── scripts/
│   ├── download_dataset.py     ← get public EV battery data from Roboflow
│   ├── remap_labels.py         ← remap Roboflow 7-class → paper 2-class (REQUIRED)
│   ├── augment_busbars.py      ← busbar-targeted recall-boost augmentation
│   ├── auto_crop_modules.py    ← crop module ROIs from detector for classifier sorting
│   ├── train_detector.py       ← Stage 1 + Stage 2 YOLOv8n training
│   ├── train_classifier.py     ← ResNet18 binary classifier training
│   ├── pipeline_inference.py   ← full two-stage inference + triage output
│   ├── webcam_demo.py          ← live webcam / video two-stage demo
│   └── evaluate.py             ← detector + classifier + lighting evaluation
├── outputs/results/            ← annotated output images saved here
├── dataset.yaml                ← YOLO dataset config
└── requirements.txt
```

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
python scripts/download_dataset.py --api_key YOUR_KEY

# Verify class distribution
python scripts/download_dataset.py --check_classes
```

---

## Dataset preparation (REQUIRED before training)

The public Roboflow dataset uses a 7-class scheme
(`Aluminum-frame, Battery Module, Bolt, Bus-bar, Cable, Nut, Screw`). This
project detects only **module** and **busbar**, so the labels must be remapped
to the 2-class scheme in `dataset.yaml` (`0=module`, `1=busbar`):

```bash
# Preview the remap (changes nothing)
python scripts/remap_labels.py --dry_run

# Apply: Battery Module(1)->module(0), Bus-bar(3)->busbar(1); drop the rest
python scripts/remap_labels.py

# Clear stale YOLO caches so the new labels take effect
find data/detector/labels -name '*.cache' -delete
```

`remap_labels.py` is idempotent — it skips any split already in the 2-class
scheme. Validation and test labels are remapped too so evaluation is valid.

---

## Training

### Step 1 — Generate busbar-targeted augmentations

```bash
python scripts/augment_busbars.py --n_augments 6
# Copies augmented images to data/detector/images/train_busbar_aug/
# Copy or symlink into train/ before Stage 2
```

### Step 2 — Train detector (both stages)

```bash
# Run both stages
python scripts/train_detector.py --stage all

# Or individually
python scripts/train_detector.py --stage 1   # 100 epochs, 640px, SGD
python scripts/train_detector.py --stage 2   # 30 epochs, 768px, AdamW recall-boost
```

### Step 3 — Train condition classifier

```bash
python scripts/train_classifier.py
python scripts/train_classifier.py --epochs 30 --batch 8
```

Expects images in `data/classifier/train/good/` and `data/classifier/train/bad/`.

---

## Inference

```bash
# Single image
python scripts/pipeline_inference.py --input path/to/image.jpg

# Folder of images
python scripts/pipeline_inference.py --input data/detector/images/test/

# Custom confidence threshold (paper F1-optimal: 0.21)
python scripts/pipeline_inference.py --input image.jpg --conf 0.21
```

Output images saved to `outputs/results/`.

### Live webcam / video demo

```bash
python scripts/webcam_demo.py                 # default camera, imgsz 640
python scripts/webcam_demo.py --imgsz 480     # faster / smoother on CPU
python scripts/webcam_demo.py --input clip.mp4  # run on a video file instead
```

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
python scripts/evaluate.py

# Skip lighting robustness (faster)
python scripts/evaluate.py --skip_lighting

# Classifier only
python scripts/evaluate.py --skip_detector
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

## References

Paper methodology (unpublished dissertation, 2025/2026):
- Architecture: Jocher et al., Ultralytics YOLOv8
- Classifier backbone: He et al., Deep Residual Learning (ResNet), CVPR 2016
- Framework context: RESCu-M2 circular manufacturing
- EU Battery Regulation: 2023/1542
