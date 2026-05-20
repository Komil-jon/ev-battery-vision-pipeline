# EV Battery CV Pipeline

CPU-deployable two-stage computer vision pipeline for EV battery module and busbar
localisation and condition assessment, replicating the paper methodology on Apple M1.

**Stage 1 — Detection:** YOLOv8n localises battery modules and busbars  
**Stage 2 — Classification:** ResNet18 binary classifier (good/bad) on each module crop  
**Output:** Grade A / B / C triage per detected module via confidence thresholds

---

## Key results from paper

| Metric | Value |
|---|---|
| Detector mAP50 | 0.901 |
| Detector mAP50-95 | 0.715 |
| Detector latency (CPU) | 78.9 ms / 12.7 FPS |
| Classifier accuracy | 91.7% |
| Classifier weighted F1 | 0.912 |
| Full pipeline latency | 149.4 ms / 6.7 FPS |

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
│   ├── augment_busbars.py      ← busbar-targeted recall-boost augmentation
│   ├── train_detector.py       ← Stage 1 + Stage 2 YOLOv8n training
│   ├── train_classifier.py     ← ResNet18 binary classifier training
│   ├── pipeline_inference.py   ← full two-stage inference + triage output
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

From paper Section 3.4 and Appendix F, Table F2:

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

- Optimal confidence threshold: **~0.21** (F1-confidence curve peak, Appendix C)
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
