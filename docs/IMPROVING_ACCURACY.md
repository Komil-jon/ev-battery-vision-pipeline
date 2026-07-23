# Improving Accuracy & Generalization

This document collects **external labeled datasets** and **synthetic data generation
methods** that can push this pipeline past its two current limits:

1. **Classifier bottleneck** — only 16 real damaged-module crops exist; classifier
   accuracy is 0.771 vs the paper's 0.917. Purely a data limit.
2. **Detector generalization** — trained on a single pack family; mAP50 0.818
   overall, dropping to 0.672 under bright/glare conditions.

---

## A. External labeled datasets

### 1. Zenodo — Battery Image Dataset for EV Circularity (19 battery types) ⭐ best pick

- **Link:** https://zenodo.org/records/19818270 (DOI: 10.5281/zenodo.19818270)
- **Content:** annotated images + videos of EV/hybrid batteries from **19 vehicle
  types** (Tesla Model 3/S, Ford Transit, and more). ~5.4 GB download.
- **Tasks:** classification, object detection, representation learning.
- **License:** CC BY 4.0 (same as our current dataset — attribution required).
- **Why it matters:** the paper recommends ≥3 pack variants for a robust
  classifier; this single dataset provides 19. Biggest available generalization win
  for the detector, and its damaged-pack images can enlarge the real "bad" pool.
- **How to use:** `python scripts/download_external_datasets.py --zenodo`, then
  inspect the class scheme in its README and remap to `0=module, 1=busbar`
  (the script has a `--remap_yaml` hook; pattern follows `scripts/remap_labels.py`).

### 2. Roboflow Universe candidates

| Dataset | Size | Notes |
|---|---|---|
| [EV battery (Machine workspace)](https://universe.roboflow.com/machine-iimx4/ev-battery-iceh6) | 643 images | Check class list on page; remap needed |
| [EV Battery pack (MTech)](https://universe.roboflow.com/mtech-project-ohj8a/ev-battery-pack) | current source | Check for newer versions with more images |
| [Battery class search](https://universe.roboflow.com/search?q=class:battery) | varies | Browse for module/busbar-adjacent sets |

Roboflow sets download via the existing API-key flow
(`scripts/download_external_datasets.py --roboflow workspace/project --api_key KEY`).
**Always check the license badge on each dataset page** — most are CC BY 4.0 but
some are more restrictive.

### 3. Defect-texture datasets (classifier transfer learning)

These do not show battery *modules*, but contain real battery-adjacent defect
textures (corrosion, cracks, coating damage). Use them to pre-train / fine-tune the
ResNet18 backbone before final training on module crops:

- **[Kaggle — Li-ion Battery Electrode Coating Defect Dataset](https://www.kaggle.com/datasets/vigneshirtt/li-ion-battery-coating-defect-dataset)**
  — surface cracks, delamination, pinholes on electrode coatings.
- **[Scientific Reports lithium battery surface defect dataset](https://www.nature.com/articles/s41598-025-18315-0)**
  — 1,300 images, 6 defect types (tab damage, impurities, electrode fold/damage/crack).
  Check the paper's data-availability statement for the download link.

Recipe: train ResNet18 on defect-vs-normal with these textures first, then fine-tune
on our good/bad module crops. Expected gain: better damage-feature priors when real
bad crops are scarce.

### 4. EV disassembly research datasets (availability TBD)

- **Nissan e-NV200 disassembly set** — 226 images, classes: cable, **busbar**,
  leaf cell, service plug ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12515223/)).
  Busbar class matches ours directly. Check the paper's data-availability section
  or email the authors.
- **EV screw-detection sets** (~2,500 and ~900 images) exist in the robotic
  disassembly literature — useful only if the project ever expands beyond
  module/busbar.

---

## B. Synthetic data generation methods (ranked by effort → payoff)

### 1. Copy-paste compositing (detector) — cheap, proven

Cut labeled module/busbar instances from training images and paste them onto other
training backgrounds with scale/rotation/blend jitter. Published results show
**+1–5% mAP**, with the biggest gains for minority/small classes — exactly our
busbar situation.

- References: [Background-instance copy-paste (MDPI)](https://www.mdpi.com/2079-9292/12/18/3781),
  [CutPaste (arXiv 2104.04015)](https://arxiv.org/abs/2104.04015).
- **Implemented:** `python scripts/synth_copy_paste.py --n_images 300`.
  Outputs go to `data/detector/images/train_copy_paste/` with matching YOLO labels.

### 2. Diffusion inpainting for damaged crops (classifier) — highest payoff for the bottleneck

Fine-tune a Stable Diffusion **inpainting** model on the 16 real bad crops (LoRA,
few-shot), then inpaint damage into masked regions of *good* crops. This is the
strongest published approach for few-shot industrial defect synthesis — e.g.
detection recall improvements from 0.656→0.908 reported for stableIDG.

- References: [stableIDG](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115093/),
  [inpainting-diffusion defect generation (JIM)](https://link.springer.com/article/10.1007/s10845-026-02915-2),
  [Awesome-Few-Shot-Defect-Image-Generation](https://github.com/bcmi/Awesome-Few-Shot-Defect-Image-Generation).
- **Implemented (Colab):** `notebooks/colab_defect_inpainting.ipynb` — runs on a
  free Colab T4. Upload the real bad crops, fine-tune, batch-generate, download a
  zip of synthetic bad crops into `data/classifier/bad_synth/`.
- Quality control: manually review generated crops; discard implausible ones.
  Keep synthetic fraction ≤50% of the bad class and always evaluate on the
  **real-only** test set.

### 3. Procedural damage overlays (classifier) — zero-setup fallback

OpenCV-generated corrosion blotches (green/white noise patches), burn marks
(darkened ellipses with soft edges), and scratches (thin bright polylines) composited
onto good crops. Less realistic than diffusion output but instant and CPU-only;
CutPaste-style "scar" augmentation from the anomaly-detection literature shows even
crude synthetic damage teaches useful decision boundaries.

- **Implemented:** `python scripts/synth_damage_overlay.py --n_per_image 4`.

### 4. Targeted glare/lighting augmentation (detector)

Our biggest measured weakness is bright/glare (mAP50 0.816→0.672). Attack it
directly: composite specular highlight blobs and overexposure gradients onto
training images (labels unchanged). `synth_copy_paste.py --glare` adds this pass.
Retrain Stage 2 and re-run `evaluate.py` lighting robustness to measure the delta.

### 5. 3D rendering + domain randomization — high effort, documented only

Render battery packs from CAD/3D models with randomized textures, lighting, and
camera poses; labels come free from the renderer.

- Tools: [BlenderProc2](https://github.com/DLR-RM/BlenderProc),
  [Unity Perception](https://arxiv.org/abs/2107.04259).
- Worth it only if a pack CAD model is available; sim-to-real gap needs care
  (mix ≥10–20% real images into training).

---

## Suggested execution order

1. Download Zenodo 19-type dataset → remap → merge → retrain detector → `evaluate.py`.
2. Generate copy-paste + glare images → retrain Stage 2 → measure lighting robustness delta.
3. Run the Colab inpainting notebook → curate synthetic bad crops → retrain classifier
   (keep `--class_weight` on) → evaluate on the real-only test set.
4. Procedural overlays as an ablation baseline against the diffusion crops.
5. Only then consider 3D rendering.

**Golden rule:** never let synthetic or external images leak into the *test* split.
All reported metrics must stay real-and-held-out.

---

## C. Modern methods (2024-2026 research) + what we tested

Researched the current SOTA for "accurate AND general from little data" and tested
the runnable routes. Findings:

### Route 1 — Open-vocabulary detection (YOLO-World) — FAILS here
Zero-shot text-prompted detection. Tested `yolov8s-worldv2` on the 225-img diverse
test with prompts "ev battery module", "busbar": **mAP50 = 0.004** (total failure).
EV battery components are too niche/out-of-distribution for open-vocab detectors.
Confirms the literature ([OVOD robustness, arXiv 2405.14874](https://arxiv.org/html/2405.14874v3)).
**Not usable as a detector.** ([YOLO-World](https://blog.roboflow.com/what-is-yolo-world/))

### Route 2 — Open-vocab AUTO-LABELING (Grounding DINO) — VIABLE (human-in-loop)
`grounding-dino-tiny` grounds modules/busbars/cells at 0.26-0.41 conf (vs
YOLO-World's ~0). Good on clean pack images, rough on hard grayscale macro shots.
Verdict: a **label accelerator that needs human verification**, not fully automatic.
Fixes the annotation-consistency problem (one model + one prompt = one standard) and
can label previously-unlabeled images (Zenodo 712, YouTube frames).
- Implemented: `scripts/autolabel_grounding_dino.py` (runs CPU or GPU; `--preview`
  to eyeball quality). Refs: [Grounding DINO](https://arxiv.org/abs/2303.05499),
  [Autodistill Grounded-SAM-2](https://github.com/autodistill/autodistill-grounded-sam-2).

### Route 3 — Frozen foundation-model backbone detector (RF-DETR / DINOv2) — MOST PROMISING
The 2024-2026 consensus: a **frozen self-supervised backbone (DINOv2/DINOv3) + light
detection head** gives SOTA generalization and resists small-data overfitting
(DINOv3 hits 66.1 mAP COCO frozen; DINO-YOLO hybrids report up to +88% mAP in
low-data regimes). RF-DETR (Roboflow, 2025) is the practical, pip-installable
version with a DINOv2 backbone.
- Implemented: `notebooks/colab_train_rfdetr.ipynb` (converts diverse YOLO data to
  COCO, trains RF-DETR, evaluates on the diverse test). Compare vs specialist 0.277
  and the YOLO11n generalist.
- Refs: [DINOv3 (arXiv 2508.10104)](https://arxiv.org/html/2508.10104v1),
  [DINO-YOLO few-shot detection](https://www.scitepress.org/Papers/2026/144164/144164.pdf),
  [Mind the Backbone](https://arxiv.org/pdf/2303.14744).

### Route 4 — Data-centric AI (the meta-lesson, already proven here)
Labels beat quantity: we showed twice that merging inconsistent-label data *hurt*.
Priorities: label-consistency audits (done), active learning, hard-negative mining,
a bigger diverse test set (done: 225 imgs). ([small-object survey 2023-2025](https://www.mdpi.com/2076-3417/15/22/11882))

**Recommended stack:** Grounding DINO auto-label (consistent labels at scale) ->
RF-DETR / DINOv2-backbone train (accurate + general) -> distil to YOLO11n (fast CPU
deployment). Evaluate everything on the diverse 225-img test, never the narrow one.
