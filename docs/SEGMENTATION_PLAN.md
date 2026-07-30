# Plan: instance segmentation for robotic EV battery disassembly

Goal: a model that outputs a **precise polygon per module and busbar**, good enough
to drive a robot (grasp point + orientation), and fast enough to run on an edge
device. This document is the execution plan. Update the checkboxes as work lands.

---

## Why segmentation, not detection

A bounding box gives a robot a rectangle. For a busbar lying diagonally, the box is
mostly background — a gripper aimed at the box centre grabs the casing, not the part.
A mask gives:

| | box | mask |
|---|---|---|
| object present | yes | yes |
| exact boundary | no | **yes** |
| true centroid | box centre ≠ object centre | **yes** |
| grasp orientation | no | **yes** (`cv2.minAreaRect` / PCA on the mask) |

**Yes — a `-seg` model returns polygons for every detection at inference.**
`results[0].masks.xy` is a list of point arrays, one polygon per instance, in pixel
coordinates. Same call, richer output. That is the shape you can feed to a robot.

---

## Where the data actually stands (measured 2026-07-30)

| | images | annotations |
|---|---|---|
| already polygon (segmentation-ready) | 2,486 | 16,226 polygons (module 11,766, busbar 4,460) |
| mixed polygon + box | 128 | |
| box only | 1,001 | 3,296 boxes |
| empty label file | 1,145 | — |
| **total** | **4,760** | **19,522** |

**55% of the dataset is already segmentation-ready with zero new labelling.** The
polygons are genuine masks (5–151 vertices, tracing busbars along their winding
paths), not 4-point rectangles.

---

## The one thing to get straight first

**You cannot convert a box into a polygon by reshaping it.** A box holds no boundary
information. Writing the 4 box corners as a "polygon" would teach the model that
every object is a rectangle — worse than not training on it at all.

The fix is to *re-derive* the boundary from the image. SAM 2 does this: prompt it
with the existing box, it returns the object's mask.

Tested on this data:

| model | result |
|---|---|
| `mobile_sam.pt` | fragmented blobs, unusable |
| `sam2.1_b.pt` | coherent, usable masks — **use this** |

Tool: `scripts/data_prep/boxes_to_masks_sam.py` (writes to a new label dir, never overwrites;
falls back to keeping the original box when SAM's mask is implausibly small).

**SAM cannot fix a bad box.** If a box covers an arbitrary region of a cell array,
the mask will too. Box quality caps mask quality.

---

## Do NOT hand-review all 4,760 images

Careful polygon work runs 30–60 s per object. Reviewing every existing annotation
means 19,522 objects — **160–320 hours**. That is not a plan, it is a way to abandon
the project.

Spend the effort where it changes the model:

| priority | what | volume | est. effort |
|---|---|---|---|
| **P0** | Fix unlabelled images that DO contain objects | ~1,145 to triage | 4–8 h |
| **P1** | SAM-convert box-only, then review | 1,001 imgs / 3,296 objects | 6–10 h review |
| **P2** | Spot-check existing polygons (sample ~200, fix systematic errors) | 200 imgs | 3–5 h |
| — | Re-drawing all existing polygons | 16,226 objects | **skip** |

Roughly **15–25 hours**, versus 160–320. Same model, a tenth of the cost.

---

## Execution

### Phase 0 — Baseline first (do this before any labelling)
Train `yolo11n-seg` on the 2,614 images that already have masks. This costs one Colab
run and answers the only question that matters: **do the existing masks train a
useful segmentation model?**

- If yes → the labelling effort is justified, and you have a baseline to beat.
- If no → find out now, not after 25 hours of annotation.

- [ ] Build `notebooks/colab_train_yolo_seg.ipynb`
- [ ] Train on polygon-only subset, evaluate mask mAP50 on a held-out diverse split
- [ ] Record the number in `CHANGELOG.md`

### Phase 1 — Triage the unlabelled (P0)
```bash
python scripts/data_prep/audit_dataset.py --images data/detector/images/train \
    --labels data/detector/labels/train --out outputs/audit
```
Open `outputs/audit/ambiguous_unlabelled.jpg` and `anomaly_unlabelled.jpg` and sort
each image into: **true background** (keep, unlabelled — ~10% background is healthy
and cuts false positives) or **under-labelled** (label it, or drop it).

Confirmed under-labelled examples exist: a `final_mobilenet_results` frame shows an
open pack with a clearly visible orange busbar assembly and no labels at all. Those
frames actively teach the model that busbars are background, and are a plausible
cause of the persistently weak busbar scores (0.30–0.35 mAP50).

- [ ] Triage the ~1,145 empty-label images
- [ ] Label the under-labelled ones (polygon, in the tool from "Tooling" below)
- [ ] Re-run the audit to confirm the counts moved

### Phase 2 — Box → mask (P1)
```bash
# ALWAYS preview first
python scripts/data_prep/boxes_to_masks_sam.py --images data/detector/images/train \
    --labels data/detector/labels/train --preview --limit 20
# then convert into a NEW directory
python scripts/data_prep/boxes_to_masks_sam.py --images data/detector/images/train \
    --labels data/detector/labels/train --out data/detector/labels_seg/train
```
- [ ] Preview and sanity-check on 20 images
- [ ] Convert train / val / test into `labels_seg/`
- [ ] Human-review the converted masks; fix or delete bad ones
- [ ] Discard masks whose underlying box was wrong (SAM faithfully segments a bad box)

### Phase 3 — Quality gate before training
- [ ] Every image is either fully polygon-labelled or a deliberate background
- [ ] Class balance recorded (busbar is the minority class — watch it)
- [ ] Dedup check (`dhash`) so no near-duplicate spans train and test
- [ ] Held-out test split is diverse and untouched by any auto-labelling

### Phase 4 — Train
- [ ] `yolo11n-seg` (edge target) and `yolo11s-seg` (accuracy reference)
- [ ] Report **mask mAP50 / mAP50-95**, not just box mAP
- [ ] Per-class: module vs busbar, and per-source
- [ ] Compare against the Phase 0 baseline to prove the labelling work paid off

### Phase 5 — Robot-facing output
From each mask, derive what a robot consumes:
```python
rect = cv2.minAreaRect(mask_pts)      # ((cx,cy),(w,h),angle) -> grasp centre + angle
M    = cv2.moments(mask_pts)          # true centroid, not the box centre
```
- [ ] Extend `scripts/inference/inference_api.py` to emit `polygon`, `centroid`, `angle_deg`,
      `min_area_rect` per instance
- [ ] Sanity-check angles on busbars at known orientations

**Honest scope limit:** this gives **2D** pose. A real arm needs 3D (6-DoF). Lifting
2D → 3D requires depth — an RGB-D camera (RealSense / Zivid) or stereo. The vision
half is what this project can deliver; say so plainly in the paper rather than
implying full grasp planning.

### Phase 6 — Edge deployment
- [ ] Export `yolo11n-seg` → ONNX → TensorRT
- [ ] Benchmark on the target device (Jetson Orin Nano is the common choice)
- [ ] Measure INT8/FP16 accuracy drop against the FP32 mask mAP; keep the FP32 number
      as the reference in the paper
- [ ] Report latency **and** mask mAP together — a fast model that loses the boundary
      is useless for grasping

---

## Prior art: does an EV battery segmentation model already exist? (searched 2026-07-30)

**No. Every public EV-battery dataset is object detection (boxes).** Searching
Roboflow Universe for "ev battery" returns these, and the only one tagged
segmentation is for cables:

| EV battery dataset (Roboflow) | Images | Task | Classes |
|---|---|---|---|
| EV Battery Component Detection | 1,360 | Object Detection | BMS_Unit, **Busbars**, Coolant_tubes, battery_module, battery_tray, fasteners… |
| Ev Battery Components | 1,010 | Object Detection | nut, b_casing, battery_module, bms_unit, **busbar**, connector, cables |
| EV battery (Machine) | 643 | Object Detection | unnamed 0–8 |
| EV Battery pack (MTech) | 145 | Object Detection | Battery Module, **Bus-bar**, Bolt, Nut, Screw… |
| EV Battery Sample | 94 | Object Detection | Battery_Module, **Busbar**, Cooling_Channel… |
| **validation Ev battery** | 85 | **Semantic Segmentation** | `evcable` — **cables only, not modules/busbars** |

We already hold most of these in `data/sources/`.

Academic systems are detection-based too, and all single-rig:

| Work | What | Limit |
|---|---|---|
| [RAPID](https://arxiv.org/abs/2603.18520v1) (2026) | Open-vocab **detection**, 0.9757 mAP50 on screws/nuts/busbars, RGB-D | Detection not segmentation; one cell. Its grasp numbers matter more — see below |
| [Li-ion module disassembly](https://link.springer.com/article/10.1007/s11740-023-01231-5) (Springer) | Instance segmentation + point-cloud registration, **demonstrated grasping busbars** | Paywalled, no public model or dataset, one module type |
| [Robotised disassembly review](https://www.sciencedirect.com/science/article/pii/S0278612524001109) | Systematic review | Confirms no cross-pack-type model |

Non-EV battery segmentation sets exist (AA cells, 9V, pack-teardown rigs such as
`battery project 101` at 99.2% mAP50 on 1.4k near-duplicate frames of one battery),
but they are a different object and their in-domain scores do not transfer — exactly
the effect we measured ourselves (specialist 0.818 in-domain → 0.277 cross-facility).

**Consequence:** our own polygon labels — 16,226 masks over module and busbar across
multiple pack types, sourced from `roboflow_battery_comp` (6,267),
`roboflow_last_exp4` (3,115), `roboflow_automated-disassembly` (2,600),
`roboflow_ev-battery-pack-62ig0` (2,116), `edfw3` (825), `ue_rav4_module` (96) — are
plausibly the largest EV module/busbar segmentation annotation set assembled anywhere.
Releasing them is a contribution in its own right.

## From RGB mask to a robot grasp

A mask alone gives 2D: centroid and in-plane angle. A robot needs 3D. Three ways to
bridge it, in order of practicality here:

1. **RGB-D camera (recommended).** RealSense / Zivid. The mask crops the point cloud
   to just that object; from the cropped cloud you get a metric 3D centroid and a
   surface normal, i.e. an approach vector. This is what the Springer busbar-grasping
   work and RAPID both do.
2. **Calibrated RGB + planar assumption (cheapest, and viable here).** Battery packs
   sit flat on a table or fixture. With a calibrated camera and a known pack-surface
   height, a 2D mask point back-projects to a unique 3D point. Good enough for
   top-down picks of parts lying on a known plane; breaks on stacked or tilted parts.
3. **Monocular depth (Depth Anything V2 etc.).** No extra hardware, but the depth is
   relative, not metric — scale must be recovered from a known reference. Do not put
   this in a control loop without validation.

**The honest number to remember:** RAPID reports 0.9757 mAP50 detection, yet
**one-shot vision-to-grasp succeeded only 57%** of the time. Taught-in poses hit 97%
and visual servoing 83%. Perception accuracy is not manipulation success — the gap is
calibration, hand-eye error, occlusion and control. Any claim we make should be about
perception, with grasping framed as future work unless we actually run a robot.

## Tooling for the manual work

Use a tool with SAM built in, so most polygons come from a click rather than tracing:

- **Roboflow Annotate** — "Smart Polygon" is SAM-backed; already hosts these datasets.
- **CVAT** — free, self-hostable, has SAM integration; best if the data must stay local.
- **Label Studio** — good if the labelling gets split across several people.

Do not trace polygons by hand point-by-point. It is 5–10× slower for no gain.

---

## Risks, and what to do about them

| risk | mitigation |
|---|---|
| SAM masks look plausible but are subtly wrong | Never train on unreviewed SAM output; Phase 0 baseline shows what "good" looks like |
| Bad boxes produce bad masks | Fix the box first, or drop the instance |
| Busbar stays weak | It is the minority class *and* the under-labelled one — Phase 1 targets it directly |
| Effort balloons | The P0/P1/P2 split is the budget; re-drawing all 16k polygons is explicitly out of scope |
| Segmentation is slower than detection on edge | `yolo11n-seg` + TensorRT; measure before assuming |
| Annotation drift across sources (already burned us twice) | One convention, written down, applied by one person or one tool |

---

## What "done" looks like

1. A `yolo11n-seg` model reporting mask mAP50 on a diverse held-out split.
2. Polygons + centroid + grasp angle emitted per instance from the inference API.
3. A TensorRT/ONNX build with measured latency and measured accuracy loss.
4. Every number reproducible from this repo, logged in `CHANGELOG.md`.
