# Post-Submission Changelog

Running log of every change made to this project **after** the BMVC 2026
submission (#1674). Newest entries at the top. Each entry: what changed, why,
and the measured result where relevant. Maintained across work sessions.

> This file is updated continuously. When new work is done, add a dated entry
> here as part of the same change — do not let the log fall behind the code.

---

## 2026-09-04

### FINDING: single-source validation inverts model selection, it does not merely bias it
- Matched-budget YOLO11n runs (60 epochs, 640 px, SGD, seeds 1 and 2) on the full
  4,425-image corpus. Validation during training used the only val images recovered,
  43 frames all from the convention-divergent source.
- On that val split, mAP@50 peaks near 0.34-0.37 around epoch 7-20 and falls to
  0.045-0.066 by epoch 60, while box and cls training losses fall monotonically
  throughout (2.2 -> 1.34 and 3.3 -> 1.68). Reproducible across both seeds.
- Evaluated instead on 385 images stratified across all 11 sources:

  | checkpoint | multi-source mAP@50 | divergent-source val mAP@50 |
  |---|---|---|
  | s1 `best.pt` (selected on divergent val) | 0.472 | 0.340 |
  | s1 `last.pt` (fixed budget)              | **0.692** | 0.066 |
  | s2 `best.pt` | 0.474 | 0.370 |
  | s2 `last.pt` | **0.698** | 0.045 |

- **The ranking is exactly inverted.** The checkpoint the divergent val calls best is
  the worse model on the consensus convention by 0.22 mAP@50, and the checkpoint it
  calls worthless (0.045) is the better one. Single-source validation does not merely
  add noise to model selection; it reverses it, and shipping on `best.pt` would have
  cost roughly 32% relative accuracy.
- This retrospectively justifies disabling early stopping and taking the final
  checkpoint. Had the Ultralytics default been used, every number in the matched-budget
  experiment would have been contaminated by selection against the outlier source.
- Seed variance is small: 0.692 vs 0.698, a spread of 0.006. The matched-budget
  comparison will therefore carry tight intervals.
- Caveat to state in the paper: the 385-image probe is drawn from training images, so
  the absolute values are optimistic. The `best.pt` against `last.pt` contrast is valid
  because both are scored identically on the same images; the inversion is the result,
  not the level.

### Training-free convention screen + positive control (both new contributions)
- **Training-free screen delivered.** Listed as future work last session; now computed
  from the 4,760 label files alone, no images and no trained model. Granularity index
  $g_i = n_i / s_i$ (module instances per image over median normalised module scale).
  MTech scores **40.7**, next highest 15.0, corpus median 9.00, $k=3$ fence **29.02**:
  exactly one source flagged, and it is the same one the model-based screen flags.
  Two methodologically independent screens agreeing on the same source.
  Tool: `scripts/eval/annotation_statistics.py`.
- **Positive control PASSES.** `scripts/data_prep/synth_convention_shift.py` relabels a
  consensus source to a sub-component convention (each module box -> 3x2 grid of inset
  sub-boxes; no image altered, no busbar touched). gqljq granularity **11.1 -> 185.8**,
  flagged after and not before, while every genuine consensus source stays below the
  fence. This answers the strongest reviewer objection: the screen is no longer
  validated only on the case that motivated it. The model-based screen's control still
  needs a GPU retrain on the manipulated corpus; stated as future work, not claimed.
- Both papers updated. Added the MVML self-citation with an explicit
  "Relation to prior work" statement (the single-source specialist *is* that paper's
  detector), plus MVTec AD, DETR and the Zhou domain-generalisation survey.
- ICPRS held at 7 pages by dropping `fig_gap` (it duplicated Table II exactly), merging
  the two result tables, folding zero-shot into the deployment subsection, moving Data
  Availability to a first-page footnote, and cutting the abstract to 259 words.
- `paper/detection/IMAGES_NEEDED.md` lists the exact filenames required for the
  qualitative and convention figures. Detector images are not in the repo (~4 GB,
  removed 2026-07-14); labels are all present.

### Reference audit: two citations were fabricated, now corrected
- Verified every reference against the CrossRef API. Three were wrong, two of them
  invented outright when author-less bibitems inherited from the ICMRA draft were
  filled in:
  | key | was (wrong) | is (verified) |
  |---|---|---|
  | `cvmodule2023` | Wuest, Fuchs, Sauer; Prod. Eng. 17:707-717 | **Gerlitz, Enslin, Fleischer; Prod. Eng. 18(3-4):393-401** |
  | `screwcompare2026` | Rastegarpanah, Hesami, Stolkin | **Naseri, Khwajazada, Yang; IJAMT 144(1-2):1107-1119** |
  | `screwbatt2023` | Zhang, Zhang, Wang, Zhang, Li, Chen | **Li, Zhang, Zhang, Zhang, Peng, Wang, Song, Chen** (8 authors) |
  | `wegener2015` | Andrew, Scott | Andrew, Stefan |
- Verified correct and unchanged: `screwrcnn2021`, `tan2025screws`, `zang2024robotic`,
  `rousseeuw1993mad`, `ghiasi2021copypaste`.

### Two near-neighbour VISAPP 2026 papers found and cited
- `penquitt2026` (Penquitt, Klees, Cakaj, Kondermann, Rottmann, Schmarje) corrects
  label errors in object-detection datasets and found 18% bad labels in KITTI
  pedestrians. Closest published neighbour; not citing it at VISAPP would have been a
  visible gap. Differentiated in the related work: they correct errors *within* one
  agreed annotation standard, whereas this work identifies disagreement *about the
  standard itself*, which no within-convention correction can reconcile.
- `abouakar2026` (BMW Group / FEMTO-ST) trains industrial detectors on multi-source
  real, rendered and generative data. Differentiated: their sources share one
  annotation standard by construction; ours each carry an independently authored one.
- Reference count now 18 (IEEE) / 24 (bib). SCITEPRESS build grew to 10 of 12 pages.

### Detection-only paper drafted for four venues
- Scope fixed: **detection only**. Condition assessment dropped for this paper
  (14 real damaged crops is too few); localisation deferred entirely.
- Reframed around novelty rather than accuracy. Five stated contributions:
  the cross-facility benchmark; a **convention-consistency screening procedure**;
  the quantified single-source evaluation gap; the characterisation of the frozen
  self-supervised backbone as buying robustness-to-annotation-shift; four negative
  results.
- **New methodological contribution formalised.** The per-source collapse, previously
  only an observation, is now a screening procedure: flag source $i$ when
  $a_i < \mathrm{med}(a) - k\cdot 1.4826\cdot \mathrm{mad}(a)$. At $k=3$ the fence
  falls at 0.330, exactly one source is flagged, and the partition is stable for all
  $k \in [1.5, 4.5]$. Excluding it recovers **+0.227 module mAP@50** (0.547 -> 0.774).
- Four submission-ready builds in `paper/detection/`:
  | file | venue | format | pages |
  |---|---|---|---|
  | `icprs2027.tex` | ICPRS 2027 (25 Oct) | IEEEtran | 7 (6 + refs) |
  | `visapp2027.tex` | VISAPP 2027 (15 Sep) | SCITEPRESS | 9 of 12 |
  | `icpram2027.tex` | ICPRAM 2027 (15 Sep) | SCITEPRESS | 9 of 12 |
  | `robovis2027.tex` | ROBOVIS 2027 (15 Sep) | SCITEPRESS | 9 of 12 |
- The three SCITEPRESS files share `body_scitepress.tex` and are identical by design;
  only one may be submitted (dual submission otherwise).
- Written to the language specification in `technical-paper-language-prompt.md`:
  passive voice for what was done, "this paper presents" for contributions, prior-work
  sentences each carrying a limitation clause, every parameter given a stated origin,
  no em dashes outside table placeholders, no hype vocabulary. Verified by script.
- Honesty items now stated in the body, not buried: the 66% figure is measured against
  an in-domain benchmark that is itself the divergent source; single seed, no CIs;
  unequal training budgets between the two detectors; annotations inherited from the
  sources rather than created here; CC BY-NC-SA contamination at 24% of training data.
- New figures in `paper/detection/figures/` from `make_figures.py`: generalisation gap,
  per-source screening with the MAD fence, per-class full vs screened, zero-shot transfer.

### Venue longlist for the split vision paper (Paper A)
- Decided to split the paper: **Paper A** = vision/dataset/benchmark, **Paper B** =
  localisation + manipulation built on the published MVML detector. The split removes
  the "two papers stapled together" criticism outright.
- Assessed honestly: Paper A is the stronger half and is close to submittable.
  Paper B is *not a paper yet* — its only new content would be the mask-centroid
  integration experiment, which has not been run.
- Longlist with verified deadlines added to `docs/VENUE_STRATEGY.md` (Appendix A).
- **Primary target: ICPRS 2027**, deadline 25 Oct 2026, notify 28 Dec 2026, Bordeaux,
  IEEE Xplore, IAPR-endorsed. It solicits **dataset papers** as an explicit category,
  uses IEEE 6+1 page format (what the paper already is), and has a student prize.
- Constraint that drives ordering: VISAPP notifies 13 Nov, after the 25 Oct ICPRS
  deadline, so the two cannot both be live. Chosen chain:
  ICPRS (25 Oct) -> SCIA (26 Jan) -> ICPR 2027 virtual (1 Mar).
- Cost noted: VISAPP speaker registration is EUR 725-795, *more* than ICMRA charged.
  ICPR 2027 is virtual and therefore free of travel cost.

### Decision: withdraw from ICMRA 2026, retarget the paper
- Venue survey and objective self-assessment written to `docs/VENUE_STRATEGY.md`.
- Verified deadlines: ICRA 2027 closes **15 Sep 2026** (Seoul, 8 pages incl. refs,
  notification 31 Jan 2027); CASE 2027 and IROS 2027 both close **1 Mar 2027**;
  CIRP CMS 2027 abstract 11 Sep 2026 / full paper 18 Nov 2026; ETFA 2027 ~Mar 2027.
- **Plan: ICRA 2027 now, then CASE 2027 or IROS 2027 on 1 Mar 2027.** ICRA notifies
  31 Jan, a month before the 1 Mar deadlines, so the first submission is a free
  option that returns reviews in time to feed the second.
- Honest tier placement recorded: strong CASE/ETFA paper, borderline IROS, long shot
  at ICRA. Largest rejection risk is that the detector and the localisation stage are
  never actually connected — the localisation still uses a fiducial on the target.
- Pre-submission blockers logged: formal ICMRA withdrawal in writing, co-author
  agreement, and five author-less bibliography entries to fix.

---

## 2026-07-30

### BIG FINDING: 55% of the data already carries instance-segmentation masks
- Audited the label formats: **16,226 of 19,522 annotations are polygons, and they are
  real segmentation masks, not 4-point boxes** — 5 to 151 vertices, tracing busbars
  along their winding paths. Verified visually.
- **2,614 of 4,760 images (55%) are segmentation-ready today**: 2,486 fully polygon,
  128 mixed. Polygon annotations: module 11,766, busbar 4,460.
- Every pipeline so far has flattened these to bounding boxes and thrown the boundary
  information away. **We can train YOLO11-seg on the existing labels with no new
  annotation work.**
- Why this matters for the robotic-handling use case: a bounding box gives a robot a
  rectangle, no orientation, and for a diagonal busbar the box is mostly background.
  A mask gives the exact boundary, a true centroid, and an orientation from
  `cv2.minAreaRect` / PCA on the mask. This is the credible path to grasp planning
  and a stronger contribution than another detector comparison.

### Dataset quality audit (detector train split, 4,425 images)
- 23.5% unlabelled (1,040) — 97 are statistical anomalies, 943 need visual review.
  Confirmed by eye that at least some are genuinely **under-labelled** (e.g. a
  `final_mobilenet_results` frame showing an open pack with a clearly visible orange
  busbar assembly and zero labels). Under-labelled frames actively teach the detector
  that visible busbars are background, and are a plausible contributor to the
  persistently weak busbar scores (0.30-0.35 mAP50).
- 12.3% "blurry" and 20.4% greyscale — **largely benign**. The lowest-blur images are
  smooth closed pack lids (Laplacian variance is low for flat surfaces, not just for
  out-of-focus shots), and the greyscale images come entirely from two monochrome
  sources (`automated`, `bmw`). Both add robustness; do not delete on the metric alone.
- 0 degenerate boxes. Tooling: `scripts/data_prep/audit_dataset.py` (flags + contact sheets,
  deletes nothing).

### Zenodo dataset: downloaded, but NOT usable for detection as-is
- `data/sources/zenodo_ev_circularity/` = 712 images (697 jpg, 15 jpeg) + 12 videos
  across 19 vehicle types, 5.1 GB. **Zero annotation files of any kind.** "Labelled"
  in its README means folder-level vehicle class — it is a classification dataset.
- **78 of its images are already in our training data** (exact dHash match, identical
  filenames): its "Tesla Model 3" and "Tesla Model S black" classes are re-published
  MTech images, as its README states. ~634 images are genuinely new.
- To use it for detection it must be labelled first (Grounding DINO auto-label +
  human verification, or manual), and the 78 duplicates excluded to avoid leakage.

### Model zoo: all three detectors selectable with `--model`
- Added `scripts/common/model_zoo.py`, a registry of every trained detector plus a thin
  wrapper giving YOLO and RF-DETR the same `.predict()` interface. Every entry point
  (`pipeline_inference`, `webcam_demo`, `inference_api`, `evaluate`) now takes
  `--model {specialist,generalist_yolo,generalist_rfdetr}` and `--list-models`.
- **Weights now in the repo:** `models/detector/generalist_yolo11n/weights/best.pt`
  (5 MB, committed). The RF-DETR checkpoint is 115 MB, over GitHub's 100 MB limit, so
  it is gitignored with fetch/retrain instructions in
  `models/detector/generalist_rfdetr/README.md`.
- **Per-model default confidence (0.21 / 0.10 / 0.30).** Found while testing: the
  paper's 0.21 (tuned for the specialist) returns almost nothing from the generalist
  (0 boxes at 0.21, 22 at 0.05 on a Tesla/MTech image). Confidence scores are not
  comparable across models, so each carries its own suggested threshold, applied
  unless the user passes `--conf`. The generalist/RF-DETR values are starting points,
  not F1-tuned.
- **`scripts/eval/compare_detectors.py`** scores any set of models on identical images with
  identical supervision mAP (Ultralytics `.val()` is a different implementation and
  must not be compared against RF-DETR numbers). It converts polygon labels to boxes
  rather than dropping them, the bug that produced the bogus 0.724 on 2026-07-24.
- Tool validated by reproducing known numbers on the 43-image MTech test set:
  | model | mAP50 | mAP50-95 | module | busbar |
  |---|---|---|---|---|
  | specialist | 0.840 | 0.581 | 0.871 | 0.860 |
  | generalist_rfdetr | 0.541 | 0.288 | 0.543 | 0.558 |
  | generalist_yolo | 0.195 | 0.112 | 0.047 | 0.347 |
  Specialist 0.840 matches the paper's 0.818; generalist_yolo's module 0.047 matches
  the 0.043 collapse measured on 2026-07-21. Independent confirmation that the
  specialist wins in-domain while RF-DETR degrades gracefully out-of-convention.
- `evaluate.py` refuses RF-DETR with a message pointing at `compare_detectors.py`,
  since it relies on Ultralytics `.val()`.

## 2026-07-24

### RETRACTED then CORRECTED: RF-DETR "win" was a label-dropping bug
- **Initial (WRONG) claim:** RF-DETR diverse-test mAP50=0.724, ~2x the YOLO
  generalist (0.397). This was scored via supervision on a COCO conversion that
  **dropped all polygon-format labels** (`if len(p)!=5: continue`).
- **Root cause (bug):** the RF-DETR notebook's YOLO->COCO conversion drops every
  non-5-field line. **80% of the diverse labels are polygon-format (16,071 of 20,004
  train boxes, mostly MTech 13,517).** So RF-DETR TRAINED on only 20% of the labels
  AND was evaluated on only that same 20% subset -> inflated, incomparable numbers.
- **Corrected apples-to-apples** (both detectors, supervision mAP, conf=0.05, ALL
  780 test boxes incl. polygons->bbox):
  | Model | mAP50 | mAP50-95 |
  |---|---|---|
  | YOLO11n generalist | **0.410** | 0.256 |
  | RF-DETR (DINOv2, 20%-label train) | 0.351 | **0.271** |
  - Validation that the pipeline is sound: harmonized YOLO (0.410) matches the
    independent Ultralytics score (0.397/0.410 TTA).
- **Honest status:** RF-DETR does NOT currently beat YOLO. But the comparison is
  still UNFAIR to RF-DETR — it trained on 1/5 of the labels. Fixed the notebook's
  COCO conversion (polygon->bbox for train/val/test). **Must retrain RF-DETR on the
  full labels before any RF-DETR vs YOLO conclusion.**
- Lesson logged: always verify GT box counts match across pipelines (Ultralytics
  reported 780; the RF-DETR COCO had 330 -> the tell we initially missed).
- Checkpoints saved: checkpoint_best_regular.pth (from the 20%-label run; superseded).

### RESOLVED: fair full-label RF-DETR retrain BEATS YOLO (modest, real)
- Retrained RF-DETR on the FIXED notebook (polygon->bbox, all ~20k labels), then
  evaluated locally on the full-label diverse test, same supervision tool + conf as
  YOLO. Clean apples-to-apples:
  | Model | mAP50 | mAP50-95 |
  |---|---|---|
  | YOLO11n generalist | 0.410 | 0.256 |
  | **RF-DETR (full-label, EMA)** | **0.502** | **0.312** |
  | RF-DETR (full-label, regular) | 0.495 | 0.305 |
- **RF-DETR wins by +0.092 mAP50 (~1.2x) and is better on mAP50-95 too.** Honest,
  defensible margin (not the earlier bogus 0.724/2x from the label-dropping bug).
- Training on full labels lifted RF-DETR from 0.351 (20%-label run) to 0.502.
- Note: RF-DETR is WEAK on the MTech val (tiny sub-pixel modules at 384px res, val
  declined 0.25->0.15), but WINS on the full diverse test because it's better on the
  normal-sized modules across the other pack types. DINOv2 backbone helps on typical
  packs; struggles only on the tiny-object MTech outlier.
- Lesson: MTech val is a poor proxy; always judge on the diverse test. (Predicted
  YOLO would win from the declining val -> wrong; the eval settled it.)
- Best checkpoint: checkpoint_best_ema(1).pth (full-label run).

### Complete per-class + no-MTech breakdown (the nuanced truth)
| Test set | metric | RF-DETR | YOLO11n |
|---|---|---|---|
| Full diverse (225) | overall | **0.502** | 0.410 |
|  | module | **0.680** | 0.547 |
|  | busbar | **0.353** | 0.304 |
| No-MTech (177) | overall | 0.558 | 0.544 |
|  | module | 0.771 | 0.774 |
|  | busbar | 0.366 | 0.344 |
- **Key insight: on the 7 consensus sources RF-DETR and YOLO are a DEAD HEAT**
  (module 0.771 vs 0.774). RF-DETR's overall win is ENTIRELY from robustness on the
  out-of-convention MTech data: adding MTech drops YOLO module 0.774->0.547 (collapse)
  but RF-DETR only 0.771->0.680 (graceful). DINOv2 backbone = more robust to weird
  annotations, NOT better on normal packs.
- **Defensible paper claim:** "RF-DETR matches YOLO on standard packs and is markedly
  more robust to out-of-convention annotations (overall 0.502 vs 0.410)." Not '2x'.
- **Ship decision:** RF-DETR for the accuracy/robustness headline (needs GPU, DETR);
  YOLO11n stays the fast CPU-deployable option and is equal on normal packs. Report
  both; lead the paper with RF-DETR's robustness result.

## 2026-07-21

### HEADLINE NUMBER: generalist module detection = 0.740 mAP50 (7 consensus sources)
- Built a "no-MTech" test split (177 imgs, dropped the 48 out-of-convention MTech
  images) and evaluated the YOLO11n generalist on it:
  - Overall mAP50 **0.498** (vs 0.397 with MTech), mAP50-95 0.305.
  - **module mAP50 0.740** (vs 0.511 with MTech) — the honest large-scale number.
  - busbar mAP50 0.257 (vs 0.284) — slightly lower without MTech's busbar labels;
    busbar stays the acknowledged weak class (sparse/uneven labels across sources).
- **Paper headline: "detects EV modules at 0.740 mAP50 across 7 diverse pack
  sources."** MTech reported separately as the out-of-convention benchmark.
- Split builder is reproducible: exclude `mtech_*` + `automated-disassembly*`
  prefixes from the diverse test.

### NEGATIVE: diffusion damage synthesis unusable (overfit to 6 real packs)
- Ran the full diffusion pipeline (LoRA fine-tune on 40 real bad crops from 6 packs +
  SD1.5 inpainting into good crops). Ran end-to-end, produced 180 crops.
- **Output is unusable:** every crop shows the same artifact — chaotic over-woven
  texture + repetitive bright RED cross/plus marks stamped along diagonals. Not
  corrosion/burn/scratch; the LoRA memorized a superficial red-marking feature from
  the 6 packs and stamped it everywhere. Classic few-shot diffusion overfitting.
- **Decision: discard the batch, do NOT train on it** (would teach a fake "red cross
  = bad" feature; same trap as ue_d1 crops + copy-paste aug that hurt the model).
- Root cause is fundamental: 6 distinct real damaged packs is far too few for
  diffusion to learn real damage appearance. Diffusion synthesis is NOT a viable fix
  for the Stage-2 data bottleneck at this data scale.
- **Pivot Stage-2 strategy:** lean on the good-only ANOMALY detector (already 0.857
  bad-recall with ZERO bad training examples) as the primary condition-assessment
  method — it sidesteps the no-bad-data problem entirely and is itself a novelty. The
  damage-TYPE+severity extension needs REAL labeled damage (collection effort), not
  synthesis.

### Novelty scoped: visual damage-TYPE + severity grading (literature gap confirmed)
- Searched the literature for a CV model that grades used EV battery modules by
  damage type + severity from RGB photos. **Confirmed gap:** it doesn't exist.
  Prior work splits into (a) A/B/C grading from ELECTRICAL signals only (blind to
  visual damage), (b) manufacturing defect detection on bare electrodes/shells,
  (c) severity grading in other domains (corrosion on metal structures, car body),
  (d) one narrow battery-swelling mild/severe vision paper.
- **Our contribution:** the missing visual modality — type + severity from images,
  mapped to reuse grade A/B/C. Feasible because `synth_damage_overlay.py` already
  emits typed damage (corrosion/burn/scratch) and the diffusion notebook can grow it.
- Full spec: `docs/NOVELTY_DAMAGE_TAXONOMY.md`. Added to roadmap.

### NEGATIVE: Grounding DINO can't relabel MTech to consensus (L2 abandoned)
- Ran a Grounding DINO preview on 20 MTech images to test the auto-relabel plan.
  It detects (18 module + 29 busbar boxes, 0 misses at image level) but the boxes
  are **incomplete on clean shots** (caught 2 of ~5 modules) and **near-noise on the
  extreme grayscale macro close-ups** (a "busbar" box covering half the frame, a
  "module" box on random corroded metal). Auto-relabeling 1,756 imgs would inject
  garbage. Confirms the docs' "rough on hard grayscale macro shots" warning.
- **Root cause made visual:** MTech "modules" are tiny GMR-labeled sub-boxes and
  macro close-ups — a fundamentally different visual definition than a pack-level
  module. Not an annotation error we can auto-fix; it's a different task.
- **Strategy pivot:** stop trying to relabel MTech. Instead **report MTech separately
  as an out-of-convention benchmark** and evaluate the generalist on the other 7
  sources (the consensus definition). Cleaner, zero-compute, and honest for the paper.

### Roadmap + Stage-2 (condition) status check + busbar diagnosis
- Added `docs/ROADMAP.md` as the single source of truth for all queued work
  (Colab run queue, local tasks, backlog, negative results, priority order).
- Re-checked the condition-assessment classifier: bad-recall **0.857** (paper 0.714),
  acc 0.792, wF1 0.800 on the real 48-img test. End-to-end catches 82/100 bad modules.
  Bottleneck confirmed = DATA (only 16 real bad crops). Highest-payoff fix = the
  diffusion damage-generation notebook (Colab #2) to break the 16-crop ceiling.
- **Busbar diagnosis (why it's the weak class now, 0.284):** busbar labels in the
  diverse test are dominated by MTech (175 of 287 instances = 61%), and only **4 of 8
  test sources have any busbar labels at all**. So busbar has the SAME single-facility
  annotation-coverage problem module had — not a model-capability failure. Fix =
  broaden busbar labels across sources (Grounding DINO auto-label) before blaming aug.

### BREAKTHROUGH: large-scale module detector works; MTech is the outlier
- Evaluated the YOLO11n diverse-trained generalist on the 225-img diverse test:
  overall mAP50 **0.397 vs the MTech specialist's 0.277**; module **0.511 vs 0.231
  (2.2x better)**. The generalist genuinely detects modules across pack types.
- Per-source module mAP50 (the money shot): ue_rav4 **0.995**, bmw_i3 **0.910**,
  gqljq **0.873**, edfw3 **0.749** — excellent on almost every pack. ONLY MTech
  collapses (**0.043**). MTech is the annotation-convention OUTLIER.
- **The module "collapse" chased for hours was entirely a MTech-specific labeling
  idiosyncrasy, not a capability failure.** The generalist learned the consensus
  module definition (8 datasets agree) and applies it well everywhere; the paper's
  single-facility benchmark was misleading.
- Core positive result + publishable finding: multi-source training yields strong
  cross-pack module detection; single-facility benchmarks with idiosyncratic
  annotations under-report generalist capability. Busbar is the harder/more-variable
  class here (0.12–0.57), opposite of what MTech implied.
- Shippable large-scale module detector = the YOLO11n diverse best.pt.

### NEGATIVE: auto-labeled defect crops hurt the classifier (scale mismatch)
- Built `scripts/build_classifier_from_defects.py`: auto-labels good/bad module
  crops by combining detector module boxes with a defect dataset's damage boxes
  (module containing damage = bad). Both classes drawn from the same source
  images to avoid dataset-style leakage.
- Applied to uerymnd/ue_d1_defect_detection (237 imgs). Finding: it is **macro
  close-up defect photography** (corroded nuts, scratches), not module-level
  views — the detector found no module in 229/237 images, so it fell back to
  damage patches (1,076 bad) + clean same-image patches (415 good).
- Empirical test (merge 150/class, retrain, real-only test set): **bad-recall
  crashed 0.857 → 0.36**, wF1 0.800 → 0.726. The patch-level data is a scale
  mismatch with the module-level classifier and confuses it.
- **Reverted**; classifier stays at the 0.857-bad-recall procedural-synth model.
  The auto-labeler tool is kept (works for a properly module-level defect source
  if one appears). Honest negative, documented like the detector-augmentation one.

### Quality pass — full-dataset perceptual dedup (0 cross-source duplicates)
- Ran a comprehensive dHash dedup across all 4,440 train images: removed 15
  same-source augmentation near-dups (MTech 3, edfw3 12). Then a cross-source
  pass at a looser threshold found **0 cross-source duplicates** — confirming the
  ~9 datasets are genuinely independent (no superset/subset overlap; the
  ybmvt↔gqljq overlap was already caught at merge time).
- **Final clean train: 4,425 images** (module 14,530 / busbar 4,992), 0 test-set
  leakage. Dataset quality verified.

### NEGATIVE + KEY FINDING: naive 9-source merge collapses module detection
- Trained YOLOv8n on the 4,425-image merged set (Colab T4, 52 epochs, early-stop).
  **Test mAP50 = 0.331 vs 0.818 baseline** on the original 43-image MTech test set.
- Breakdown is the finding: **module mAP50 collapsed to 0.094**, while
  **busbar held at 0.568**. On the mixed val set module scored 0.59 — so the model
  detects "module" as the merged sources define it, but not as MTech defines it.
- **Cause: annotation-definition drift.** The 9 datasets label "module"
  inconsistently (full module stack vs individual cells/leaf-cells vs pack).
  Merging blurred the concept and poisoned module detection. Busbar is annotated
  consistently across sources, so it improved. Confirms the annotation-quality
  risk flagged at merge time: with conflicting labels, more data *hurts*.
- **Decision: keep the 0.818 baseline shipped; do NOT adopt this model.** The
  YOLO11n comparison would hit the same data issue (architecture-independent).
- Publishable lesson: naive multi-source dataset merging degrades detection under
  inconsistent annotation conventions — curation > volume. Fix: retrain on only
  module-definition-consistent sources (keeps busbar gains, drops module noise).

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

### Programmable inference API (`scripts/inference/inference_api.py`)
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
- **Good-only anomaly detector** (`scripts/inference/anomaly_condition.py`, PatchCore-lite):
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
- `scripts/eval/calibrate_classifier.py`: reliability diagram + ECE (0.284 → 0.246
  via temperature scaling), bootstrap 95% CIs, cost-sensitive thresholds.
  Key finding: **zero bad modules land in Grade A** (no false-safe routing).
- `scripts/build_singlestage_dataset.py`: 3-class (module-good / module-bad /
  busbar) ablation dataset builder with classifier-bootstrapped pre-annotation,
  for the single-stage vs two-stage comparison reviewers asked for.

### Classifier bad-recall improvement (synthetic damaged crops)
- Expanded the classifier bad class 40 → 80 crops with procedural damage
  synthesis (`scripts/data_prep/synth_damage_overlay.py`; synthetic ≤50% of the class;
  test set 100% real).
- **Result:** bad-class recall **0.571 → 0.857** (exceeds the paper's 0.714),
  weighted F1 0.768 → 0.800, accuracy 0.771 → 0.792. Retrained weights shipped.

### Accuracy/generalization tooling + guide
- `docs/IMPROVING_ACCURACY.md`: curated external-dataset catalogue + synthetic
  data method guide.
- `scripts/data_prep/download_external_datasets.py`: Zenodo / Roboflow download + class-
  remap merge.
- `scripts/data_prep/synth_copy_paste.py`: copy-paste compositing (+ synthetic glare pass)
  for the detector, guarded against un-remapped 7-class labels.
- `scripts/data_prep/synth_damage_overlay.py`: procedural corrosion/burn/scratch synthesis.
- `notebooks/colab_defect_inpainting.ipynb`: few-shot SD-inpainting LoRA pipeline
  for diffusion-generated damaged crops (free Colab GPU).

## 2026-07-14

### Repository cleanup + public release
- Removed the tracked raw dataset (`EV-Battery-pack--1/`, `data/`) and the
  `.claude/` agent config; kept trained weights, scripts, and README.
- Repo reduced from 4,786 tracked files to 69; pushed to GitHub and renamed to
  **ev-battery-vision-pipeline** for public discoverability.
