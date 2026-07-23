# Project Roadmap — single source of truth

This file is the master to-do so nothing gets forgotten across sessions. Update it
whenever a task changes state. Companion docs: `IMPROVING_ACCURACY.md` (methods +
datasets + modern-methods research), `REVIEWER_RESPONSE_PLAN.md` (BMVC replies),
`../CHANGELOG.md` (running log of every change).

The project has **two deliverables**, both must be accurate AND general:
1. **Detection** — find EV modules + busbars across many pack types (large-scale).
2. **Condition assessment** — classify each module good/bad → Grade A/B/C.

---

## STATUS SNAPSHOT (2026-07-23)

### Stage 1 — Detection
| Model | Diverse-test mAP50 | module | busbar | Verdict |
|---|---|---|---|---|
| MTech specialist (paper) | 0.277 | 0.231 | — | narrow; only knows one pack |
| **YOLO11n generalist** (current best) | **0.397** | **0.511** | 0.284 | WORKS at scale; ship candidate |
| YOLO11n + TTA | 0.410 | 0.541 | 0.279 | small free bump |
| RF-DETR (DINOv2) | _pending Colab_ | — | — | expected to beat 0.397 |

- Per-source module mAP50 is excellent: ue_rav4 0.995, bmw_i3 0.910, gqljq 0.873,
  edfw3 0.749. **Only MTech collapses (0.043)** — it is the annotation-convention
  outlier, not a capability failure.
- **busbar is now the weak/variable class (0.12–0.57)** — opposite of the MTech story.

### Stage 2 — Condition assessment (classifier)
| Metric | Current | Paper | Target |
|---|---|---|---|
| Bad-class recall | **0.857** | 0.714 | ≥0.892 (85% e2e), ≥0.944 (90% e2e) |
| Accuracy | 0.792 | 0.917* | ≥0.85 |
| Weighted F1 | 0.800 | 0.912* | ≥0.85 |

\* paper split likely leaky/easier. **Root limit: only 16 real "bad" crops, 48-img
test.** This is a DATA problem — fix it with synthetic damage + more real crops.

---

## COLAB RUN QUEUE (what to train, in order)

> Upload the data zip to Drive, open the notebook via its Colab badge, Run all,
> send me the printed metrics + the downloaded weights. I do the analysis locally.

1. **RF-DETR detector** — `notebooks/colab_train_rfdetr.ipynb`
   - Needs `ev_diverse_data.zip` in Drive (same zip as the YOLO11n run).
   - Goal: beat the YOLO11n generalist's 0.397 on the diverse test via a frozen
     DINOv2 backbone. Try `RFDETRNano`; if promising, rerun with `RFDETRSmall`.
   - Deliverable: diverse-test mAP50 + `ev_rfdetr_best.pth`.

2. **Diffusion damage generation (classifier)** — `notebooks/colab_defect_inpainting.ipynb`
   - Needs the 16 real bad crops zipped from `data/classifier/.../bad/`.
   - Goal: generate realistic damaged module crops → expand the bad class → lift
     bad-recall past 0.892. HIGHEST-payoff lever for condition assessment.
   - Deliverable: a zip of synthetic bad crops; I curate + retrain + eval on the
     REAL-ONLY test set (never let synthetic leak into test).

3. **(after 1) MTech-relabel retrain** — retrain the winning detector after MTech
   labels are fixed to the consensus convention (see local task L2). Confirms the
   outlier disappears and lifts the overall/mtech number.

4. **(optional) Ensemble / distillation** — if RF-DETR wins, ensemble it with
   YOLO11n for the paper number, then distil to YOLO11n for CPU deployment.

---

## LOCAL TASKS (I do these, no Colab needed)

- **L1 — Per-source & busbar diagnosis.** Break down busbar mAP by source to find
  why it is variable (0.12–0.57); decide if it needs relabeling or copy-paste aug.
- **L2 — MTech relabel to consensus.** Run `scripts/autolabel_grounding_dino.py` on
  MTech images to regenerate module labels in the consensus convention, human-spot-
  check, swap into the diverse train/test. Removes the 0.043 outlier.
- **L3 — Build a diverse classifier crop set.** Crop modules from the diverse packs
  to give the classifier varied GOOD examples (more pack styles, not just MTech).
- **L4 — k-fold CV for the classifier.** The 48-img test is tiny; add k-fold +
  bootstrap CIs so the reported number is trustworthy for the paper.
- **L5 — Anomaly detector write-up.** Formalize the good-only PatchCore-lite result
  (0.857 bad-recall with ZERO bad training examples) as the novel Stage-2 method —
  strong story for reviewers who want novelty.
- **L6 — Diverse val split.** Model selection currently uses MTech val (biases
  toward the outlier). Build a diverse val split so we pick models on the real goal.

---

## BACKLOG / IDEAS (don't forget, lower priority)

- Copy-paste compositing for busbar only (minority class; the general aug hurt, but
  busbar-targeted may help) — `scripts/synth_copy_paste.py`.
- Hyperparameter tuning (`model.tune()`) once the architecture is fixed.
- Grounding DINO auto-label the unlabeled Zenodo (712) + YouTube frames to grow data.
- Defect-texture pretraining for ResNet18 (Kaggle coating-defect, Sci.Reports set).
- Nissan e-NV200 busbar dataset (226 imgs) — email authors for availability.
- Procedural damage overlay as an ablation baseline vs diffusion crops.
- Cost-sensitive threshold + temperature scaling already done (`calibrate_classifier.py`).

## NEGATIVE RESULTS (proven, do NOT repeat — see CHANGELOG)
- Naive multi-source merge → module mAP collapse (label inconsistency).
- Copy-paste + glare aug on full detector → mAP collapse.
- ue_d1 defect crops into classifier → bad-recall crashed (scale mismatch).
- YOLO-World zero-shot detection → 0.004 mAP (unusable as a detector).
- DINOv2 frozen features for the classifier → worse bad-recall than ResNet18.

---

## PRIORITY ORDER (agreed with user)
1. RF-DETR (Colab #1) — push detection accuracy/generality further.
2. MTech relabel (L2 + Colab #3) — kill the outlier.
3. Busbar improvement (L1 → targeted aug).
4. Condition assessment: diffusion damage (Colab #2) + L3/L4/L5 — make Stage 2 brilliant.
