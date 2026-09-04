# The paper: scope, thesis and structure

Written 2026-09-04. Supersedes the ICMRA framing entirely. This paper is the
follow-on to BMVC #1674, built from work logged in `CHANGELOG.md` between
2026-07-14 and 2026-07-30.

---

## 1. The thesis

> In EV battery vision the binding constraint is **data, not model capacity** — and
> it binds differently at each stage. At detection the constraint is **annotation
> convention**. At condition assessment it is **the absence of the damaged class**.
> Each demands a different response, and in both cases the response is a frozen
> self-supervised representation rather than a bigger model.

This is one claim tested twice, not two papers stapled together. The unifying
technical thread is DINOv2: its frozen backbone gives graceful degradation under
annotation shift at Stage 1, and its patch tokens carry the good-only anomaly
detector at Stage 2.

## 2. What we actually have (all measured, all in the repo)

### Stage 1 — detection
| Result | Number |
|---|---|
| Single-facility specialist, in-domain | 0.818 mAP@50 |
| Same specialist, cross-facility (225 imgs) | 0.277 |
| Multi-source YOLO11n, cross-facility | 0.410 |
| Multi-source RF-DETR (frozen DINOv2) | **0.502** |
| Both, on 7 convention-consistent sources | 0.544 / 0.558 (dead heat) |
| Per-source module mAP@50 spread | 0.995 -> 0.043 |
| Naive 9-source merge | 0.331 (module 0.094) |
| Zero-shot probe, 17 unseen pack types | 0.76 mean detection rate |
| INT8 ONNX | 44.8 ms, 3.4 MB, -0.007 mAP |

### Stage 2 — condition
| Result | Number |
|---|---|
| Backbone benchmark (identical protocol) | DINOv2 0.792 = EffNet-B0 0.792 > ShuffleNet 0.771 > MobileNetV3 0.708 > **ResNet18 0.688** |
| Procedural synthetic damage | bad-recall 0.571 -> **0.857** |
| **Good-only anomaly, ResNet18 layer3** | AUROC 0.603, bad-recall 0.357 |
| **Good-only anomaly, DINOv2 patch tokens** | **AUROC 0.702, bad-recall 0.857, zero damaged training data** |
| Calibration | ECE 0.284 -> 0.246 (T=0.55); bootstrap 95% CIs reported |
| Cost-sensitive triage, 5:1 miss cost | bad-recall 0.929 |
| Safety property | **zero bad modules routed to Grade A** |

### Documented negative results (a genuine asset — four of them)
1. Diffusion damage synthesis unusable at 6 real damaged packs (memorised red-cross artefact).
2. Copy-paste + glare augmentation degrades the detector (val 0.503 -> 0.27, declining).
3. Auto-labelled defect crops crash the classifier (bad-recall 0.857 -> 0.36, scale mismatch).
4. Grounding DINO cannot relabel the out-of-convention source; YOLO-World scores 0.004 zero-shot.

## 3. Which BMVC criticisms this answers

| # | BMVC criticism | Answered? |
|---|---|---|
| 1 | No methodological novelty (all 4 reviewers) | **Yes** — good-only anomaly detection |
| 3 | No lightweight/quantised benchmarking | **Yes** — 5 backbones + INT8 |
| 5 | Small test sets (54 imgs / 14 bad) | **Partly** — detection now 225 imgs; condition still 48/14 |
| 6 | Weak bad-module recall (0.714) | **Yes** — 0.857, and 0.929 cost-calibrated |
| 7 | No calibration / CIs / cost analysis | **Yes** |
| 8 | Single facility | **Yes** — 13 sources, cross-facility benchmark, 17 unseen variants |
| 2 | No single-stage vs two-stage ablation | No — needs per-box condition labels |
| 9 | Single annotator | No — needs a second annotator pass |

Eight of eleven. The two that remain are the ones to declare in Limitations.

## 4. Structure

1. Introduction — the data-centric claim, stated up front
2. Related work — battery disassembly perception; domain shift; anomaly detection
3. Data and benchmarks — 13 sources, 4,425 train, 225-image cross-facility test, audit
4. **Part I. Detection: the constraint is annotation convention**
   - the generalisation gap, the architecture comparison, the per-source collapse,
     the naive-merge failure
5. **Part II. Condition: the constraint is the missing class**
   - backbone benchmark, good-only anomaly detection, calibration and triage
6. What did not work — the four negative results
7. Limitations
8. Conclusion

## 5. What must be done before submission

Ranked by value per hour.

- [ ] **Bootstrap CIs on every detection number.** Tooling already exists for the
      classifier (`scripts/eval/calibrate_classifier.py`); extend to detection. Hours.
- [ ] **Seed reruns (n=3) for YOLO11n and RF-DETR at matched epochs and resolution.**
      The current comparison is confounded (150 ep @640 vs 60 ep @384). Colab time.
- [ ] **Enlarge the condition test set.** 14 bad crops is the single biggest weakness
      and reviewers flagged it twice. Any real damaged data helps.
- [ ] **Gold re-label ~100 test images to a written convention.** Turns the paper's
      internal contradiction (label quality is the thesis, labels are unvalidated)
      into a contribution. 1-2 days.
- [ ] **Fix the annotation-contribution wording.** Polygons are inherited from the
      sources, not created here — `DATASETS.md` says so. Free, and a credibility issue.
- [ ] **Decide the licence.** gqljq is CC BY-NC-SA 4.0 and is 24% of training data;
      ShareAlike is viral. Free, and blocking for release.
- [ ] **Reframe the 66% claim.** The specialist's in-domain benchmark is itself the
      convention-divergent source. State this, or add a second in-domain reference.

## 6. Explicitly out of scope

ArUco localisation, the UR5e, monocular metric pose — all deferred to a later paper
(see `INTEGRATION_WITH_LOCALISATION.md`). Damage type + severity taxonomy: not run.
Segmentation training: not run. Single-stage ablation: not run.

## 7. Venue

The unified paper wants **10-14 pages**, which rules out a 6+1 page container.

- **SCIA 2027** — deadline 26 Jan 2027, Springer LNCS, Gjovik. Best fit for the
  full-length version, and the extra time covers the experiment list in §5.
- **ICPR 2027** — deadline 1 Mar 2027, Springer LNCS, virtual (no travel cost).
- **ICPRS 2027** — deadline 25 Oct 2026, IEEE 6+1 pp. Only viable if the paper is
  cut to Part I *or* Part II, not both.

See `VENUE_STRATEGY.md` Appendix A for the full longlist.
