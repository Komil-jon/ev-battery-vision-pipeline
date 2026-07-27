# Novelty: Visual Damage-Type + Severity Grading for Used EV Battery Modules

## The gap (literature-verified 2026-07-24)
No published CV model classifies **external damage TYPE + severity** on *used*
EV battery modules from ordinary RGB photos and maps it to a reuse grade. The
pieces exist only separately:

| Prior work | What it does | Why it's not this |
|---|---|---|
| AI-Integrated Smart Grading (retired Li-ion) | Grade A/B/C output | Uses SOH/DCIR/temp/voltage — **electrical only, no vision** |
| Li-battery surface defect detectors (YOLO/transformer) | pits/cracks/tabs on electrodes | **Manufacturing QC** on bare cells/shells, no severity→grade |
| Corrosion grading Level A–D | 4 severity levels | **Hydraulic metal structures**, not batteries |
| Car damage + severity | dents/scratches + severity | **Automotive body**, not batteries |
| Battery swelling detection (2025) | mild vs severe swelling | **One type, 2 levels**, mostly ESS packs |

Refs: AI grading (ResearchGate 397467392); Li-battery seg (Sci.Rep s41598-025-18315-0);
corrosion grading (MDPI applsci 14/24/12009); swelling (ScienceDirect S2352152X25044597);
car damage (Roboflow car-damage-kadad).

Also checked (2026-07-27): Kaggle "EV Battery Defect Detection | ML + SHAP"
(haritmengar) — sklearn (RF/GBM/LogReg/SVC)+SMOTE+SHAP on a SYNTHETIC TABULAR QC csv
(1,000 rows; temp, anode overhang, electrolyte volume, internal resistance, capacity,
retention). No images. Falls squarely in the "electrical/sensor-only" bucket — does
NOT overlap the visual damage-type+severity gap; it reinforces the motivation.

## Our contribution (one sentence)
A CV model that reads external damage **type** and **severity** on used EV battery
modules from RGB images, supplying the visual modality that electrical A/B/C grading
systems are blind to (corrosion, burns, casing cracks, swelling, leakage).

## Proposed taxonomy
**Damage types (multi-label — a module can have several):**
1. corrosion / oxidation (terminals, busbars)
2. thermal / burn marks
3. physical deformation — dent / casing crack
4. swelling / bulging
5. electrolyte leakage / residue
6. surface scratch / abrasion

**Severity (per module, 3 levels → reuse grade):**
- minor → Grade A (reusable)
- moderate → Grade B (repurposable)
- severe → Grade C (recycle)

## Why feasible here
- `scripts/synth_damage_overlay.py` ALREADY generates corrosion/burn/scratch as
  distinct types — the typed labels come for free.
- The diffusion generator (Colab #2, `colab_defect_inpainting.ipynb`) can synthesize
  typed + severity-varied damage to populate minority classes.
- Model change: binary good/bad classifier → shared backbone with **two heads**
  (multi-label damage-type + ordinal severity). Keep the real-only held-out test.

## Honest caveats to state in the paper
- Real damaged data is tiny (6 distinct damaged packs). Type/severity labels will
  lean on synthetic generation → must validate on a real, human-graded test set,
  however small, and report per-type support honestly.
- Severity is subjective; define a rubric (area-% affected, functional risk) and
  report inter-rater agreement if possible.

## Factory reality (informs scope; literature-verified 2026-07-24)
Retired pack → **module** (primary grading/inspection point for second-life) →
**cell** (only for suspect modules or recycling pathways). Not every module is torn
down; inspection stops at the reusable unit. Refs: teardown steps (ScienceDirect
S2666386421002484), remanufacture-to-cell (Springer s13243-020-00088-6), disassembly
obstacles (MDPI processes 13/1/123).

Implication for our scope: a module-level RGB model sees EXTERNAL damage (corrosion,
burn, casing crack, dent, leakage, gross swelling). Cell-venting / subtle per-cell
swelling is better at cell level; internal shorts/dendrites need electrical/CT.
**State the scope as "module-level external damage from RGB," complementary to the
electrical channel — not a replacement for it.**

## Damage-type table (commonality / visibility / CV fit)
| Type | Common | Module-visible | Current tech | CV difficulty | CV better? |
|---|---|---|---|---|---|
| Corrosion/oxidation | very | yes | electrical partial | medium | yes (earlier) |
| Thermal/burn marks | moderate | yes | electrical mostly no | medium | yes (CV-only) |
| Casing crack/dent | common | yes (hairline hard) | no | medium-hard | yes |
| Swelling/bulging | common | partial (gross) | electrical partial | HARD (3D cue) | complementary |
| Electrolyte leakage | moderate | yes (stains) | no | medium | yes |
| Surface scratch | very | yes | no | easy | grade-only |
| Connector/busbar dmg | moderate | yes | electrical partial | medium | yes |
| Internal short/dendrite | dangerous | NO | electrical/CT only | out of scope | no |

CV sweet spot = corrosion, burn, crack/dent, leakage, connector damage (electrical
is blind to these). Swelling is the hard RGB case (needs side view). Internal faults
are the honest out-of-scope boundary.

## Two-head classifier feasibility
Architecture is trivial (shared ResNet18 → multi-label type head + ordinal severity
head; loss = BCE(type)+λ·CE(severity)). REAL work is labels: only 6 real damaged
packs, so lean on synthetic (synth_damage_overlay already emits typed damage; severity
= overlay intensity/area) and VALIDATE on a small human-graded real test set with
honest per-type support. Rollout: ship binary good/bad→A/B/C first, add type+severity
heads as the novelty extension.

## Status
- [ ] Define rubric + relabel existing bad crops with type + severity
- [ ] Diffusion-generate typed/severity crops (Colab #2)
- [ ] Two-head classifier + eval on real-only test
- [ ] Compare vs binary baseline (does typing help the A/B/C decision?)
