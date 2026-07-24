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

## Status
- [ ] Define rubric + relabel existing bad crops with type + severity
- [ ] Diffusion-generate typed/severity crops (Colab #2)
- [ ] Two-head classifier + eval on real-only test
- [ ] Compare vs binary baseline (does typing help the A/B/C decision?)
