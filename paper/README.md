# ICMRA 2026 paper

**Venue:** 8th International Conference on Mechatronics, Robotics and Automation,
13--15 November 2026, Suzhou, China. Proceedings go to IEEE Xplore, indexed by EI
Compendex and Scopus.

**Format:** IEEE conference template, 4--10 pages including figures, tables and
references. **Submission deadline: 5 August 2026.**

## Files

```
icmra2026_ev_battery.tex     the paper (IEEEtran, conference option)
figures/fig_gap.pdf          generalisation gap, in-domain vs cross-facility
figures/fig_persource.pdf    per-source module mAP, the convention outlier
figures/fig_perclass.pdf     per-class comparison across the three detectors
make_figures.py              regenerates all figures
```

## Compiling

No LaTeX is installed locally. Easiest route is Overleaf:

1. Create a project from the official **IEEE Conference Template** (or upload
   `IEEEtran.cls`).
2. Upload `icmra2026_ev_battery.tex` and the `figures/` folder.
3. Compile with pdfLaTeX.

Locally instead:

```bash
brew install --cask mactex-no-gui   # large; or basictex + tlmgr install ieeetran
pdflatex icmra2026_ev_battery.tex
pdflatex icmra2026_ev_battery.tex   # twice, for references
```

If ICMRA supplies its own Word template and requires it, the section order and
content transfer directly; only the markup changes.

## Every number in the paper, and where it came from

| Claim | Value | Source |
|---|---|---|
| Specialist in-domain | 0.818 mAP@50 | prior submission; reproduced at 0.840 with `compare_detectors.py` |
| Specialist cross-facility | 0.277 | CHANGELOG 2026-07-21 |
| YOLO11n cross-facility | 0.410 | harmonised eval, 2026-07-24 |
| RF-DETR cross-facility | 0.502 / 0.312 | full-label retrain, 2026-07-24 |
| Per-class (module/busbar) | see Table III | `eval_breakdown.py` |
| Consensus subset (177 imgs) | 0.558 vs 0.544 | no-MTech split |
| Per-source module mAP | 0.995 … 0.043 | CHANGELOG 2026-07-21 |
| Anomaly detector recall | 0.857, AUROC 0.702, 0 damaged examples | `anomaly_condition.py` |
| INT8 quantisation | −20% latency, −46% size, −0.007 mAP | `benchmark_detector_cpu.py` |
| YOLO-World zero-shot | 0.004 mAP@50 | negative result, logged |
| Diffusion synthesis failure | 180 unusable crops | negative result, logged |
| 78 duplicate images | dHash exact match | dedup check, 2026-07-30 |

## Before submitting

- [ ] Confirm the author block: affiliation, and whether supervisors are co-authors
- [ ] Check ICMRA's exact template (Word vs LaTeX) and any anonymity requirement
- [ ] Complete the reference entries — several need full author lists, DOIs and page
      numbers, currently abbreviated
- [ ] Make the GitHub repository public if the data-availability statement stays
- [ ] Confirm the licence position on redistributed annotations (one source is
      CC BY-NC-SA, see `../docs/DATASETS.md`)
- [ ] Re-read Limitations: it is deliberately explicit, which strengthens the paper,
      but confirm you are comfortable with each admission
