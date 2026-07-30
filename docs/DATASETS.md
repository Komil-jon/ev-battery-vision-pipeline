# Datasets: sources, licences, attribution

Every image in this project comes from a third-party dataset. This file is the
attribution record — CC BY 4.0 **requires** attribution when redistributing, so this
table must ship with any release of the data.

## Sources

| Directory | Source | Images | Licence |
|---|---|---|---|
| `roboflow_ev-battery-component-detection-gqljq` | [ev-battery/ev-battery-component-detection-gqljq](https://universe.roboflow.com/ev-battery/ev-battery-component-detection-gqljq/dataset/1) | 1,045 | **CC BY-NC-SA 4.0** |
| `roboflow_ev-battery-components-edfw3` | [academic-lsrwt/ev-battery-components-edfw3](https://universe.roboflow.com/academic-lsrwt/ev-battery-components-edfw3/dataset/1) | 680 | CC BY 4.0 |
| `roboflow_last_exp4` | [ca-2kt9o/last_exp4](https://universe.roboflow.com/ca-2kt9o/last_exp4/dataset/1) | 483 | CC BY 4.0 |
| `roboflow_ev-battery-pack-62ig0` | [mtech-project-ohj8a/ev-battery-pack](https://universe.roboflow.com/mtech-project-ohj8a/ev-battery-pack/dataset/1) | 435 | CC BY 4.0 |
| `roboflow_battery_comp` | [ca-2kt9o/battery_comp](https://universe.roboflow.com/ca-2kt9o/battery_comp/dataset/1) | 280 | CC BY 4.0 |
| `roboflow_ue_d1_defect_detection` | [uerymnd/ue_d1_defect_detection](https://universe.roboflow.com/uerymnd/ue_d1_defect_detection/dataset/1) | 237 | CC BY 4.0 |
| `roboflow_final_mobilenet_results` | [tina-eslami/final_mobilenet_results](https://universe.roboflow.com/tina-eslami/final_mobilenet_results/dataset/1) | 216 | CC BY 4.0 |
| `roboflow_automated-disassembly` | [auto-dissasembly/automated-disassembly](https://universe.roboflow.com/auto-dissasembly/automated-disassembly/dataset/1) | 117 | CC BY 4.0 |
| `roboflow_bmw_i3` | [uerymnd/bmw_i3](https://universe.roboflow.com/uerymnd/bmw_i3/dataset/1) | 111 | CC BY 4.0 |
| `roboflow_ue_rav4_module` | [uerymnd/ue_rav4_module](https://universe.roboflow.com/uerymnd/ue_rav4_module/dataset/1) | 96 | CC BY 4.0 |
| `roboflow_ev-battery-sample-ybmvt` | [ev-battery/ev-battery-sample-ybmvt](https://universe.roboflow.com/ev-battery/ev-battery-sample-ybmvt/dataset/1) | 92 | MIT |
| `roboflow_battery-modules` | [tina-eslami/battery-modules](https://universe.roboflow.com/tina-eslami/battery-modules/dataset/1) | 85 | CC BY 4.0 |
| `roboflow_radbs-yda1j` | [rabds/radbs-yda1j](https://universe.roboflow.com/rabds/radbs-yda1j/dataset/2) | 11 | CC BY 4.0 |
| `zenodo_ev_circularity` | [Zenodo 19818270](https://zenodo.org/records/19818270) (RECIRCULATE) | 712 | CC BY 4.0 |

## Licence constraint to be aware of before publishing

**`gqljq` is CC BY-NC-SA 4.0**, not CC BY 4.0. That is the largest single source
(1,045 images) and it carries two conditions the others do not:

* **NonCommercial** — the combined dataset cannot be licensed for commercial use.
* **ShareAlike** — derivative works must be released under the same licence. This is
  viral: a merged dataset containing gqljq must itself be BY-NC-SA.

Three options, in order of preference:

1. **Release without gqljq.** The remaining sources are CC BY 4.0 / MIT, so the
   release can be CC BY 4.0. Cleanest, and costs 1,045 of ~3,900 images.
2. **Release the whole thing under CC BY-NC-SA 4.0.** Keeps all data, but the release
   becomes non-commercial and ShareAlike-encumbered.
3. **Release labels only, keep gqljq's images out.** See below.

Not deciding is also a decision — publishing a merged CC BY 4.0 archive that contains
gqljq would be a licence violation.

## Overlap already found

The Zenodo set's "Tesla Model 3" and "Tesla Model S black" classes are re-published
MTech images: **78 of its 712 images are byte-identical duplicates** of images already
in our training data (verified by dHash). Deduplicate before any merge, or the same
images end up in both train and test.

## What is ours

The trained models, the merged/cleaned splits, the evaluation code and the analysis
in this repository are our own work. The **polygon annotations** are inherited from
the sources above, not created here — any claim about "the largest EV module/busbar
segmentation set" must be phrased as *assembled and curated from public sources*, with
this table attached.

## Distribution plan

| Artifact | Size | Where | Why |
|---|---|---|---|
| Labels (`*.txt`) | ~2 MB | **GitHub** | Tiny, text, diffable; the curated part |
| Code, notebooks, docs | small | GitHub | — |
| Model weights | 6–120 MB each | GitHub (small) / Drive-Zenodo (RF-DETR) | 100 MB is GitHub's hard per-file limit |
| Images | ~4 GB | **Zenodo** | Too large for GitHub; Zenodo gives a citable DOI |
| Zenodo videos | 2.1 GB | not redistributed | Already public at the original DOI |

**Do not push the images to GitHub.** GitHub rejects files over 100 MB, recommends
repositories stay under ~1 GB, and the free Git-LFS tier gives 1 GB of storage and
1 GB/month of bandwidth — well under the ~4 GB of images here. Zenodo is free,
designed for research data, and issues a DOI you can cite in the paper.
