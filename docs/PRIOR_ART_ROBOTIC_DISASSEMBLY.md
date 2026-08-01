# Prior art: robotic EV battery disassembly, and how they got their data

Surveyed 2026-07-30, prompted by "how did the Birmingham unscrewing pipeline get its
dataset?". The short answer that matters for us: **none of them found a public
dataset — every group photographed and labelled their own, and almost none released
it.**

## The Birmingham group

Prof. **Duc Truong Pham** (Chance Chair of Engineering, School of Mechanical
Engineering) leads this line of work, usually with Wuhan University of Technology.

| Paper | What it does | Data |
|---|---|---|
| [Robotic Removal and Collection of Screws in Collaborative Disassembly of EoL EV Batteries](https://pubmed.ncbi.nlm.nih.gov/40862925/) (Biomimetics, 2025) — Tan, Huang, Jiang, Fang, Liu, **Pham** | Cobot + electric spindle + screw-collection device + **3D camera** + 6-axis force/torque sensor. Human-inspired: unfasten then grasp, fusing **vision and touch**. Position + force control. | **YOLOv8** for screw detection. Dataset self-collected; no public release. |
| [Robotic disassembly of EV batteries: technologies and opportunities](https://research.birmingham.ac.uk/en/publications/robotic-disassembly-of-electric-vehicle-batteries-technologies-an/) (Computers & Industrial Engineering, 2024) — Zang, Qu, **Pham**, Dixon, Goli, Zhang, Wang | Open-access review of the whole field (repair / remanufacture / recycle) | Review, no dataset |
| Projects: **ATARI** (2022-24, self-learning robotics for contact-rich tasks), Royal Society (2019-22, disassembly replanning), **REBELION** (Birmingham leads automation/robotics, vision-guided dismantling + adaptive digital twin) | | |

Their vision-based unfastening for pack-to-module disassembly reports **>90%
efficiency in unscrewing** — but note that is the *unscrewing* step with force
feedback, not vision alone.

## How everyone actually builds the dataset

| Work | Dataset | Public? |
|---|---|---|
| [An Accurate Activate Screw Detection Method](https://www.mdpi.com/2313-0105/9/3/187) (Batteries, 2023) | **1,719 RGB images** photographed from different EV batteries; labelled by hand in **LabelImg**; 6,992 external hexagonal screws + 687 hexagonal nuts | No link found |
| [Comparative analysis of YOLO vs Faster R-CNN for EV battery screw detection](https://link.springer.com/article/10.1007/s00170-026-17984-z) (IJAMT, 2026) | A **proxy dataset of laptop screws** plus a **custom-collected EV screw set**. States plainly that the field is "hampered by a lack of public datasets" | No |
| [Accurate screw detection: Faster R-CNN + rotation edge similarity](https://www.tandfonline.com/doi/abs/10.1080/0951192X.2021.1963476) (IJCIM, 2021) | Own images | No |
| [RAPID](https://arxiv.org/abs/2603.18520v1) (2026) | Own rig, RGB-D | Platform open-source; dataset unclear |

**Pattern:** photograph your own packs → label by hand (LabelImg / similar) → train →
keep the data. A dataset in the 1,500–1,700 image range with hand labels is the
*normal* size for a credible paper in this area.

## What this means for us

1. **Our plan is the standard method, not a shortcut.** Self-photographing and
   hand-labelling 1,000–1,500 polygon-annotated images is exactly what these groups
   did (1,719 images in the Batteries 2023 paper). We are not under-resourced.
2. **Releasing our annotations is a real contribution.** Every paper above notes the
   absence of public data and none fixes it. Ours would.
3. **They all pair vision with another modality.** Birmingham's cobot fuses vision +
   force/torque; RAPID uses RGB-D. Nobody trusts RGB detection alone to drive the
   arm — consistent with RAPID's 57% one-shot vision grasp vs 83% with visual
   servoing. Reinforces framing our contribution as perception, with manipulation as
   future work.
4. **Different target.** They detect **screws/fasteners** for unfastening. We detect
   **modules and busbars** for extraction and grading. Complementary, not competing —
   in a full line both are needed, and the user already has the unscrewing stage.
