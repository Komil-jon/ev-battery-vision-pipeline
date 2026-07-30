# EV battery module/busbar annotations

The annotation half of this project's detector dataset: **4,760 label files,
16,945 polygon (instance-segmentation) masks and 3,889 bounding boxes** for two
classes across 13 EV battery pack sources.

The images are **not** here — they are ~4 GB, far past what GitHub is meant to hold.
Rebuild the image set with `scripts/download_external_datasets.py`, then pair it with
these labels by filename (`<stem>.jpg` ↔ `<stem>.txt`).

## Layout

```
detector/
├── train/   4,425 label files
├── val/       292
└── test/       43
```

## Format

YOLO text format, one object per line, all coordinates normalised to `[0, 1]`.
Two kinds of line appear:

```
# bounding box:  cls  cx cy w h
0 0.512 0.433 0.180 0.121

# polygon mask:  cls  x1 y1 x2 y2 x3 y3 ...
1 0.31 0.22 0.34 0.21 0.37 0.25 0.36 0.31 ...
```

Classes: `0 = module`, `1 = busbar`.

Polygons are genuine masks (5–151 vertices, tracing busbars along their winding
paths), not rectangles written as four corners. 55% of images are fully
polygon-annotated and can train an instance-segmentation model as-is.

## Licence — read before redistributing

**CC BY-NC-SA 4.0.**

These annotations are derived from third-party datasets. Most are CC BY 4.0, but one
source (`ev-battery-component-detection-gqljq`, 941 of these label files) is
**CC BY-NC-SA 4.0**, whose ShareAlike term is viral: any collection containing it must
carry the same licence. The most restrictive licence therefore governs the whole set.

Consequences: **non-commercial use only**, and derivatives must be shared alike.

If you need a CC BY 4.0 (commercially usable) subset, exclude every file whose name
begins with the gqljq source prefix; the remainder is CC BY 4.0 / MIT.

Full per-source attribution and URLs: [`docs/DATASETS.md`](../../docs/DATASETS.md).
Attribution is a licence requirement, not a courtesy — keep that table with any copy.
