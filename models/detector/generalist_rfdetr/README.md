# generalist_rfdetr weights

The RF-DETR checkpoint (`weights/checkpoint_best_ema.pth`, ~115 MB) is **not stored
in git** — GitHub rejects files over 100 MB. Everything else (code, config, the two
YOLO models) is in the repo; only this one file must be fetched separately.

## Get the weights

Download `ev_rfdetr_best.pth` from the project Drive folder (or ask the maintainer),
then:

```bash
mkdir -p models/detector/generalist_rfdetr/weights
cp /path/to/ev_rfdetr_best.pth models/detector/generalist_rfdetr/weights/checkpoint_best_ema.pth
```

Verify it is registered:

```bash
python scripts/model_zoo.py
```

`generalist_rfdetr` should show `(OK)` instead of `(MISSING)`.

## Reproduce instead of downloading

`notebooks/colab_train_rfdetr.ipynb` retrains it from scratch on a Colab GPU
(~60 epochs, several hours on a T4). The notebook writes
`checkpoint_best_ema.pth` and `checkpoint_best_regular.pth`; either works, take
whichever the final log reports as best.

## Runtime requirement

```bash
pip install rfdetr
```

Without it, `--model generalist_rfdetr` raises a clear ImportError and the YOLO
models still work.

## What this model is

RF-DETR-Nano with a frozen DINOv2 backbone, trained on the 10-source diverse
dataset with **all** labels (polygon annotations converted to boxes, not dropped).

| Test set | mAP50 | module | busbar |
|---|---|---|---|
| Full diverse (225 imgs) | 0.502 | 0.680 | 0.353 |
| No-MTech (177 imgs) | 0.558 | 0.771 | 0.366 |

It ties `generalist_yolo` on standard pack images (module 0.771 vs 0.774) and wins
overall by degrading gracefully on the out-of-convention MTech data, where the YOLO
models collapse. See `CHANGELOG.md` (2026-07-24).
