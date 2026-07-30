# Repository structure

```
ev-battery-vision-pipeline/
│
├── README.md                     project overview and quick start
├── CHANGELOG.md                  every post-submission change, newest first
├── requirements.txt
├── dataset.yaml                  YOLO dataset config for the merged detector splits
│
├── data/
│   ├── sources/                  ONE FOLDER PER ORIGIN DATASET (see below)
│   ├── detector/                 merged train/val/test used for training
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── classifier/               good/bad module crops
│   │   ├── train/{good,bad}/  test/{good,bad}/
│   └── labels_release/           annotations published on GitHub (17 MB, no images)
│
├── models/
│   ├── detector/
│   │   ├── baseline_yolov8n_stage1/    first training stage (historical)
│   │   ├── specialist_yolov8n/         paper baseline, single-facility
│   │   ├── generalist_yolo11n/         10-source diverse training
│   │   └── generalist_rfdetr/          DINOv2 backbone, best diverse score
│   └── classifier_resnet18/            good/bad condition classifier
│
├── scripts/
│   ├── common/       model_zoo.py — the model registry every script selects from
│   ├── data_prep/    download, remap, audit, build, auto-label, synthesise (14)
│   ├── train/        train_detector.py, train_classifier.py
│   ├── eval/         evaluate, compare, benchmark, calibrate (7)
│   └── inference/    pipeline, webcam, HTTP api, anomaly detector (4)
│
├── notebooks/        Colab training notebooks (GPU work)
├── docs/             plans, dataset provenance, method notes
└── outputs/          generated results (gitignored)
```

## Model registry

Every inference and evaluation script selects a detector **by name**, never by
hardcoded path:

```bash
python scripts/inference/pipeline_inference.py --input img.jpg --model generalist_yolo
python scripts/eval/evaluate.py --list-models
```

| Name | Architecture | Diverse-test mAP50 |
|---|---|---|
| `specialist` | YOLOv8n, single facility | 0.277 (0.818 in-domain) |
| `generalist_yolo` *(default)* | YOLO11n, 10 sources | 0.410 |
| `generalist_rfdetr` | RF-DETR, DINOv2 backbone | **0.502** |

Confidence scores are not comparable across architectures, so each model carries its
own sensible default threshold; `--conf` overrides it.

## `data/sources/` — one folder per origin

Raw datasets are kept exactly as downloaded, one directory per source, so provenance
is never ambiguous and any single source can be excluded (which matters: licences
differ, and one is non-commercial).

```
data/sources/
├── roboflow_battery_comp/            280 imgs   CC BY 4.0
├── roboflow_last_exp4/               483 imgs   CC BY 4.0
├── roboflow_ev-battery-component-detection-gqljq/  1,045 imgs   CC BY-NC-SA 4.0
└── ...                               (13 Roboflow sources + Zenodo)
```

Full attribution, URLs and licence terms: [`DATASETS.md`](DATASETS.md).

The merged `data/detector/` splits are **built from** these sources — treat
`data/sources/` as read-only input and the merged splits as regenerable output.

## Which files live in git

| | In git | Why |
|---|---|---|
| Code, notebooks, docs | yes | small, text |
| Labels (`data/labels_release/`) | yes, 17 MB | the curated artefact; text, diffable |
| Model weights | yes, 5–45 MB each | under the 100 MB per-file limit |
| Images (~4 GB) | see below | too large for a comfortable clone |
| Zenodo videos (2.1 GB) | no | already public at the original DOI |
| Training-batch mosaics | no | regenerated on every run |
| Pretrained bases (`yolov8n.pt` …) | no | auto-downloaded by ultralytics |

## Reproducing the dataset

```bash
python scripts/data_prep/download_external_datasets.py     # fetch the sources
python scripts/data_prep/remap_labels.py                   # 7-class -> module/busbar
python scripts/data_prep/audit_dataset.py \
    --images data/detector/images/train --labels data/detector/labels/train
```
