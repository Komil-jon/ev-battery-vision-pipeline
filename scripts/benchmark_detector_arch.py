"""
benchmark_detector_arch.py
==========================
Fair architecture comparison for the detector: trains YOLOv8n and YOLO11n (and
optionally YOLOv8s) from COCO-pretrained weights with an IDENTICAL reduced
schedule on the same dataset, then reports test-split mAP, CPU latency, and
parameter count. Answers the reviewer criticism that no alternative detector
architecture was compared on the same data.

This uses a REDUCED epoch budget (default 40) so it finishes on CPU — it is a
like-for-like architecture comparison at a fixed budget, NOT a reproduction of
the shipped 100-epoch Stage 1 model. Both architectures get the exact same
budget, so the comparison is fair; absolute numbers will be below the shipped
model's.

Usage:
    python scripts/benchmark_detector_arch.py                     # v8n vs 11n, 40 epochs
    python scripts/benchmark_detector_arch.py --epochs 60 --models yolov8n yolo11n yolov8s
Results printed as a markdown table + saved to outputs/detector_arch_benchmark.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "detector" / "arch_benchmark"
CSV_OUT = ROOT / "outputs" / "detector_arch_benchmark.csv"
TEST_IMGS = ROOT / "data" / "detector" / "images" / "test"


def cpu_latency_ms(model, imgs, imgsz, n_warm=2):
    for p in imgs[:n_warm]:
        model(str(p), imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for p in imgs:
        t0 = time.perf_counter()
        model(str(p), imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_one(name, epochs, imgsz, batch, yaml_path):
    from ultralytics import YOLO
    print(f"\n=== training {name} ({epochs} epochs @ {imgsz}) ===")
    model = YOLO(f"{name}.pt")
    model.train(
        data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=batch,
        optimizer="SGD", lr0=0.01, weight_decay=0.0005,
        hsv_v=0.30, hsv_s=0.25, hsv_h=0.0, mosaic=0.5,
        degrees=2.0, translate=0.06, fliplr=0.5,
        device="cpu", project=str(OUT_DIR), name=name, exist_ok=True,
        verbose=False, plots=False,
    )
    # Evaluate on the TEST split for the reported number
    metrics = model.val(data=str(yaml_path), split="test", imgsz=imgsz, device="cpu",
                        verbose=False, plots=False, project=str(OUT_DIR),
                        name=f"{name}_test", exist_ok=True)

    imgs = sorted(TEST_IMGS.glob("*.jpg"))[:20]
    lat = cpu_latency_ms(model, imgs, imgsz)
    n_params = sum(p.numel() for p in model.model.parameters())
    return {
        "model": name,
        "params_M": round(n_params / 1e6, 2),
        "test_mAP50": round(float(metrics.box.map50), 3),
        "test_mAP50_95": round(float(metrics.box.map), 3),
        "latency_ms": round(lat, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Fair YOLOv8n vs YOLO11n detector comparison")
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data", type=str, default=str(ROOT / "dataset.yaml"),
                    help="Dataset YAML (default: dataset.yaml)")
    args = ap.parse_args()

    results = [run_one(m, args.epochs, args.imgsz, args.batch, args.data) for m in args.models]
    results.sort(key=lambda r: r["test_mAP50"], reverse=True)

    print(f"\n\n## Detector architecture comparison "
          f"(identical {args.epochs}-epoch budget @ {args.imgsz}, test split)\n")
    print("| Model | Params (M) | Test mAP50 | Test mAP50-95 | CPU latency (ms) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['params_M']} | {r['test_mAP50']} "
              f"| {r['test_mAP50_95']} | {r['latency_ms']} |")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {CSV_OUT.relative_to(ROOT)}")
    print(f"NOTE: reduced {args.epochs}-epoch budget for CPU; fair v8-vs-v11 "
          "comparison, not the shipped 100-epoch model.")


if __name__ == "__main__":
    main()
