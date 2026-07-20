"""
benchmark_detector_cpu.py
=========================
CPU deployment benchmark for the detector, answering the reviewer criticism
that the YOLOv8n choice was never compared against quantised/exported variants
on the same data.

Compares, on the held-out test split and the same M-series CPU:
  1. YOLOv8n PyTorch (current shipped model)
  2. ONNX FP32 export (onnxruntime CPU EP)
  3. ONNX INT8 dynamic quantisation

reporting mAP50 / mAP50-95 (ultralytics val on the exported model) and median
per-image latency, plus file size — the deployment trade-off table reviewers
asked for.

Usage:
    python scripts/benchmark_detector_cpu.py
    python scripts/benchmark_detector_cpu.py --skip_map    # latency + size only
Requires: pip install onnx onnxruntime onnxslim
"""

import argparse
import time
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
YAML    = ROOT / "dataset.yaml"
TEST_IMGS = ROOT / "data" / "detector" / "images" / "test"
OUT_DIR = ROOT / "outputs"

IMGSZ = 768   # matches pipeline_inference.py


def size_mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def latency_ms_yolo(model, image_paths, n_warm=2):
    for p in image_paths[:n_warm]:
        model(str(p), imgsz=IMGSZ, device="cpu", verbose=False)
    times = []
    for p in image_paths:
        t0 = time.perf_counter()
        model(str(p), imgsz=IMGSZ, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def val_map(model_path, name):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    r = m.val(data=str(YAML), split="test", imgsz=IMGSZ, device="cpu",
              verbose=False, plots=False, project=str(OUT_DIR), name=f"val_{name}",
              exist_ok=True)
    return r.box.map50, r.box.map


def main():
    ap = argparse.ArgumentParser(description="Detector CPU deployment benchmark")
    ap.add_argument("--skip_map", action="store_true",
                    help="Skip mAP validation (latency and size only)")
    ap.add_argument("--n_images", type=int, default=20,
                    help="Test images used for latency timing (default: 20)")
    args = ap.parse_args()

    from ultralytics import YOLO
    imgs = sorted(TEST_IMGS.glob("*.jpg"))[:args.n_images]
    if not imgs:
        raise SystemExit(f"No test images in {TEST_IMGS}")

    rows = []

    # 1 — PyTorch
    print("=== YOLOv8n PyTorch ===")
    pt_model = YOLO(str(WEIGHTS))
    rows.append({"variant": "PyTorch FP32", "file": WEIGHTS,
                 "size_mb": round(size_mb(WEIGHTS), 1),
                 "latency_ms": round(latency_ms_yolo(pt_model, imgs), 1)})

    # 2 — ONNX FP32
    print("=== Exporting ONNX FP32 ===")
    onnx_path = Path(pt_model.export(format="onnx", imgsz=IMGSZ, device="cpu"))
    onnx_model = YOLO(str(onnx_path), task="detect")
    rows.append({"variant": "ONNX FP32", "file": onnx_path,
                 "size_mb": round(size_mb(onnx_path), 1),
                 "latency_ms": round(latency_ms_yolo(onnx_model, imgs), 1)})

    # 3 — ONNX INT8 dynamic quantisation
    print("=== Quantising ONNX INT8 (dynamic) ===")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
    int8_model = YOLO(str(int8_path), task="detect")
    rows.append({"variant": "ONNX INT8", "file": int8_path,
                 "size_mb": round(size_mb(int8_path), 1),
                 "latency_ms": round(latency_ms_yolo(int8_model, imgs), 1)})

    if not args.skip_map:
        for row, (path, name) in zip(rows, [(WEIGHTS, "pt"), (onnx_path, "onnx"),
                                            (int8_path, "int8")]):
            print(f"=== mAP validation: {row['variant']} ===")
            m50, m5095 = val_map(path, name)
            row["mAP50"], row["mAP50_95"] = round(m50, 3), round(m5095, 3)

    print("\n\n## Detector CPU deployment benchmark "
          f"(imgsz={IMGSZ}, median latency over {len(imgs)} test images)\n")
    cols = ["variant", "size_mb", "latency_ms"] + \
           (["mAP50", "mAP50_95"] if not args.skip_map else [])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")


if __name__ == "__main__":
    main()
