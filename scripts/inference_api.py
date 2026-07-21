"""
inference_api.py
================
Programmable interface to the two-stage pipeline. Turns an image (file path,
numpy array, or raw bytes) into structured results — bounding-box positions plus
per-module condition grade — as plain Python dicts / JSON. This is the layer you
call from other software (a conveyor controller, a web service, a robot cell).

What each detection contains:
    class        "module" | "busbar"
    confidence   detector confidence, 0-1
    box_xyxy     [x1, y1, x2, y2]  pixel coords (top-left, bottom-right)
    box_xywh     [x, y, w, h]      pixel coords (top-left + size)
    center       [cx, cy]          pixel coords of the box centre
    box_norm     same as xyxy but divided by image size (0-1, resolution-independent)
    grade        "A" | "B" | "C"   (modules only; None for busbars)
    p_bad        damaged probability 0-1 (modules only)

Three ways to use it:

1. As a library (import and call):
       from scripts.inference_api import BatteryInspector
       insp = BatteryInspector()
       result = insp.infer("photo.jpg")          # -> dict
       for d in result["detections"]:
           print(d["class"], d["box_xyxy"], d.get("grade"))

2. As a CLI returning JSON (pipe into anything):
       python scripts/inference_api.py --input photo.jpg
       python scripts/inference_api.py --input photo.jpg --json_only > result.json

3. As a local HTTP service (POST an image, get JSON back):
       python scripts/inference_api.py --serve --port 8000
       curl -F image=@photo.jpg http://localhost:8000/infer
"""

import argparse
import io
import json
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

ROOT               = Path(__file__).resolve().parent.parent
DETECTOR_WEIGHTS   = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = ROOT / "models" / "classifier" / "resnet18_binary.pth"
CLASS_MAP_PATH     = ROOT / "models" / "classifier" / "class_map.json"

GRADE_A_THRESHOLD = 0.30
GRADE_C_THRESHOLD = 0.70

_CLF_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _grade(p_bad: float) -> str:
    if p_bad < GRADE_A_THRESHOLD:
        return "A"
    if p_bad < GRADE_C_THRESHOLD:
        return "B"
    return "C"


class BatteryInspector:
    """Loads both models once; call .infer() repeatedly (e.g. per camera frame)."""

    def __init__(self, detector_weights=DETECTOR_WEIGHTS,
                 classifier_weights=CLASSIFIER_WEIGHTS,
                 class_map_path=CLASS_MAP_PATH, conf=0.21, imgsz=768):
        self.conf = conf
        self.imgsz = imgsz
        self.detector = YOLO(str(detector_weights))
        self.names = self.detector.names

        self.classifier = models.resnet18()
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 2)
        self.classifier.load_state_dict(torch.load(str(classifier_weights), map_location="cpu"))
        self.classifier.eval()

        self.bad_idx = 0
        if Path(class_map_path).exists():
            self.bad_idx = json.load(open(class_map_path)).get("bad", 0)

    def _to_bgr(self, image: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """Accept a path, an OpenCV/numpy BGR array, or raw image bytes."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
            return img
        if isinstance(image, bytes):
            arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        if isinstance(image, np.ndarray):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def infer(self, image: Union[str, Path, np.ndarray, bytes],
              crop_padding: int = 10) -> dict:
        """Run the full pipeline on one image. Returns a JSON-serialisable dict."""
        img = self._to_bgr(image)
        h, w = img.shape[:2]
        t0 = time.perf_counter()

        det_res = self.detector(img, conf=self.conf, imgsz=self.imgsz,
                                device="cpu", verbose=False)[0]

        detections = []
        for box in det_res.boxes:
            cls_id = int(box.cls)
            label = self.names[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            rec = {
                "class": label,
                "confidence": round(conf, 4),
                "box_xyxy": [x1, y1, x2, y2],
                "box_xywh": [x1, y1, x2 - x1, y2 - y1],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                "box_norm": [round(x1 / w, 5), round(y1 / h, 5),
                             round(x2 / w, 5), round(y2 / h, 5)],
                "grade": None,
                "p_bad": None,
            }

            if label == "module":
                px1, py1 = max(0, x1 - crop_padding), max(0, y1 - crop_padding)
                px2, py2 = min(w, x2 + crop_padding), min(h, y2 + crop_padding)
                crop = img[py1:py2, px1:px2]
                if crop.size:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    probs = torch.softmax(self.classifier(_CLF_TF(pil).unsqueeze(0)), dim=1)[0]
                    p_bad = float(probs[self.bad_idx])
                    rec["p_bad"] = round(p_bad, 4)
                    rec["grade"] = _grade(p_bad)
            detections.append(rec)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "image_size": {"width": w, "height": h},
            "n_modules": sum(1 for d in detections if d["class"] == "module"),
            "n_busbars": sum(1 for d in detections if d["class"] == "busbar"),
            "detections": detections,
            "latency_ms": round(elapsed, 1),
        }


def _serve(inspector: "BatteryInspector", host: str, port: int):
    """Minimal stdlib HTTP server: POST an image to /infer, get JSON back."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter logs
            pass

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # Accept either raw image bytes or multipart/form-data 'image'
            ctype = self.headers.get("Content-Type", "")
            data = body
            if "multipart/form-data" in ctype and b"\r\n\r\n" in body:
                # crude multipart: take the payload between the first blank line and last boundary
                parts = body.split(b"\r\n\r\n", 1)[1]
                data = parts.rsplit(b"\r\n--", 1)[0]
            try:
                result = inspector.infer(data)
                payload = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(400, str(e))

    print(f"Serving inspector on http://{host}:{port}/infer  (POST an image)")
    HTTPServer((host, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Programmable EV battery inspection API")
    ap.add_argument("--input", type=str, help="Image file to run once")
    ap.add_argument("--conf", type=float, default=0.21)
    ap.add_argument("--json_only", action="store_true", help="Print only JSON")
    ap.add_argument("--serve", action="store_true", help="Run as an HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    inspector = BatteryInspector(conf=args.conf)

    if args.serve:
        _serve(inspector, args.host, args.port)
        return

    if not args.input:
        ap.error("provide --input IMAGE (or --serve)")

    result = inspector.infer(args.input)
    if args.json_only:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
        print(f"\n{result['n_modules']} module(s), {result['n_busbars']} busbar(s), "
              f"{result['latency_ms']} ms")


if __name__ == "__main__":
    main()
