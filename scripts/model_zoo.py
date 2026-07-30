"""
model_zoo.py
============
Central registry of every trained detector in this project, plus a thin wrapper
that gives YOLO and RF-DETR models the same `.predict(...)` interface. Every
inference script (pipeline_inference.py, webcam_demo.py, evaluate.py,
inference_api.py) selects a model by NAME through this file instead of hardcoding
a weights path, so switching models is one flag: `--model NAME`.

Models:
  specialist          paper baseline, YOLOv8n trained on one facility (MTech).
                       Best on that facility's images (mAP50 0.818 in-domain), but
                       collapses on other pack types (0.277 on the diverse test).
  generalist_yolo      YOLO11n trained on 10 diverse EV pack sources. mAP50 0.410
                       on the diverse test, 0.740 module-only on the 7 consensus
                       sources (excl. the MTech outlier). Fast, CPU-friendly.
  generalist_rfdetr    RF-DETR-Nano (frozen DINOv2 backbone) trained on the same
                       diverse data (full labels). mAP50 0.502 on the diverse test
                       -- tied with generalist_yolo on normal packs, but far more
                       robust on out-of-convention data. Needs a GPU for real-time
                       use and the `rfdetr` package (pip install rfdetr).

See CHANGELOG.md (2026-07-21 to 2026-07-24 entries) for the full evaluation history.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = {0: "module", 1: "busbar"}


@dataclass
class ModelInfo:
    kind: str          # "yolo" or "rfdetr"
    label: str          # human-readable description
    weights: Path
    default_conf: float = 0.21
    notes: str = ""


MODEL_REGISTRY = {
    "specialist": ModelInfo(
        kind="yolo",
        label="Specialist (paper baseline, single-facility MTech)",
        weights=ROOT / "models" / "detector" / "specialist_yolov8n" / "weights" / "best.pt",
        default_conf=0.21,  # paper F1-confidence optimum (Appendix I)
        notes="mAP50 0.818 in-domain (MTech) / 0.277 on the diverse cross-facility test.",
    ),
    "generalist_yolo": ModelInfo(
        kind="yolo",
        label="Generalist YOLO11n (10-source diverse training)",
        weights=ROOT / "models" / "detector" / "generalist_yolo11n" / "weights" / "best.pt",
        default_conf=0.10,  # NOT tuned: scores lower than the specialist, 0.21 misses most boxes
        notes="mAP50 0.410 diverse test / 0.740 module-only on 7 consensus sources. CPU-friendly.",
    ),
    "generalist_rfdetr": ModelInfo(
        kind="rfdetr",
        label="Generalist RF-DETR (DINOv2 backbone, full-label diverse training)",
        weights=ROOT / "models" / "detector" / "generalist_rfdetr" / "weights" / "checkpoint_best_ema.pth",
        default_conf=0.30,  # DETR-style scores; 0.30 used in the Colab eval
        notes="mAP50 0.502 diverse test. Tied with generalist_yolo on normal packs, "
              "more robust on out-of-convention data. Needs `pip install rfdetr` + GPU for real-time.",
    ),
}

DEFAULT_MODEL = "generalist_yolo"


def list_models() -> str:
    lines = ["Available detector models (--model NAME):", ""]
    for name, info in MODEL_REGISTRY.items():
        default_tag = "  [default]" if name == DEFAULT_MODEL else ""
        present = "OK" if info.weights.exists() else "MISSING"
        lines.append(f"  {name}{default_tag}")
        lines.append(f"      {info.label}")
        lines.append(f"      {info.notes}")
        lines.append(f"      suggested --conf {info.default_conf}")
        lines.append(f"      weights: {info.weights}  ({present})")
        lines.append("")
    lines.append("Note: confidence scores are not comparable across models. Each model's")
    lines.append("suggested --conf is applied automatically unless you pass --conf yourself.")
    return "\n".join(lines)


class Detection:
    __slots__ = ("cls_id", "label", "conf", "xyxy")

    def __init__(self, cls_id: int, label: str, conf: float, xyxy: tuple):
        self.cls_id = cls_id
        self.label = label
        self.conf = conf
        self.xyxy = xyxy  # (x1, y1, x2, y2) ints


class Detector:
    """Unified predict() interface over an Ultralytics YOLO model or an RF-DETR model."""

    def __init__(self, name: str = None, weights_path: str = None):
        if name is None and weights_path is None:
            name = DEFAULT_MODEL

        if name is not None:
            if name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model '{name}'. " + list_models())
            info = MODEL_REGISTRY[name]
            self.kind, weights_path, self.label = info.kind, str(info.weights), info.label
            self.default_conf = info.default_conf
        else:
            # explicit path override: assume yolo (ultralytics) for backward compatibility
            self.kind, self.label = "yolo", f"custom ({weights_path})"
            self.default_conf = 0.21

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {weights_path}\n"
                f"({self.label})\n"
                "If this is generalist_rfdetr, download the checkpoint separately "
                "(see models/detector/generalist_rfdetr/README.md) -- it's too large "
                "for a plain git checkout."
            )

        self.names = CLASS_NAMES
        if self.kind == "yolo":
            from ultralytics import YOLO
            self._model = YOLO(str(weights_path))
        elif self.kind == "rfdetr":
            try:
                from rfdetr import RFDETRNano
            except ImportError as e:
                raise ImportError(
                    f"The '{self.label}' model needs the rfdetr package.\n"
                    "Install it with:  pip install rfdetr\n"
                    "Or use a YOLO model instead: --model generalist_yolo"
                ) from e
            self._model = RFDETRNano(pretrain_weights=str(weights_path))
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    def predict(self, image, conf: float = 0.21, imgsz: int = 768):
        """image: a file path (str/Path) or a BGR numpy array (cv2 frame).
        Returns a list of Detection."""
        if self.kind == "yolo":
            return self._predict_yolo(image, conf, imgsz)
        return self._predict_rfdetr(image, conf)

    def _predict_yolo(self, image, conf, imgsz):
        results = self._model(image, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append(Detection(cls_id, self._model.names[cls_id], float(box.conf), (x1, y1, x2, y2)))
        return out

    def _predict_rfdetr(self, image, conf):
        from PIL import Image
        import numpy as np
        import cv2

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image

        det = self._model.predict(pil_img, threshold=conf)
        out = []
        for xyxy, cls_id, score in zip(det.xyxy, det.class_id, det.confidence):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(int, xyxy)
            out.append(Detection(cls_id, self.names.get(cls_id, str(cls_id)), float(score), (x1, y1, x2, y2)))
        return out


def load_detector(name: str = None, weights_path: str = None) -> Detector:
    return Detector(name=name, weights_path=weights_path)


if __name__ == "__main__":
    print(list_models())
