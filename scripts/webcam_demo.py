"""
webcam_demo.py
==============
Live two-stage inference from a laptop webcam, using the exact same pipeline as
pipeline_inference.py:
  Stage 1 — YOLOv8n detects module / busbar in each frame
  Stage 2 — ResNet18 classifies each module crop -> Grade A / B / C

Controls (while the video window is focused):
  q  — quit
  s  — save the current annotated frame to outputs/results/

Usage:
    python scripts/webcam_demo.py
    python scripts/webcam_demo.py --camera 1 --conf 0.21 --imgsz 640
    python scripts/webcam_demo.py --input demo.mp4        # run on a video file

macOS note: the terminal app you launch this from needs camera permission
(System Settings -> Privacy & Security -> Camera -> enable your terminal).
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

# Reuse the shared pipeline building blocks so behaviour matches the batch script.
from pipeline_inference import (
    load_detector, load_classifier, get_grade,
    CLASSIFIER_TRANSFORM, GRADE_COLOURS,
    DETECTOR_WEIGHTS, CLASSIFIER_WEIGHTS, CLASS_MAP_PATH, OUTPUT_DIR,
)


def process_frame(frame, detector, classifier, bad_idx, conf, imgsz, crop_padding=10):
    """Annotate a single BGR frame in place; return (annotated, n_modules, n_busbars)."""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    n_mod = n_bus = 0

    results = detector(frame, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
    for r in results:
        for box in r.boxes:
            label = detector.names[int(box.cls)]
            det_conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "module":
                n_mod += 1
                px1, py1 = max(0, x1 - crop_padding), max(0, y1 - crop_padding)
                px2, py2 = min(w, x2 + crop_padding), min(h, y2 + crop_padding)
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = CLASSIFIER_TRANSFORM(pil).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(classifier(tensor), dim=1)[0]
                    p_bad = probs[bad_idx].item()
                grade, colour = get_grade(p_bad)
                text = f"module Grade {grade} p_bad={p_bad:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, text, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            elif label == "busbar":
                n_bus += 1
                colour = (255, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, f"busbar {det_conf:.2f}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    return annotated, n_mod, n_bus


def main():
    ap = argparse.ArgumentParser(description="Live webcam two-stage EV battery pipeline")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--input", type=str, default=None, help="Optional video file instead of webcam")
    ap.add_argument("--conf", type=float, default=0.21, help="Detection confidence (paper optimum ~0.21)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size; lower = faster (e.g. 480)")
    ap.add_argument("--detector", type=str, default=str(DETECTOR_WEIGHTS))
    ap.add_argument("--classifier", type=str, default=str(CLASSIFIER_WEIGHTS))
    args = ap.parse_args()

    print("Loading models...")
    detector = load_detector(Path(args.detector))
    classifier, bad_idx = load_classifier(Path(args.classifier), CLASS_MAP_PATH)
    print(f"  conf={args.conf}  imgsz={args.imgsz}  bad_idx={bad_idx}")

    source = args.input if args.input else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\nERROR: could not open {'video '+args.input if args.input else f'camera {args.camera}'}.")
        print("On macOS, grant camera permission to your terminal:")
        print("  System Settings -> Privacy & Security -> Camera -> enable your terminal app,")
        print("  then fully quit and reopen the terminal and rerun.")
        return

    print("Running. Focus the video window, then press 'q' to quit, 's' to save a frame.")
    fps_smooth, saved = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream / no frame.")
            break

        t0 = time.perf_counter()
        annotated, n_mod, n_bus = process_frame(
            frame, detector, classifier, bad_idx, args.conf, args.imgsz)
        dt = time.perf_counter() - t0
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if dt > 0 else fps_smooth

        hud = f"{fps_smooth:4.1f} FPS | modules:{n_mod} busbars:{n_bus} | imgsz {args.imgsz}"
        cv2.putText(annotated, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("EV Battery Pipeline (q=quit, s=save)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / f"webcam_{int(time.time())}.jpg"
            cv2.imwrite(str(out), annotated)
            saved += 1
            print(f"  saved {out}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} frame(s).")


if __name__ == "__main__":
    main()
