"""
harvest_youtube_frames.py
=========================
Harvests training frames from EV battery teardown / repair videos and
pre-labels them with the current detector, producing a human-review queue.
This is the cheapest route to new pack variants, camera poses, and lighting
conditions beyond the single-facility dataset.

LICENSING: only harvest videos you are allowed to reuse. Filter YouTube search
by Creative Commons licence (Filters → Features → Creative Commons), or use
your own recordings. Downloading arbitrary videos may violate YouTube's ToS;
frames of others' footage also need care at publication time. The tool takes
explicit URLs only — it does not crawl.

Pipeline per video:
  1. yt-dlp downloads at ≤720p                     (pip install yt-dlp)
  2. Frames are sampled every --interval seconds, then de-duplicated by
     perceptual difference (mean absolute pixel delta)
  3. The current detector pseudo-labels each kept frame; frames with no
     detection at conf ≥ --min_conf are dropped
  4. Frames + YOLO label files land in data/youtube_harvest/review/
     — these are PSEUDO-labels: review and correct them (LabelImg/Roboflow)
     before ever merging into data/detector/ train (never into val/test).

Usage:
    python scripts/harvest_youtube_frames.py --urls URL [URL ...]
    python scripts/harvest_youtube_frames.py --urls URL --interval 3 --min_conf 0.4
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "detector" / "stage2_recall_boost" / "weights" / "best.pt"
OUT_DIR = ROOT / "data" / "youtube_harvest" / "review"


def download(url: str, tmpdir: Path) -> Path:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    out_tmpl = str(tmpdir / "%(id)s.%(ext)s")
    subprocess.check_call([
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4",
        "-o", out_tmpl, url,
    ])
    vids = list(tmpdir.glob("*.mp4")) + list(tmpdir.glob("*.mkv")) + list(tmpdir.glob("*.webm"))
    if not vids:
        raise RuntimeError(f"yt-dlp produced no video file for {url}")
    return vids[0]


def sample_frames(video: Path, interval_s: float, dedup_thresh: float):
    """Yield (timestamp_s, frame) roughly every interval_s, skipping near-duplicates."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * interval_s))
    prev_small = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (64, 64)).astype(np.float32)
            if prev_small is None or np.abs(small - prev_small).mean() > dedup_thresh:
                yield idx / fps, frame
                prev_small = small
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(description="Harvest + pseudo-label frames from EV battery videos")
    ap.add_argument("--urls", nargs="+", required=True,
                    help="Video URLs (Creative-Commons or own footage only)")
    ap.add_argument("--interval",  type=float, default=2.0, help="Seconds between samples")
    ap.add_argument("--min_conf",  type=float, default=0.40,
                    help="Keep frames with ≥1 detection above this confidence")
    ap.add_argument("--dedup",     type=float, default=8.0,
                    help="Min mean pixel delta (0-255) vs last kept frame")
    ap.add_argument("--max_frames", type=int, default=200, help="Cap per video")
    args = ap.parse_args()

    from ultralytics import YOLO
    detector = YOLO(str(WEIGHTS))
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    total_kept = 0
    for url in args.urls:
        with tempfile.TemporaryDirectory(dir=str(OUT_DIR.parent)) as td:
            print(f"\n=== {url} ===")
            video = download(url, Path(td))
            vid_id = video.stem
            kept = 0
            for ts, frame in sample_frames(video, args.interval, args.dedup):
                if kept >= args.max_frames:
                    break
                res = detector(frame, conf=args.min_conf, imgsz=768,
                               device="cpu", verbose=False)[0]
                if len(res.boxes) == 0:
                    continue
                h, w = frame.shape[:2]
                stem = f"yt_{vid_id}_{int(ts):05d}s"
                cv2.imwrite(str(OUT_DIR / "images" / f"{stem}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = []
                for box in res.boxes:
                    cls = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (OUT_DIR / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                kept += 1
            print(f"kept {kept} frames with detections")
            total_kept += kept

    print(f"\nDone. {total_kept} pseudo-labelled frames → {OUT_DIR.relative_to(ROOT)}")
    print("\nNEXT STEPS (required before training on these):")
    print("  1. Review EVERY pseudo-label (LabelImg / Roboflow Annotate) — the")
    print("     detector's own mistakes must not be trained back into it.")
    print("  2. Move reviewed pairs into data/detector/{images,labels}/train/.")
    print("  3. Never place harvested frames in val/ or test/.")


if __name__ == "__main__":
    main()
