"""
download_external_datasets.py
=============================
Downloads external labeled datasets that improve detector generalization and
classifier accuracy. See docs/IMPROVING_ACCURACY.md for the full catalogue,
licenses, and rationale.

Supported sources:
  --zenodo               Battery Image Dataset for EV Circularity (19 battery
                         types, CC BY 4.0, ~5.4 GB) from zenodo.org/records/19818270
  --roboflow WS/PROJECT  Any Roboflow Universe dataset (needs --api_key)

Downloaded sets land in data/external/<source>/ — they are NOT auto-merged into
data/detector/. Inspect the class scheme first, then merge with --merge, which
remaps classes to the project scheme (0=module, 1=busbar) using the mapping you
provide via --class_map (same idea as scripts/remap_labels.py).

Usage:
    python scripts/download_external_datasets.py --dry_run              # show plan only
    python scripts/download_external_datasets.py --zenodo
    python scripts/download_external_datasets.py --roboflow machine-iimx4/ev-battery-iceh6 --api_key KEY
    # After inspecting data/external/<name>/data.yaml:
    python scripts/download_external_datasets.py --merge data/external/zenodo --class_map "2:0,5:1"
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "external"

ZENODO_RECORD = "19818270"
ZENODO_API    = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def download_zenodo(dry_run: bool):
    """Fetch file listing from the Zenodo API and download each archive."""
    import json

    dest = EXTERNAL_DIR / "zenodo_ev_circularity"
    print(f"Zenodo record {ZENODO_RECORD} → {dest.relative_to(ROOT)}")
    if dry_run:
        print("  [dry run] would query the Zenodo API and download ~5.4 GB of archives")
        return

    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API) as resp:
        record = json.load(resp)

    files = record.get("files", [])
    if not files:
        print("  No files listed on the record — download manually from "
              f"https://zenodo.org/records/{ZENODO_RECORD}")
        return

    for f in files:
        name = f["key"]
        url  = f["links"]["self"]
        size_mb = f.get("size", 0) / 1e6
        out = dest / name
        if out.exists():
            print(f"  {name} already exists — skipped")
            continue
        print(f"  downloading {name} ({size_mb:.0f} MB) ...")
        urllib.request.urlretrieve(url, out)

    print("\nDone. Unzip the archives, read the dataset README for the class scheme,")
    print("then merge with:  python scripts/download_external_datasets.py --merge "
          f"{dest.relative_to(ROOT)} --class_map 'SRC:DST,...'")


def download_roboflow(slug: str, api_key: str, version: int, dry_run: bool):
    """Download any Roboflow Universe dataset in YOLOv8 format."""
    try:
        workspace, project = slug.split("/")
    except ValueError:
        sys.exit(f"--roboflow expects 'workspace/project', got: {slug}")

    dest = EXTERNAL_DIR / f"roboflow_{project}"
    print(f"Roboflow {slug} v{version} → {dest.relative_to(ROOT)}")
    if dry_run:
        print("  [dry run] would download via the roboflow package (needs --api_key)")
        return
    if not api_key:
        sys.exit("Roboflow downloads need --api_key (free at https://roboflow.com)")

    try:
        from roboflow import Roboflow
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download("yolov8")
    src = Path(dataset.location)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    print(f"\nDone → {dest.relative_to(ROOT)}")
    print("Inspect data.yaml for the class order, then merge with --merge/--class_map.")


def parse_class_map(spec: str) -> dict:
    """Parse '2:0,5:1' into {2: 0, 5: 1} (source class id -> project class id)."""
    mapping = {}
    for pair in spec.split(","):
        src, dst = pair.split(":")
        dst = int(dst)
        if dst not in (0, 1):
            sys.exit(f"Destination class must be 0 (module) or 1 (busbar), got {dst}")
        mapping[int(src)] = dst
    return mapping


def merge(source_dir: Path, class_map: dict, dry_run: bool):
    """
    Merge an external YOLO-format dataset into data/detector/, remapping classes.
    Only train/valid splits are merged — the project test split stays untouched so
    reported metrics remain comparable. Filenames get a source prefix to avoid
    collisions.
    """
    if not source_dir.exists():
        sys.exit(f"Source dataset not found: {source_dir}")

    prefix = source_dir.name
    split_map = {"train": "train", "valid": "val", "val": "val"}
    total_imgs = total_kept = total_dropped = 0

    for src_split, dst_split in split_map.items():
        img_dir = source_dir / src_split / "images"
        lab_dir = source_dir / src_split / "labels"
        if not img_dir.exists():
            continue

        dst_img = ROOT / "data" / "detector" / "images" / dst_split
        dst_lab = ROOT / "data" / "detector" / "labels" / dst_split

        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"[{src_split} → {dst_split}] {len(imgs)} images")
        total_imgs += len(imgs)
        if dry_run:
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            lab = lab_dir / (img.stem + ".txt")
            kept_lines, dropped = [], 0
            if lab.exists():
                for line in lab.read_text().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    src_cls = int(float(parts[0]))
                    if src_cls in class_map:
                        parts[0] = str(class_map[src_cls])
                        kept_lines.append(" ".join(parts))
                    else:
                        dropped += 1
            total_kept += len(kept_lines)
            total_dropped += dropped

            shutil.copy(img, dst_img / f"{prefix}_{img.name}")
            # Keep empty label files: valid background images for YOLO.
            (dst_lab / f"{prefix}_{img.stem}.txt").write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else ""))

    if dry_run:
        print(f"\nDRY RUN — would merge {total_imgs} images (nothing copied).")
    else:
        print(f"\nMerged {total_imgs} images: {total_kept} instances kept, "
              f"{total_dropped} dropped by class map.")
        print("Delete stale YOLO caches before retraining:")
        print("  find data/detector/labels -name '*.cache' -delete")


def main():
    ap = argparse.ArgumentParser(
        description="Download/merge external datasets (see docs/IMPROVING_ACCURACY.md)")
    ap.add_argument("--zenodo",   action="store_true",
                    help="Download the 19-battery-type Zenodo dataset (~5.4 GB)")
    ap.add_argument("--roboflow", type=str, metavar="WORKSPACE/PROJECT",
                    help="Download a Roboflow Universe dataset")
    ap.add_argument("--api_key",  type=str, default=None, help="Roboflow API key")
    ap.add_argument("--version",  type=int, default=1, help="Roboflow dataset version")
    ap.add_argument("--merge",    type=str, metavar="DIR",
                    help="Merge a downloaded YOLO-format dataset into data/detector/")
    ap.add_argument("--class_map", type=str, default=None, metavar="SRC:DST,...",
                    help="Class remap for --merge, e.g. '1:0,3:1'")
    ap.add_argument("--dry_run",  action="store_true", help="Show plan; download/copy nothing")
    args = ap.parse_args()

    if not (args.zenodo or args.roboflow or args.merge):
        ap.print_help()
        print("\nCatalogue of recommended datasets: docs/IMPROVING_ACCURACY.md")
        return

    if args.zenodo:
        download_zenodo(args.dry_run)
    if args.roboflow:
        download_roboflow(args.roboflow, args.api_key, args.version, args.dry_run)
    if args.merge:
        if not args.class_map:
            sys.exit("--merge requires --class_map (e.g. '1:0,3:1'); inspect the "
                     "source data.yaml to build it")
        merge(ROOT / args.merge, parse_class_map(args.class_map), args.dry_run)


if __name__ == "__main__":
    main()
