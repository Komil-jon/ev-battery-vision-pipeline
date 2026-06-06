"""
download_dataset.py
===================
Downloads the public EV Battery Pack dataset from Roboflow Universe.

Dataset: "EV Battery pack" by MTech project
URL: https://universe.roboflow.com/mtech-project-ohj8a/ev-battery-pack
Classes: Battery Module, Bus-bar (plus Screw, Nut, Bolt, Cable, Aluminum-frame)
License: CC BY 4.0

This gives you a real labelled EV battery dataset with both Battery Module
and Bus-bar annotations — the closest public equivalent to the paper's dataset.

Usage:
    python scripts/download_dataset.py --api_key YOUR_ROBOFLOW_API_KEY

Get a free API key at: https://roboflow.com  (no credit card needed)
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def download(api_key: str, version: int = 1):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow package not found. Installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("mtech-project-ohj8a").project("ev-battery-pack")
    dataset = project.version(version).download("yolov8")

    src = Path(dataset.location)
    print(f"\nDataset downloaded to: {src}")
    print("Reorganising into project structure...")

    # Map Roboflow folders → project folders
    split_map = {
        "train": "train",
        "valid": "val",
        "test":  "test",
    }

    for rf_split, proj_split in split_map.items():
        rf_img_dir = src / rf_split / "images"
        rf_lab_dir = src / rf_split / "labels"

        proj_img_dir = ROOT / "data" / "detector" / "images" / proj_split
        proj_lab_dir = ROOT / "data" / "detector" / "labels" / proj_split

        proj_img_dir.mkdir(parents=True, exist_ok=True)
        proj_lab_dir.mkdir(parents=True, exist_ok=True)

        if rf_img_dir.exists():
            imgs = list(rf_img_dir.glob("*.*"))
            for f in imgs:
                shutil.copy(f, proj_img_dir / f.name)
            print(f"  {rf_split}/images → {proj_img_dir.relative_to(ROOT)}  ({len(imgs)} files)")

        if rf_lab_dir.exists():
            labs = list(rf_lab_dir.glob("*.txt"))
            for f in labs:
                shutil.copy(f, proj_lab_dir / f.name)
            print(f"  {rf_split}/labels → {proj_lab_dir.relative_to(ROOT)}  ({len(labs)} files)")

    print("\nDataset ready. Review class IDs in the label files and update dataset.yaml if needed.")
    print("Roboflow class order may differ from paper (module=0, busbar=1).")
    print("Check with:  python scripts/download_dataset.py --check_classes")


def check_classes():
    """Print class distribution in training labels."""
    lab_dir = ROOT / "data" / "detector" / "labels" / "train"
    if not lab_dir.exists():
        print(f"Labels directory not found: {lab_dir}")
        return
    counts = {}
    for lab_file in lab_dir.glob("*.txt"):
        with open(lab_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    print(f"\nClass distribution in training labels ({lab_dir}):")
    for cls_id in sorted(counts):
        print(f"  Class {cls_id}: {counts[cls_id]} instances ({100*counts[cls_id]/total:.1f}%)")
    print("\nExpected: class 0 = module (majority), class 1 = busbar (minority)")
    print("If classes are different, update dataset.yaml names accordingly.")


def main():
    parser = argparse.ArgumentParser(description="Download EV battery dataset from Roboflow")
    parser.add_argument("--api_key",       type=str, default=None, help="Roboflow API key")
    parser.add_argument("--version",       type=int, default=1,    help="Dataset version (default: 1)")
    parser.add_argument("--check_classes", action="store_true",    help="Check class distribution in existing labels")
    args = parser.parse_args()

    if args.check_classes:
        check_classes()
        return

    if not args.api_key:
        print(
            "Roboflow API key required.\n"
            "1. Sign up free at https://roboflow.com\n"
            "2. Get your key from https://app.roboflow.com/settings/api\n"
            "3. Run:  python scripts/download_dataset.py --api_key YOUR_KEY\n\n"
            "Alternatively, download manually from:\n"
            "https://universe.roboflow.com/mtech-project-ohj8a/ev-battery-pack\n"
            "and place images in data/detector/images/{train,val,test}/\n"
            "and labels in data/detector/labels/{train,val,test}/"
        )
        return

    download(args.api_key, args.version)


if __name__ == "__main__":
    main()
