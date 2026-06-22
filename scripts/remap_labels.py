"""
remap_labels.py
===============
Remap the public Roboflow "EV Battery pack" 7-class label scheme to the paper's
2-class scheme used throughout this project.

Roboflow source classes (EV-Battery-pack--1/data.yaml):
    0 = Aluminum-frame   3 = Bus-bar
    1 = Battery Module   4 = Cable      6 = Screw
    2 = Bolt             5 = Nut

Paper / dataset.yaml scheme:
    0 = module           1 = busbar

Mapping applied:
    Battery Module (1) -> module (0)
    Bus-bar        (3) -> busbar (1)
    all other classes  -> dropped

Works for both YOLO bounding-box (5 tokens) and YOLO segmentation/polygon
(class + polygon points) label lines: only the class id is rewritten, the
geometry is preserved. A label file that ends up empty is kept as an empty
file (a valid "background" image for YOLO), so the image is not silently lost.

The script is idempotent-safe: if every label file in a split already contains
only classes 0/1 it is treated as already-remapped and skipped (re-run a backup
restore first if you need to remap from scratch).

Usage:
    python scripts/remap_labels.py            # remap train/val/test in place
    python scripts/remap_labels.py --dry_run  # report only, change nothing
"""

import argparse
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "detector" / "labels"
SPLITS     = ["train", "val", "test"]

# Roboflow source index -> paper index
REMAP = {1: 0, 3: 1}          # Battery Module -> module, Bus-bar -> busbar
SOURCE_NAMES = {
    0: "Aluminum-frame", 1: "Battery Module", 2: "Bolt", 3: "Bus-bar",
    4: "Cable", 5: "Nut", 6: "Screw",
}


def remap_file(path: Path, dry_run: bool) -> dict:
    """Remap one label file. Returns per-file stats."""
    lines = path.read_text().splitlines()
    kept_lines, dropped = [], 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        src_cls = int(float(parts[0]))
        if src_cls in REMAP:
            parts[0] = str(REMAP[src_cls])
            kept_lines.append(" ".join(parts))
        else:
            dropped += 1
    if not dry_run:
        # Keep an empty file for background images rather than deleting it.
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return {"kept": len(kept_lines), "dropped": dropped}


def split_already_remapped(label_files) -> bool:
    """True if no file references a class > 1 (already in 2-class scheme)."""
    for f in label_files:
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts and int(float(parts[0])) > 1:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Remap Roboflow 7-class labels to paper 2-class scheme")
    ap.add_argument("--dry_run", action="store_true", help="Report only; do not modify files")
    args = ap.parse_args()

    print("Class mapping: Battery Module(1)->module(0), Bus-bar(3)->busbar(1); others dropped\n")
    grand_module = grand_busbar = grand_dropped = 0

    for split in SPLITS:
        split_dir = LABELS_DIR / split
        if not split_dir.exists():
            print(f"[{split}] directory missing: {split_dir} — skipped")
            continue
        label_files = sorted(split_dir.glob("*.txt"))
        if not label_files:
            print(f"[{split}] no label files — skipped")
            continue
        if split_already_remapped(label_files):
            print(f"[{split}] already in 2-class scheme (no class>1) — skipped")
            continue

        # Count source classes before remap for a transparency report
        src_counts = {}
        mod = bus = dropped = empties = 0
        for f in label_files:
            for line in f.read_text().splitlines():
                p = line.strip().split()
                if p:
                    c = int(float(p[0]))
                    src_counts[c] = src_counts.get(c, 0) + 1
            stats = remap_file(f, args.dry_run)
            mod += sum(1 for ln in (f.read_text().splitlines() if not args.dry_run else [])
                       if ln.split() and ln.split()[0] == "0")
            bus += sum(1 for ln in (f.read_text().splitlines() if not args.dry_run else [])
                       if ln.split() and ln.split()[0] == "1")
            dropped += stats["dropped"]
            if stats["kept"] == 0:
                empties += 1

        print(f"[{split}] {len(label_files)} files")
        print("   source instances: " +
              ", ".join(f"{SOURCE_NAMES[c]}={n}" for c, n in sorted(src_counts.items())))
        if not args.dry_run:
            print(f"   -> module={mod}, busbar={bus}, dropped={dropped}, "
                  f"now-empty(background) files={empties}")
            grand_module += mod
            grand_busbar += bus
        grand_dropped += dropped

    print("\n" + ("DRY RUN — no files changed." if args.dry_run else
          f"Done. Total module={grand_module}, busbar={grand_busbar}, dropped={grand_dropped}."))
    if not args.dry_run:
        print("Remember to delete stale YOLO caches before retraining:")
        print("  find data/detector/labels -name '*.cache' -delete")


if __name__ == "__main__":
    main()
