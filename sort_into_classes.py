"""
Auto-sort photos into class folders based on results.csv
==========================================================
Reads the CSV from estimate.py and copies/moves each image
into the correct class subfolder for CNN training.

Usage:
  python sort_into_classes.py --csv results.csv --src exp_photos_jpg --dest dataset --oil castrol
"""

import os
import csv
import shutil
import argparse
from pathlib import Path


CLASS_NAMES = [
    "Fresh",
    "Lightly_heated",
    "Moderately_heated",
    "Heavily_heated",
    "Severely_degraded",
]


def main():
    parser = argparse.ArgumentParser(description="Sort photos into class folders")
    parser.add_argument("--csv", type=str, required=True, help="CSV from estimate.py")
    parser.add_argument("--src", type=str, required=True, help="Source image folder")
    parser.add_argument("--dest", type=str, default="dataset", help="Destination root folder")
    parser.add_argument("--oil", type=str, default="", help="Oil prefix for filenames (e.g. castrol)")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()

    # Create class folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(args.dest, cls), exist_ok=True)

    # Read CSV
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    moved = 0
    skipped = 0

    for row in rows:
        filename = row["file"]
        label = row["label"]

        if label not in CLASS_NAMES:
            print(f"[SKIP] Unknown label '{label}' for {filename}")
            skipped += 1
            continue

        src_path = Path(args.src) / filename
        if not src_path.exists():
            print(f"[SKIP] Not found: {src_path}")
            skipped += 1
            continue

        # Add oil prefix to filename
        if args.oil:
            dest_name = f"{args.oil}_{filename}"
        else:
            dest_name = filename

        dest_path = Path(args.dest) / label / dest_name

        if args.copy:
            shutil.copy2(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)  # Always copy to be safe

        moved += 1

    print(f"\nSorted {moved} images into {args.dest}/")
    print(f"Skipped {skipped} images")

    # Show counts
    for cls in CLASS_NAMES:
        cls_dir = Path(args.dest) / cls
        count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
