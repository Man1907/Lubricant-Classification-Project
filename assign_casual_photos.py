"""
Auto-assign casual photos to class folders
=============================================
Each casual photo gets the same label as its nearest reference photo
(by image number). Photos between two reference shots get assigned
to the earlier one (same heating stage).

Usage:
  python assign_casual_photos.py --csv results_castrol.csv --src casual_photos --dest dataset --oil castrol
  python assign_casual_photos.py --csv results_mobil.csv --src casual_photos --dest dataset --oil mobil
"""

import os
import re
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


def extract_number(filename):
    """Extract the numeric part from filenames like IMG_2040.jpg"""
    match = re.search(r'(\d{3,})', filename)
    if match:
        return int(match.group(1))
    return None


def load_reference_labels(csv_path):
    """Load reference photo labels from CSV, keyed by image number."""
    refs = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = extract_number(row["file"])
            if num is not None:
                refs[num] = {
                    "file": row["file"],
                    "label": row["label"],
                    "ASTM": row["ASTM"],
                }
    return refs


def find_nearest_reference(photo_num, ref_numbers):
    """
    Find the nearest reference photo by number.
    Prefer the reference photo just BEFORE this one (same heating stage).
    """
    best_ref = None
    best_dist = float("inf")

    for ref_num in ref_numbers:
        # Prefer earlier reference (casual photo was taken at same stage)
        dist = photo_num - ref_num
        if dist >= 0 and dist < best_dist:
            best_dist = dist
            best_ref = ref_num

    # If no earlier reference found, use the closest one
    if best_ref is None:
        for ref_num in ref_numbers:
            dist = abs(photo_num - ref_num)
            if dist < best_dist:
                best_dist = dist
                best_ref = ref_num

    return best_ref


def main():
    parser = argparse.ArgumentParser(description="Assign casual photos to class folders")
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV from estimate_castrol.py or estimate_mobil.py")
    parser.add_argument("--src", type=str, required=True,
                        help="Folder with casual photos")
    parser.add_argument("--dest", type=str, default="dataset",
                        help="Destination dataset folder")
    parser.add_argument("--oil", type=str, default="",
                        help="Oil prefix for filenames (e.g. castrol, mobil)")
    args = parser.parse_args()

    # Create class folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(args.dest, cls), exist_ok=True)

    # Load reference labels
    refs = load_reference_labels(args.csv)
    ref_numbers = sorted(refs.keys())

    if not ref_numbers:
        print("No reference photos found in CSV.")
        return

    print(f"Reference photos found ({len(ref_numbers)}):")
    for num in ref_numbers:
        r = refs[num]
        print(f"  IMG_{num}: ASTM {r['ASTM']} -> {r['label']}")
    print()

    # Process casual photos
    src_dir = Path(args.src)
    extensions = (".jpg", ".jpeg", ".png")
    assigned = 0
    skipped = 0

    for img_file in sorted(src_dir.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue

        photo_num = extract_number(img_file.name)
        if photo_num is None:
            print(f"  [SKIP] No number in: {img_file.name}")
            skipped += 1
            continue

        # Skip if this IS a reference photo (already in CSV)
        if photo_num in refs:
            continue

        # Find nearest reference
        nearest_ref = find_nearest_reference(photo_num, ref_numbers)
        if nearest_ref is None:
            print(f"  [SKIP] No reference match for: {img_file.name}")
            skipped += 1
            continue

        label = refs[nearest_ref]["label"]
        if label not in CLASS_NAMES:
            print(f"  [SKIP] Unknown label '{label}' for {img_file.name}")
            skipped += 1
            continue

        # Copy with oil prefix
        if args.oil:
            dest_name = f"{args.oil}_{img_file.name}"
        else:
            dest_name = img_file.name

        dest_path = Path(args.dest) / label / dest_name
        shutil.copy2(img_file, dest_path)
        assigned += 1

        ref_info = refs[nearest_ref]
        print(f"  {img_file.name} -> {label}  (matched to IMG_{nearest_ref}, ASTM {ref_info['ASTM']})")

    print(f"\nDone.")
    print(f"  Assigned: {assigned} casual photos")
    print(f"  Skipped: {skipped}")
    print()

    # Show final counts
    print("Dataset counts:")
    for cls in CLASS_NAMES:
        cls_dir = Path(args.dest) / cls
        count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
