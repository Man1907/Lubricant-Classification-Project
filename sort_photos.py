"""
Auto-sort photos into reference (top-down) vs casual (side-angle)
==================================================================
Checks two things in the expected ROI positions:
  1. Is the white plate bright and visible where expected?
  2. Is the ASTM chart's yellow top-bar visible where expected?

If both pass → reference (top-down) photo
Otherwise  → casual (training) photo

Usage:
  python sort_ref_vs_casual.py --src exp_photos_jpg --ref reference_photos --casual casual_photos
"""

import os
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path


# Same fractions as estimate.py
GREY_PATCH_FRAC = (0.081, 0.290, 0.190, 0.363)
CHART_TOP_FRAC  = (0.574, 0.039, 0.829, 0.150)

# Thresholds (tuned from your sample images)
PLATE_BRIGHTNESS_MIN = 150   # White plate should be bright
CHART_YELLOW_PCT_MIN = 25.0  # Top chart bar should have yellow pixels


def is_reference_photo(img_path: str) -> bool:
    """Return True if the image looks like a top-down reference shot."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Check 1: Is the white plate visible and bright?
    px1 = int(GREY_PATCH_FRAC[0] * w)
    py1 = int(GREY_PATCH_FRAC[1] * h)
    px2 = int(GREY_PATCH_FRAC[2] * w)
    py2 = int(GREY_PATCH_FRAC[3] * h)
    plate = img[py1:py2, px1:px2]

    if plate.size == 0:
        return False

    plate_brightness = plate.mean()

    # Check 2: Is the yellow ASTM 0.5 bar visible at the top of the chart?
    cx1 = int(CHART_TOP_FRAC[0] * w)
    cy1 = int(CHART_TOP_FRAC[1] * h)
    cx2 = int(CHART_TOP_FRAC[2] * w)
    cy2 = int(CHART_TOP_FRAC[3] * h)
    chart_top = img[cy1:cy2, cx1:cx2]

    if chart_top.size == 0:
        return False

    hsv = cv2.cvtColor(chart_top, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 30, 150], dtype=np.uint8),
        np.array([40, 200, 255], dtype=np.uint8),
    )
    total_pixels = yellow_mask.shape[0] * yellow_mask.shape[1]
    yellow_pct = (cv2.countNonZero(yellow_mask) / total_pixels) * 100

    is_ref = (plate_brightness >= PLATE_BRIGHTNESS_MIN and
              yellow_pct >= CHART_YELLOW_PCT_MIN)

    return is_ref


def main():
    parser = argparse.ArgumentParser(
        description="Sort photos into reference vs casual folders"
    )
    parser.add_argument("--src", type=str, required=True,
                        help="Source folder with all JPGs")
    parser.add_argument("--ref", type=str, default="reference_photos",
                        help="Output folder for top-down reference photos")
    parser.add_argument("--casual", type=str, default="casual_photos",
                        help="Output folder for casual/side-angle photos")
    args = parser.parse_args()

    os.makedirs(args.ref, exist_ok=True)
    os.makedirs(args.casual, exist_ok=True)

    src_dir = Path(args.src)
    extensions = (".jpg", ".jpeg", ".png")

    ref_count = 0
    casual_count = 0

    for img_file in sorted(src_dir.iterdir()):
        if not img_file.suffix.lower() in extensions:
            continue

        if is_reference_photo(str(img_file)):
            dest = Path(args.ref) / img_file.name
            label = "REFERENCE"
            ref_count += 1
        else:
            dest = Path(args.casual) / img_file.name
            label = "CASUAL"
            casual_count += 1

        shutil.copy2(img_file, dest)
        print(f"  [{label:9s}] {img_file.name}")

    print(f"\nDone.")
    print(f"  Reference (top-down): {ref_count} photos -> {args.ref}/")
    print(f"  Casual (side-angle):  {casual_count} photos -> {args.casual}/")
    print(f"\nNext steps:")
    print(f"  1. Run estimate.py on {args.ref}/ to get ASTM labels")
    print(f"  2. Use sort_into_classes.py to sort {args.casual}/ into class folders")


if __name__ == "__main__":
    main()