"""
ASTM Chip Lab Measurement Helper
==================================
Measures mean CIE-Lab values from photos of individual ASTM color bars.
Outputs a JSON file and a copy-paste Python snippet for estimate.py.

Usage:
  python measure_astm_lab.py chip1.jpeg chip2.jpeg ...
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

# ============================================================
# CONFIG — match the grey-card fractions from estimate.py
# ============================================================

# Set to None if chip images do NOT contain the grey card.
GREY_PATCH_FRAC: Optional[Tuple[float, float, float, float]] = (0.05, 0.04, 0.23, 0.19)

# Central crop fraction for each chip image
ROI_FRAC_W = 0.6
ROI_FRAC_H = 0.6

# ============================================================
# HELPERS
# ============================================================

def frac_to_rect(img_shape, frac_rect):
    h, w = img_shape[:2]
    x0, y0, x1, y1 = frac_rect
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def apply_simple_white_balance_bgr(img, frac_rect):
    x1, y1, x2, y2 = frac_to_rect(img.shape, frac_rect)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        raise RuntimeError("Grey patch ROI is empty.")
    mean_b, mean_g, mean_r = patch.mean(axis=(0, 1))
    eps = 1e-6
    gain_b = mean_g / (mean_b + eps)
    gain_r = mean_g / (mean_r + eps)
    out = img.astype(np.float32)
    out[..., 0] *= gain_b
    out[..., 2] *= gain_r
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_lab_mean(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = float(lab[..., 0].mean())
    a = float(lab[..., 1].mean())
    b = float(lab[..., 2].mean())
    return L, a, b


def crop_center_roi(img, frac_w, frac_h):
    h, w = img.shape[:2]
    roi_w = int(w * frac_w)
    roi_h = int(h * frac_h)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    return img[y1:y1 + roi_h, x1:x1 + roi_w], (x1, y1, x1 + roi_w, y1 + roi_h)


def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_astm_lab.py chip1.jpeg chip2.jpeg ...")
        sys.exit(1)

    image_paths = [Path(p) for p in sys.argv[1:]]
    chips_data = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        print(f"\n=== Processing {path.name} ===")

        # Optional white balance
        if GREY_PATCH_FRAC is not None:
            try:
                img = apply_simple_white_balance_bgr(img, GREY_PATCH_FRAC)
                print("Applied grey-card white balance.")
            except RuntimeError as e:
                print(f"[WARN] {e} — continuing without WB")

        # Crop central ROI
        roi, rect = crop_center_roi(img, ROI_FRAC_W, ROI_FRAC_H)
        if roi.size == 0:
            print("[WARN] ROI is empty, skipping.")
            continue

        # Visual check
        x1, y1, x2, y2 = rect
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Chip image (ROI in green)", vis)
        cv2.imshow("ROI", roi)
        print("Check that the ROI is uniform chip color (no text/edges).")
        print("Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        L_mean, a_mean, b_mean = compute_lab_mean(roi)
        print(f"Mean Lab: L*={L_mean:.2f}, a*={a_mean:.2f}, b*={b_mean:.2f}")

        while True:
            val_str = input(
                "Enter ASTM value for this chip (e.g. 0.5, 1.0, 1.5) "
                "or leave blank to skip: "
            ).strip()
            if val_str == "":
                print("Skipping (no ASTM value entered).")
                astm_val = None
                break
            try:
                astm_val = float(val_str)
                break
            except ValueError:
                print("Could not parse number, please try again.")

        if astm_val is None:
            continue

        chips_data.append({
            "filename": path.name,
            "astm_value": astm_val,
            "L": L_mean,
            "a": a_mean,
            "b": b_mean,
        })

    if not chips_data:
        print("\nNo chip data collected. Exiting.")
        return

    out_json = Path("astm_lab_chips.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(chips_data, f, indent=2)
    print(f"\nSaved Lab data for {len(chips_data)} chips to {out_json}")

    print("\n--- Copy-paste snippet for estimate.py ---\n")
    print("from dataclasses import dataclass\n")
    print("@dataclass")
    print("class ASTMChip:")
    print("    value: float")
    print("    L: float")
    print("    a: float")
    print("    b: float\n")
    print("ASTM_REFERENCE_CHIPS = [")
    for chip in chips_data:
        print(
            f"    ASTMChip("
            f"value={chip['astm_value']}, "
            f"L={chip['L']:.2f}, "
            f"a={chip['a']:.2f}, "
            f"b={chip['b']:.2f}"
            f"),  # {chip['filename']}"
        )
    print("]\n")


if __name__ == "__main__":
    main()