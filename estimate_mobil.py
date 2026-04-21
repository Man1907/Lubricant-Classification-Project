"""
Mobil Aviation Oil — ASTM Color Estimator
==========================================
Auto-detects oil cup via circle detection.
Compares oil color against pre-measured ASTM chip Lab values.
Fresh baseline: ASTM 2.5

Usage:
  python estimate_mobil.py image1.jpg [image2.jpg ...]
  python estimate_mobil.py --csv results.csv image1.jpg ...
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# MOBIL CONFIG
# ============================================================

OIL_NAME = "Mobil Aviation Oil"
FRESH_ASTM = 2.5
CALIBRATION_OFFSET = 0.0  # Run fresh image first, then set: offset = raw - 2.5

SHOW_PREVIEW = True

# ============================================================
# ASTM REFERENCE CHIPS (from measure_astm_lab.py)
# ============================================================

@dataclass
class ASTMChip:
    value: float
    L: float
    a: float
    b: float

ASTM_CHIPS = [
    ASTMChip(0.5, 208.42, 119.75, 161.38),
    ASTMChip(1.0, 193.24, 116.91, 203.91),
    ASTMChip(1.5, 176.36, 122.72, 197.76),
    ASTMChip(2.0, 148.39, 132.58, 179.54),
    ASTMChip(2.5, 116.07, 145.99, 161.37),
    ASTMChip(3.0, 102.51, 151.79, 152.93),
    ASTMChip(3.5, 89.72, 150.29, 146.18),
    ASTMChip(4.0, 74.85, 142.48, 138.78),
    ASTMChip(4.5, 65.34, 144.90, 132.47),
    ASTMChip(5.0, 52.22, 137.12, 129.73),
    ASTMChip(5.5, 47.87, 134.31, 130.05),
    ASTMChip(6.0, 40.71, 132.56, 128.18),
    ASTMChip(6.5, 35.42, 129.69, 129.07),
    ASTMChip(7.0, 33.25, 127.93, 128.85),
    ASTMChip(7.5, 35.69, 127.49, 128.48),
    ASTMChip(8.0, 43.89, 126.01, 127.08),
]

# ============================================================
# AUTO-DETECTION + HELPERS
# ============================================================

def detect_oil_cup(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    h, w = img.shape[:2]
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=300,
        param1=80, param2=60,
        minRadius=int(0.05 * w), maxRadius=int(0.15 * w),
    )
    if circles is None:
        return None
    c = np.round(circles[0][0]).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def get_wb_patch(img, cx, cy, cr):
    h, w = img.shape[:2]
    gx1 = max(0, cx - int(cr * 3.0))
    gy1 = max(0, cy - int(cr * 2.5))
    gx2 = max(gx1 + 50, cx - int(cr * 1.5))
    gy2 = max(gy1 + 50, cy - int(cr * 1.5))
    return gx1, gy1, min(gx2, w), min(gy2, h)


def apply_white_balance(img, rect):
    x1, y1, x2, y2 = rect
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return img
    mb, mg, mr = patch.mean(axis=(0, 1))
    eps = 1e-6
    out = img.astype(np.float32)
    out[..., 0] *= mg / (mb + eps)
    out[..., 2] *= mg / (mr + eps)
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_oil_lab(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 15, 10]), np.array([50, 255, 255]))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    if cv2.countNonZero(mask) >= 20:
        pixels = lab[mask > 0]
    else:
        pixels = lab.reshape(-1, 3)
    return pixels.mean(axis=0)


def match_astm(oil_lab):
    best_val, best_dist = 0.5, float("inf")
    for chip in ASTM_CHIPS:
        d = float(np.sqrt(
            (oil_lab[0] - chip.L) ** 2 +
            (oil_lab[1] - chip.a) ** 2 +
            (oil_lab[2] - chip.b) ** 2
        ))
        if d < best_dist:
            best_dist = d
            best_val = chip.value
    return best_val


def classify(astm):
    if astm <= 2.5:
        return "Fresh"
    elif astm <= 4.0:
        return "Lightly_heated"
    elif astm <= 5.0:
        return "Moderately_heated"
    elif astm <= 6.5:
        return "Heavily_heated"
    else:
        return "Severely_degraded"


# ============================================================
# MAIN
# ============================================================

def analyse(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    h, w = img.shape[:2]

    cup = detect_oil_cup(img)
    if cup is None:
        raise RuntimeError(f"No oil cup detected in {path}")
    cx, cy, cr = cup

    wb_rect = get_wb_patch(img, cx, cy, cr)
    img_wb = apply_white_balance(img, wb_rect)

    ir = int(cr * 0.30)
    roi = img_wb[max(0, cy - ir):min(h, cy + ir),
                 max(0, cx - ir):min(w, cx + ir)]
    if roi.size == 0:
        raise RuntimeError("Oil ROI is empty.")

    oil_lab = extract_oil_lab(roi)
    raw = match_astm(oil_lab)
    corrected = max(0.5, min(8.0, raw - CALIBRATION_OFFSET))

    return {
        "file": Path(path).name,
        "oil_type": OIL_NAME,
        "L_mean": float(oil_lab[0]),
        "a_mean": float(oil_lab[1]),
        "b_mean": float(oil_lab[2]),
        "raw_ASTM": raw,
        "ASTM": corrected,
        "label": classify(corrected),
    }, img_wb, wb_rect, (max(0, cx - ir), max(0, cy - ir),
                          min(w, cx + ir), min(h, cy + ir)), cup


def main():
    parser = argparse.ArgumentParser(description=f"{OIL_NAME} ASTM estimator")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    show = SHOW_PREVIEW and not args.no_preview
    print(f"Oil: {OIL_NAME}  |  Fresh baseline: {FRESH_ASTM}  |  Offset: {CALIBRATION_OFFSET}\n")

    results = []
    for img_path in args.images:
        try:
            m, img_wb, wb_rect, oil_rect, cup_info = analyse(img_path)
        except RuntimeError as e:
            print(f"[SKIP] {e}")
            continue

        results.append(m)
        print(f"=== {m['file']} ===")
        print(f"  L*={m['L_mean']:.2f}  a*={m['a_mean']:.2f}  b*={m['b_mean']:.2f}")
        print(f"  Raw ASTM: {m['raw_ASTM']}  Corrected: {m['ASTM']}")
        print(f"  Class: {m['label']}\n")

        if show:
            vis = img_wb.copy()
            gx1, gy1, gx2, gy2 = wb_rect
            ox1, oy1, ox2, oy2 = oil_rect
            ccx, ccy, ccr = cup_info
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 4)
            cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (0, 255, 0), 5)
            cv2.circle(vis, (ccx, ccy), ccr, (0, 255, 255), 3)
            cv2.putText(vis, f"ASTM {m['ASTM']}", (ox1, max(oy1 - 15, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            scale = 800 / vis.shape[0]
            small = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow(f"{OIL_NAME} - {m['file']}", small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if args.csv and results:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file", "oil_type", "L_mean", "a_mean", "b_mean",
                "raw_ASTM", "ASTM", "label",
            ])
            w.writeheader()
            w.writerows(results)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()