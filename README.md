# Lubricant Degradation Classification using Computer Vision and AI

A computer vision system that classifies engine lubricants as **Fresh** or **Used/Heated** by estimating their **ASTM D1500 colour number** from photographs. The system auto-detects the oil cup in the image, performs white-balance correction, and compares the oil colour against pre-calibrated ASTM reference values.

---

## Project Structure

```
MultidisciplinaryProj/
├── estimate_castrol.py        # ASTM estimator for Castrol 10W-30
├── estimate_mobil.py          # ASTM estimator for Mobil Aviation Oil
├── measure_astm_lab.py        # Helper to measure Lab values from ASTM chip photos
├── convert_dng.py             # Batch convert DNG raw photos to JPEG
├── sort_ref_vs_casual.py      # Auto-sort photos into reference vs casual folders
├── sort_into_classes.py       # Sort reference photos into class folders using CSV
├── assign_casual_photos.py    # Assign casual photos to classes by nearest reference
├── train_classical.py         # Train Random Forest classifier + predict
├── train_classifier.py        # Train MobileNetV3 CNN classifier (needs large dataset)
├── predict.py                 # CNN inference script
├── astm_chips/                # Cropped ASTM colour chip images
├── exp_photos_jpg/            # Converted experiment photos (JPEG)
├── reference_photos/          # Top-down calibrated reference photos
│   ├── castrol/
│   └── mobil/
├── casual_photos/             # Side-angle training photos
│   ├── castrol/
│   └── mobil/
├── dataset/                   # Organised training data
│   ├── Fresh/
│   ├── Lightly_heated/
│   ├── Moderately_heated/
│   └── Heavily_heated/
├── results_castrol.csv        # ASTM estimation results
├── model_classical.pkl        # Trained Random Forest model
└── astm_lab_chips.json        # Measured ASTM chip Lab values
```

---

## Requirements

```
pip install opencv-python numpy scikit-learn rawpy
```

For the CNN classifier (optional, needs large dataset):
```
pip install tensorflow
```

---

## Quick Start

### 1. Convert DNG photos to JPEG

```powershell
python convert_dng.py
```

Reads from `exp_photos/`, outputs to `exp_photos_jpg/`.

### 2. Measure ASTM reference chip values (one-time setup)

```powershell
python measure_astm_lab.py (Get-ChildItem -File .\astm_chips\*.jpeg | ForEach-Object { $_.FullName })
```

Enter the ASTM value (0.5, 1.0, ..., 8.0) for each chip when prompted. Outputs `astm_lab_chips.json` and a copy-paste Python snippet.

### 3. Sort photos into reference vs casual

```powershell
python sort_ref_vs_casual.py --src exp_photos_jpg --ref reference_photos --casual casual_photos
```

Automatically detects top-down reference photos (white plate + ASTM chart visible) vs side-angle casual photos.

### 4. Run ASTM estimation on reference photos

For Castrol:
```powershell
python estimate_castrol.py --csv results_castrol.csv (Get-ChildItem -File .\reference_photos\castrol\*.jpg | ForEach-Object { $_.FullName })
```

For Mobil:
```powershell
python estimate_mobil.py --csv results_mobil.csv (Get-ChildItem -File .\reference_photos\mobil\*.jpg | ForEach-Object { $_.FullName })
```

Add `--no-preview` to skip the OpenCV preview windows on large batches.

### 5. Build the training dataset

Sort reference photos into class folders:
```powershell
python sort_into_classes.py --csv results_castrol.csv --src reference_photos\castrol --dest dataset --oil castrol
```

Assign casual photos to the same classes by image number:
```powershell
python assign_casual_photos.py --csv results_castrol.csv --src casual_photos\castrol --dest dataset --oil castrol
```

### 6. Train the classifier

**Recommended (small datasets):**
```powershell
python train_classical.py --data_dir dataset/ --output model_classical.pkl
```

**CNN alternative (needs 500+ images):**
```powershell
python train_classifier.py --data_dir dataset/ --epochs 30 --batch_size 16
```

### 7. Predict on new images

Classical model:
```powershell
python train_classical.py predict --model model_classical.pkl image1.jpg image2.jpg
```

CNN model:
```powershell
python predict.py --oil castrol --model lubricant_model.keras image1.jpg
```

---

## How It Works

### ASTM Estimation Pipeline

1. **Auto-detect oil cup** using Hough Circle Transform — no fixed ROI coordinates needed
2. **White-balance** using a neutral patch on the white plate surface, located relative to the detected cup
3. **Extract oil colour** from the inner 30% of the detected circle, filtered by an HSV mask to exclude glass rim and reflections
4. **Convert to CIE-Lab** colour space and compute mean L\*, a\*, b\*
5. **Match against ASTM chips** by computing ΔE (Euclidean distance in Lab space) to 16 pre-measured reference values
6. **Apply calibration offset** to correct for per-oil baseline differences

### Classification

The degradation classes are:

| Class | ASTM Range (Castrol) | ASTM Range (Mobil) |
|-------|---------------------|--------------------|
| Fresh | ≤ 1.0 | ≤ 2.5 |
| Lightly heated | 1.5 – 2.5 | 3.0 – 4.0 |
| Moderately heated | 3.0 – 4.0 | 4.5 – 5.5 |
| Heavily heated | 4.5 – 5.5 | 6.0 – 7.0 |
| Severely degraded | 6.0+ | 7.5+ |

---

## Oil Profiles

Each oil type has its own estimation script with a hardcoded calibration offset:

| Oil | Fresh ASTM | Calibration Offset | Script |
|-----|-----------|-------------------|--------|
| Castrol 10W-30 | 1.0 | 0.5 | `estimate_castrol.py` |
| Mobil Aviation Oil | 2.5 | Set after first run | `estimate_mobil.py` |

To calibrate a new oil: run its fresh image with offset 0, note the raw ASTM, then set `CALIBRATION_OFFSET = raw - true_fresh_value`.

---

## Imaging Setup

- **Container:** Transparent glass cup on a white styrofoam plate
- **References:** 18% grey card on the plate, printed ASTM D1500 colour chart to the right
- **Camera:** Smartphone, top-down, auto-flash disabled, consistent lighting
- **Reference photos:** Top-down with grey card and chart visible (for ASTM labelling)
- **Casual photos:** Various angles without references (for ML training)

---

## Results

| Metric | Value |
|--------|-------|
| ASTM estimation | Correct monotonic progression across heating series |
| ML classifier accuracy | 63.48% cross-validation (Random Forest, 104 images) |
| CNN accuracy | 18% (insufficient data, not used) |
| Dataset size | 104 images across 4 classes |

---

## Author

**Manthan Bengani**
B.Tech Information Technology, First Year
