"""
Lubricant Classifier — Classical ML (Small Dataset Friendly)
==============================================================
Uses color features (Lab, HSV stats) + Random Forest.
Works well with as few as 30-50 images, unlike CNNs which need hundreds.

Usage:
  python train_classical.py --data_dir dataset/ --output model_classical.pkl
  python predict_classical.py --model model_classical.pkl --oil castrol image.jpg
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.metrics import classification_report

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

CLASS_NAMES = [
    "Fresh",
    "Lightly_heated",
    "Moderately_heated",
    "Heavily_heated",
    "Severely_degraded",
]

OIL_TYPES = ["castrol", "mobil", "unknown"]

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(img_path: str) -> np.ndarray:
    """
    Extract color-based features from an oil image.
    Returns a 1D feature vector.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to Lab and HSV
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    features = []

    # Lab channel stats: mean, std, median, percentiles
    for ch in range(3):
        channel = lab[:, :, ch].flatten()
        features.extend([
            channel.mean(),
            channel.std(),
            np.median(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
        ])

    # HSV channel stats
    for ch in range(3):
        channel = hsv[:, :, ch].flatten()
        features.extend([
            channel.mean(),
            channel.std(),
            np.median(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
        ])

    # Darkness features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    features.append(gray.mean())
    features.append(gray.std())
    features.append((gray < 50).sum() / gray.size)   # dark pixel %
    features.append((gray < 100).sum() / gray.size)   # medium-dark pixel %
    features.append((gray > 200).sum() / gray.size)   # bright pixel %

    # Color histograms (compact)
    for ch in range(3):
        hist = cv2.calcHist([lab], [ch], None, [8], [0, 256])
        hist = hist.flatten() / hist.sum()  # normalize
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def extract_oil_type(filename: str) -> str:
    fname_lower = filename.lower()
    for oil in OIL_TYPES:
        if oil != "unknown" and fname_lower.startswith(oil):
            return oil
    return "unknown"


# ============================================================
# TRAINING
# ============================================================

def load_dataset(data_dir: str):
    X_features = []
    y_labels = []
    filenames = []

    active_classes = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(files) == 0:
            continue

        active_classes.append(class_name)

        for fname in sorted(files):
            fpath = os.path.join(class_dir, fname)
            try:
                feats = extract_features(fpath)
                X_features.append(feats)
                y_labels.append(class_name)
                filenames.append(fname)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")

    X = np.array(X_features)
    y = np.array(y_labels)

    print(f"Loaded {len(X)} images across {len(active_classes)} classes")
    for cls in active_classes:
        count = np.sum(y == cls)
        print(f"  {cls}: {count}")

    return X, y, filenames, active_classes


def train_main():
    parser = argparse.ArgumentParser(description="Train classical ML classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="model_classical.pkl")
    args = parser.parse_args()

    X, y, filenames, active_classes = load_dataset(args.data_dir)

    if len(X) == 0:
        print("No images found.")
        return

    # Skip empty classes
    if len(active_classes) < 2:
        print("Need at least 2 classes with images.")
        return

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",  # handles imbalanced classes
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    n_samples = len(X)
    min_class_count = min(np.sum(y == cls) for cls in active_classes)

    if min_class_count >= 3 and n_samples >= 10:
        n_folds = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"\nCross-validation accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
    else:
        print(f"\nToo few samples for cross-validation, training on all data.")

    # Train on full dataset
    clf.fit(X, y)

    # Training accuracy
    train_preds = clf.predict(X)
    print(f"Training accuracy: {(train_preds == y).mean():.2%}")
    print(f"\nClassification report (on training data):")
    print(classification_report(y, train_preds))

    # Save model
    model_data = {
        "classifier": clf,
        "active_classes": active_classes,
        "img_size": IMG_SIZE,
        "feature_dim": X.shape[1],
    }

    with open(args.output, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to: {args.output}")

    # Also save metadata as JSON
    meta_path = args.output.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "active_classes": active_classes,
            "img_size": IMG_SIZE,
            "n_samples": len(X),
            "feature_dim": int(X.shape[1]),
        }, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


# ============================================================
# PREDICTION
# ============================================================

def predict_main():
    parser = argparse.ArgumentParser(description="Predict with classical ML model")
    parser.add_argument("--model", type=str, default="model_classical.pkl")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model_data = pickle.load(f)

    clf = model_data["classifier"]
    active_classes = model_data["active_classes"]

    print(f"Model: {args.model}")
    print(f"Classes: {active_classes}\n")

    results = []
    for img_path in args.images:
        try:
            feats = extract_features(img_path)
        except Exception as e:
            print(f"[SKIP] {img_path}: {e}")
            continue

        pred = clf.predict([feats])[0]
        proba = clf.predict_proba([feats])[0]
        confidence = float(max(proba))

        result = {
            "file": Path(img_path).name,
            "predicted_class": pred,
            "confidence": confidence,
        }
        results.append(result)

        print(f"=== {result['file']} ===")
        print(f"  Prediction: {pred}")
        print(f"  Confidence: {confidence:.2%}")
        for cls, prob in zip(clf.classes_, proba):
            print(f"    {cls}: {prob:.2%}")
        print()

    if args.csv and results:
        import csv as csvmod
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=["file", "predicted_class", "confidence"])
            w.writeheader()
            for r in results:
                r_out = dict(r)
                r_out["confidence"] = f"{r_out['confidence']:.4f}"
                w.writerow(r_out)
        print(f"Saved to {args.csv}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        sys.argv.pop(1)
        predict_main()
    else:
        train_main()
