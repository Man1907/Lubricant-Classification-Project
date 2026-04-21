"""
Lubricant Degradation Classifier — Inference
==============================================
Predicts degradation class from any oil photo (no grey card/chart needed).

Usage:
  python predict.py --oil castrol image1.jpg image2.jpg
  python predict.py --oil mobil --csv results.csv image1.jpg
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="lubricant_model.keras")
    p.add_argument("--oil", type=str, required=True, help="Oil type (e.g. castrol, mobil)")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("images", nargs="+")
    return p.parse_args()


def load_meta(model_path):
    meta_path = model_path.replace(".keras", "_meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta["class_names"], meta["oil_types"], meta["img_size"]
    except FileNotFoundError:
        return [
            "Fresh", "Lightly_heated", "Moderately_heated",
            "Heavily_heated", "Severely_degraded",
        ], ["castrol", "mobil", "unknown"], 224


def main():
    args = parse_args()

    class_names, oil_types, img_size = load_meta(args.model)
    model = tf.keras.models.load_model(args.model)

    oil_encoder = LabelEncoder()
    oil_encoder.fit(oil_types)

    oil_type = args.oil.lower()
    if oil_type not in oil_types:
        print(f"[WARN] Unknown oil type '{oil_type}', using 'unknown'")
        oil_type = "unknown"

    oil_encoded = oil_encoder.transform([oil_type]).astype("float32").reshape(1, 1)

    print(f"Model: {args.model}")
    print(f"Oil type: {oil_type}")
    print(f"Classes: {class_names}\n")

    results = []

    for img_path in args.images:
        img = tf.keras.utils.load_img(img_path, target_size=(img_size, img_size))
        img_arr = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = model.predict(
            {"image_input": img_arr, "oil_input": oil_encoded}, verbose=0
        )[0]
        pred_idx = int(np.argmax(preds))

        result = {
            "file": Path(img_path).name,
            "oil_type": oil_type,
            "predicted_class": class_names[pred_idx],
            "confidence": float(preds[pred_idx]),
        }
        results.append(result)

        print(f"=== {result['file']} ===")
        print(f"  Oil Type: {result['oil_type']}")
        print(f"  Prediction: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        for name, prob in zip(class_names, preds):
            print(f"    {name}: {prob:.2%}")
        print()

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "oil_type", "predicted_class", "confidence"])
            w.writeheader()
            for r in results:
                r_out = dict(r)
                r_out["confidence"] = f"{r_out['confidence']:.4f}"
                w.writerow(r_out)
        print(f"Saved to {args.csv}")


if __name__ == "__main__":
    main()
