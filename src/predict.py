"""Predict loan default risk on new data using the saved pipeline.

Usage:
    python3 src/predict.py --input data/new_applications.csv --output predictions.csv
"""
import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_default
from run_improved import add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to new data CSV")
    ap.add_argument("--output", default="predictions.csv", help="Output CSV path")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold (default: use saved optimal threshold)")
    a = ap.parse_args()

    # Load saved artifacts
    pipe = joblib.load(os.path.join(MODEL_DIR, "best_model_pipeline.joblib"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    threshold = a.threshold if a.threshold is not None else joblib.load(
        os.path.join(MODEL_DIR, "optimal_threshold_10_1.joblib"))

    # Load new data
    df = pd.read_csv(a.input)

    # Apply same feature engineering (row-wise, no fitted statistics)
    df, _ = add_features(df)

    # Select only the columns the model was trained on
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feat_cols]

    # Predict probabilities (preprocessing inside pipeline, fitted on training data only)
    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Save results
    out = pd.DataFrame({
        "default_probability": probs.round(4),
        "predict_default": preds,
        "threshold_used": threshold,
    })
    out = pd.concat([df.reset_index(drop=True), out], axis=1)
    out.to_csv(a.output, index=False)

    print(f"Predictions written to {a.output}")
    print(f"Threshold used: {threshold:.4f}")
    print(f"Predicted defaults: {preds.sum()} / {len(preds)} ({preds.mean():.1%})")


if __name__ == "__main__":
    main()