"""
Phase 12B model correction sanity check.

Uses >=20 REAL and >=20 FAKE videos that are in the TEST partition only,
so they were never seen during training or model selection.

Requires corrected model artifacts to already exist.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.video.full_ml_benchmark import load_and_split_connected_components
from src.video.model_correction import _apply_threshold


CSV_PATH   = "data/videos/features/video_features_stage_b.csv"
MODELS_DIR = Path("data/videos/models")
RESULTS_DIR = Path("data/videos/results")
THRESHOLD_JSON = MODELS_DIR / "corrected_threshold.json"
MODEL_PATH = MODELS_DIR / "corrected_best_model.joblib"
SCALER_PATH = MODELS_DIR / "corrected_scaler.joblib"
RANDOM_STATE = 42


def run_corrected_sanity_check(n_each: int = 20, random_state: int = RANDOM_STATE):
    """Sample n_each REAL and n_each FAKE from TEST partition, run inference."""

    # ── 1. Load split ────────────────────────────────────────────────────────
    train_df, val_df, test_df, feature_names = load_and_split_connected_components(
        csv_path=CSV_PATH, random_state=random_state
    )

    real_test = test_df[test_df["target"] == 0]
    fake_test = test_df[test_df["target"] == 1]

    n_real = min(n_each, len(real_test))
    n_fake = min(n_each, len(fake_test))
    print(f"Sampling {n_real} REAL and {n_fake} FAKE videos from TEST partition.")

    sample_real = real_test.sample(n=n_real, random_state=random_state)
    sample_fake = fake_test.sample(n=n_fake, random_state=random_state)
    sample_df = pd.concat([sample_real, sample_fake]).reset_index(drop=True)

    # ── 2. Load corrected artifacts ──────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"[ERROR] Corrected model not found at {MODEL_PATH}. Run model_correction first.")
        sys.exit(1)

    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    thresh_info = json.loads(THRESHOLD_JSON.read_text())
    threshold = float(thresh_info["threshold"])
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Threshold: {threshold:.4f} (strategy: {thresh_info.get('threshold_strategy','?')})")

    # ── 3. Inference ─────────────────────────────────────────────────────────
    X = scaler.transform(sample_df[feature_names].values.astype(np.float64))
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
    else:
        y_score = clf.predict(X).astype(float)
    y_pred = _apply_threshold(y_score, threshold)

    sample_df = sample_df.copy()
    sample_df["predicted_label"] = y_pred
    sample_df["fake_probability"] = y_score
    sample_df["correct"] = (sample_df["predicted_label"] == sample_df["target"]).astype(int)

    # ── 4. Report ────────────────────────────────────────────────────────────
    print("\n=== CORRECTED MODEL SANITY CHECK ===")
    for cls_label, cls_int in [("REAL", 0), ("FAKE", 1)]:
        cls_df = sample_df[sample_df["target"] == cls_int]
        correct = cls_df["correct"].sum()
        total   = len(cls_df)
        print(f"  {cls_label}: {correct}/{total} correctly classified ({100*correct/total:.1f}%)")

    # Category-level breakdown for FAKE
    if "category" in sample_df.columns:
        print("\n  Category-level (FAKE partition):")
        fake_samp = sample_df[sample_df["target"] == 1]
        for cat, grp in fake_samp.groupby("category"):
            c = grp["correct"].sum()
            t = len(grp)
            print(f"    {cat}: {c}/{t} ({100*c/t:.1f}%)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "corrected_sanity_check.csv"
    sample_df[["video_path", "category", "label", "target", "predicted_label", "fake_probability", "correct"]].to_csv(out_csv, index=False)
    print(f"\nSanity check results saved to: {out_csv}")
    return sample_df


if __name__ == "__main__":
    run_corrected_sanity_check()
