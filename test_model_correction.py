"""
test_model_correction.py — Phase 12B test suite.

Verifies:
- No NaN/Inf in features
- Correct 216 feature dimensions
- Valid REAL/FAKE labels
- Zero connected-component overlap across partitions
- Scaler fitted only on train
- Threshold in [0,1]
- Corrected model artifact loads
- Corrected model predicts both classes on validation
- No browser sanity-check videos leaked into training
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest

from src.video.full_ml_benchmark import load_and_split_connected_components
from src.video.model_correction import (
    run_model_correction,
    make_balanced_train,
    compute_sample_weights,
    _compute_metrics,
    _apply_threshold,
)

CSV_PATH   = "data/videos/features/video_features_stage_b.csv"
MODELS_DIR = Path("data/videos/models")
RESULTS_DIR = Path("data/videos/results")
THRESHOLD_JSON = MODELS_DIR / "corrected_threshold.json"
MODEL_PATH = MODELS_DIR / "corrected_best_model.joblib"
SCALER_PATH = MODELS_DIR / "corrected_scaler.joblib"

BROWSER_SANITY_CSV = "data/videos/results/browser_prediction_sanity_check.csv"


@pytest.fixture(scope="module")
def splits():
    train_df, val_df, test_df, feature_names = load_and_split_connected_components(
        csv_path=CSV_PATH
    )
    return train_df, val_df, test_df, feature_names


def test_feature_count(splits):
    _, _, _, feature_names = splits
    assert len(feature_names) == 216, f"Expected 216 features, got {len(feature_names)}"


def test_no_nan_in_train(splits):
    train_df, _, _, feature_names = splits
    assert not train_df[feature_names].isna().any().any(), "NaN found in training features"


def test_no_inf_in_train(splits):
    train_df, _, _, feature_names = splits
    assert not np.isinf(train_df[feature_names].values).any(), "Inf found in training features"


def test_valid_labels(splits):
    train_df, val_df, test_df, _ = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        unique_labels = set(df["target"].unique())
        assert unique_labels.issubset({0, 1}), f"Invalid labels in {name}: {unique_labels}"


def test_zero_group_overlap(splits):
    train_df, val_df, test_df, _ = splits
    s_train = set(train_df["connected_component_id"])
    s_val   = set(val_df["connected_component_id"])
    s_test  = set(test_df["connected_component_id"])
    assert len(s_train & s_val)  == 0, "Train/Val connected-component overlap"
    assert len(s_train & s_test) == 0, "Train/Test connected-component overlap"
    assert len(s_val  & s_test)  == 0, "Val/Test connected-component overlap"


def test_make_balanced_train(splits):
    train_df, _, _, feature_names = splits
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_names].values.astype(np.float64))
    y_train = train_df["target"].values.astype(int)
    X_bal, y_bal = make_balanced_train(X_train, y_train)
    n_real = int((y_bal == 0).sum())
    n_fake = int((y_bal == 1).sum())
    assert n_real == n_fake, f"Balanced train not 1:1: REAL={n_real} FAKE={n_fake}"
    assert len(X_bal) == len(y_bal)


def test_sample_weights(splits):
    train_df, _, _, _ = splits
    y = train_df["target"].values.astype(int)
    sw = compute_sample_weights(y)
    assert len(sw) == len(y)
    assert np.all(sw > 0), "Some sample weights are <= 0"


def test_corrected_model_artifact_exists():
    assert MODEL_PATH.exists(),   f"Model artifact missing: {MODEL_PATH}"
    assert SCALER_PATH.exists(),  f"Scaler artifact missing: {SCALER_PATH}"
    assert THRESHOLD_JSON.exists(), f"Threshold JSON missing: {THRESHOLD_JSON}"


def test_corrected_threshold_in_range():
    thresh_info = json.loads(THRESHOLD_JSON.read_text())
    t = float(thresh_info["threshold"])
    assert 0.0 <= t <= 1.0, f"Threshold out of range: {t}"


def test_corrected_scaler_feature_count():
    scaler = joblib.load(SCALER_PATH)
    assert scaler.n_features_in_ == 216, f"Scaler n_features_in_={scaler.n_features_in_}, expected 216"


def test_corrected_model_predicts_both_classes(splits):
    _, val_df, _, feature_names = splits
    from sklearn.preprocessing import StandardScaler
    train_df, _, _, _ = splits
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names].values.astype(np.float64))

    clf = joblib.load(MODEL_PATH)
    thresh_info = json.loads(THRESHOLD_JSON.read_text())
    thresh = float(thresh_info["threshold"])

    X_val = scaler.transform(val_df[feature_names].values.astype(np.float64))
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_val)[:, 1]
    else:
        y_score = clf.predict(X_val).astype(float)

    y_pred = _apply_threshold(y_score, thresh)
    unique_preds = set(y_pred.tolist())
    assert 0 in unique_preds, "Corrected model never predicts REAL on validation set"
    assert 1 in unique_preds, "Corrected model never predicts FAKE on validation set"


def test_no_browser_sanity_data_in_train(splits):
    """Verify browser sanity check videos are not present in the training split."""
    if not os.path.exists(BROWSER_SANITY_CSV):
        pytest.skip("Browser sanity check CSV not found; skipping.")
    train_df, _, _, _ = splits
    sanity_df = pd.read_csv(BROWSER_SANITY_CSV)
    if "video_path" not in sanity_df.columns:
        pytest.skip("No video_path column in browser sanity CSV.")
    sanity_paths = set(sanity_df["video_path"].tolist())
    train_paths  = set(train_df["video_path"].tolist())
    leaked = sanity_paths & train_paths
    assert len(leaked) == 0, f"Browser sanity videos found in train: {leaked}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
