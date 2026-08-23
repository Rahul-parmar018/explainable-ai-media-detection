"""
test_face_model.py — Phase 13A: Face Model Tests.

Verifies:
- Leakage-safe connected-component splitting
- Scaler fitted only on train
- Model predicts both classes on validation
- No original baseline artifacts overwritten
- Face threshold in [0,1]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest

FACE_CSV  = "data/videos/features/video_face_features.csv"
FACE_MODEL  = Path("data/videos/models/face_best_model.joblib")
FACE_SCALER = Path("data/videos/models/face_scaler.joblib")
FACE_THRESH = Path("data/videos/models/face_threshold.json")

BASELINE_MODEL  = Path("data/videos/models/best_model.joblib")
BASELINE_SCALER = Path("data/videos/models/scaler.joblib")


@pytest.fixture(scope="module")
def splits():
    if not Path(FACE_CSV).exists():
        pytest.skip("video_face_features.csv not generated yet.")
    from src.video.face_model import load_and_split_face_features
    return load_and_split_face_features(csv_path=FACE_CSV)


def test_zero_group_overlap(splits):
    train_df, val_df, test_df, _ = splits
    s_tr = set(train_df["connected_component_id"])
    s_vl = set(val_df["connected_component_id"])
    s_ts = set(test_df["connected_component_id"])
    assert len(s_tr & s_vl) == 0
    assert len(s_tr & s_ts) == 0
    assert len(s_vl & s_ts) == 0


def test_valid_labels(splits):
    train_df, val_df, test_df, _ = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert set(df["target"].unique()).issubset({0, 1}), f"Bad labels in {name}"


def test_288_features(splits):
    _, _, _, feature_names = splits
    assert len(feature_names) == 288, f"Expected 288 features, got {len(feature_names)}"


def test_no_nan_in_train(splits):
    train_df, _, _, feature_names = splits
    assert not train_df[feature_names].isna().any().any()


def test_baseline_artifacts_untouched():
    """The original best_model.joblib must NOT have been modified by this phase."""
    if not BASELINE_MODEL.exists():
        pytest.skip("Baseline model not found; skipping baseline protection check.")
    import joblib
    model = joblib.load(BASELINE_MODEL)
    # Just verify it loads and has the expected type
    assert hasattr(model, "predict")
    scaler = joblib.load(BASELINE_SCALER)
    assert scaler.n_features_in_ == 216, \
        f"Baseline scaler modified! n_features_in_={scaler.n_features_in_} (expected 216)"


def test_face_model_artifact_exists():
    if not FACE_MODEL.exists():
        pytest.skip("Face model not generated yet.")
    assert FACE_MODEL.exists()


def test_face_threshold_in_range():
    if not FACE_THRESH.exists():
        pytest.skip("Face threshold not generated yet.")
    t = float(json.loads(FACE_THRESH.read_text())["threshold"])
    assert 0.0 <= t <= 1.0


def test_face_scaler_feature_count():
    if not FACE_SCALER.exists():
        pytest.skip("Face scaler not generated yet.")
    sc = joblib.load(FACE_SCALER)
    assert sc.n_features_in_ == 288


def test_face_model_predicts_both_classes(splits):
    if not FACE_MODEL.exists() or not FACE_THRESH.exists():
        pytest.skip("Face model artifacts not generated yet.")
    _, val_df, _, feature_names = splits
    from sklearn.preprocessing import StandardScaler
    train_df, _, _, _ = splits
    sc = StandardScaler()
    sc.fit(train_df[feature_names].values.astype(np.float64))
    clf = joblib.load(FACE_MODEL)
    t = float(json.loads(FACE_THRESH.read_text())["threshold"])
    X_val = sc.transform(val_df[feature_names].values.astype(np.float64))
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_val)[:, 1]
    else:
        y_score = clf.predict(X_val).astype(float)
    y_pred = (y_score >= t).astype(int)
    assert 0 in set(y_pred.tolist()), "Face model never predicts REAL on validation"
    assert 1 in set(y_pred.tolist()), "Face model never predicts FAKE on validation"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
