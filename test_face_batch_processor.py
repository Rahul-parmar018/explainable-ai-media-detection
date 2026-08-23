"""
test_face_batch_processor.py — Phase 13A: Batch Processor Tests.

Tests face feature CSV integrity after extraction.
Skips if video_face_features.csv not yet generated.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FACE_CSV = "data/videos/features/video_face_features.csv"
FACE_MODEL = Path("data/videos/models/face_best_model.joblib")
FACE_SCALER = Path("data/videos/models/face_scaler.joblib")
FACE_THRESH = Path("data/videos/models/face_threshold.json")


@pytest.fixture(scope="module")
def face_df():
    if not Path(FACE_CSV).exists():
        pytest.skip("video_face_features.csv not yet generated; run face_batch_processor first.")
    return pd.read_csv(FACE_CSV)


def test_no_duplicate_video_paths(face_df):
    n_dups = face_df["video_path"].duplicated().sum()
    assert n_dups == 0, f"{n_dups} duplicate video_paths found"


def test_valid_labels(face_df):
    assert set(face_df["label"].unique()).issubset({"REAL", "FAKE"}), \
        f"Unexpected labels: {face_df['label'].unique()}"


def test_face_detection_rate_in_range(face_df):
    assert (face_df["face_detection_rate"] >= 0).all()
    assert (face_df["face_detection_rate"] <= 1).all()


def test_no_nan_in_features(face_df):
    meta_cols = {"video_path", "video_name", "category", "label", "source_id",
                 "frames_sampled", "faces_detected", "face_detection_rate"}
    feat_cols = [c for c in face_df.columns if c not in meta_cols]
    assert not face_df[feat_cols].isna().any().any(), "NaN found in feature columns"


def test_no_inf_in_features(face_df):
    meta_cols = {"video_path", "video_name", "category", "label", "source_id",
                 "frames_sampled", "faces_detected", "face_detection_rate"}
    feat_cols = [c for c in face_df.columns if c not in meta_cols]
    assert not np.isinf(face_df[feat_cols].values).any(), "Inf found in feature columns"


def test_consistent_feature_count(face_df):
    meta_cols = {"video_path", "video_name", "category", "label", "source_id",
                 "frames_sampled", "faces_detected", "face_detection_rate"}
    feat_cols = [c for c in face_df.columns if c not in meta_cols]
    # 72 frame features × 4 aggregations = 288
    assert len(feat_cols) == 288, f"Expected 288 feature columns, got {len(feat_cols)}"


def test_expected_categories(face_df):
    expected = {"original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
                "NeuralTextures", "DeepFakeDetection"}
    actual = set(face_df["category"].unique())
    missing = expected - actual
    assert len(missing) == 0, f"Missing categories: {missing}"


def test_face_model_artifacts_exist():
    if not FACE_MODEL.exists():
        pytest.skip("Face model artifacts not yet generated; run face_model.py first.")
    assert FACE_MODEL.exists(),  f"Missing: {FACE_MODEL}"
    assert FACE_SCALER.exists(), f"Missing: {FACE_SCALER}"
    assert FACE_THRESH.exists(), f"Missing: {FACE_THRESH}"


def test_face_threshold_in_range():
    if not FACE_THRESH.exists():
        pytest.skip("Face threshold JSON not yet generated.")
    t = json.loads(FACE_THRESH.read_text())["threshold"]
    assert 0.0 <= float(t) <= 1.0, f"Threshold out of range: {t}"


def test_face_scaler_feature_count():
    if not FACE_SCALER.exists():
        pytest.skip("Face scaler not yet generated.")
    import joblib
    scaler = joblib.load(FACE_SCALER)
    assert scaler.n_features_in_ == 288, \
        f"Scaler n_features_in_={scaler.n_features_in_}, expected 288"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
