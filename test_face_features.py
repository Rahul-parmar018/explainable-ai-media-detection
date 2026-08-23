"""
test_face_features.py — Phase 13A: Face Feature Extraction Unit Tests.
"""
import sys
import numpy as np
import pytest

from src.video.face_detection import detect_and_crop_face, FACE_CROP_SIZE
from src.video.face_feature_extraction import (
    extract_face_color_features,
    extract_face_texture_features,
    extract_face_glcm_features,
    extract_face_dct_features,
    extract_face_edge_features,
    extract_face_forensic_features,
    extract_face_features,
    TOTAL_FACE_FEATURES,
)


def _make_face_result(seed: int = 0):
    """Synthesize a face detection result with random image data."""
    rng = np.random.default_rng(seed)
    w, h = FACE_CROP_SIZE
    bgr  = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, (h, w), dtype=np.uint8)
    hsv  = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (hsv[:, :, 0] * 180 / 255).astype(np.uint8)  # Hue range 0-179
    return {
        "detected": True,
        "crop_bgr": bgr.copy(),
        "crop_gray": gray.copy(),
        "crop_hsv": hsv.copy(),
        "bbox_raw": (10, 10, 50, 60),
        "bbox_padded": (8, 8, 62, 72),
        "face_area_ratio": 0.12,
        "backend": "test",
    }


def test_color_feature_count():
    face = _make_face_result()
    feats = extract_face_color_features(face["crop_hsv"])
    assert len(feats) == 30, f"Expected 30 color features, got {len(feats)}"


def test_texture_feature_count():
    face = _make_face_result()
    feats = extract_face_texture_features(face["crop_gray"])
    assert len(feats) == 12, f"Expected 12 texture features, got {len(feats)}"


def test_glcm_feature_count():
    face = _make_face_result()
    feats = extract_face_glcm_features(face["crop_gray"])
    assert len(feats) == 8, f"Expected 8 GLCM features, got {len(feats)}"


def test_dct_feature_count():
    face = _make_face_result()
    feats = extract_face_dct_features(face["crop_gray"])
    assert len(feats) == 5, f"Expected 5 DCT features, got {len(feats)}"


def test_edge_feature_count():
    face = _make_face_result()
    feats = extract_face_edge_features(face["crop_gray"])
    assert len(feats) == 7, f"Expected 7 edge features, got {len(feats)}"


def test_forensic_feature_count():
    face = _make_face_result()
    feats = extract_face_forensic_features(face["crop_bgr"], face["crop_gray"])
    assert len(feats) == 10, f"Expected 10 forensic features, got {len(feats)}"


def test_total_feature_count():
    face = _make_face_result()
    feats = extract_face_features(face)
    assert len(feats) == TOTAL_FACE_FEATURES, \
        f"Expected {TOTAL_FACE_FEATURES} total features, got {len(feats)}"


def test_deterministic_feature_names():
    """Same input must produce identical feature names and values."""
    face1 = _make_face_result(seed=99)
    face2 = _make_face_result(seed=99)
    f1 = extract_face_features(face1)
    f2 = extract_face_features(face2)
    assert list(f1.keys()) == list(f2.keys())
    for k in f1:
        assert abs(f1[k] - f2[k]) < 1e-9, f"Non-deterministic feature: {k}"


def test_no_nan_in_features():
    for seed in range(5):
        face = _make_face_result(seed=seed)
        feats = extract_face_features(face)
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in feature {k} (seed={seed})"


def test_no_inf_in_features():
    for seed in range(5):
        face = _make_face_result(seed=seed)
        feats = extract_face_features(face)
        for k, v in feats.items():
            assert not np.isinf(v), f"Inf in feature {k} (seed={seed})"


def test_empty_face_result_returns_empty():
    result = {"detected": False}
    feats = extract_face_features(result)
    assert feats == {}


def test_none_crops_return_empty():
    bad_result = {"detected": True, "crop_bgr": None, "crop_gray": None, "crop_hsv": None}
    feats = extract_face_features(bad_result)
    assert feats == {}


def test_feature_names_have_face_prefix():
    face = _make_face_result()
    feats = extract_face_features(face)
    for k in feats:
        assert k.startswith("face_"), f"Feature missing 'face_' prefix: {k}"


def test_aggregated_feature_count():
    """After mean/std/min/max aggregation, 72 × 4 = 288 features."""
    from src.video.face_batch_processor import aggregate_face_features
    face_list = [extract_face_features(_make_face_result(seed=s)) for s in range(5)]
    agg = aggregate_face_features(face_list)
    assert len(agg) == TOTAL_FACE_FEATURES * 4, \
        f"Expected {TOTAL_FACE_FEATURES * 4} aggregated features, got {len(agg)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
