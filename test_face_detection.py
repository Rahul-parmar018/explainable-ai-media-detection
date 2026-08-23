"""
test_face_detection.py — Phase 13A: Face Detection Unit Tests.
"""
import sys
import numpy as np
import cv2
import pytest

from src.video.face_detection import (
    detect_and_crop_face,
    get_backend,
    FACE_CROP_SIZE,
    _expand_bbox,
    _clamp,
)


def _make_dummy_frame(h=240, w=320, with_face=False) -> np.ndarray:
    """Generate a synthetic BGR frame."""
    frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    if with_face:
        # Draw a skin-tone filled ellipse as a synthetic face
        center = (w // 2, h // 2)
        axes = (w // 6, h // 5)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 140, 120), -1)
    return frame


def test_detect_none_frame():
    """None input must not crash."""
    result = detect_and_crop_face(None)
    assert result["detected"] is False
    assert result["crop_bgr"] is None


def test_detect_empty_frame():
    """Empty array must not crash."""
    result = detect_and_crop_face(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result["detected"] is False


def test_detect_returns_correct_keys():
    """Result always contains all required keys."""
    frame = _make_dummy_frame()
    result = detect_and_crop_face(frame)
    for key in ["detected", "crop_bgr", "crop_gray", "crop_hsv", "bbox_raw",
                "bbox_padded", "face_area_ratio", "backend"]:
        assert key in result, f"Missing key: {key}"


def test_crop_shape_when_detected():
    """When a face IS detected, crop must match target size."""
    frame = _make_dummy_frame(480, 640, with_face=False)
    result = detect_and_crop_face(frame)
    if result["detected"]:
        w, h = FACE_CROP_SIZE
        assert result["crop_bgr"].shape[:2] == (h, w)
        assert result["crop_gray"].shape == (h, w)
        assert result["crop_hsv"].shape[:2] == (h, w)


def test_face_area_ratio_valid():
    """face_area_ratio must be in [0, 1]."""
    frame = _make_dummy_frame(240, 320)
    result = detect_and_crop_face(frame)
    assert 0.0 <= result["face_area_ratio"] <= 1.0


def test_expand_bbox_clamp():
    """_expand_bbox must not exceed image bounds."""
    # Face at top-left corner — padding might go negative
    x0, y0, x1, y1 = _expand_bbox(0, 0, 30, 30, img_h=100, img_w=100, pad=0.20)
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 100 and y1 <= 100
    assert x1 > x0 and y1 > y0


def test_expand_bbox_no_degenerate():
    """Expanded bbox width/height must be positive."""
    for x, y, w, h, img_h, img_w in [(5, 5, 20, 20, 100, 100), (0, 0, 5, 5, 50, 50)]:
        x0, y0, x1, y1 = _expand_bbox(x, y, w, h, img_h, img_w, pad=0.20)
        assert x1 > x0
        assert y1 > y0


def test_clamp():
    assert _clamp(-5, 0, 100) == 0
    assert _clamp(200, 0, 100) == 100
    assert _clamp(50, 0, 100) == 50


def test_backend_string():
    """Backend name must be a non-empty string after a detection call."""
    frame = _make_dummy_frame()
    detect_and_crop_face(frame)
    backend = get_backend()
    assert isinstance(backend, str)
    assert len(backend) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
