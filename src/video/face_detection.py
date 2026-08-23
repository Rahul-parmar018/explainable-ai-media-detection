"""
src/video/face_detection.py — Phase 13A Face Detection Module.

Provides robust per-frame face detection using MediaPipe FaceDetection as the
primary backend, with an OpenCV Haar Cascade fallback.

Design principles:
- Deterministic: identical input → identical output
- Resilient: no face detected → graceful None return, never crash
- Minimal dependency surface: MediaPipe is already installed
- Face crop includes a configurable padding margin to preserve boundary context
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe initialisation (lazy, module-level singleton)
# ──────────────────────────────────────────────────────────────────────────────

_mp_face_detection = None
_haar_cascade: Optional[cv2.CascadeClassifier] = None
_BACKEND: str = "none"


def _init_mediapipe() -> bool:
    """Initialise MediaPipe FaceDetection singleton. Returns True on success."""
    global _mp_face_detection, _BACKEND
    if _mp_face_detection is not None:
        return True
    try:
        import mediapipe as mp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,     # 0 = short-range (<2 m), better for close-up faces
                min_detection_confidence=0.5,
            )
        _BACKEND = "mediapipe"
        return True
    except Exception:
        return False


def _init_haar() -> bool:
    """Initialise OpenCV Haar Cascade fallback. Returns True on success."""
    global _haar_cascade, _BACKEND
    if _haar_cascade is not None:
        return True
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return False
    _haar_cascade = cascade
    _BACKEND = "haar"
    return True


def get_backend() -> str:
    """Return the active face detection backend name."""
    return _BACKEND


# ──────────────────────────────────────────────────────────────────────────────
# Core detection
# ──────────────────────────────────────────────────────────────────────────────

# Face crop target size (pixels)
FACE_CROP_SIZE: Tuple[int, int] = (128, 128)

# Padding factor: expand bounding box by this fraction of face dimension on each side
PADDING_FACTOR: float = 0.20


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _expand_bbox(
    x: int, y: int, w: int, h: int,
    img_h: int, img_w: int,
    pad: float = PADDING_FACTOR,
) -> Tuple[int, int, int, int]:
    """Expand and clamp a bounding box with padding, staying within image bounds."""
    pad_x = int(w * pad)
    pad_y = int(h * pad)
    x0 = _clamp(x - pad_x, 0, img_w - 1)
    y0 = _clamp(y - pad_y, 0, img_h - 1)
    x1 = _clamp(x + w + pad_x, 1, img_w)
    y1 = _clamp(y + h + pad_y, 1, img_h)
    return x0, y0, x1, y1


def _detect_mediapipe(bgr_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run MediaPipe face detection on a BGR frame.

    Returns (x, y, w, h) of the largest detected face (pixels), or None.
    """
    global _mp_face_detection
    if _mp_face_detection is None:
        if not _init_mediapipe():
            return None

    h, w = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    import mediapipe as mp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _mp_face_detection.process(rgb)

    if not results.detections:
        return None

    # Select the detection with the largest bounding box area
    best = None
    best_area = -1
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        bx = int(bbox.xmin * w)
        by = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        area = bw * bh
        if area > best_area and bw > 0 and bh > 0:
            best_area = area
            best = (bx, by, bw, bh)

    return best


def _detect_haar(bgr_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run OpenCV Haar Cascade face detection on a BGR frame.

    Returns (x, y, w, h) of the largest detected face (pixels), or None.
    """
    global _haar_cascade
    if _haar_cascade is None:
        if not _init_haar():
            return None

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None

    # Return largest face
    areas = [w * h for (_, _, w, h) in faces]
    best_idx = int(np.argmax(areas))
    x, y, w, h = faces[best_idx]
    return int(x), int(y), int(w), int(h)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(
    bgr_frame: np.ndarray,
    target_size: Tuple[int, int] = FACE_CROP_SIZE,
    padding: float = PADDING_FACTOR,
) -> Dict[str, Any]:
    """
    Detect the primary face in a BGR frame and return a cropped, resized crop.

    Args:
        bgr_frame:   OpenCV BGR frame (H, W, 3).
        target_size: (width, height) to resize the face crop.
        padding:     Fractional expansion of the detected bounding box.

    Returns:
        dict with keys:
            detected      (bool)  — whether a face was found
            crop_bgr      (ndarray | None) — resized face crop in BGR
            crop_gray     (ndarray | None) — grayscale face crop
            crop_hsv      (ndarray | None) — HSV face crop
            bbox_raw      (tuple | None)   — raw (x, y, w, h) before padding
            bbox_padded   (tuple | None)   — padded (x0, y0, x1, y1)
            face_area_ratio (float)        — fraction of frame area occupied by face
            backend       (str)            — detection backend used
    """
    null_result = {
        "detected": False,
        "crop_bgr": None,
        "crop_gray": None,
        "crop_hsv": None,
        "bbox_raw": None,
        "bbox_padded": None,
        "face_area_ratio": 0.0,
        "backend": _BACKEND,
    }

    if bgr_frame is None or bgr_frame.ndim != 3 or bgr_frame.size == 0:
        return null_result

    img_h, img_w = bgr_frame.shape[:2]
    frame_area = img_h * img_w

    # ── 1. Try MediaPipe first ───────────────────────────────────────────────
    bbox = _detect_mediapipe(bgr_frame)
    backend_used = "mediapipe"

    # ── 2. Fallback to Haar if MediaPipe fails ───────────────────────────────
    if bbox is None:
        bbox = _detect_haar(bgr_frame)
        backend_used = "haar"

    if bbox is None:
        return {**null_result, "backend": backend_used}

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return {**null_result, "backend": backend_used}

    # ── 3. Expand + clamp bbox ───────────────────────────────────────────────
    x0, y0, x1, y1 = _expand_bbox(x, y, w, h, img_h, img_w, pad=padding)
    if x1 <= x0 or y1 <= y0:
        return {**null_result, "backend": backend_used}

    # ── 4. Crop ──────────────────────────────────────────────────────────────
    crop = bgr_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return {**null_result, "backend": backend_used}

    # ── 5. Resize ────────────────────────────────────────────────────────────
    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    crop_gray    = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
    crop_hsv     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

    face_area = w * h
    face_area_ratio = float(face_area) / float(frame_area) if frame_area > 0 else 0.0

    return {
        "detected": True,
        "crop_bgr": crop_resized,
        "crop_gray": crop_gray,
        "crop_hsv": crop_hsv,
        "bbox_raw": (x, y, w, h),
        "bbox_padded": (x0, y0, x1, y1),
        "face_area_ratio": round(face_area_ratio, 6),
        "backend": backend_used,
    }
