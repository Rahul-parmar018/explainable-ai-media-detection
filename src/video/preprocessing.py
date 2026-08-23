import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (256, 256)
) -> Dict[str, Any]:
    """
    Preprocesses a single video frame by validating input, resizing to standard
    dimensions, and extracting color representations (BGR, Grayscale, HSV).

    Args:
        frame (np.ndarray): Input OpenCV BGR frame array.
        target_size (Tuple[int, int]): Target (width, height) for resizing (default: 256x256).

    Returns:
        Dict[str, Any]: Preprocessed representations and metadata dictionary.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return {
            "bgr": None,
            "gray": None,
            "hsv": None,
            "shape_bgr": None,
            "shape_gray": None,
            "shape_hsv": None,
            "valid": False,
            "error": "Invalid or empty frame array"
        }

    # Resize BGR frame to target standardized resolution
    resized_bgr = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    # Convert BGR to Grayscale
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)

    return {
        "bgr": resized_bgr,
        "gray": gray,
        "hsv": hsv,
        "shape_bgr": resized_bgr.shape,
        "shape_gray": gray.shape,
        "shape_hsv": hsv.shape,
        "valid": True,
        "error": None
    }


def preprocess_video_frames(
    sampled_frames: List[np.ndarray],
    target_size: Tuple[int, int] = (256, 256)
) -> List[Dict[str, Any]]:
    """
    Preprocesses a list of sampled BGR frames from a video stream.

    Args:
        sampled_frames (List[np.ndarray]): List of OpenCV BGR frame arrays.
        target_size (Tuple[int, int]): Target (width, height) for resizing (default: 256x256).

    Returns:
        List[Dict[str, Any]]: List of preprocessed frame dictionaries.
    """
    preprocessed = []
    for frame in sampled_frames:
        processed = preprocess_frame(frame, target_size=target_size)
        preprocessed.append(processed)
    return preprocessed
