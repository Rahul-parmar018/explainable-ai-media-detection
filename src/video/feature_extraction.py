import cv2
import numpy as np
from typing import Dict, Any, Optional
from skimage.feature import local_binary_pattern

def extract_color_features(hsv_frame: np.ndarray, num_bins: int = 8) -> Dict[str, float]:
    """
    Extracts color statistical metrics and normalized HSV histograms.

    Args:
        hsv_frame (np.ndarray): Input HSV frame array (H: 0-179, S: 0-255, V: 0-255).
        num_bins (int): Number of histogram bins per channel (default: 8).

    Returns:
        Dict[str, float]: Color features dictionary.
    """
    if hsv_frame is None or hsv_frame.ndim != 3:
        return {}

    h_channel = hsv_frame[:, :, 0]
    s_channel = hsv_frame[:, :, 1]
    v_channel = hsv_frame[:, :, 2]

    features = {
        "h_mean": float(np.mean(h_channel)),
        "h_std": float(np.std(h_channel)),
        "s_mean": float(np.mean(s_channel)),
        "s_std": float(np.std(s_channel)),
        "v_mean": float(np.mean(v_channel)),
        "v_std": float(np.std(v_channel)),
    }

    # Hue histogram (0 to 180 in OpenCV)
    h_hist = cv2.calcHist([hsv_frame], [0], None, [num_bins], [0, 180]).flatten()
    h_sum = np.sum(h_hist)
    h_hist = h_hist / (h_sum + 1e-7)
    for i in range(num_bins):
        features[f"h_hist_{i}"] = float(h_hist[i])

    # Saturation histogram (0 to 256)
    s_hist = cv2.calcHist([hsv_frame], [1], None, [num_bins], [0, 256]).flatten()
    s_sum = np.sum(s_hist)
    s_hist = s_hist / (s_sum + 1e-7)
    for i in range(num_bins):
        features[f"s_hist_{i}"] = float(s_hist[i])

    # Value histogram (0 to 256)
    v_hist = cv2.calcHist([hsv_frame], [2], None, [num_bins], [0, 256]).flatten()
    v_sum = np.sum(v_hist)
    v_hist = v_hist / (v_sum + 1e-7)
    for i in range(num_bins):
        features[f"v_hist_{i}"] = float(v_hist[i])

    return features


def extract_texture_features(gray_frame: np.ndarray, P: int = 8, R: int = 1) -> Dict[str, float]:
    """
    Extracts grayscale statistical metrics and normalized Local Binary Pattern (LBP) histogram.

    Args:
        gray_frame (np.ndarray): Input grayscale frame array (H, W).
        P (int): Number of circularly symmetric neighbor set points (default: 8).
        R (int): Radius of circle (default: 1).

    Returns:
        Dict[str, float]: Texture features dictionary.
    """
    if gray_frame is None or gray_frame.ndim != 2:
        return {}

    features = {
        "gray_mean": float(np.mean(gray_frame)),
        "gray_std": float(np.std(gray_frame)),
    }

    # Uniform LBP calculation
    lbp = local_binary_pattern(gray_frame, P=P, R=R, method="uniform")
    num_lbp_bins = P + 2  # 10 bins for P=8 uniform LBP
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, num_lbp_bins + 1),
        range=(0, num_lbp_bins)
    )
    lbp_hist = lbp_hist.astype(float)
    lbp_sum = np.sum(lbp_hist)
    lbp_hist = lbp_hist / (lbp_sum + 1e-7)

    for i in range(num_lbp_bins):
        features[f"lbp_hist_{i}"] = float(lbp_hist[i])

    return features


def extract_edge_features(
    gray_frame: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> Dict[str, float]:
    """
    Extracts structural edge features using Canny edge detection.

    Args:
        gray_frame (np.ndarray): Input grayscale frame array.
        low_threshold (int): First threshold for hysteresis procedure (default: 100).
        high_threshold (int): Second threshold for hysteresis procedure (default: 200).

    Returns:
        Dict[str, float]: Edge features dictionary.
    """
    if gray_frame is None or gray_frame.ndim != 2:
        return {}

    edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
    total_pixels = gray_frame.shape[0] * gray_frame.shape[1]
    edge_pixels = np.count_nonzero(edges)

    return {
        "edge_density": float(edge_pixels / total_pixels) if total_pixels > 0 else 0.0,
        "edge_mean": float(np.mean(edges)),
        "edge_std": float(np.std(edges)),
    }


def extract_frame_features(preprocessed_frame: Dict[str, Any]) -> Dict[str, float]:
    """
    Combines Color, Texture, and Edge feature families into a single feature dictionary for a frame.

    Args:
        preprocessed_frame (Dict[str, Any]): Preprocessed frame dictionary containing 'hsv' and 'gray'.

    Returns:
        Dict[str, float]: Unified dictionary of extracted visual features.
    """
    if not preprocessed_frame or not preprocessed_frame.get("valid", False):
        return {}

    hsv_frame = preprocessed_frame.get("hsv")
    gray_frame = preprocessed_frame.get("gray")

    color_feats = extract_color_features(hsv_frame)
    texture_feats = extract_texture_features(gray_frame)
    edge_feats = extract_edge_features(gray_frame)

    combined_features = {}
    combined_features.update(color_feats)
    combined_features.update(texture_feats)
    combined_features.update(edge_feats)

    return combined_features
