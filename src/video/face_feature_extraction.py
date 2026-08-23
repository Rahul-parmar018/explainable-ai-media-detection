"""
src/video/face_feature_extraction.py — Phase 13A Face-Region Forensic Features.

Extracts forensic features ONLY from a detected face crop (BGR/gray/HSV).
All features have deterministic names. Total per-frame: 72 features.

Feature families:
  A. HSV Color     — 30 features  (channel stats + histogram bins)
  B. Gray+LBP      — 12 features  (gray stats + LBP histogram)
  C. GLCM Texture  —  8 features  (multi-property co-occurrence)
  D. Frequency/DCT —  5 features  (energy bands + face-size-aware ratio)
  E. Edge          —  7 features  (Canny + Laplacian sharpness + Sobel)
  F. Forensic      — 10 features  (noise residual, high-freq anomaly, chroma)
                                   Total = 72
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# ──────────────────────────────────────────────────────────────────────────────
# A. HSV Color (30)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_color_features(hsv_crop: np.ndarray, num_bins: int = 8) -> Dict[str, float]:
    """
    Extract HSV color statistics and histograms from a face crop.
    Returns 30 features (6 stats + 8+8+8 hist bins).
    """
    if hsv_crop is None or hsv_crop.ndim != 3:
        return {}

    h_ch = hsv_crop[:, :, 0].astype(np.float32)
    s_ch = hsv_crop[:, :, 1].astype(np.float32)
    v_ch = hsv_crop[:, :, 2].astype(np.float32)

    features: Dict[str, float] = {
        "face_h_mean": float(np.mean(h_ch)),
        "face_h_std":  float(np.std(h_ch)),
        "face_s_mean": float(np.mean(s_ch)),
        "face_s_std":  float(np.std(s_ch)),
        "face_v_mean": float(np.mean(v_ch)),
        "face_v_std":  float(np.std(v_ch)),
    }

    h_hist = cv2.calcHist([hsv_crop], [0], None, [num_bins], [0, 180]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-7)
    for i, v in enumerate(h_hist):
        features[f"face_h_hist_{i}"] = float(v)

    s_hist = cv2.calcHist([hsv_crop], [1], None, [num_bins], [0, 256]).flatten()
    s_hist = s_hist / (s_hist.sum() + 1e-7)
    for i, v in enumerate(s_hist):
        features[f"face_s_hist_{i}"] = float(v)

    v_hist = cv2.calcHist([hsv_crop], [2], None, [num_bins], [0, 256]).flatten()
    v_hist = v_hist / (v_hist.sum() + 1e-7)
    for i, v in enumerate(v_hist):
        features[f"face_v_hist_{i}"] = float(v)

    return features  # 30 features


# ──────────────────────────────────────────────────────────────────────────────
# B. Gray + LBP (12)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_texture_features(gray_crop: np.ndarray, P: int = 8, R: int = 1) -> Dict[str, float]:
    """
    Extract grayscale statistics and uniform LBP histogram.
    Returns 12 features (2 stats + 10 LBP bins for P=8 uniform).
    """
    if gray_crop is None or gray_crop.ndim != 2:
        return {}

    features: Dict[str, float] = {
        "face_gray_mean": float(np.mean(gray_crop)),
        "face_gray_std":  float(np.std(gray_crop)),
    }

    lbp = local_binary_pattern(gray_crop, P=P, R=R, method="uniform")
    num_bins = P + 2  # 10 for P=8
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_bins + 1), range=(0, num_bins))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    for i, v in enumerate(lbp_hist):
        features[f"face_lbp_hist_{i}"] = float(v)

    return features  # 12 features


# ──────────────────────────────────────────────────────────────────────────────
# C. GLCM Texture (8)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_glcm_features(gray_crop: np.ndarray, levels: int = 64) -> Dict[str, float]:
    """
    Extract GLCM co-occurrence texture properties at distances [1, 2].
    Returns 8 features (4 properties × mean over 2 distances × 4 angles = averaged to 8).
    Actually: 4 properties mean-averaged over distances and angles + 4 more = 8.
    """
    if gray_crop is None or gray_crop.ndim != 2:
        return {}

    scale = 256.0 / levels
    quantized = (gray_crop.astype(np.float32) / scale).clip(0, levels - 1).astype(np.uint8)

    features: Dict[str, float] = {}
    for dist, dname in [(1, "d1"), (2, "d2")]:
        glcm = graycomatrix(
            quantized,
            distances=[dist],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        features[f"face_glcm_contrast_{dname}"]     = float(np.mean(graycoprops(glcm, "contrast")))
        features[f"face_glcm_homogeneity_{dname}"]  = float(np.mean(graycoprops(glcm, "homogeneity")))
        features[f"face_glcm_energy_{dname}"]       = float(np.mean(graycoprops(glcm, "energy")))
        features[f"face_glcm_correlation_{dname}"]  = float(np.mean(graycoprops(glcm, "correlation")))

    return features  # 8 features


# ──────────────────────────────────────────────────────────────────────────────
# D. Frequency / DCT (5)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_dct_features(gray_crop: np.ndarray) -> Dict[str, float]:
    """
    Extract DCT frequency band energies from face crop.
    Uses face-size-aware frequency thresholds.
    Returns 5 features.
    """
    if gray_crop is None or gray_crop.ndim != 2:
        return {}

    h, w = gray_crop.shape
    gray_f = gray_crop.astype(np.float32)
    dct = cv2.dct(gray_f)
    energy = dct ** 2

    u, v = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    freq_order = u + v

    # For 128×128 face crop: thresholds ~10 and ~40
    r1 = max(4, min(h, w) // 12)
    r2 = max(16, min(h, w) // 3)

    low_mask  = freq_order < r1
    mid_mask  = (freq_order >= r1) & (freq_order < r2)
    high_mask = freq_order >= r2

    dct_low  = float(np.mean(energy[low_mask]))  if np.any(low_mask)  else 0.0
    dct_mid  = float(np.mean(energy[mid_mask]))  if np.any(mid_mask)  else 0.0
    dct_high = float(np.mean(energy[high_mask])) if np.any(high_mask) else 0.0
    ratio_hl = float(dct_high / (dct_low + 1e-7))

    # DC component (top-left)
    dct_dc = float(energy[0, 0])

    return {
        "face_dct_low_energy":   dct_low,
        "face_dct_mid_energy":   dct_mid,
        "face_dct_high_energy":  dct_high,
        "face_dct_high_low_ratio": ratio_hl,
        "face_dct_dc":           dct_dc,
    }  # 5 features


# ──────────────────────────────────────────────────────────────────────────────
# E. Edge (7)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_edge_features(gray_crop: np.ndarray) -> Dict[str, float]:
    """
    Extract edge and sharpness features from face crop.
    Returns 7 features.
    """
    if gray_crop is None or gray_crop.ndim != 2:
        return {}

    total_px = gray_crop.shape[0] * gray_crop.shape[1]

    # Canny edge density
    edges_canny = cv2.Canny(gray_crop, 80, 180)
    edge_density = float(np.count_nonzero(edges_canny)) / total_px

    # Laplacian sharpness (variance of Laplacian = focus measure)
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
    lap_var  = float(np.var(lap))
    lap_mean = float(np.mean(np.abs(lap)))

    # Sobel gradient magnitude
    sobel_x = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mean = float(np.mean(grad_mag))
    sobel_std  = float(np.std(grad_mag))

    return {
        "face_edge_density":    edge_density,
        "face_edge_mean":       float(np.mean(edges_canny)),
        "face_edge_std":        float(np.std(edges_canny)),
        "face_laplacian_var":   lap_var,
        "face_laplacian_mean":  lap_mean,
        "face_sobel_mean":      sobel_mean,
        "face_sobel_std":       sobel_std,
    }  # 7 features


# ──────────────────────────────────────────────────────────────────────────────
# F. Forensic (10)
# ──────────────────────────────────────────────────────────────────────────────

def extract_face_forensic_features(bgr_crop: np.ndarray, gray_crop: np.ndarray) -> Dict[str, float]:
    """
    Extract deepfake-sensitive forensic features.

    Includes:
    - Noise residual statistics (high-pass filter minus image)
    - Chroma channel saturation anomaly
    - Block artifact measure
    - High-frequency anomaly in YCrCb

    Returns 10 features.
    """
    if bgr_crop is None or bgr_crop.ndim != 3 or gray_crop is None:
        return {}

    features: Dict[str, float] = {}

    # ── Noise residual (Wiener-like: subtract 3×3 median blur) ───────────────
    blurred = cv2.medianBlur(gray_crop, 3)
    residual = gray_crop.astype(np.float32) - blurred.astype(np.float32)
    features["face_noise_mean"]    = float(np.mean(np.abs(residual)))
    features["face_noise_std"]     = float(np.std(residual))
    features["face_noise_energy"]  = float(np.mean(residual ** 2))

    # ── Chroma channel (YCrCb) ───────────────────────────────────────────────
    ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr_ch = ycrcb[:, :, 1]
    cb_ch = ycrcb[:, :, 2]
    features["face_chroma_cr_mean"] = float(np.mean(cr_ch))
    features["face_chroma_cr_std"]  = float(np.std(cr_ch))
    features["face_chroma_cb_mean"] = float(np.mean(cb_ch))
    features["face_chroma_cb_std"]  = float(np.std(cb_ch))

    # ── Block artifact (8×8 DCT block boundary discontinuity) ───────────────
    h, w = gray_crop.shape
    gray_f = gray_crop.astype(np.float32)
    # Mean absolute difference at 8-pixel horizontal boundaries
    block_diffs = []
    for row in range(8, h, 8):
        diff = np.abs(gray_f[row, :] - gray_f[row - 1, :])
        block_diffs.append(float(np.mean(diff)))
    for col in range(8, w, 8):
        diff = np.abs(gray_f[:, col] - gray_f[:, col - 1])
        block_diffs.append(float(np.mean(diff)))
    features["face_block_artifact"] = float(np.mean(block_diffs)) if block_diffs else 0.0

    # ── High-frequency power ratio (Fourier based) ───────────────────────────
    f_shift = np.fft.fftshift(np.fft.fft2(gray_f))
    magnitude = np.abs(f_shift)
    ch, cw = h // 2, w // 2
    r_inner = min(ch, cw) // 4
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - ch) ** 2 + (xx - cw) ** 2)
    low_mask  = dist <= r_inner
    high_mask = dist > r_inner
    low_power  = float(np.mean(magnitude[low_mask]))  if np.any(low_mask)  else 0.0
    high_power = float(np.mean(magnitude[high_mask])) if np.any(high_mask) else 0.0
    features["face_hf_power_ratio"] = float(high_power / (low_power + 1e-7))

    # ── Saturation anomaly (mean-normalised chroma spread) ───────────────────
    bgr_f = bgr_crop.astype(np.float32)
    channel_means = bgr_f.mean(axis=(0, 1))
    channel_stds  = bgr_f.std(axis=(0, 1))
    # Coefficient of variation across channels → measures chroma uniformity
    chroma_spread = float(np.std(channel_stds) / (np.mean(channel_stds) + 1e-7))
    features["face_chroma_spread"] = chroma_spread

    return features  # 10 features


# ──────────────────────────────────────────────────────────────────────────────
# Combined extractor
# ──────────────────────────────────────────────────────────────────────────────

TOTAL_FACE_FEATURES = 72  # 30 + 12 + 8 + 5 + 7 + 10


def extract_face_features(face_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Run all forensic feature families on a face detection result.

    Args:
        face_result: dict returned by detect_and_crop_face()

    Returns:
        Dict of 72 named features, or empty dict if face was not detected.
    """
    if not face_result or not face_result.get("detected", False):
        return {}

    bgr  = face_result.get("crop_bgr")
    gray = face_result.get("crop_gray")
    hsv  = face_result.get("crop_hsv")

    if bgr is None or gray is None or hsv is None:
        return {}

    combined: Dict[str, float] = {}
    combined.update(extract_face_color_features(hsv))
    combined.update(extract_face_texture_features(gray))
    combined.update(extract_face_glcm_features(gray))
    combined.update(extract_face_dct_features(gray))
    combined.update(extract_face_edge_features(gray))
    combined.update(extract_face_forensic_features(bgr, gray))

    return combined
