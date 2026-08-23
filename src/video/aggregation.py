import math
import numpy as np
from typing import List, Dict, Any

def aggregate_frame_features(
    frame_features_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregates a temporal sequence of frame-level feature dictionaries into a single
    video-level feature vector using 4 summary statistics (Mean, Std, Min, Max).

    Args:
        frame_features_list (List[Dict[str, float]]): List of frame-level feature dictionaries.

    Returns:
        Dict[str, float]: Video-level aggregated feature dictionary (54 features x 4 stats = 216 features).
    """
    if not frame_features_list:
        return {}

    # Extract all feature keys from the first frame dictionary (maintaining consistent ordering)
    feature_keys = list(frame_features_list[0].keys())

    # Build matrix of values: shape (num_frames, num_features)
    matrix = []
    for f_dict in frame_features_list:
        row = [float(f_dict.get(key, 0.0)) for key in feature_keys]
        matrix.append(row)

    matrix_np = np.array(matrix, dtype=np.float64)

    # Calculate summary statistics across temporal frame axis (axis 0)
    means = np.mean(matrix_np, axis=0)
    stds = np.std(matrix_np, axis=0)
    mins = np.min(matrix_np, axis=0)
    maxs = np.max(matrix_np, axis=0)

    # Sanitize any NaN / Inf values
    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
    mins = np.nan_to_num(mins, nan=0.0, posinf=0.0, neginf=0.0)
    maxs = np.nan_to_num(maxs, nan=0.0, posinf=0.0, neginf=0.0)

    aggregated = {}
    for idx, key in enumerate(feature_keys):
        aggregated[f"{key}_mean"] = float(means[idx])
        aggregated[f"{key}_std"] = float(stds[idx])
        aggregated[f"{key}_min"] = float(mins[idx])
        aggregated[f"{key}_max"] = float(maxs[idx])

    return aggregated
