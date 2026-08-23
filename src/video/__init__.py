"""
Video processing, frame sampling, preprocessing, and visual feature extraction module.
"""

from .dataset_utils import get_video_info, get_sample_videos_info
from .frame_extraction import sample_video_frames
from .preprocessing import preprocess_frame, preprocess_video_frames
from .feature_extraction import (
    extract_color_features,
    extract_texture_features,
    extract_edge_features,
    extract_frame_features,
)

__all__ = [
    "get_video_info",
    "get_sample_videos_info",
    "sample_video_frames",
    "preprocess_frame",
    "preprocess_video_frames",
    "extract_color_features",
    "extract_texture_features",
    "extract_edge_features",
    "extract_frame_features",
]
