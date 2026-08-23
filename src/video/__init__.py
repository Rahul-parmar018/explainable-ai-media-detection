"""
Video processing, frame sampling, preprocessing, visual feature extraction, temporal aggregation, batch processing, and ML training module.
"""

from .dataset_utils import get_video_info, get_sample_videos_info
from .frame_extraction import sample_video_frames
from .preprocessing import preprocess_frame, preprocess_video_frames
from .feature_extraction import (
    extract_color_features,
    extract_texture_features,
    extract_edge_features,
    extract_glcm_features,
    extract_dct_features,
    extract_frame_features,
)
from .aggregation import aggregate_frame_features
from .batch_processor import (
    select_stage_b_videos,
    process_single_video,
    run_batch_processing,
)
from .ml_training import (
    load_and_split_data,
    prepare_scaled_features,
    evaluate_predictions,
    train_and_evaluate_baselines,
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
    "extract_glcm_features",
    "extract_dct_features",
    "extract_frame_features",
    "aggregate_frame_features",
    "select_stage_b_videos",
    "process_single_video",
    "run_batch_processing",
    "load_and_split_data",
    "prepare_scaled_features",
    "evaluate_predictions",
    "train_and_evaluate_baselines",
]
