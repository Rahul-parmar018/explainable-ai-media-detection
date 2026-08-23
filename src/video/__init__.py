"""
Video processing, frame sampling, preprocessing, visual feature extraction, temporal aggregation, batch processing, ML training, SHAP explainability, ML experiments, dataset audit, and full ML benchmark module.
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
from .explainability import (
    map_feature_to_family,
    generate_feature_families_mapping,
    explain_video,
    compute_shap_explanations,
)
from .ml_experiments import (
    get_hyperparameter_search_space,
    run_model_experiments,
    run_feature_ablation_experiment,
)
from .full_batch_processor import (
    discover_all_videos,
    load_existing_processed_paths,
    run_full_batch_processing,
)
from .dataset_audit import (
    extract_actors_from_path,
    build_connected_component_groups,
    audit_dataset_integrity,
)
from .full_ml_benchmark import (
    load_and_split_connected_components,
    prepare_benchmark_scaled_features,
    evaluate_benchmark_predictions,
    run_full_ml_benchmark,
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
    "map_feature_to_family",
    "generate_feature_families_mapping",
    "explain_video",
    "compute_shap_explanations",
    "get_hyperparameter_search_space",
    "run_model_experiments",
    "run_feature_ablation_experiment",
    "discover_all_videos",
    "load_existing_processed_paths",
    "run_full_batch_processing",
    "extract_actors_from_path",
    "build_connected_component_groups",
    "audit_dataset_integrity",
    "load_and_split_connected_components",
    "prepare_benchmark_scaled_features",
    "evaluate_benchmark_predictions",
    "run_full_ml_benchmark",
]
