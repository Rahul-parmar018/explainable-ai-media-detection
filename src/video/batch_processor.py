import os
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.video.dataset_utils import CATEGORY_LABEL_MAP, get_video_info
from src.video.frame_extraction import sample_video_frames
from src.video.preprocessing import preprocess_frame
from src.video.feature_extraction import extract_frame_features
from src.video.aggregation import aggregate_frame_features

def extract_source_id(file_name: str, category: str) -> str:
    """
    Extracts the source video ID from a filename based on category conventions.

    Args:
        file_name (str): Video file name (e.g. '006.mp4', '006_002.mp4', '01_02__...mp4').
        category (str): Category folder name.

    Returns:
        str: Extracted source ID.
    """
    stem = Path(file_name).stem
    if category == "original":
        return stem
    elif category in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
        return stem.split("_")[0] if "_" in stem else stem
    else:  # DeepFakeDetection
        return stem.split("__")[0] if "__" in stem else stem


def select_stage_b_videos(
    dataset_dir: str = "Dataset/Video",
    samples_per_category: int = 100,
    random_state: int = 42
) -> List[Dict[str, str]]:
    """
    Selects Stage B videos (100 per category = 700 videos) preserving source ID grouping
    to prevent data leakage across categories.

    Args:
        dataset_dir (str): Base dataset directory.
        samples_per_category (int): Target videos per category (default: 100).
        random_state (int): Fixed seed for reproducibility (default: 42).

    Returns:
        List[Dict[str, str]]: List of selected video descriptors.
    """
    base_dir = Path(dataset_dir)
    orig_dir = base_dir / "original"
    if not orig_dir.exists():
        raise FileNotFoundError(f"Original directory {orig_dir} not found.")

    orig_files = sorted([f for f in os.listdir(orig_dir) if f.lower().endswith(".mp4")])
    source_ids = [Path(f).stem for f in orig_files]

    # Reproducibly select 100 source IDs
    rng = random.Random(random_state)
    selected_source_ids = sorted(rng.sample(source_ids, samples_per_category))

    selected_videos = []

    # 1. Standard FF++ Categories (original + 5 manipulated variants matching source IDs)
    ff_categories = ["original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    for cat in ff_categories:
        cat_dir = base_dir / cat
        if not cat_dir.exists():
            continue
        cat_files = set(os.listdir(cat_dir))

        for sid in selected_source_ids:
            if cat == "original":
                vfile = f"{sid}.mp4"
            else:
                matches = [f for f in cat_files if f.startswith(f"{sid}_")]
                vfile = matches[0] if matches else None

            if vfile and vfile in cat_files:
                selected_videos.append({
                    "video_path": str(cat_dir / vfile),
                    "video_name": vfile,
                    "category": cat,
                    "label": CATEGORY_LABEL_MAP[cat],
                    "source_id": sid
                })

    # 2. DeepFakeDetection (DFD) Category (100 reproducibly sampled videos)
    dfd_dir = base_dir / "DeepFakeDetection"
    if dfd_dir.exists():
        dfd_files = sorted([f for f in os.listdir(dfd_dir) if f.lower().endswith(".mp4")])
        rng_dfd = random.Random(random_state)
        selected_dfd_files = sorted(rng_dfd.sample(dfd_files, samples_per_category))

        for vfile in selected_dfd_files:
            sid = extract_source_id(vfile, "DeepFakeDetection")
            selected_videos.append({
                "video_path": str(dfd_dir / vfile),
                "video_name": vfile,
                "category": "DeepFakeDetection",
                "label": CATEGORY_LABEL_MAP["DeepFakeDetection"],
                "source_id": sid
            })

    return selected_videos


def process_single_video(
    video_meta: Dict[str, str],
    num_frames: int = 10,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Processes a single video stream through frame sampling, preprocessing,
    feature extraction, and temporal aggregation.

    Args:
        video_meta (Dict[str, str]): Video metadata dictionary.
        num_frames (int): Number of frames to sample (default: 10).
        target_size (Tuple[int, int]): Preprocessing target size (default: 256x256).

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            (row_dict, None) on success or (None, error_dict) on failure.
    """
    vpath = video_meta["video_path"]
    try:
        # Get metadata
        info = get_video_info(vpath)
        if not info["readable"]:
            return None, {
                "video_path": vpath,
                "category": video_meta["category"],
                "label": video_meta["label"],
                "error": info.get("error", "Unreadable video stream")
            }

        # 1. Sample frames
        sampling_res = sample_video_frames(vpath, num_frames=num_frames)
        sampled_frames = sampling_res["sampled_frames"]
        if not sampled_frames or len(sampled_frames) == 0:
            return None, {
                "video_path": vpath,
                "category": video_meta["category"],
                "label": video_meta["label"],
                "error": "No frames extracted"
            }

        # 2. Preprocess & extract frame features
        frame_feats_list = []
        for frame in sampled_frames:
            pf = preprocess_frame(frame, target_size=target_size)
            ff = extract_frame_features(pf)
            if ff:
                frame_feats_list.append(ff)

        if not frame_feats_list:
            return None, {
                "video_path": vpath,
                "category": video_meta["category"],
                "label": video_meta["label"],
                "error": "Feature extraction failed on sampled frames"
            }

        # 3. Aggregate frame features
        agg_features = aggregate_frame_features(frame_feats_list)
        if not agg_features:
            return None, {
                "video_path": vpath,
                "category": video_meta["category"],
                "label": video_meta["label"],
                "error": "Aggregation produced empty feature set"
            }

        # Build complete row dictionary
        row_dict = {
            "video_path": vpath,
            "category": video_meta["category"],
            "label": video_meta["label"],
            "source_id": video_meta["source_id"],
            "frame_count": info["frame_count"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
        }
        row_dict.update(agg_features)

        return row_dict, None

    except Exception as e:
        return None, {
            "video_path": vpath,
            "category": video_meta["category"],
            "label": video_meta["label"],
            "error": str(e)
        }


def run_batch_processing(
    dataset_dir: str = "Dataset/Video",
    output_dir: str = "data/videos/features",
    samples_per_category: int = 100,
    random_state: int = 42,
    num_frames: int = 10
) -> Dict[str, Any]:
    """
    Executes controlled batch processing for Stage B video dataset.

    Args:
        dataset_dir (str): Input video dataset directory.
        output_dir (str): Output CSV directory.
        samples_per_category (int): Videos per category (default: 100).
        random_state (int): Random seed (default: 42).
        num_frames (int): Number of frames to sample (default: 10).

    Returns:
        Dict[str, Any]: Processing summary dictionary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features_csv_path = out_path / "video_features_stage_b.csv"
    errors_csv_path = out_path / "video_processing_errors.csv"

    selected_videos = select_stage_b_videos(
        dataset_dir=dataset_dir,
        samples_per_category=samples_per_category,
        random_state=random_state
    )

    total_selected = len(selected_videos)
    print(f"Starting Stage B Batch Processing: {total_selected} videos selected ({samples_per_category} per category)...")

    successful_rows = []
    error_rows = []

    for idx, video_meta in enumerate(selected_videos, 1):
        print(f"[{idx:03d}/{total_selected:03d}] Processing [{video_meta['category']:18s}]: {video_meta['video_name']}", end="\r")

        row_dict, err_dict = process_single_video(video_meta, num_frames=num_frames)

        if row_dict is not None:
            successful_rows.append(row_dict)
        else:
            error_rows.append(err_dict)

    print(f"\nProcessing complete: {len(successful_rows)} successful, {len(error_rows)} errors.")

    # Save features CSV
    if successful_rows:
        fieldnames = list(successful_rows[0].keys())
        with open(features_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(successful_rows)
        print(f"Features saved to: {features_csv_path}")

    # Save errors CSV
    with open(errors_csv_path, mode="w", newline="", encoding="utf-8") as f:
        err_fieldnames = ["video_path", "category", "label", "error"]
        writer = csv.DictWriter(f, fieldnames=err_fieldnames)
        writer.writeheader()
        if error_rows:
            writer.writerows(error_rows)
    print(f"Error log saved to: {errors_csv_path}")

    return {
        "total_selected": total_selected,
        "successful_count": len(successful_rows),
        "error_count": len(error_rows),
        "features_csv": str(features_csv_path),
        "errors_csv": str(errors_csv_path)
    }

if __name__ == "__main__":
    run_batch_processing()
