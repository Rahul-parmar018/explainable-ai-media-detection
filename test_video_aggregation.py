import os
import sys
import math
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.dataset_utils import CATEGORY_LABEL_MAP
from src.video.frame_extraction import sample_video_frames
from src.video.preprocessing import preprocess_frame
from src.video.feature_extraction import extract_frame_features
from src.video.aggregation import aggregate_frame_features

def run_video_aggregation_test(samples_per_category: int = 4, num_frames: int = 10):
    base_dir = Path("Dataset/Video")
    print("=" * 80)
    print(f"VIDEO FEATURE AGGREGATION TEST REPORT ({samples_per_category} SAMPLES PER CATEGORY, {num_frames} FRAMES PER VIDEO)")
    print("=" * 80)

    total_videos = 0
    successful_videos = 0
    failed_videos = 0
    nan_count = 0
    inf_count = 0

    expected_frame_feature_count = 54
    expected_video_feature_count = 216

    example_video_features = None

    for folder_name, label in CATEGORY_LABEL_MAP.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Warning: Directory {folder_path} not found.")
            continue

        video_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".mp4")
        ])

        selected_videos = video_files[:samples_per_category]

        for vfile in selected_videos:
            vpath = folder_path / vfile
            total_videos += 1

            # 1. Sample frames
            sampling_res = sample_video_frames(str(vpath), num_frames=num_frames)
            sampled_frames = sampling_res["sampled_frames"]

            # 2. Preprocess & extract frame features
            frame_feats_list = []
            for frame in sampled_frames:
                pf = preprocess_frame(frame, target_size=(256, 256))
                ff = extract_frame_features(pf)
                if ff:
                    frame_feats_list.append(ff)

            # 3. Aggregate 10 frame feature vectors into 1 video feature vector
            agg_feats = aggregate_frame_features(frame_feats_list)

            # Verification checks
            is_valid = True
            if len(agg_feats) != expected_video_feature_count:
                is_valid = False

            v_nan = 0
            v_inf = 0
            for k, val in agg_feats.items():
                if math.isnan(val):
                    v_nan += 1
                    nan_count += 1
                    is_valid = False
                if math.isinf(val):
                    v_inf += 1
                    inf_count += 1
                    is_valid = False

            if is_valid:
                successful_videos += 1
                if example_video_features is None:
                    example_video_features = agg_feats
            else:
                failed_videos += 1

            frame_count_str = f"{len(frame_feats_list)}"
            ff_count_str = f"{len(frame_feats_list[0])}" if frame_feats_list else "0"
            vf_count_str = f"{len(agg_feats)}"

            print(f"Video {total_videos:02d}/28 [{folder_name:18s}]: {vfile:40s} | Frames: {frame_count_str} | Frame Feats: {ff_count_str} | Video Feats: {vf_count_str} | Valid: {'Yes' if is_valid else 'No'}")

    print("\n" + "=" * 80)
    print("VIDEO FEATURE AGGREGATION SUMMARY:")
    print(f"  Total videos:             {total_videos}")
    print(f"  Successful:               {successful_videos}")
    print(f"  Failed:                   {failed_videos}")
    print(f"  Expected video features:  {expected_video_feature_count}")
    print(f"  Actual video features:    {len(example_video_features) if example_video_features else 0}")
    print(f"  NaN count:                {nan_count}")
    print(f"  Inf count:                {inf_count}")
    print("=" * 80)

    if example_video_features:
        print("\n--- EXAMPLE VIDEO-LEVEL FEATURE DICTIONARY (216 FEATURES) ---")
        for key, val in list(example_video_features.items())[:20]:
            print(f"  {key:28s}: {val:.6f}")
        print(f"  ... [{len(example_video_features) - 20} additional features omitted for brevity]")
        print("=" * 80)

if __name__ == "__main__":
    run_video_aggregation_test(samples_per_category=4, num_frames=10)
