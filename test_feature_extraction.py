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

def run_feature_extraction_test(samples_per_category: int = 4, num_frames: int = 10):
    base_dir = Path("Dataset/Video")
    print("=" * 80)
    print(f"ADVANCED FEATURE EXTRACTION TEST REPORT ({samples_per_category} SAMPLES PER CATEGORY, {num_frames} FRAMES PER VIDEO)")
    print("=" * 80)

    total_videos = 0
    total_frames = 0
    successful_extractions = 0
    failed_extractions = 0
    expected_feature_count = None
    example_feature_vector = None

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

            video_valid = True
            for frame in sampled_frames:
                total_frames += 1

                # 2. Preprocess frame
                pf = preprocess_frame(frame, target_size=(256, 256))

                # 3. Extract features (Color + LBP + Edge + GLCM + DCT)
                feats = extract_frame_features(pf)

                # Verification rules:
                # a) non-empty
                # b) no NaN, no Inf
                # c) numeric values
                # d) consistent vector length
                if not feats:
                    video_valid = False
                    failed_extractions += 1
                    continue

                if expected_feature_count is None:
                    expected_feature_count = len(feats)
                    example_feature_vector = feats
                elif len(feats) != expected_feature_count:
                    video_valid = False
                    failed_extractions += 1
                    continue

                # Check numerical integrity
                has_invalid_val = False
                for val in feats.values():
                    if not isinstance(val, (int, float, np.number)) or math.isnan(val) or math.isinf(val):
                        has_invalid_val = True
                        break

                if has_invalid_val:
                    video_valid = False
                    failed_extractions += 1
                else:
                    successful_extractions += 1

            print(f"Video {total_videos:02d}/28 [{folder_name:18s}]: {vfile:45s} | Frames: {len(sampled_frames)} | Feature Extraction: {'Passed' if video_valid else 'Failed'}")

    print("\n" + "=" * 80)
    print("ADVANCED FEATURE EXTRACTION TEST SUMMARY:")
    print(f"  Total videos:                     {total_videos}")
    print(f"  Total frames:                     {total_frames}")
    print(f"  Successful feature extractions:   {successful_extractions}")
    print(f"  Failed feature extractions:       {failed_extractions}")
    print(f"  Number of features per frame:     {expected_feature_count}")
    print("=" * 80)

    if example_feature_vector:
        print("\n--- EXAMPLE FEATURE VECTOR (COMPLETE 54-FEATURE FRAME DESCRIPTOR) ---")
        for key, val in example_feature_vector.items():
            print(f"  {key:22s}: {val:.6f}")
        print("=" * 80)

if __name__ == "__main__":
    run_feature_extraction_test(samples_per_category=4, num_frames=10)
