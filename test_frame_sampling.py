import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.dataset_utils import CATEGORY_LABEL_MAP
from src.video.frame_extraction import sample_video_frames

def run_frame_sampling_test(samples_per_category: int = 4, num_frames: int = 10):
    base_dir = Path("Dataset/Video")
    print("=" * 80)
    print(f"FRAME SAMPLING TEST REPORT ({samples_per_category} SAMPLES PER CATEGORY, {num_frames} FRAMES PER VIDEO)")
    print("=" * 80)

    total_tested = 0
    total_sampled_count = 0
    failures = 0
    sample_indices_example = []

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
            total_tested += 1

            result = sample_video_frames(str(vpath), num_frames=num_frames)

            print(f"\n--- Sample {total_tested} [{folder_name}] ---")
            print(f"Video:                   {result['video_path']}")
            print(f"Label:                   {label}")
            print(f"Total frames:            {result['total_frames']}")
            print(f"Sampled indices:         {result['sampled_indices']}")
            print(f"Frames successfully read: {result['actual_sampled_count']}")
            print(f"Frame shape:             {result['frame_shape']}")

            if result['actual_sampled_count'] == num_frames:
                total_sampled_count += result['actual_sampled_count']
            else:
                failures += 1
                total_sampled_count += result['actual_sampled_count']

            if not sample_indices_example and result['sampled_indices']:
                sample_indices_example = result['sampled_indices']

    avg_sampled = round(total_sampled_count / total_tested, 2) if total_tested > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"FRAME SAMPLING TEST SUMMARY:")
    print(f"  Total Videos Tested:               {total_tested}")
    print(f"  Requested Frames Per Video:        {num_frames}")
    print(f"  Average Successfully Read Frames:  {avg_sampled} / {num_frames}")
    print(f"  Sampling Failures:                 {failures}")
    print(f"  Sample Indices Example:            {sample_indices_example}")
    print("=" * 80)

if __name__ == "__main__":
    run_frame_sampling_test(samples_per_category=4, num_frames=10)
