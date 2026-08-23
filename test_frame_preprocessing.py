import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.dataset_utils import CATEGORY_LABEL_MAP
from src.video.frame_extraction import sample_video_frames
from src.video.preprocessing import preprocess_frame, preprocess_video_frames

def run_frame_preprocessing_test(samples_per_category: int = 4, num_frames: int = 10):
    base_dir = Path("Dataset/Video")
    print("=" * 80)
    print(f"FRAME PREPROCESSING TEST REPORT ({samples_per_category} SAMPLES PER CATEGORY, {num_frames} FRAMES PER VIDEO)")
    print("=" * 80)

    total_videos = 0
    total_frames_processed = 0
    successful_preprocessing = 0
    failed_preprocessing = 0

    sample_bgr_shape = None
    sample_gray_shape = None
    sample_hsv_shape = None

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

            # Phase 2: Sample 10 frames
            sampling_res = sample_video_frames(str(vpath), num_frames=num_frames)
            sampled_frames = sampling_res["sampled_frames"]
            orig_shape = sampling_res["frame_shape"]

            # Phase 3: Preprocess each frame
            processed_frames = preprocess_video_frames(sampled_frames, target_size=(256, 256))

            video_all_valid = True
            for pf in processed_frames:
                total_frames_processed += 1
                if pf["valid"]:
                    successful_preprocessing += 1
                    if sample_bgr_shape is None:
                        sample_bgr_shape = pf["shape_bgr"]
                        sample_gray_shape = pf["shape_gray"]
                        sample_hsv_shape = pf["shape_hsv"]
                else:
                    failed_preprocessing += 1
                    video_all_valid = False

            bgr_shape_str = str(processed_frames[0]["shape_bgr"]) if processed_frames else "N/A"
            gray_shape_str = str(processed_frames[0]["shape_gray"]) if processed_frames else "N/A"
            hsv_shape_str = str(processed_frames[0]["shape_hsv"]) if processed_frames else "N/A"

            print(f"\n--- Video {total_videos} [{folder_name}] ---")
            print(f"Video:               {vpath}")
            print(f"Label:               {label}")
            print(f"Original shape:      {orig_shape}")
            print(f"Processed BGR shape: {bgr_shape_str}")
            print(f"Processed Gray shape:{gray_shape_str}")
            print(f"Processed HSV shape: {hsv_shape_str}")
            print(f"Valid:               {'Yes' if video_all_valid else 'No'}")

    print("\n" + "=" * 80)
    print("FRAME PREPROCESSING SUMMARY:")
    print(f"  Total videos:             {total_videos}")
    print(f"  Total frames:             {total_frames_processed}")
    print(f"  Successful preprocessing: {successful_preprocessing}")
    print(f"  Failed preprocessing:     {failed_preprocessing}")
    print(f"  BGR shape:                {sample_bgr_shape}")
    print(f"  GRAY shape:               {sample_gray_shape}")
    print(f"  HSV shape:                {sample_hsv_shape}")
    print("=" * 80)

if __name__ == "__main__":
    run_frame_preprocessing_test(samples_per_category=4, num_frames=10)
