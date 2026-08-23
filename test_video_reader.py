import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.dataset_utils import get_sample_videos_info, CATEGORY_LABEL_MAP

def run_test(samples_per_category: int = 4):
    print("=" * 80)
    print(f"OPENCV VIDEO READER TEST REPORT ({samples_per_category} SAMPLES PER CATEGORY)")
    print("=" * 80)

    results = get_sample_videos_info(dataset_dir="Dataset/Video", samples_per_folder=samples_per_category)

    total_tested = len(results)
    readable_count = 0

    for idx, info in enumerate(results, 1):
        print(f"\n--- Sample {idx}/{total_tested} [{info['folder']}] ---")
        print(f"Video:      {info['video_path']}")
        print(f"Label:      {info['label']}")
        print(f"FPS:        {info['fps']}")
        print(f"Frame Count:{info['frame_count']}")
        print(f"Width:      {info['width']}")
        print(f"Height:     {info['height']}")
        print(f"Duration:   {info['duration']}s")
        print(f"Readable:   {'Yes' if info['readable'] else 'No'}")

        if info['readable']:
            readable_count += 1

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {readable_count}/{total_tested} sample videos successfully opened and read.")
    print("=" * 80)

if __name__ == "__main__":
    run_test(samples_per_category=4)
