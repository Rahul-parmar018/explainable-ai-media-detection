import os
import sys
import time
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

from src.video.batch_processor import process_single_video, extract_source_id

CATEGORIES_MAP = {
    "original": "REAL",
    "Deepfakes": "FAKE",
    "Face2Face": "FAKE",
    "FaceShifter": "FAKE",
    "FaceSwap": "FAKE",
    "NeuralTextures": "FAKE",
    "DeepFakeDetection": "FAKE",
}

def discover_all_videos(dataset_dir: str = "Dataset/Video") -> List[Tuple[Path, str, str]]:
    """
    Discovers all MP4 videos across the 7 dataset categories.

    Returns:
        List[Tuple[video_path, category, label]]
    """
    ds_path = Path(dataset_dir)
    discovered = []

    for cat, label in CATEGORIES_MAP.items():
        cat_dir = ds_path / cat
        if cat_dir.exists():
            vids = sorted(list(cat_dir.glob("*.mp4")))
            for v in vids:
                discovered.append((v, cat, label))

    return discovered


def load_existing_processed_paths(csv_path: Path) -> Tuple[Set[str], List[str]]:
    """
    Loads already-processed video paths from existing CSV to enable resumable processing.

    Returns:
        Tuple[set_of_processed_video_paths, header_fieldnames]
    """
    processed = set()
    header = []

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                for row in reader:
                    vp = row.get("video_path", "")
                    if vp:
                        processed.add(str(Path(vp).as_posix()))
        except Exception as e:
            print(f"Warning reading existing CSV {csv_path}: {e}")

    return processed, header


def run_full_batch_processing(
    dataset_dir: str = "Dataset/Video",
    output_dir: str = "data/videos/features",
    num_frames: int = 10,
    target_size: Tuple[int, int] = (256, 256),
    flush_interval: int = 10,
) -> Dict[str, Any]:
    """
    Executes resumable full batch processing across all 7,000 video files.
    """
    start_time = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features_csv_path = out_path / "video_features_full.csv"
    errors_csv_path = out_path / "full_video_processing_errors.csv"

    # 1. Discover all videos in dataset
    all_videos = discover_all_videos(dataset_dir)
    total_discovered = len(all_videos)

    print("=" * 80)
    print("PHASE 10A: FULL 7,000-VIDEO RESUMABLE BATCH FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Total Videos Discovered: {total_discovered}")

    # 2. Check resumability: load already-processed video paths
    already_processed, existing_header = load_existing_processed_paths(features_csv_path)
    initial_processed_count = len(already_processed)

    print(f"Already Processed Videos Found: {initial_processed_count} / {total_discovered}")

    videos_to_process = [
        (vpath, cat, lbl)
        for vpath, cat, lbl in all_videos
        if str(vpath.as_posix()) not in already_processed
    ]

    remaining_count = len(videos_to_process)
    print(f"Remaining Videos to Process:   {remaining_count}")
    print("=" * 80)

    if remaining_count == 0:
        print("All 7,000 videos have already been processed into CSV!")

    # Prepare file handles
    file_exists = features_csv_path.exists() and features_csv_path.stat().st_size > 0
    csv_file = open(features_csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8")
    writer = None

    err_file_exists = errors_csv_path.exists() and errors_csv_path.stat().st_size > 0
    err_file = open(errors_csv_path, mode="a" if err_file_exists else "w", newline="", encoding="utf-8")
    err_writer = csv.DictWriter(err_file, fieldnames=["video_path", "category", "label", "error_message"])

    if not err_file_exists:
        err_writer.writeheader()

    successful_count = initial_processed_count
    failed_count = 0
    newly_processed = 0
    run_start_time = time.time()

    try:
        for idx, (vpath, cat, lbl) in enumerate(videos_to_process, start=1):
            vpath_str = str(vpath.as_posix())
            sid = extract_source_id(vpath, cat)

            video_meta = {
                "video_path": vpath_str,
                "video_name": vpath.name,
                "category": cat,
                "label": lbl,
                "source_id": sid,
            }

            # Format progress & ETA
            elapsed_run = time.time() - run_start_time
            avg_per_video = elapsed_run / max(newly_processed, 1)
            eta_seconds = avg_per_video * (remaining_count - newly_processed)
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if newly_processed > 0 else "Calculating..."

            overall_idx = initial_processed_count + idx
            print(
                f"[{overall_idx:04d}/{total_discovered}] Processing [{cat:18s}]: {vpath.name:18s} "
                f"(Run Elapsed: {int(elapsed_run)}s, ETA: {eta_str})",
                end="\r",
                flush=True
            )

            try:
                row_dict, err_dict = process_single_video(
                    video_meta=video_meta,
                    num_frames=num_frames,
                    target_size=target_size
                )

                if row_dict is not None:
                    if writer is None:
                        fieldnames = list(row_dict.keys())
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()

                    writer.writerow(row_dict)
                    successful_count += 1
                    newly_processed += 1

                    if newly_processed % flush_interval == 0:
                        csv_file.flush()
                else:
                    failed_count += 1
                    err_msg = err_dict.get("error", "Unknown processing error") if err_dict else "Feature extraction returned None"
                    err_writer.writerow({
                        "video_path": vpath_str,
                        "category": cat,
                        "label": lbl,
                        "error_message": err_msg
                    })
                    err_file.flush()
            except Exception as ex:
                failed_count += 1
                err_writer.writerow({
                    "video_path": vpath_str,
                    "category": cat,
                    "label": lbl,
                    "error_message": str(ex)
                })
                err_file.flush()

    finally:
        csv_file.close()
        err_file.close()

    total_processing_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("FULL BATCH EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total Discovered:       {total_discovered}")
    print(f"Total Processed:        {successful_count}")
    print(f"Newly Processed in Run: {newly_processed}")
    print(f"Failed Count:           {failed_count}")
    print(f"Total Processing Time:  {total_processing_time:.2f} seconds ({total_processing_time/60:.2f} minutes)")
    print(f"Features CSV Path:      {features_csv_path}")
    print(f"Errors CSV Path:        {errors_csv_path}")
    print("=" * 80)

    return {
        "total_discovered": total_discovered,
        "successful_count": successful_count,
        "newly_processed": newly_processed,
        "failed_count": failed_count,
        "processing_time_seconds": round(total_processing_time, 2),
        "features_csv_path": str(features_csv_path),
        "errors_csv_path": str(errors_csv_path),
    }

if __name__ == "__main__":
    run_full_batch_processing()
