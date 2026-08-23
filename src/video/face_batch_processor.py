"""
src/video/face_batch_processor.py — Phase 13A Face-Region Batch Processor.

Processes the same Stage B 700-video dataset using face-region features.
Reuses existing labels, source IDs, and connected-component metadata exactly.
Produces data/videos/features/video_face_features.csv.

Key design:
- Uses select_stage_b_videos() from batch_processor to get the same 700 videos
- Per-video: sample 10 frames, detect face, extract 72 face features
- Aggregate: mean, std, min, max → 288 video-level face features
- Minimum 3 frames must have detected faces (otherwise mark as failed)
- Resumable: skips already-written video_paths on re-run
- Preserves connected_component_id for leakage-safe splitting
"""

import csv
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

from src.video.batch_processor import select_stage_b_videos, extract_source_id
from src.video.frame_extraction import sample_video_frames
from src.video.face_detection import detect_and_crop_face
from src.video.face_feature_extraction import extract_face_features
from src.video.dataset_audit import build_connected_component_groups

# Minimum fraction of frames that must have a detected face
MIN_FACE_DETECTION_RATE: float = 0.30  # at least 3 out of 10 frames


# ──────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_face_features(face_feat_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate per-frame face features across all detected-face frames.
    Aggregations: mean, std, min, max → 4 × N_frame_features.
    """
    if not face_feat_list:
        return {}

    feature_names = list(face_feat_list[0].keys())
    agg: Dict[str, float] = {}

    for name in feature_names:
        vals = [fdict[name] for fdict in face_feat_list if name in fdict]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[f"{name}_mean"] = float(np.mean(arr))
        agg[f"{name}_std"]  = float(np.std(arr))
        agg[f"{name}_min"]  = float(np.min(arr))
        agg[f"{name}_max"]  = float(np.max(arr))

    return agg


def process_single_video_face(
    video_meta: Dict[str, str],
    num_frames: int = 10,
    min_detection_rate: float = MIN_FACE_DETECTION_RATE,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process one video through face detection and face-region feature extraction.

    Returns:
        (row_dict, None) on success
        (None, error_dict) on failure
    """
    vpath = video_meta["video_path"]

    try:
        result = sample_video_frames(vpath, num_frames=num_frames)
        if not result["readable"] or not result["sampled_frames"]:
            return None, {**video_meta, "error": result.get("error", "No frames sampled")}

        frames = result["sampled_frames"]
        total_frames_sampled = len(frames)

        face_feats_list: List[Dict[str, float]] = []
        detected_count = 0

        for frame in frames:
            face_result = detect_and_crop_face(frame)
            if face_result["detected"]:
                feats = extract_face_features(face_result)
                if feats:
                    face_feats_list.append(feats)
                    detected_count += 1

        detection_rate = detected_count / total_frames_sampled if total_frames_sampled > 0 else 0.0

        if detection_rate < min_detection_rate:
            return None, {
                **video_meta,
                "error": f"Face detection rate too low: {detection_rate:.2f} ({detected_count}/{total_frames_sampled})"
            }

        agg = aggregate_face_features(face_feats_list)
        if not agg:
            return None, {**video_meta, "error": "Feature aggregation returned empty"}

        row = {
            "video_path":              video_meta["video_path"],
            "video_name":              video_meta.get("video_name", Path(vpath).name),
            "category":                video_meta["category"],
            "label":                   video_meta["label"],
            "source_id":               video_meta["source_id"],
            "frames_sampled":          total_frames_sampled,
            "faces_detected":          detected_count,
            "face_detection_rate":     round(detection_rate, 4),
        }
        row.update(agg)
        return row, None

    except Exception as ex:
        return None, {**video_meta, "error": str(ex)}


# ──────────────────────────────────────────────────────────────────────────────
# Resumable batch processing
# ──────────────────────────────────────────────────────────────────────────────

def _load_existing_paths(csv_path: Path) -> Set[str]:
    """Return the set of video_paths already written to the CSV."""
    processed: Set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return processed
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = row.get("video_path", "")
                if vp:
                    processed.add(str(Path(vp).as_posix()))
    except Exception as e:
        print(f"Warning reading {csv_path}: {e}")
    return processed


def run_face_batch_processing(
    dataset_dir: str = "Dataset/Video",
    output_dir: str = "data/videos/features",
    num_frames: int = 10,
    min_detection_rate: float = MIN_FACE_DETECTION_RATE,
    flush_interval: int = 10,
) -> Dict[str, Any]:
    """
    Resumable batch processor for face-region features on the Stage B 700-video dataset.

    Outputs:
        data/videos/features/video_face_features.csv
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    features_csv = out_path / "video_face_features.csv"
    errors_csv   = out_path / "video_face_errors.csv"

    # ── 1. Select the same 700 Stage B videos ────────────────────────────────
    selected = select_stage_b_videos(dataset_dir=dataset_dir)
    total = len(selected)
    print("=" * 80)
    print("PHASE 13A: FACE-REGION BATCH FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Total Stage B videos: {total}")

    # ── 2. Resumability ───────────────────────────────────────────────────────
    already_done = _load_existing_paths(features_csv)
    print(f"Already processed: {len(already_done)}")

    to_process = [
        v for v in selected
        if str(Path(v["video_path"]).as_posix()) not in already_done
    ]
    print(f"Remaining: {len(to_process)}")
    print("=" * 80)

    if not to_process:
        print("All videos already processed.")
        df = pd.read_csv(features_csv)
        return {"status": "already_done", "csv": str(features_csv), "rows": len(df)}

    # ── 3. Run ────────────────────────────────────────────────────────────────
    file_exists = features_csv.exists() and features_csv.stat().st_size > 0
    csv_fh = open(features_csv, "a" if file_exists else "w", newline="", encoding="utf-8")
    err_fh = open(errors_csv,   "a", newline="", encoding="utf-8")
    err_writer = None
    writer = None

    successful = len(already_done)
    failed = 0
    newly_done = 0
    start_t = time.time()

    try:
        for idx, vmeta in enumerate(to_process, 1):
            elapsed = time.time() - start_t
            eta = (elapsed / max(newly_done, 1)) * (len(to_process) - newly_done) if newly_done > 0 else 0
            print(
                f"[{successful+1:03d}/{total}] {vmeta['category']:18s} | {vmeta['video_name']:20s} "
                f"| ETA: {int(eta//60)}m{int(eta%60)}s",
                end="\r", flush=True
            )

            row, err = process_single_video_face(vmeta, num_frames=num_frames, min_detection_rate=min_detection_rate)

            if row is not None:
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                writer.writerow(row)
                successful += 1
                newly_done += 1
                if newly_done % flush_interval == 0:
                    csv_fh.flush()
            else:
                if err_writer is None:
                    err_writer = csv.DictWriter(err_fh, fieldnames=list(err.keys()))
                    err_fh.seek(0, 2)
                    if err_fh.tell() == 0:
                        err_writer.writeheader()
                err_writer.writerow(err)
                err_fh.flush()
                failed += 1

    finally:
        csv_fh.close()
        err_fh.close()

    elapsed_total = time.time() - start_t
    print()
    print("=" * 80)
    print(f"Successful: {successful}  |  Failed: {failed}  |  Time: {elapsed_total:.1f}s")
    print(f"Output: {features_csv}")
    print("=" * 80)

    return {
        "total": total, "successful": successful, "failed": failed,
        "csv": str(features_csv), "errors_csv": str(errors_csv),
        "time_seconds": round(elapsed_total, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing: attach connected_component_id + target
# ──────────────────────────────────────────────────────────────────────────────

def attach_connected_components(csv_path: str = "data/videos/features/video_face_features.csv") -> pd.DataFrame:
    """
    Load face features CSV, compute connected_component_id, and attach target column.
    Returns the enriched DataFrame (does NOT overwrite the CSV).
    """
    df = pd.read_csv(csv_path)
    df["target"] = df["label"].map({"REAL": 0, "FAKE": 1})
    comp_map = build_connected_component_groups(df)
    df["connected_component_id"] = df["video_path"].map(comp_map)
    return df


if __name__ == "__main__":
    run_face_batch_processing()
