"""
src/video/face_detection_audit.py — Phase 13A Face Detection Quality Audit.

Samples 10 videos from each category in the Stage B 700-video dataset
and measures per-category face detection rates before committing to
full feature extraction.
"""

import csv
import time
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.video.batch_processor import select_stage_b_videos
from src.video.frame_extraction import sample_video_frames
from src.video.face_detection import detect_and_crop_face, get_backend

OUTPUT_CSV = "data/videos/results/face_detection_audit.csv"
NUM_AUDIT_PER_CATEGORY = 10
NUM_FRAMES = 10
RANDOM_STATE = 42


def run_face_detection_audit(
    dataset_dir: str = "Dataset/Video",
    output_csv: str = OUTPUT_CSV,
    n_per_category: int = NUM_AUDIT_PER_CATEGORY,
    num_frames: int = NUM_FRAMES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Audit face detection quality on a sample of videos per category.

    Returns a DataFrame with per-video detection results.
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Get Stage B videos and sample n per category
    all_videos = select_stage_b_videos(dataset_dir=dataset_dir)
    by_category: Dict[str, List] = {}
    for v in all_videos:
        cat = v["category"]
        by_category.setdefault(cat, []).append(v)

    rng = random.Random(random_state)
    sample: List[Dict] = []
    for cat, vids in sorted(by_category.items()):
        n = min(n_per_category, len(vids))
        sample.extend(rng.sample(vids, n))

    print("=" * 70)
    print(f"PHASE 13A: FACE DETECTION AUDIT  ({len(sample)} videos, {num_frames} frames each)")
    print(f"Backend: {get_backend() or 'initialising...'}")
    print("=" * 70)

    rows = []
    for idx, vmeta in enumerate(sample, 1):
        vpath = vmeta["video_path"]
        cat   = vmeta["category"]
        lbl   = vmeta["label"]

        result = sample_video_frames(vpath, num_frames=num_frames)
        if not result["readable"] or not result["sampled_frames"]:
            rows.append({
                "video_path": vpath, "category": cat, "label": lbl,
                "frames_sampled": 0, "faces_detected": 0,
                "detection_rate": 0.0, "avg_face_area_ratio": 0.0,
                "backend": "N/A", "readable": False,
            })
            continue

        frames = result["sampled_frames"]
        n_sam  = len(frames)
        n_det  = 0
        area_ratios = []
        backend_used = "N/A"

        for frame in frames:
            fr = detect_and_crop_face(frame)
            backend_used = fr["backend"]
            if fr["detected"]:
                n_det += 1
                area_ratios.append(fr["face_area_ratio"])

        det_rate = n_det / n_sam if n_sam > 0 else 0.0
        avg_area = float(np.mean(area_ratios)) if area_ratios else 0.0

        rows.append({
            "video_path": vpath, "category": cat, "label": lbl,
            "frames_sampled": n_sam, "faces_detected": n_det,
            "detection_rate": round(det_rate, 4),
            "avg_face_area_ratio": round(avg_area, 4),
            "backend": backend_used, "readable": True,
        })

        print(
            f"  [{idx:02d}/{len(sample)}] {cat:18s} | {Path(vpath).name:20s} "
            f"| det={n_det}/{n_sam} ({100*det_rate:.0f}%) | area={avg_area:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n=== CATEGORY-LEVEL DETECTION RATES ===")
    agg = df.groupby("category").agg(
        n_videos=("video_path", "count"),
        avg_detection_rate=("detection_rate", "mean"),
        avg_face_area=("avg_face_area_ratio", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))
    overall = df["detection_rate"].mean()
    print(f"\nOverall average detection rate: {overall:.4f}")
    print(f"Audit saved to: {output_csv}")

    if overall < 0.50:
        print("\n[WARNING] Overall face detection rate is below 50%. Consider adjusting detection parameters.")
    else:
        print("\n[OK] Face detection rate is acceptable. Proceeding to full feature extraction.")

    return df


if __name__ == "__main__":
    run_face_detection_audit()
