import os
import sys
import csv
import math
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

def run_batch_processor_validation():
    features_csv = Path("data/videos/features/video_features_stage_b.csv")
    errors_csv = Path("data/videos/features/video_processing_errors.csv")

    print("=" * 80)
    print("STAGE B BATCH PROCESSOR VALIDATION REPORT")
    print("=" * 80)

    if not features_csv.exists():
        print(f"Error: Features CSV {features_csv} does not exist!")
        return

    with open(features_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    total_rows = len(rows)
    print(f"Total Rows in CSV:             {total_rows}")
    print(f"Total Columns in CSV:          {len(fieldnames)}")

    metadata_cols = ["video_path", "category", "label", "source_id", "frame_count", "fps", "width", "height"]
    feature_cols = [c for c in fieldnames if c not in metadata_cols]

    print(f"Metadata Columns Count:       {len(metadata_cols)}")
    print(f"Feature Columns Count:        {len(feature_cols)} (Expected: 216)")

    # Category counts
    category_counts = {}
    label_counts = {}
    video_paths = set()
    duplicate_paths = 0
    nan_count = 0
    inf_count = 0

    for r in rows:
        cat = r["category"]
        lbl = r["label"]
        vpath = r["video_path"]

        category_counts[cat] = category_counts.get(cat, 0) + 1
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if vpath in video_paths:
            duplicate_paths += 1
        else:
            video_paths.add(vpath)

        for fc in feature_cols:
            val_str = r[fc]
            try:
                val = float(val_str)
                if math.isnan(val):
                    nan_count += 1
                if math.isinf(val):
                    inf_count += 1
            except ValueError:
                nan_count += 1

    print("\n--- CATEGORY DISTRIBUTION ---")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:22s}: {count}")

    print("\n--- BINARY LABEL DISTRIBUTION ---")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl:10s}: {count}")

    print("\n--- INTEGRITY CHECKS ---")
    print(f"  Duplicate Video Paths:      {duplicate_paths}")
    print(f"  NaN Values Count:           {nan_count}")
    print(f"  Inf Values Count:           {inf_count}")

    # Check errors CSV
    error_count = 0
    if errors_csv.exists():
        with open(errors_csv, mode="r", encoding="utf-8") as f:
            err_reader = csv.DictReader(f)
            error_count = len(list(err_reader))
    print(f"  Processing Failures Logged: {error_count}")

    print("\n" + "=" * 80)
    validation_passed = (
        total_rows == 700 and
        len(feature_cols) == 216 and
        duplicate_paths == 0 and
        nan_count == 0 and
        inf_count == 0 and
        error_count == 0 and
        label_counts.get("REAL", 0) == 100 and
        label_counts.get("FAKE", 0) == 600
    )
    print(f"BATCH PROCESSOR VALIDATION: {'PASSED' if validation_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_batch_processor_validation()
