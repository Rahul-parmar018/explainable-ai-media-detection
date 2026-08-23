import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.dataset_audit import audit_dataset_integrity

def run_test_dataset_audit():
    print("=" * 80)
    print("RUNNING DATASET INTEGRITY & LEAKAGE AUDIT TEST SUITE")
    print("=" * 80)

    # Prefer completed full CSV if 7000 rows, else use verified stage B CSV
    csv_target = "data/videos/features/video_features_stage_b.csv"
    f_full = Path("data/videos/features/video_features_full.csv")
    if f_full.exists() and f_full.stat().st_size > 0:
        with open(f_full, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f) - 1
            if line_count >= 7000:
                csv_target = str(f_full)

    res = audit_dataset_integrity(csv_path=csv_target, output_dir="data/videos/results")

    print(f"\nAudited CSV Path:               {res['csv_path']}")
    print(f"Total Audited Rows:             {res['total_rows']}")
    print(f"Feature Count Verified:         {res['feature_count']} (Expected: 216)")
    print(f"Duplicate Video Paths:          {res['duplicate_paths']}")
    print(f"Duplicate Feature Vectors:      {res['duplicate_rows']}")
    print(f"NaN Values Count:               {res['nan_count']}")
    print(f"Inf Values Count:               {res['inf_count']}")

    summary_file = Path(res["summary_csv"])
    summary_exists = summary_file.exists() and summary_file.stat().st_size > 0
    print(f"Source Group Summary CSV:       {'PASSED' if summary_exists else 'FAILED'}")

    all_passed = (
        res["audit_passed"] and
        res["feature_count"] == 216 and
        res["duplicate_paths"] == 0 and
        res["duplicate_rows"] == 0 and
        res["nan_count"] == 0 and
        res["inf_count"] == 0 and
        summary_exists
    )

    print("\n" + "=" * 80)
    print(f"TEST DATASET AUDIT RESULT:      {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_test_dataset_audit()
