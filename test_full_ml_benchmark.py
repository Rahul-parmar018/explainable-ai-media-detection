import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.full_ml_benchmark import (
    load_and_split_connected_components,
    prepare_benchmark_scaled_features,
    run_full_ml_benchmark,
)

def run_test_full_ml_benchmark():
    print("=" * 80)
    print("PHASE 10C FULL DATASET ML BENCHMARK VALIDATION SUITE")
    print("=" * 80)

    # 1. Test dataset loading and connected-component group splitting
    csv_target = "data/videos/features/video_features_stage_b.csv"
    f_full = Path("data/videos/features/video_features_full.csv")
    if f_full.exists() and f_full.stat().st_size > 0:
        with open(f_full, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f) - 1
            if line_count >= 7000:
                csv_target = str(f_full)

    train_df, val_df, test_df, feature_names = load_and_split_connected_components(
        csv_path=csv_target, random_state=42
    )

    print(f"Dataset CSV Evaluated:           {csv_target}")
    print(f"Feature Columns Verified:        {len(feature_names)} (Expected: 216)")
    print(f"Train Partition Rows:            {len(train_df)}")
    print(f"Validation Partition Rows:       {len(val_df)}")
    print(f"Test Partition Rows:             {len(test_df)}")

    # 2. Check 0 connected component group overlap
    s_train = set(train_df["connected_component_id"])
    s_val = set(val_df["connected_component_id"])
    s_test = set(test_df["connected_component_id"])

    overlap_tr_val = len(s_train.intersection(s_val))
    overlap_tr_test = len(s_train.intersection(s_test))
    overlap_val_test = len(s_val.intersection(s_test))

    print(f"Group Overlap (Train/Val):       {overlap_tr_val}")
    print(f"Group Overlap (Train/Test):      {overlap_tr_test}")
    print(f"Group Overlap (Val/Test):        {overlap_val_test}")

    # 3. Check feature scaling fit safety (fitted ONLY on train)
    X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, scaler = prepare_benchmark_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    nan_scaled = np.isnan(X_tr_sc).sum() + np.isnan(X_val_sc).sum() + np.isnan(X_te_sc).sum()
    inf_scaled = np.isinf(X_tr_sc).sum() + np.isinf(X_val_sc).sum() + np.isinf(X_te_sc).sum()

    print(f"NaN in Scaled Features:          {nan_scaled}")
    print(f"Inf in Scaled Features:          {inf_scaled}")

    # 4. Run full benchmark suite
    res = run_full_ml_benchmark(csv_path=csv_target, output_dir="data/videos/results", models_dir="data/videos/models", random_state=42)

    val_comp_df = res["val_comp_df"]
    test_results_df = res["test_results_df"]

    print(f"\nModels Evaluated in Benchmark:  {len(val_comp_df)}")
    print(f"Best Model Selected:             {res['best_model_name']}")

    # Check persistence of model and scaler
    f_model = Path(res["model_save_path"])
    f_scaler = Path(res["scaler_save_path"])
    persisted = f_model.exists() and f_scaler.exists()
    print(f"Model and Scaler Saved:          {'PASSED' if persisted else 'FAILED'}")

    all_passed = (
        len(feature_names) == 216 and
        overlap_tr_val == 0 and
        overlap_tr_test == 0 and
        overlap_val_test == 0 and
        nan_scaled == 0 and
        inf_scaled == 0 and
        len(val_comp_df) >= 6 and
        persisted
    )

    print("\n" + "=" * 80)
    print(f"TEST FULL ML BENCHMARK RESULT:  {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_test_full_ml_benchmark()
