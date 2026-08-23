import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.ml_training import (
    load_and_split_data,
    prepare_scaled_features,
    train_and_evaluate_baselines,
)

def run_ml_training_validation():
    csv_path = "data/videos/features/video_features_stage_b.csv"

    print("=" * 80)
    print("PHASE 7 ML TRAINING VALIDATION SUITE")
    print("=" * 80)

    # 1. Load data & test source-aware split
    train_df, val_df, test_df, feature_names = load_and_split_data(csv_path=csv_path, random_state=42)

    print(f"Feature columns count:          {len(feature_names)} (Expected: 216)")
    print(f"Train split size:               {len(train_df)}")
    print(f"Validation split size:          {len(val_df)}")
    print(f"Test split size:                {len(test_df)}")

    # Check target values (only 0 and 1)
    all_targets = pd.concat([train_df["target"], val_df["target"], test_df["target"]])
    unique_targets = set(all_targets.unique())
    print(f"Unique Target Values:           {unique_targets} (Expected: {{0, 1}})")

    # Check source ID overlaps
    s_train = set(train_df["source_id"])
    s_val = set(val_df["source_id"])
    s_test = set(test_df["source_id"])

    overlap_train_val = len(s_train.intersection(s_val))
    overlap_train_test = len(s_train.intersection(s_test))
    overlap_val_test = len(s_val.intersection(s_test))

    print(f"Source ID Overlap (Train/Val):  {overlap_train_val}")
    print(f"Source ID Overlap (Train/Test): {overlap_train_test}")
    print(f"Source ID Overlap (Val/Test):   {overlap_val_test}")

    # 2. Test scaling & preprocessing leakage rules
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = prepare_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    nan_in_scaled = np.isnan(X_train_scaled).sum() + np.isnan(X_val_scaled).sum() + np.isnan(X_test_scaled).sum()
    inf_in_scaled = np.isinf(X_train_scaled).sum() + np.isinf(X_val_scaled).sum() + np.isinf(X_test_scaled).sum()

    print(f"NaN in Scaled Features:         {nan_in_scaled}")
    print(f"Inf in Scaled Features:         {inf_in_scaled}")

    # 3. Train models and generate results
    res = train_and_evaluate_baselines(csv_path=csv_path, random_state=42)

    val_df_results = res["val_results_df"]
    test_df_results = res["test_results_df"]

    # Verify metrics are finite
    metrics_finite = True
    for df_m in [val_df_results, test_df_results]:
        for col in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if col in df_m.columns:
                for v in df_m[col]:
                    if math.isnan(v) or math.isinf(v):
                        metrics_finite = False

    print(f"Finite Metric Check:            {'PASSED' if metrics_finite else 'FAILED'}")

    # Verify output files exist
    f_comp = Path("data/videos/results/model_comparison.csv")
    f_test = Path("data/videos/results/test_results.csv")
    f_fi = Path("data/videos/results/feature_importance.csv")

    files_exist = f_comp.exists() and f_test.exists() and f_fi.exists()
    print(f"Results CSV Files Created:      {'PASSED' if files_exist else 'FAILED'}")

    print("\n" + "=" * 80)
    all_passed = (
        len(feature_names) == 216 and
        unique_targets == {0, 1} and
        overlap_train_val == 0 and
        overlap_train_test == 0 and
        overlap_val_test == 0 and
        nan_in_scaled == 0 and
        inf_in_scaled == 0 and
        metrics_finite and
        files_exist
    )
    print(f"ML TRAINING VALIDATION SUITE:   {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_ml_training_validation()
