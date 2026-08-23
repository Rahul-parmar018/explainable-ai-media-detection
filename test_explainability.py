import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.explainability import compute_shap_explanations, explain_video

def run_explainability_validation():
    print("=" * 80)
    print("PHASE 8 SHAP EXPLAINABILITY VALIDATION SUITE")
    print("=" * 80)

    res = compute_shap_explanations(
        csv_path="data/videos/features/video_features_stage_b.csv",
        output_dir="data/videos/results",
        random_state=42
    )

    global_df = res["global_df"]
    family_summary = res["family_summary_df"]
    local_df = res["local_df"]

    print(f"\nGlobal Features Count:          {len(global_df)} (Expected: 216)")
    print(f"Test Videos Explained:           {res['test_sample_count']}")

    # 1. NaN and Inf checks
    nan_global = global_df["mean_abs_shap"].isna().sum()
    inf_global = np.isinf(global_df["mean_abs_shap"]).sum()

    print(f"NaN in Global SHAP:              {nan_global}")
    print(f"Inf in Global SHAP:              {inf_global}")

    # 2. Check feature family percentage sum
    perc_sum = family_summary["percentage_contribution"].sum()
    print(f"Feature Family Percentage Sum:   {perc_sum:.2f}% (Expected: 100.0%)")

    # 3. Test local explain_video function
    dummy_video_feats = {f: 1.0 for f in res["feature_names"]}
    single_exp = explain_video(
        dummy_video_feats,
        res["model"],
        res["scaler"],
        res["explainer"],
        res["feature_names"],
        top_k=5
    )

    print("\n--- SINGLE VIDEO EXPLANATION FUNCTION OUTPUT SAMPLE ---")
    print(f"  Prediction:              {single_exp['prediction']}")
    print(f"  Probability:             {single_exp['probability']}")
    print(f"  Top FAKE Evidence Count: {len(single_exp['top_fake_evidence'])}")
    print(f"  Top REAL Evidence Count: {len(single_exp['top_real_evidence'])}")

    # Check output files exist
    f_global = Path("data/videos/results/shap_global_importance.csv")
    f_fam = Path("data/videos/results/shap_family_importance.csv")
    f_local = Path("data/videos/results/shap_local_explanations.csv")

    files_exist = f_global.exists() and f_fam.exists() and f_local.exists()
    print(f"Explainability CSV Files Created: {'PASSED' if files_exist else 'FAILED'}")

    print("\n" + "=" * 80)
    all_passed = (
        len(global_df) == 216 and
        nan_global == 0 and
        inf_global == 0 and
        abs(perc_sum - 100.0) < 0.1 and
        files_exist and
        len(single_exp["top_fake_evidence"]) > 0
    )
    print(f"SHAP EXPLAINABILITY VALIDATION:  {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_explainability_validation()
