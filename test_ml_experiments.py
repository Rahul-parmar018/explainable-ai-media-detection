import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.ml_experiments import run_model_experiments, run_feature_ablation_experiment

def run_experiments_validation():
    print("=" * 80)
    print("PHASE 9 ML EXPERIMENTS VALIDATION SUITE")
    print("=" * 80)

    # 1. Run model hyperparameter search experiments
    res_exp = run_model_experiments(
        csv_path="data/videos/features/video_features_stage_b.csv",
        output_dir="data/videos/results",
        random_state=42
    )

    exp_df = res_exp["exp_df"]
    best_results_df = res_exp["best_results_df"]

    print(f"\nTotal Model Configs Evaluated: {len(exp_df)}")
    print(f"Best Config Selected:           {res_exp['best_config_name']}")

    # Check finite metrics
    exp_nan = exp_df["Val_Balanced_Accuracy"].isna().sum()
    print(f"NaN in Validation Metrics:       {exp_nan}")

    # Check output CSV files
    f_exp = Path("data/videos/results/ml_model_experiments.csv")
    f_best = Path("data/videos/results/best_model_results.csv")
    exp_files_exist = f_exp.exists() and f_best.exists()

    # 2. Run feature ablation experiments
    ablation_df = run_feature_ablation_experiment(
        csv_path="data/videos/features/video_features_stage_b.csv",
        output_dir="data/videos/results",
        random_state=42
    )

    f_abl = Path("data/videos/results/feature_ablation_results.csv")
    abl_files_exist = f_abl.exists()

    print(f"\nAblation Configurations Evaluated: {len(ablation_df)} (Expected: 5)")
    print(f"Model Experiments CSV Created:     {'PASSED' if exp_files_exist else 'FAILED'}")
    print(f"Feature Ablation CSV Created:      {'PASSED' if abl_files_exist else 'FAILED'}")

    print("\n" + "=" * 80)
    all_passed = (
        len(exp_df) >= 20 and
        exp_nan == 0 and
        exp_files_exist and
        len(ablation_df) == 5 and
        abl_files_exist
    )
    print(f"ML EXPERIMENTS VALIDATION:        {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiments_validation()
