import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.video.ml_training import (
    load_and_split_data,
    prepare_scaled_features,
    evaluate_predictions,
    METADATA_COLUMNS,
)
from src.video.explainability import map_feature_to_family


def get_hyperparameter_search_space(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Defines a targeted, lightweight hyperparameter search grid across traditional ML model families.
    """
    configs = []

    # 1. Logistic Regression
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append({
            "model_family": "Logistic Regression",
            "model_name": f"Logistic Regression (C={c})",
            "model_obj": LogisticRegression(C=c, class_weight="balanced", random_state=random_state, max_iter=2000),
            "requires_scaling": True,
            "params": f"C={c}",
        })

    # 2. Support Vector Machine (SVM)
    for c in [0.1, 1.0, 10.0]:
        for kernel in ["linear", "rbf"]:
            configs.append({
                "model_family": "SVM",
                "model_name": f"SVM (kernel={kernel}, C={c})",
                "model_obj": SVC(C=c, kernel=kernel, class_weight="balanced", probability=True, random_state=random_state),
                "requires_scaling": True,
                "params": f"kernel={kernel}, C={c}",
            })

    # 3. Random Forest
    for n_est in [200, 500]:
        for depth in [None, 10, 20]:
            configs.append({
                "model_family": "Random Forest",
                "model_name": f"Random Forest (n_est={n_est}, max_depth={depth})",
                "model_obj": RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, class_weight="balanced", random_state=random_state, n_jobs=-1
                ),
                "requires_scaling": False,
                "params": f"n_estimators={n_est}, max_depth={depth}",
            })

    # 4. Extra Trees
    for n_est in [200, 500]:
        for depth in [None, 10, 20]:
            configs.append({
                "model_family": "Extra Trees",
                "model_name": f"Extra Trees (n_est={n_est}, max_depth={depth})",
                "model_obj": ExtraTreesClassifier(
                    n_estimators=n_est, max_depth=depth, class_weight="balanced", random_state=random_state, n_jobs=-1
                ),
                "requires_scaling": False,
                "params": f"n_estimators={n_est}, max_depth={depth}",
            })

    # 5. HistGradientBoosting
    for lr in [0.03, 0.1]:
        for max_i in [100, 200]:
            configs.append({
                "model_family": "HistGradientBoosting",
                "model_name": f"HistGradientBoosting (lr={lr}, max_iter={max_i})",
                "model_obj": HistGradientBoostingClassifier(
                    learning_rate=lr, max_iter=max_i, class_weight="balanced", random_state=random_state
                ),
                "requires_scaling": False,
                "params": f"learning_rate={lr}, max_iter={max_i}",
            })

    return configs


def run_model_experiments(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Phase 9 model hyperparameter experiments across 5 traditional ML model families.
    Selects best model based on Validation Balanced Accuracy and evaluates ONCE on held-out test set.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & execute source-aware split
    train_df, val_df, test_df, feature_names = load_and_split_data(csv_path=csv_path, random_state=random_state)

    # 2. Fit StandardScaler ONLY on train set for models requiring scaling
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = prepare_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    X_train_raw = train_df[feature_names].values.astype(np.float64)
    X_val_raw = val_df[feature_names].values.astype(np.float64)
    X_test_raw = test_df[feature_names].values.astype(np.float64)

    search_space = get_hyperparameter_search_space(random_state=random_state)
    exp_rows = []
    trained_models = {}

    print("=" * 80)
    print("PHASE 9: TRADITIONAL ML MODEL IMPROVEMENT EXPERIMENTS")
    print("=" * 80)
    print(f"Total Model Configurations to Test: {len(search_space)}")
    print(f"Train Set: {len(train_df)} | Val Set: {len(val_df)} | Test Set: {len(test_df)}")
    print("=" * 80)

    for cfg in search_space:
        model = cfg["model_obj"]
        name = cfg["model_name"]
        use_scaled = cfg["requires_scaling"]

        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_va = X_val_scaled if use_scaled else X_val_raw

        model.fit(X_tr, y_train)

        y_val_pred = model.predict(X_va)
        try:
            y_val_proba = model.predict_proba(X_va)
        except Exception:
            y_val_proba = np.column_stack([1 - y_val_pred, y_val_pred])

        metrics_val = evaluate_predictions(y_val, y_val_pred, y_val_proba)

        trained_models[name] = {
            "model": model,
            "requires_scaling": use_scaled,
            "metrics_val": metrics_val,
            "config": cfg,
        }

        exp_rows.append({
            "Model": cfg["model_family"],
            "Configuration": name,
            "Hyperparameters": cfg["params"],
            "Requires_Scaling": use_scaled,
            "Val_Accuracy": metrics_val["Accuracy"],
            "Val_Balanced_Accuracy": metrics_val["Balanced Accuracy"],
            "Val_Precision": metrics_val["Precision"],
            "Val_Recall": metrics_val["Recall"],
            "Val_F1": metrics_val["F1"],
            "Val_ROC_AUC": metrics_val["ROC-AUC"],
        })

    exp_df = pd.DataFrame(exp_rows).sort_values(
        by=["Val_Balanced_Accuracy", "Val_F1", "Val_ROC_AUC"], ascending=False
    ).reset_index(drop=True)

    exp_csv_path = out_path / "ml_model_experiments.csv"
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"Experiment results saved to: {exp_csv_path}")

    # 3. Select best model based strictly on Validation Balanced Accuracy
    best_config_name = str(exp_df.iloc[0]["Configuration"])
    best_item = trained_models[best_config_name]
    best_model = best_item["model"]
    use_scaled_best = best_item["requires_scaling"]

    print(f"\nBEST MODEL SELECTED (by Validation Balanced Accuracy): {best_config_name}")

    # 4. Evaluate Selected Best Model ONCE on Held-out Test Set
    X_te = X_test_scaled if use_scaled_best else X_test_raw
    y_test_pred = best_model.predict(X_te)
    try:
        y_test_proba = best_model.predict_proba(X_te)
    except Exception:
        y_test_proba = np.column_stack([1 - y_test_pred, y_test_pred])

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    best_results_df = pd.DataFrame([{
        "Selected_Model": best_config_name,
        "Model_Family": best_item["config"]["model_family"],
        "Hyperparameters": best_item["config"]["params"],
        "Val_Balanced_Accuracy": best_item["metrics_val"]["Balanced Accuracy"],
        "Val_F1": best_item["metrics_val"]["F1"],
        "Val_ROC_AUC": best_item["metrics_val"]["ROC-AUC"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Balanced_Accuracy": test_metrics["Balanced Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
        "Test_ROC_AUC": test_metrics["ROC-AUC"],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }])

    best_csv_path = out_path / "best_model_results.csv"
    best_results_df.to_csv(best_csv_path, index=False)
    print(f"Best model evaluation saved to: {best_csv_path}")

    return {
        "exp_df": exp_df,
        "best_config_name": best_config_name,
        "best_model": best_model,
        "use_scaled_best": use_scaled_best,
        "best_results_df": best_results_df,
        "feature_names": feature_names,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def run_feature_ablation_experiment(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Executes Feature Ablation Experiment comparing 5 feature subset configurations
    to determine which traditional visual feature families contribute to detection.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, all_feature_names = load_and_split_data(csv_path=csv_path, random_state=random_state)

    # Define Feature Sets
    color_feats = [f for f in all_feature_names if map_feature_to_family(f) == "Color (HSV)"]
    lbp_feats = [f for f in all_feature_names if map_feature_to_family(f) == "Texture (LBP & Gray)"]
    glcm_feats = [f for f in all_feature_names if map_feature_to_family(f) == "Texture (GLCM)"]
    dct_feats = [f for f in all_feature_names if map_feature_to_family(f) == "Frequency (DCT)"]
    edge_feats = [f for f in all_feature_names if map_feature_to_family(f) == "Edge (Canny)"]

    feature_subsets = {
        "Set A: All 216 Features": all_feature_names,
        "Set B: Remove Color Features": [f for f in all_feature_names if f not in color_feats],
        "Set C: Remove LBP/Gray Features": [f for f in all_feature_names if f not in lbp_feats],
        "Set D: Color + LBP/Gray Only": color_feats + lbp_feats,
        "Set E: GLCM + DCT + Edge Only": glcm_feats + dct_feats + edge_feats,
    }

    ablation_rows = []

    print("\n" + "=" * 80)
    print("PHASE 9: FEATURE ABLATION EXPERIMENTS")
    print("=" * 80)

    for set_name, subset_cols in feature_subsets.items():
        # Prepare subset data
        X_train_sub = train_df[subset_cols].values.astype(np.float64)
        y_train = train_df["target"].values.astype(int)

        X_val_sub = val_df[subset_cols].values.astype(np.float64)
        y_val = val_df["target"].values.astype(int)

        X_test_sub = test_df[subset_cols].values.astype(np.float64)
        y_test = test_df["target"].values.astype(int)

        # Fit scaler ONLY on train subset
        scaler_sub = StandardScaler()
        X_train_sub_sc = scaler_sub.fit_transform(X_train_sub)
        X_val_sub_sc = scaler_sub.transform(X_val_sub)
        X_test_sub_sc = scaler_sub.transform(X_test_sub)

        # Evaluate using baseline Logistic Regression model
        model_sub = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=2000)
        model_sub.fit(X_train_sub_sc, y_train)

        # Validation evaluation
        y_val_pred = model_sub.predict(X_val_sub_sc)
        y_val_proba = model_sub.predict_proba(X_val_sub_sc)
        val_m = evaluate_predictions(y_val, y_val_pred, y_val_proba)

        # Test evaluation
        y_test_pred = model_sub.predict(X_test_sub_sc)
        y_test_proba = model_sub.predict_proba(X_test_sub_sc)
        test_m = evaluate_predictions(y_test, y_test_pred, y_test_proba)

        ablation_rows.append({
            "Feature_Subset_Configuration": set_name,
            "Feature_Count": len(subset_cols),
            "Val_Accuracy": val_m["Accuracy"],
            "Val_Balanced_Accuracy": val_m["Balanced Accuracy"],
            "Val_Precision": val_m["Precision"],
            "Val_Recall": val_m["Recall"],
            "Val_F1": val_m["F1"],
            "Val_ROC_AUC": val_m["ROC-AUC"],
            "Test_Balanced_Accuracy": test_m["Balanced Accuracy"],
            "Test_F1": test_m["F1"],
            "Test_ROC_AUC": test_m["ROC-AUC"],
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = out_path / "feature_ablation_results.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)

    print(ablation_df.to_string(index=False))
    print(f"\nAblation results saved to: {ablation_csv_path}")
    print("=" * 80)

    return ablation_df

if __name__ == "__main__":
    res_exp = run_model_experiments()
    res_abl = run_feature_ablation_experiment()
