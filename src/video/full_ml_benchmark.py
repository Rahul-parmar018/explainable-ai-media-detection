import os
import sys
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
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
    average_precision_score,
    confusion_matrix,
)

from src.video.ml_training import METADATA_COLUMNS
from src.video.dataset_audit import build_connected_component_groups


def load_and_split_connected_components(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    test_size: float = 0.15,
    val_size_relative: float = 0.1765,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads dataset CSV, computes leak-free connected component actor groups,
    maps target labels (REAL=0, FAKE=1), and executes GroupShuffleSplit.

    Returns:
        Tuple[train_df, val_df, test_df, feature_names]
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in target dataset.")

    df["target"] = df["label"].map({"REAL": 0, "FAKE": 1})

    feature_names = [c for c in df.columns if c not in METADATA_COLUMNS and c != "target" and c != "connected_component_id"]

    # Compute connected component actor groups to eliminate 100% subject leakage
    video_comp_map = build_connected_component_groups(df)
    df["connected_component_id"] = df["video_path"].map(video_comp_map)
    groups = df["connected_component_id"]

    # Step 1: Split 85% Train/Val and 15% Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Step 2: Split Train/Val into 70% Train and 15% Val
    groups_train_val = df_train_val["connected_component_id"]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=random_state)
    train_idx_sub, val_idx_sub = next(gss2.split(df_train_val, groups=groups_train_val))

    train_df = df_train_val.iloc[train_idx_sub].copy()
    val_df = df_train_val.iloc[val_idx_sub].copy()

    # Verify zero group overlap across partitions
    s_train = set(train_df["connected_component_id"])
    s_val = set(val_df["connected_component_id"])
    s_test = set(test_df["connected_component_id"])

    assert len(s_train.intersection(s_val)) == 0, "Train & Val group overlap detected!"
    assert len(s_train.intersection(s_test)) == 0, "Train & Test group overlap detected!"
    assert len(s_val.intersection(s_test)) == 0, "Val & Test group overlap detected!"

    return train_df, val_df, test_df, feature_names


def prepare_benchmark_scaled_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler EXCLUSIVELY on training data features and transforms Train, Val, and Test.
    """
    X_train = train_df[feature_names].values.astype(np.float64)
    y_train = train_df["target"].values.astype(int)

    X_val = val_df[feature_names].values.astype(np.float64)
    y_val = val_df["target"].values.astype(int)

    X_test = test_df[feature_names].values.astype(np.float64)
    y_test = test_df["target"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


def evaluate_benchmark_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Computes benchmark classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_proba[:, 1])
    except Exception:
        pr_auc = 0.5

    return {
        "Accuracy": round(float(acc), 4),
        "Balanced Accuracy": round(float(bal_acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4),
    }


def run_full_ml_benchmark(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    models_dir: str = "data/videos/models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Phase 10C full dataset leakage-safe ML benchmark using connected-component grouping.
    Saves validation comparison, held-out test evaluation, confusion matrix CSVs, and persisted models.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    # 1. Perform leak-free connected-component grouping and split
    train_df, val_df, test_df, feature_names = load_and_split_connected_components(
        csv_path=csv_path, random_state=random_state
    )

    # 2. Fit StandardScaler ONLY on training partition
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = prepare_benchmark_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    # 3. Define candidate traditional ML benchmark models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=2000
        ),
        "Linear SVM": SVC(
            kernel="linear", class_weight="balanced", probability=True, random_state=random_state
        ),
        "RBF SVM": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        ),
    }

    val_rows = []
    trained_models = {}

    print("=" * 80)
    print("PHASE 10C: FULL DATASET LEAKAGE-SAFE ML BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Target Dataset CSV: {csv_path}")
    print(f"Train Set:      {len(train_df)} samples (Groups: {train_df['connected_component_id'].nunique()})")
    print(f"Validation Set: {len(val_df)} samples (Groups: {val_df['connected_component_id'].nunique()})")
    print(f"Test Set:       {len(test_df)} samples (Groups: {test_df['connected_component_id'].nunique()})")
    print("=" * 80)

    # 4. Train and evaluate models on Validation partition
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)

        metrics = evaluate_benchmark_predictions(y_val, y_val_pred, y_val_proba)
        metrics["Model"] = name
        val_rows.append(metrics)
        trained_models[name] = model

    val_comp_df = pd.DataFrame(val_rows)[
        ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ].sort_values(by=["Balanced Accuracy", "F1", "ROC-AUC"], ascending=False).reset_index(drop=True)

    comp_csv_path = out_path / "full_model_comparison.csv"
    val_comp_df.to_csv(comp_csv_path, index=False)

    print("\n--- VALIDATION BENCHMARK MODEL COMPARISON ---")
    print(val_comp_df.to_string(index=False))
    print(f"\nValidation comparison saved to: {comp_csv_path}")

    # 5. Select Best Model strictly based on Validation Balanced Accuracy
    best_model_name = str(val_comp_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]
    print(f"\nBEST MODEL SELECTED (by Validation Balanced Accuracy): {best_model_name}")

    # 6. Evaluate Selected Best Model ONCE on Held-out Test Set
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)

    test_metrics = evaluate_benchmark_predictions(y_test, y_test_pred, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    test_results_df = pd.DataFrame([{
        "Model": best_model_name,
        "Accuracy": test_metrics["Accuracy"],
        "Balanced Accuracy": test_metrics["Balanced Accuracy"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "F1": test_metrics["F1"],
        "ROC-AUC": test_metrics["ROC-AUC"],
        "PR-AUC": test_metrics["PR-AUC"],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }])

    test_csv_path = out_path / "full_test_results.csv"
    test_results_df.to_csv(test_csv_path, index=False)

    cm_df = pd.DataFrame([
        {"Actual": "REAL (0)", "Predicted_REAL": int(tn), "Predicted_FAKE": int(fp)},
        {"Actual": "FAKE (1)", "Predicted_REAL": int(fn), "Predicted_FAKE": int(tp)},
    ])
    cm_csv_path = out_path / "full_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index=False)

    print("\n--- HELD-OUT TEST EVALUATION RESULT ---")
    print(test_results_df.to_string(index=False))
    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Test evaluation saved to: {test_csv_path}")

    # 7. Persist trained model and scaler to disk
    model_save_path = mod_path / "best_model.joblib"
    scaler_save_path = mod_path / "scaler.joblib"

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"\nSaved trained best model to: {model_save_path}")
    print(f"Saved fitted scaler to:     {scaler_save_path}")
    print("=" * 80)

    return {
        "val_comp_df": val_comp_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "test_results_df": test_results_df,
        "cm_df": cm_df,
        "model_save_path": str(model_save_path),
        "scaler_save_path": str(scaler_save_path),
    }

if __name__ == "__main__":
    run_full_ml_benchmark()
