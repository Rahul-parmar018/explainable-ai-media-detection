import os
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

METADATA_COLUMNS = [
    "video_path",
    "category",
    "label",
    "source_id",
    "frame_count",
    "fps",
    "width",
    "height",
]

def load_and_split_data(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    test_size: float = 0.15,
    val_size_relative: float = 0.1765,  # 0.15 / 0.85 = ~0.1765 (yielding 70% Train, 15% Val, 15% Test)
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads feature CSV, separates metadata columns, maps labels (REAL=0, FAKE=1),
    and executes source-aware GroupShuffleSplit to prevent data leakage.

    Returns:
        Tuple[train_df, val_df, test_df, feature_names]
    """
    df = pd.read_csv(csv_path)

    # Validate target column and map REAL=0, FAKE=1
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in CSV dataset.")

    df["target"] = df["label"].map({"REAL": 0, "FAKE": 1})

    feature_names = [c for c in df.columns if c not in METADATA_COLUMNS and c != "target"]
    if len(feature_names) != 216:
        print(f"Warning: Found {len(feature_names)} feature columns (expected 216).")

    groups = df["source_id"]

    # Step 1: Split 85% Train/Val and 15% Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Step 2: Split Train/Val into 70% Train and 15% Val
    groups_train_val = df_train_val["source_id"]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=random_state)
    train_idx_sub, val_idx_sub = next(gss2.split(df_train_val, groups=groups_train_val))

    train_df = df_train_val.iloc[train_idx_sub].copy()
    val_df = df_train_val.iloc[val_idx_sub].copy()

    return train_df, val_df, test_df, feature_names


def prepare_scaled_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler ONLY on training data features and transforms Train, Val, and Test sets.
    """
    X_train = train_df[feature_names].values.astype(np.float64)
    y_train = train_df["target"].values.astype(int)

    X_val = val_df[feature_names].values.astype(np.float64)
    y_val = val_df["target"].values.astype(int)

    X_test = test_df[feature_names].values.astype(np.float64)
    y_test = test_df["target"].values.astype(int)

    # CRITICAL: Fit scaler ONLY on X_train to prevent preprocessing leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Computes classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        auc = 0.5

    return {
        "Accuracy": round(float(acc), 4),
        "Balanced Accuracy": round(float(bal_acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "ROC-AUC": round(float(auc), 4),
    }


def train_and_evaluate_baselines(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes Phase 7 traditional ML training, validation evaluation, model selection,
    and final test set evaluation. Saves comparison and feature importance CSVs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data and perform source-aware split
    train_df, val_df, test_df, feature_names = load_and_split_data(
        csv_path=csv_path, random_state=random_state
    )

    # 2. Fit scaler ONLY on train
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = prepare_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    # 3. Define traditional ML baseline models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=2000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "Support Vector Machine (SVM)": SVC(
            class_weight="balanced", probability=True, random_state=random_state
        ),
    }

    val_results = []
    trained_models = {}

    print("=" * 80)
    print("PHASE 7: TRADITIONAL MACHINE LEARNING BASELINE TRAINING")
    print("=" * 80)
    print(f"Train Set:      {len(train_df)} samples (Source IDs: {train_df['source_id'].nunique()})")
    print(f"Validation Set: {len(val_df)} samples (Source IDs: {val_df['source_id'].nunique()})")
    print(f"Test Set:       {len(test_df)} samples (Source IDs: {test_df['source_id'].nunique()})")
    print("=" * 80)

    # 4. Train and evaluate on Validation set
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)

        metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba)
        metrics["Model"] = name
        val_results.append(metrics)
        trained_models[name] = model

    val_results_df = pd.DataFrame(val_results)[
        ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ]

    # Save model_comparison.csv
    comp_path = out_path / "model_comparison.csv"
    val_results_df.to_csv(comp_path, index=False)
    print("\n--- VALIDATION MODEL COMPARISON ---")
    print(val_results_df.to_string(index=False))
    print(f"\nValidation comparison saved to: {comp_path}")

    # 5. Select best model based on Balanced Accuracy
    best_row = val_results_df.sort_values(by="Balanced Accuracy", ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_model = trained_models[best_model_name]
    print(f"\nBest Model Selected (by Balanced Accuracy): {best_model_name}")

    # 6. Evaluate Best Model ONCE on Held-out Test Set
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
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
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }])

    test_path = out_path / "test_results.csv"
    test_results_df.to_csv(test_path, index=False)
    print("\n--- HELD-OUT TEST EVALUATION RESULT ---")
    print(test_results_df.to_string(index=False))
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Test evaluation saved to: {test_path}")

    # 7. Extract Feature Importances / Coefficients for Explainability
    fi_df = pd.DataFrame({"Feature": feature_names})

    # Add Random Forest importance scores
    rf_model = trained_models["Random Forest"]
    fi_df["RF_Importance"] = rf_model.feature_importances_

    # Add Logistic Regression coefficient magnitude
    lr_model = trained_models["Logistic Regression"]
    fi_df["LR_Coefficient"] = lr_model.coef_[0]
    fi_df["LR_Abs_Coefficient"] = np.abs(lr_model.coef_[0])

    fi_df = fi_df.sort_values(by="RF_Importance", ascending=False).reset_index(drop=True)
    fi_path = out_path / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importance table saved to: {fi_path}")

    return {
        "val_results_df": val_results_df,
        "best_model_name": best_model_name,
        "test_results_df": test_results_df,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "feature_importance_df": fi_df,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

if __name__ == "__main__":
    train_and_evaluate_baselines()
