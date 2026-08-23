"""
Phase 12B — Model Correction Experiment.

Implements a comprehensive, leakage-safe imbalance-correction experiment grid to
address the critical failure of the Phase 10C baseline (Balanced Accuracy = 50.00%,
REAL recall = 0.00%).

Strategy:
1. Connected-component actor graph grouping (inherited from full_ml_benchmark.py)
2. StandardScaler fitted ONLY on training data
3. class_weight="balanced" for all models that support it
4. Explicit sample_weight for HistGradientBoosting (no native class_weight)
5. Balanced undersampling experiment (FAKE class undersampled in TRAIN only)
6. Threshold optimisation on VAL using Youden's J and F1 (never on TEST)
7. Final model evaluated ONCE on the untouched held-out TEST partition
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.utils import resample

from src.video.full_ml_benchmark import load_and_split_connected_components

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute full per-class + aggregate metric dictionary."""
    acc       = float(accuracy_score(y_true, y_pred))
    bal_acc   = float(balanced_accuracy_score(y_true, y_pred))
    real_prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    real_rec  = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    fake_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    fake_rec  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wgt_f1    = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = 0.5

    p = prefix
    return {
        f"{p}accuracy":        round(acc, 4),
        f"{p}balanced_accuracy": round(bal_acc, 4),
        f"{p}real_precision":  round(real_prec, 4),
        f"{p}real_recall":     round(real_rec, 4),
        f"{p}fake_precision":  round(fake_prec, 4),
        f"{p}fake_recall":     round(fake_rec, 4),
        f"{p}macro_f1":        round(macro_f1, 4),
        f"{p}weighted_f1":     round(wgt_f1, 4),
        f"{p}roc_auc":         round(roc_auc, 4),
        f"{p}pr_auc":          round(pr_auc, 4),
    }


def _diagnose(metrics: Dict[str, float], prefix: str = "val_") -> str:
    """Return WARNING if REAL recall, FAKE recall, or Balanced Acc are weak."""
    real_rec = metrics.get(f"{prefix}real_recall", 0.0)
    fake_rec = metrics.get(f"{prefix}fake_recall", 0.0)
    bal_acc  = metrics.get(f"{prefix}balanced_accuracy", 0.0)
    issues = []
    if real_rec < 0.50:
        issues.append(f"REAL recall={real_rec:.3f} < 0.50")
    if fake_rec < 0.50:
        issues.append(f"FAKE recall={fake_rec:.3f} < 0.50")
    if bal_acc <= 0.55:
        issues.append(f"Balanced Acc={bal_acc:.3f} <= 0.55")
    return "WARNING: " + "; ".join(issues) if issues else "OK"


def _optimise_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Select threshold maximising Youden's J on the given partition."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def _optimise_threshold_balanced_acc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    candidates: np.ndarray,
) -> float:
    """Select threshold maximising balanced accuracy on the given partition."""
    best_thresh = 0.5
    best_ba = -1.0
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        ba = float(balanced_accuracy_score(y_true, y_pred))
        if ba > best_ba:
            best_ba = ba
            best_thresh = float(t)
    return best_thresh


def _apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Build balanced training subset (undersample FAKE in TRAIN only)
# ──────────────────────────────────────────────────────────────────────────────

def make_balanced_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample FAKE class so REAL:FAKE = 1:1 in training. VAL/TEST untouched."""
    real_idx = np.where(y_train == 0)[0]
    fake_idx = np.where(y_train == 1)[0]
    n_real = len(real_idx)

    fake_idx_down = resample(fake_idx, replace=False, n_samples=n_real, random_state=random_state)
    keep = np.concatenate([real_idx, fake_idx_down])
    np.random.default_rng(random_state).shuffle(keep)
    return X_train[keep], y_train[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Sample weights for HistGradientBoosting (no native class_weight)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for class balancing."""
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    w_map = dict(zip(classes, weight_per_class))
    return np.array([w_map[yi] for yi in y])


# ──────────────────────────────────────────────────────────────────────────────
# Main correction experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_model_correction(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    models_dir: str = "data/videos/models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full Phase 12B model-correction experiment.

    Returns dict with corrected model, scaler, threshold, and all metric tables.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + split ──────────────────────────────────────────────────────
    train_df, val_df, test_df, feature_names = load_and_split_connected_components(
        csv_path=csv_path, random_state=random_state
    )

    X_train_raw = train_df[feature_names].values.astype(np.float64)
    y_train     = train_df["target"].values.astype(int)
    X_val_raw   = val_df[feature_names].values.astype(np.float64)
    y_val       = val_df["target"].values.astype(int)
    X_test_raw  = test_df[feature_names].values.astype(np.float64)
    y_test      = test_df["target"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    # Balanced undersampled training data (FAKE downsampled to REAL count)
    X_train_bal, y_train_bal = make_balanced_train(X_train, y_train, random_state)

    # Sample weights for HistGradientBoosting
    sw_train     = compute_sample_weights(y_train)
    sw_train_bal = compute_sample_weights(y_train_bal)

    print("=" * 80)
    print("PHASE 12B: MODEL CORRECTION EXPERIMENT")
    print("=" * 80)
    print(f"CSV:   {csv_path}")
    print(f"TRAIN: {len(y_train)} samples | REAL={int((y_train==0).sum())} | FAKE={int((y_train==1).sum())}")
    print(f"VAL:   {len(y_val)} samples   | REAL={int((y_val==0).sum())} | FAKE={int((y_val==1).sum())}")
    print(f"TEST:  {len(y_test)} samples  | REAL={int((y_test==0).sum())} | FAKE={int((y_test==1).sum())}")
    print(f"BALANCED TRAIN: {len(y_train_bal)} samples | REAL={int((y_train_bal==0).sum())} | FAKE={int((y_train_bal==1).sum())}")
    print("=" * 80)

    # ── 2. Experiment grid ───────────────────────────────────────────────────
    # Each entry: (label, training_strategy, callable_returning_fitted_model, X_train, y_train, sw)
    # "balanced" strategy uses undersampled training set.

    threshold_candidates = np.arange(0.10, 0.91, 0.05)

    experiments = []

    # Logistic Regression
    for C in [0.01, 0.1, 1.0, 10.0]:
        for strategy, Xtr, ytr, _sw in [
            ("full_class_weight", X_train, y_train, None),
            ("undersample", X_train_bal, y_train_bal, None),
        ]:
            experiments.append({
                "model_label": f"LR C={C}",
                "strategy": strategy,
                "clf": LogisticRegression(
                    C=C, class_weight="balanced",
                    random_state=random_state, max_iter=3000,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": None,
            })

    # Linear SVM
    for C in [0.01, 0.1, 1.0, 10.0]:
        for strategy, Xtr, ytr in [
            ("full_class_weight", X_train, y_train),
            ("undersample", X_train_bal, y_train_bal),
        ]:
            experiments.append({
                "model_label": f"LinearSVM C={C}",
                "strategy": strategy,
                "clf": SVC(
                    kernel="linear", C=C, class_weight="balanced",
                    probability=True, random_state=random_state,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": None,
            })

    # RBF SVM — fewer configs (slow)
    for C, gamma in [(1.0, "scale"), (10.0, "scale"), (1.0, "auto")]:
        for strategy, Xtr, ytr in [
            ("full_class_weight", X_train, y_train),
            ("undersample", X_train_bal, y_train_bal),
        ]:
            experiments.append({
                "model_label": f"RBFSVM C={C} g={gamma}",
                "strategy": strategy,
                "clf": SVC(
                    kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                    probability=True, random_state=random_state,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": None,
            })

    # Random Forest
    for max_depth, min_leaf in [(None, 1), (10, 2), (20, 1)]:
        for strategy, Xtr, ytr in [
            ("full_class_weight", X_train, y_train),
            ("undersample", X_train_bal, y_train_bal),
        ]:
            experiments.append({
                "model_label": f"RF depth={max_depth} leaf={min_leaf}",
                "strategy": strategy,
                "clf": RandomForestClassifier(
                    n_estimators=300, max_depth=max_depth,
                    min_samples_leaf=min_leaf, class_weight="balanced",
                    random_state=random_state, n_jobs=-1,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": None,
            })

    # Extra Trees
    for max_depth, min_leaf in [(None, 1), (10, 2), (20, 1)]:
        for strategy, Xtr, ytr in [
            ("full_class_weight", X_train, y_train),
            ("undersample", X_train_bal, y_train_bal),
        ]:
            experiments.append({
                "model_label": f"ET depth={max_depth} leaf={min_leaf}",
                "strategy": strategy,
                "clf": ExtraTreesClassifier(
                    n_estimators=300, max_depth=max_depth,
                    min_samples_leaf=min_leaf, class_weight="balanced",
                    random_state=random_state, n_jobs=-1,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": None,
            })

    # HistGradientBoosting — use sample_weight (no class_weight support in older sklearn)
    for lr, max_iter in [(0.1, 100), (0.05, 200), (0.1, 200)]:
        for strategy, Xtr, ytr, sw in [
            ("full_sample_weight", X_train, y_train, sw_train),
            ("undersample_sw", X_train_bal, y_train_bal, sw_train_bal),
        ]:
            experiments.append({
                "model_label": f"HGB lr={lr} iter={max_iter}",
                "strategy": strategy,
                "clf": HistGradientBoostingClassifier(
                    learning_rate=lr, max_iter=max_iter,
                    random_state=random_state,
                ),
                "X_tr": Xtr, "y_tr": ytr, "sw": sw,
            })

    # ── 3. Run experiments ───────────────────────────────────────────────────
    records = []
    best_val_ba = -1.0
    best_config = None

    n_exp = len(experiments)
    print(f"\nRunning {n_exp} experiment configurations...\n")

    for idx, exp in enumerate(experiments, 1):
        label    = exp["model_label"]
        strategy = exp["strategy"]
        clf      = exp["clf"]
        Xtr      = exp["X_tr"]
        ytr      = exp["y_tr"]
        sw       = exp["sw"]

        # Fit
        try:
            if sw is not None:
                clf.fit(Xtr, ytr, sample_weight=sw)
            else:
                clf.fit(Xtr, ytr)
        except Exception as e:
            print(f"  [{idx}/{n_exp}] {label} | {strategy} — FIT ERROR: {e}")
            continue

        # Predict on val (default threshold 0.5)
        if hasattr(clf, "predict_proba"):
            y_val_score = clf.predict_proba(X_val)[:, 1]
        elif hasattr(clf, "decision_function"):
            raw = clf.decision_function(X_val)
            y_val_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            y_val_score = clf.predict(X_val).astype(float)

        # Threshold optimisation on VAL only
        thresh_youden = _optimise_threshold_youden(y_val, y_val_score)
        thresh_ba     = _optimise_threshold_balanced_acc(y_val, y_val_score, threshold_candidates)

        # Evaluate with multiple thresholds
        for thresh_name, thresh in [("default_0.5", 0.5), ("youden", thresh_youden), ("ba_opt", thresh_ba)]:
            y_val_pred = _apply_threshold(y_val_score, thresh)
            m = _compute_metrics(y_val, y_val_pred, y_val_score, prefix="val_")
            diag = _diagnose(m, prefix="val_")

            row = {
                "model": label,
                "configuration": str(clf.get_params()),
                "training_strategy": strategy,
                "threshold_strategy": thresh_name,
                "threshold": round(thresh, 4),
                **m,
                "diagnostic": diag,
            }
            records.append(row)

            if thresh_name == "ba_opt":
                ba = m["val_balanced_accuracy"]
                real_rec = m["val_real_recall"]
                print(f"  [{idx}/{n_exp}] {label:<28} | {strategy:<20} | BA={ba:.4f} | REAL_rec={real_rec:.4f} | thresh={thresh:.2f} | {diag}")

                if (ba > best_val_ba and real_rec >= 0.30):
                    best_val_ba = ba
                    best_config = {
                        "model_label": label,
                        "strategy": strategy,
                        "clf": clf,
                        "X_tr": Xtr, "y_tr": ytr, "sw": sw,
                        "thresh": thresh,
                        "val_metrics": m,
                    }

    # ── 4. Save experiment table ─────────────────────────────────────────────
    records_df = pd.DataFrame(records)
    exp_csv = out_path / "model_correction_experiments.csv"
    records_df.to_csv(exp_csv, index=False)
    print(f"\nSaved {len(records_df)} experiment rows to: {exp_csv}")

    # ── 5. Select best model ─────────────────────────────────────────────────
    if best_config is None:
        print("\n[CRITICAL] No configuration found with REAL recall >= 0.30 and improved Balanced Accuracy.")
        return {"status": "FAILED", "records_df": records_df}

    print(f"\nBEST MODEL: {best_config['model_label']} | strategy={best_config['strategy']} | thresh={best_config['thresh']:.2f}")
    print(f"Validation Balanced Accuracy: {best_val_ba:.4f}")
    print(f"Validation Metrics: {best_config['val_metrics']}")

    # ── 6. Retrain selected model on TRAIN and evaluate ONCE on TEST ─────────
    best_clf = best_config["clf"]  # already fitted on same Xtr/ytr
    best_thresh = best_config["thresh"]

    if hasattr(best_clf, "predict_proba"):
        y_test_score = best_clf.predict_proba(X_test)[:, 1]
    elif hasattr(best_clf, "decision_function"):
        raw = best_clf.decision_function(X_test)
        y_test_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        y_test_score = best_clf.predict(X_test).astype(float)

    y_test_pred = _apply_threshold(y_test_score, best_thresh)
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_score, prefix="test_")
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- CORRECTED MODEL HELD-OUT TEST RESULTS ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"  Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # Comparison CSV
    corrected_compare = pd.DataFrame([{
        "phase": "Phase_10C_Baseline",
        "model": "HistGradientBoostingClassifier (default threshold)",
        "test_balanced_accuracy": 0.5000,
        "test_real_recall": 0.0000,
        "test_fake_recall": 1.0000,
        "test_roc_auc": 0.5343,
        "TN": 0, "FP": 17, "FN": 0, "TP": 85,
    }, {
        "phase": "Phase_12B_Corrected",
        "model": best_config["model_label"],
        "test_balanced_accuracy": test_metrics["test_balanced_accuracy"],
        "test_real_recall": test_metrics["test_real_recall"],
        "test_fake_recall": test_metrics["test_fake_recall"],
        "test_roc_auc": test_metrics["test_roc_auc"],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }])
    corrected_compare.to_csv(out_path / "corrected_model_comparison.csv", index=False)

    test_results_df = pd.DataFrame([{
        "model": best_config["model_label"],
        "training_strategy": best_config["strategy"],
        "threshold": best_thresh,
        **test_metrics,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }])
    test_results_df.to_csv(out_path / "corrected_test_results.csv", index=False)

    cm_df = pd.DataFrame([
        {"Actual": "REAL (0)", "Predicted_REAL": int(tn), "Predicted_FAKE": int(fp)},
        {"Actual": "FAKE (1)", "Predicted_REAL": int(fn), "Predicted_FAKE": int(tp)},
    ])
    cm_df.to_csv(out_path / "corrected_confusion_matrix.csv", index=False)

    # ── 7. Always save the best-found artifacts for test/research use ──────
    old_ba = 0.5000
    new_ba = test_metrics["test_balanced_accuracy"]
    improved = new_ba > old_ba + 0.01  # at least 1 point improvement

    # Always save: test suite needs to validate the artifacts.
    # The 'improved' flag in the JSON indicates whether this is deployment-worthy.
    joblib.dump(best_clf, mod_path / "corrected_best_model.joblib")
    joblib.dump(scaler, mod_path / "corrected_scaler.joblib")
    thresh_json = {
        "threshold": best_thresh,
        "threshold_strategy": best_config.get("thresh_strategy", "ba_opt"),
        "val_balanced_accuracy": best_val_ba,
        "test_balanced_accuracy": new_ba,
        "model_label": best_config["model_label"],
        "improved_over_baseline": improved,
        "deployment_approved": False,  # Requires explicit user approval
    }
    (mod_path / "corrected_threshold.json").write_text(json.dumps(thresh_json, indent=2))
    if improved:
        print(f"\nSaved corrected model artifacts (IMPROVED) to {mod_path}")
    else:
        print(f"\nSaved corrected model artifacts (NOT improved over baseline; BA {old_ba:.4f} -> {new_ba:.4f}) to {mod_path}")
        print("  NOTE: Artifacts saved for research/test use only. DO NOT deploy without explicit user approval.")

    return {
        "status": "OK",
        "best_config": best_config,
        "best_thresh": best_thresh,
        "val_balanced_accuracy": best_val_ba,
        "test_metrics": test_metrics,
        "test_results_df": test_results_df,
        "records_df": records_df,
        "improved": improved,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "scaler": scaler,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    run_model_correction()
