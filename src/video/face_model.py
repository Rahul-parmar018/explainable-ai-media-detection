"""
src/video/face_model.py — Phase 13A Face-Region Model Training & Evaluation.

Trains and evaluates the same model families used in Phase 12B on the new
face-region feature set. Uses identical leakage-safe connected-component
splitting, threshold optimisation, and diagnostic warnings.

Outputs:
  data/videos/results/face_model_comparison.csv
  data/videos/results/face_test_results.csv
  data/videos/results/face_confusion_matrix.csv
  data/videos/results/face_vs_fullframe_comparison.csv
  data/videos/models/face_best_model.joblib
  data/videos/models/face_scaler.joblib
  data/videos/models/face_threshold.json
"""

import json
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
from sklearn.model_selection import GroupShuffleSplit
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

from src.video.face_batch_processor import attach_connected_components
from src.video.model_correction import compute_sample_weights

# ──────────────────────────────────────────────────────────────────────────────
# Metadata columns to exclude from features
# ──────────────────────────────────────────────────────────────────────────────

_NON_FEATURE_COLS = {
    "video_path", "video_name", "category", "label", "source_id",
    "target", "connected_component_id",
    "frames_sampled", "faces_detected", "face_detection_rate",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, prefix: str = "") -> Dict[str, float]:
    acc     = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    r_prec  = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r_rec   = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f_prec  = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f_rec   = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    m_f1    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w_f1    = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = 0.5
    try:
        pr = float(average_precision_score(y_true, y_score))
    except Exception:
        pr = 0.5

    p = prefix
    return {
        f"{p}accuracy":          round(acc, 4),
        f"{p}balanced_accuracy": round(bal_acc, 4),
        f"{p}real_precision":    round(r_prec, 4),
        f"{p}real_recall":       round(r_rec, 4),
        f"{p}fake_precision":    round(f_prec, 4),
        f"{p}fake_recall":       round(f_rec, 4),
        f"{p}macro_f1":          round(m_f1, 4),
        f"{p}weighted_f1":       round(w_f1, 4),
        f"{p}roc_auc":           round(roc, 4),
        f"{p}pr_auc":            round(pr, 4),
    }


def _diagnose(m: Dict[str, float], prefix: str = "val_") -> str:
    rr = m.get(f"{prefix}real_recall", 0)
    fr = m.get(f"{prefix}fake_recall", 0)
    ba = m.get(f"{prefix}balanced_accuracy", 0)
    issues = []
    if rr < 0.50: issues.append(f"REAL recall={rr:.3f}<0.50")
    if fr < 0.50: issues.append(f"FAKE recall={fr:.3f}<0.50")
    if ba <= 0.55: issues.append(f"BA={ba:.3f}<=0.55")
    return "WARNING: " + "; ".join(issues) if issues else "OK"


def _best_threshold_ba(y_true, y_score, candidates=None):
    if candidates is None:
        candidates = np.arange(0.10, 0.91, 0.05)
    best_t, best_ba = 0.5, -1.0
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        ba = float(balanced_accuracy_score(y_true, y_pred))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def _apply_threshold(y_score: np.ndarray, t: float) -> np.ndarray:
    return (y_score >= t).astype(int)


def _make_balanced(X, y, rs=42):
    r_idx = np.where(y == 0)[0]
    f_idx = np.where(y == 1)[0]
    n = len(r_idx)
    f_down = resample(f_idx, replace=False, n_samples=n, random_state=rs)
    keep = np.concatenate([r_idx, f_down])
    np.random.default_rng(rs).shuffle(keep)
    return X[keep], y[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def load_and_split_face_features(
    csv_path: str = "data/videos/features/video_face_features.csv",
    test_size: float = 0.15,
    val_size_relative: float = 0.1765,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load face feature CSV, attach CC groups, and do leakage-safe 70/15/15 split."""
    df = attach_connected_components(csv_path)

    # Verify NaN/Inf
    feature_names = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    feat_arr = df[feature_names].values.astype(np.float64)
    if np.isnan(feat_arr).any() or np.isinf(feat_arr).any():
        # Drop rows with NaN/Inf
        bad_rows = df[feature_names].isna().any(axis=1) | np.isinf(feat_arr).any(axis=1)
        n_bad = int(bad_rows.sum())
        print(f"  Dropping {n_bad} rows with NaN/Inf features.")
        df = df[~bad_rows].reset_index(drop=True)

    groups = df["connected_component_id"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, test_idx = next(gss1.split(df, groups=groups))

    df_tv   = df.iloc[tv_idx].copy()
    test_df = df.iloc[test_idx].copy()

    groups_tv = df_tv["connected_component_id"]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=random_state)
    tr_idx, vl_idx = next(gss2.split(df_tv, groups=groups_tv))
    train_df = df_tv.iloc[tr_idx].copy()
    val_df   = df_tv.iloc[vl_idx].copy()

    # Verify no overlap
    s_tr = set(train_df["connected_component_id"])
    s_vl = set(val_df["connected_component_id"])
    s_ts = set(test_df["connected_component_id"])
    assert len(s_tr & s_vl) == 0
    assert len(s_tr & s_ts) == 0
    assert len(s_vl & s_ts) == 0

    feature_names = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    return train_df, val_df, test_df, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Main training run
# ──────────────────────────────────────────────────────────────────────────────

def run_face_model(
    csv_path: str = "data/videos/features/video_face_features.csv",
    output_dir: str = "data/videos/results",
    models_dir: str = "data/videos/models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate face-region models. Compare against whole-frame baseline.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + split ──────────────────────────────────────────────────────
    train_df, val_df, test_df, feature_names = load_and_split_face_features(
        csv_path=csv_path, random_state=random_state
    )
    n_features = len(feature_names)

    X_train_r = train_df[feature_names].values.astype(np.float64)
    y_train   = train_df["target"].values.astype(int)
    X_val_r   = val_df[feature_names].values.astype(np.float64)
    y_val     = val_df["target"].values.astype(int)
    X_test_r  = test_df[feature_names].values.astype(np.float64)
    y_test    = test_df["target"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_r)
    X_val   = scaler.transform(X_val_r)
    X_test  = scaler.transform(X_test_r)

    X_tr_bal, y_tr_bal = _make_balanced(X_train, y_train, random_state)
    sw_train   = compute_sample_weights(y_train)
    sw_bal     = compute_sample_weights(y_tr_bal)

    print("=" * 80)
    print("PHASE 13A: FACE-REGION MODEL TRAINING")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Features: {n_features}")
    print(f"TRAIN: {len(y_train)}  REAL={int((y_train==0).sum())}  FAKE={int((y_train==1).sum())}")
    print(f"VAL:   {len(y_val)}   REAL={int((y_val==0).sum())}   FAKE={int((y_val==1).sum())}")
    print(f"TEST:  {len(y_test)}  REAL={int((y_test==0).sum())}  FAKE={int((y_test==1).sum())}")
    print(f"BAL TRAIN: {len(y_tr_bal)}  REAL={int((y_tr_bal==0).sum())}  FAKE={int((y_tr_bal==1).sum())}")
    print("=" * 80)

    # ── 2. Experiment grid ───────────────────────────────────────────────────
    thresh_candidates = np.arange(0.10, 0.91, 0.05)

    experiments = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        for strat, Xtr, ytr in [("full_cw", X_train, y_train), ("undersample", X_tr_bal, y_tr_bal)]:
            experiments.append({"label": f"LR C={C}", "strategy": strat,
                "clf": LogisticRegression(C=C, class_weight="balanced", random_state=random_state, max_iter=3000),
                "Xtr": Xtr, "ytr": ytr, "sw": None})
    for C in [0.01, 0.1, 1.0, 10.0]:
        for strat, Xtr, ytr in [("full_cw", X_train, y_train), ("undersample", X_tr_bal, y_tr_bal)]:
            experiments.append({"label": f"LinearSVM C={C}", "strategy": strat,
                "clf": SVC(kernel="linear", C=C, class_weight="balanced", probability=True, random_state=random_state),
                "Xtr": Xtr, "ytr": ytr, "sw": None})
    for C, gamma in [(1.0, "scale"), (10.0, "scale")]:
        for strat, Xtr, ytr in [("full_cw", X_train, y_train), ("undersample", X_tr_bal, y_tr_bal)]:
            experiments.append({"label": f"RBFSVM C={C} g={gamma}", "strategy": strat,
                "clf": SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True, random_state=random_state),
                "Xtr": Xtr, "ytr": ytr, "sw": None})
    for md, ml in [(None, 1), (10, 2), (20, 1)]:
        for strat, Xtr, ytr in [("full_cw", X_train, y_train), ("undersample", X_tr_bal, y_tr_bal)]:
            experiments.append({"label": f"RF depth={md} leaf={ml}", "strategy": strat,
                "clf": RandomForestClassifier(n_estimators=300, max_depth=md, min_samples_leaf=ml,
                    class_weight="balanced", random_state=random_state, n_jobs=-1),
                "Xtr": Xtr, "ytr": ytr, "sw": None})
    for md, ml in [(None, 1), (10, 2), (20, 1)]:
        for strat, Xtr, ytr in [("full_cw", X_train, y_train), ("undersample", X_tr_bal, y_tr_bal)]:
            experiments.append({"label": f"ET depth={md} leaf={ml}", "strategy": strat,
                "clf": ExtraTreesClassifier(n_estimators=300, max_depth=md, min_samples_leaf=ml,
                    class_weight="balanced", random_state=random_state, n_jobs=-1),
                "Xtr": Xtr, "ytr": ytr, "sw": None})
    for lr, niter in [(0.1, 100), (0.05, 200), (0.1, 200)]:
        for strat, Xtr, ytr, sw in [("full_sw", X_train, y_train, sw_train), ("undersample_sw", X_tr_bal, y_tr_bal, sw_bal)]:
            experiments.append({"label": f"HGB lr={lr} iter={niter}", "strategy": strat,
                "clf": HistGradientBoostingClassifier(learning_rate=lr, max_iter=niter, random_state=random_state),
                "Xtr": Xtr, "ytr": ytr, "sw": sw})

    records = []
    best_ba = -1.0
    best_config = None
    n_exp = len(experiments)
    print(f"\nRunning {n_exp} configurations...\n")

    for idx, exp in enumerate(experiments, 1):
        label    = exp["label"]
        strategy = exp["strategy"]
        clf      = exp["clf"]
        Xtr, ytr, sw = exp["Xtr"], exp["ytr"], exp["sw"]

        try:
            if sw is not None:
                clf.fit(Xtr, ytr, sample_weight=sw)
            else:
                clf.fit(Xtr, ytr)
        except Exception as e:
            print(f"  [{idx}/{n_exp}] {label} | {strategy} — FIT ERROR: {e}")
            continue

        if hasattr(clf, "predict_proba"):
            y_sc = clf.predict_proba(X_val)[:, 1]
        else:
            raw = clf.decision_function(X_val)
            y_sc = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        thresh = _best_threshold_ba(y_val, y_sc, thresh_candidates)
        y_pred = _apply_threshold(y_sc, thresh)
        m = _metrics(y_val, y_pred, y_sc, prefix="val_")
        diag = _diagnose(m, "val_")

        row = {"model": label, "strategy": strategy, "threshold": round(thresh, 4), **m, "diagnostic": diag}
        records.append(row)

        ba   = m["val_balanced_accuracy"]
        rr   = m["val_real_recall"]
        print(f"  [{idx:3d}/{n_exp}] {label:<30} | {strategy:<14} | BA={ba:.4f} | REAL_rec={rr:.4f} | thresh={thresh:.2f} | {diag}")

        if ba > best_ba and rr >= 0.30:
            best_ba = ba
            best_config = {**exp, "thresh": thresh, "val_metrics": m, "clf": clf}

    # Save experiments table
    rec_df = pd.DataFrame(records)
    rec_df.to_csv(out_path / "face_model_experiments.csv", index=False)

    # ── 3. Best model test evaluation ────────────────────────────────────────
    if best_config is None:
        print("\n[CRITICAL] No configuration found with REAL recall >= 0.30.")
        return {"status": "FAILED", "records_df": rec_df}

    best_clf   = best_config["clf"]
    best_thresh = best_config["thresh"]

    print(f"\nBEST MODEL: {best_config['label']} | {best_config['strategy']} | thresh={best_thresh:.2f}")
    print(f"Validation BA: {best_ba:.4f}")

    if hasattr(best_clf, "predict_proba"):
        y_ts = best_clf.predict_proba(X_test)[:, 1]
    else:
        raw = best_clf.decision_function(X_test)
        y_ts = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    y_tp = _apply_threshold(y_ts, best_thresh)
    test_m = _metrics(y_test, y_tp, y_ts, "test_")
    cm = confusion_matrix(y_test, y_tp)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- FACE MODEL TEST RESULTS ---")
    for k, v in test_m.items():
        print(f"  {k}: {v}")
    print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    # Save test results
    test_df_out = pd.DataFrame([{"model": best_config["label"], "strategy": best_config["strategy"],
        "threshold": best_thresh, **test_m, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}])
    test_df_out.to_csv(out_path / "face_test_results.csv", index=False)

    cm_df = pd.DataFrame([
        {"Actual": "REAL (0)", "Pred_REAL": int(tn), "Pred_FAKE": int(fp)},
        {"Actual": "FAKE (1)", "Pred_REAL": int(fn), "Pred_FAKE": int(tp)},
    ])
    cm_df.to_csv(out_path / "face_confusion_matrix.csv", index=False)

    # ── 4. Comparison vs whole-frame baseline ────────────────────────────────
    compare_df = pd.DataFrame([
        {
            "method": "WholeFrame_216_features (Phase10C/12B baseline)",
            "test_balanced_accuracy": 0.5000,
            "test_real_recall": 0.0000,
            "test_fake_recall": 1.0000,
            "test_roc_auc": 0.5343,
            "TN": 0, "FP": 17, "FN": 0, "TP": 85,
        },
        {
            "method": f"FaceRegion_{n_features}_features (Phase13A)",
            "test_balanced_accuracy": test_m["test_balanced_accuracy"],
            "test_real_recall": test_m["test_real_recall"],
            "test_fake_recall": test_m["test_fake_recall"],
            "test_roc_auc": test_m["test_roc_auc"],
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        },
    ])
    compare_df.to_csv(out_path / "face_vs_fullframe_comparison.csv", index=False)

    # ── 5. Model comparison table ─────────────────────────────────────────────
    val_comp = rec_df.sort_values("val_balanced_accuracy", ascending=False).reset_index(drop=True)
    val_comp.to_csv(out_path / "face_model_comparison.csv", index=False)

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    joblib.dump(best_clf, mod_path / "face_best_model.joblib")
    joblib.dump(scaler,   mod_path / "face_scaler.joblib")
    thresh_info = {
        "threshold": best_thresh,
        "val_balanced_accuracy": best_ba,
        "test_balanced_accuracy": test_m["test_balanced_accuracy"],
        "model_label": best_config["label"],
        "n_features": n_features,
        "improved_over_baseline": test_m["test_balanced_accuracy"] > 0.5100,
        "deployment_approved": False,
    }
    (mod_path / "face_threshold.json").write_text(json.dumps(thresh_info, indent=2))

    return {
        "status": "OK",
        "best_config": best_config,
        "best_thresh": best_thresh,
        "val_ba": best_ba,
        "test_metrics": test_m,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n_features": n_features,
        "feature_names": feature_names,
        "records_df": rec_df,
        "compare_df": compare_df,
        "improved": test_m["test_balanced_accuracy"] > 0.5100,
    }


if __name__ == "__main__":
    run_face_model()
