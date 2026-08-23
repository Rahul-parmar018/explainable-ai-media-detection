import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import shap

from src.video.ml_training import load_and_split_data, prepare_scaled_features, METADATA_COLUMNS
from sklearn.linear_model import LogisticRegression

FEATURE_FAMILY_RULES = [
    (("h_", "s_", "v_"), "Color (HSV)"),
    (("gray_", "lbp_"), "Texture (LBP & Gray)"),
    (("edge_",), "Edge (Canny)"),
    (("glcm_",), "Texture (GLCM)"),
    (("dct_",), "Frequency (DCT)"),
]

def map_feature_to_family(feature_name: str) -> str:
    """
    Maps an aggregated feature name to its primary traditional visual feature family.
    """
    fname_lower = feature_name.lower()
    for prefixes, family in FEATURE_FAMILY_RULES:
        for prefix in prefixes:
            if fname_lower.startswith(prefix):
                return family
    return "Other"


def generate_feature_families_mapping(feature_names: List[str]) -> pd.DataFrame:
    """
    Generates a DataFrame mapping each feature to its designated feature family.
    """
    rows = []
    for f in feature_names:
        rows.append({"feature": f, "feature_family": map_feature_to_family(f)})
    return pd.DataFrame(rows)


def explain_video(
    video_feature_dict: Dict[str, float],
    model: LogisticRegression,
    scaler: Any,
    explainer: shap.LinearExplainer,
    feature_names: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Generates a reusable local SHAP explanation for a single video feature dictionary.

    Args:
        video_feature_dict (Dict[str, float]): 216 aggregated video features.
        model (LogisticRegression): Trained Logistic Regression classifier.
        scaler (Any): Fitted StandardScaler.
        explainer (shap.LinearExplainer): Initialized SHAP LinearExplainer.
        feature_names (List[str]): List of 216 feature names.
        top_k (int): Number of top positive/negative evidence items to return.

    Returns:
        Dict[str, Any]: Explanation dictionary containing prediction, probability, and evidence items.
    """
    # 1. Format feature vector matching feature_names order
    raw_vector = np.array([[float(video_feature_dict.get(fn, 0.0)) for fn in feature_names]], dtype=np.float64)

    # 2. Scale features using training distribution parameters
    scaled_vector = scaler.transform(raw_vector)

    # 3. Model prediction and probabilities
    pred_code = int(model.predict(scaled_vector)[0])
    probas = model.predict_proba(scaled_vector)[0]
    pred_label = "FAKE" if pred_code == 1 else "REAL"
    pred_prob = float(probas[pred_code])

    # 4. Compute single-sample SHAP values
    shap_vals = explainer.shap_values(scaled_vector)
    if isinstance(shap_vals, list):
        shap_array = shap_vals[1][0]  # SHAP for positive class (FAKE=1)
    elif shap_vals.ndim == 2:
        shap_array = shap_vals[0]
    else:
        shap_array = shap_vals.flatten()

    # 5. Extract top FAKE evidence (positive SHAP values pushing toward FAKE=1)
    fake_items = []
    real_items = []

    for idx, fn in enumerate(feature_names):
        s_val = float(shap_array[idx])
        family = map_feature_to_family(fn)
        raw_val = float(raw_vector[0, idx])
        item = {
            "feature": fn,
            "feature_family": family,
            "feature_value": round(raw_val, 6),
            "shap_value": round(s_val, 6),
            "direction": "Pushes FAKE" if s_val > 0 else "Pushes REAL"
        }
        if s_val > 0:
            fake_items.append(item)
        else:
            real_items.append(item)

    # Sort fake evidence descending by SHAP value
    fake_items.sort(key=lambda x: x["shap_value"], reverse=True)
    # Sort real evidence ascending by SHAP value (most negative pushing REAL)
    real_items.sort(key=lambda x: x["shap_value"], reverse=False)

    return {
        "prediction": pred_label,
        "probability": round(pred_prob, 4),
        "top_fake_evidence": fake_items[:top_k],
        "top_real_evidence": real_items[:top_k]
    }


def compute_shap_explanations(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes global and local SHAP explanations, generates feature family contribution rankings,
    and exports research CSV artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data & execute source-aware split
    train_df, val_df, test_df, feature_names = load_and_split_data(
        csv_path=csv_path, random_state=random_state
    )

    # 2. Fit scaler ONLY on training set to prevent data leakage
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = prepare_scaled_features(
        train_df, val_df, test_df, feature_names
    )

    # 3. Fit Logistic Regression baseline model
    model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=2000)
    model.fit(X_train_scaled, y_train)

    # 4. Initialize SHAP LinearExplainer with background distribution from X_train_scaled
    explainer = shap.LinearExplainer(model, X_train_scaled)

    # Compute SHAP values for Test set
    shap_test = explainer.shap_values(X_test_scaled)
    if isinstance(shap_test, list):
        shap_matrix = shap_test[1]
    else:
        shap_matrix = shap_test

    # 5. Global Explanation DataFrame
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_shap = np.mean(shap_matrix, axis=0)

    global_rows = []
    for idx, fn in enumerate(feature_names):
        m_abs = float(mean_abs_shap[idx])
        m_val = float(mean_shap[idx])
        family = map_feature_to_family(fn)
        global_rows.append({
            "feature": fn,
            "feature_family": family,
            "mean_abs_shap": round(m_abs, 6),
            "mean_shap": round(m_val, 6),
            "direction": "Pushes FAKE" if m_val > 0 else "Pushes REAL"
        })

    global_df = pd.DataFrame(global_rows).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    global_path = out_path / "shap_global_importance.csv"
    global_df.to_csv(global_path, index=False)

    # 6. Feature Family Mapping & Aggregated Family Importance
    fam_mapping_df = generate_feature_families_mapping(feature_names)
    fam_mapping_df["mean_abs_shap"] = fam_mapping_df["feature"].map(
        global_df.set_index("feature")["mean_abs_shap"].to_dict()
    )
    fam_mapping_path = out_path / "shap_feature_families.csv"
    fam_mapping_df.to_csv(fam_mapping_path, index=False)

    total_importance_sum = float(global_df["mean_abs_shap"].sum())
    family_summary = (
        fam_mapping_df.groupby("feature_family")["mean_abs_shap"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "total_mean_abs_shap", "count": "feature_count"})
    )
    family_summary["percentage_contribution"] = (
        (family_summary["total_mean_abs_shap"] / (total_importance_sum + 1e-7)) * 100
    ).round(2)
    family_summary = family_summary.sort_values(by="percentage_contribution", ascending=False).reset_index(drop=True)

    family_path = out_path / "shap_family_importance.csv"
    family_summary.to_csv(family_path, index=False)

    # 7. Local Explanations for Test Sample Videos
    local_rows = []
    test_sample_videos = test_df.reset_index(drop=True)

    for i in range(len(test_sample_videos)):
        row = test_sample_videos.iloc[i]
        v_feat_dict = {fn: row[fn] for fn in feature_names}
        local_exp = explain_video(v_feat_dict, model, scaler, explainer, feature_names, top_k=10)

        # Record top fake evidence items
        for item in local_exp["top_fake_evidence"]:
            local_rows.append({
                "video_path": row["video_path"],
                "category": row["category"],
                "actual_label": row["label"],
                "predicted_label": local_exp["prediction"],
                "prediction_probability": local_exp["probability"],
                "evidence_type": "Top FAKE Evidence",
                "feature": item["feature"],
                "feature_family": item["feature_family"],
                "feature_value": item["feature_value"],
                "shap_value": item["shap_value"],
                "direction": item["direction"]
            })

        # Record top real evidence items
        for item in local_exp["top_real_evidence"]:
            local_rows.append({
                "video_path": row["video_path"],
                "category": row["category"],
                "actual_label": row["label"],
                "predicted_label": local_exp["prediction"],
                "prediction_probability": local_exp["probability"],
                "evidence_type": "Top REAL Evidence",
                "feature": item["feature"],
                "feature_family": item["feature_family"],
                "feature_value": item["feature_value"],
                "shap_value": item["shap_value"],
                "direction": item["direction"]
            })

    local_df = pd.DataFrame(local_rows)
    local_path = out_path / "shap_local_explanations.csv"
    local_df.to_csv(local_path, index=False)

    print("=" * 80)
    print("PHASE 8: SHAP VIDEO EXPLAINABILITY SUITE")
    print("=" * 80)
    print(f"Features Explained:              {len(feature_names)}")
    print(f"Test Videos Explained:           {len(test_df)}")
    print(f"Global Importance CSV:           {global_path}")
    print(f"Feature Families Mapping CSV:    {fam_mapping_path}")
    print(f"Family Importance Summary CSV:   {family_path}")
    print(f"Local Explanations CSV:          {local_path}")
    print("=" * 80)
    print("\n--- FEATURE FAMILY IMPORTANCE CONTRIBUTION ---")
    print(family_summary.to_string(index=False))
    print("=" * 80)

    return {
        "explainer": explainer,
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "global_df": global_df,
        "family_summary_df": family_summary,
        "local_df": local_df,
        "test_sample_count": len(test_df)
    }

if __name__ == "__main__":
    compute_shap_explanations()
