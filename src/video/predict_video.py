import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from src.video.frame_extraction import sample_video_frames
from src.video.preprocessing import preprocess_frame
from src.video.feature_extraction import extract_frame_features
from src.video.aggregation import aggregate_frame_features
from src.video.explainability import map_feature_to_family, explain_video

# Research Benchmark Prototype Disclaimer
RESEARCH_PROTOTYPE_DISCLAIMER = (
    "This system is an explainable machine learning research prototype. "
    "Full dataset benchmark validation achieved Balanced Accuracy = 50.00% and ROC-AUC = 53.43%. "
    "Predictions and probabilities represent algorithmic decision scores on traditional visual features "
    "and should NOT be used as definitive or ground-truth proof of media authenticity."
)


def extract_features_from_video_path(
    video_path: str,
    n_frames: int = 10
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Extracts 216 aggregated traditional visual features from an arbitrary MP4 video file.

    Returns:
        Tuple[video_features_dict, video_metadata_dict]
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Target video file '{video_path}' does not exist.")

    # 1. Sample frames using OpenCV frame extraction
    sampling_res = sample_video_frames(str(path_obj), num_frames=n_frames)
    if not sampling_res.get("readable", False) or len(sampling_res.get("sampled_frames", [])) == 0:
        err_msg = sampling_res.get("error", "Failed to open or read video frames.")
        raise ValueError(f"Unreadable video stream '{video_path}': {err_msg}")

    sampled_frames = sampling_res["sampled_frames"]

    # 2. Preprocess frames & extract 54 features per frame
    frame_features_list = []
    for frame in sampled_frames:
        preprocessed = preprocess_frame(frame, target_size=(256, 256))
        if preprocessed.get("valid", False):
            f_dict = extract_frame_features(preprocessed)
            if len(f_dict) == 54:
                frame_features_list.append(f_dict)

    if not frame_features_list:
        raise ValueError(f"Feature extraction failed for video frames in '{video_path}'.")

    # 3. Temporal aggregation (54 features x 4 stats = 216 features)
    video_features = aggregate_frame_features(frame_features_list)

    if len(video_features) != 216:
        raise ValueError(f"Extracted {len(video_features)} features (expected 216).")

    # Validate NaN & Inf
    for fk, fval in video_features.items():
        if math.isnan(fval) or math.isinf(fval):
            video_features[fk] = 0.0

    meta = {
        "filename": path_obj.name,
        "video_path": str(path_obj.as_posix()),
        "total_frames_in_stream": sampling_res.get("total_frames", 0),
        "frames_analyzed": len(frame_features_list),
        "features_extracted": len(video_features),
    }

    return video_features, meta


def predict_video(
    video_path: str,
    model_path: str = "data/videos/models/best_model.joblib",
    scaler_path: str = "data/videos/models/scaler.joblib",
    n_frames: int = 10,
    top_k_evidence: int = 5
) -> Dict[str, Any]:
    """
    Executes the end-to-end raw video inference and SHAP explainability pipeline.

    Args:
        video_path (str): Path to input MP4 video.
        model_path (str): Path to trained model joblib file.
        scaler_path (str): Path to fitted scaler joblib file.
        n_frames (int): Number of frames to sample (default: 10).
        top_k_evidence (int): Number of top positive/negative SHAP evidence items.

    Returns:
        Dict[str, Any]: Structured JSON-serializable inference result dictionary.
    """
    try:
        # 1. Validate artifact existence
        mod_p = Path(model_path)
        scl_p = Path(scaler_path)

        if not mod_p.exists():
            return {
                "status": "error",
                "error_type": "ModelMissingError",
                "error_message": f"Trained model file not found at '{model_path}'.",
                "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
            }

        if not scl_p.exists():
            return {
                "status": "error",
                "error_type": "ScalerMissingError",
                "error_message": f"Fitted scaler file not found at '{scaler_path}'.",
                "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
            }

        # 2. Extract 216 video features
        video_features, meta = extract_features_from_video_path(video_path, n_frames=n_frames)
        feature_names = list(video_features.keys())

        # 3. Load model & scaler
        model = joblib.load(mod_p)
        scaler = joblib.load(scl_p)

        # Validate scaler feature dimension
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != 216:
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features (expected 216).")

        raw_vector = np.array([[float(video_features[fn]) for fn in feature_names]], dtype=np.float64)
        scaled_vector = scaler.transform(raw_vector)

        # 4. Predict label and decision probabilities
        pred_code = int(model.predict(scaled_vector)[0])
        pred_label = "FAKE" if pred_code == 1 else "REAL"

        fake_prob = None
        real_prob = None

        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(scaled_vector)[0]
                real_prob = round(float(probas[0]), 4)
                fake_prob = round(float(probas[1]), 4)
            except Exception:
                pass

        # 5. Generate SHAP Explainability if available
        explainability_dict = {"available": False}

        try:
            import shap
            if hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, np.zeros((1, 216)))
                exp_res = explain_video(
                    video_feature_dict=video_features,
                    model=model,
                    scaler=scaler,
                    explainer=explainer,
                    feature_names=feature_names,
                    top_k=top_k_evidence
                )
                top_fake_items = [
                    {"feature": item["feature"], "feature_family": item["feature_family"], "shap_value": round(float(item["shap_value"]), 6), "direction": "FAKE"}
                    for item in exp_res.get("top_fake_evidence", [])
                ]
                top_real_items = [
                    {"feature": item["feature"], "feature_family": item["feature_family"], "shap_value": round(float(item["shap_value"]), 6), "direction": "REAL"}
                    for item in exp_res.get("top_real_evidence", [])
                ]
                explainability_dict = {
                    "available": True,
                    "top_fake_evidence": top_fake_items[:top_k_evidence],
                    "top_real_evidence": top_real_items[:top_k_evidence],
                }
            else:
                # For non-linear tree models (e.g. HistGradientBoosting), compute feature attributions via TreeExplainer or Explainer
                try:
                    tree_explainer = shap.TreeExplainer(model)
                    s_vals = tree_explainer.shap_values(scaled_vector)
                    if isinstance(s_vals, list):
                        s_arr = s_vals[1][0]
                    elif s_vals.ndim == 2:
                        s_arr = s_vals[0]
                    else:
                        s_arr = s_vals.flatten()

                    fake_ev = []
                    real_ev = []
                    for idx_f, fn in enumerate(feature_names):
                        val_s = float(s_arr[idx_f])
                        fam = map_feature_to_family(fn)
                        item = {"feature": fn, "feature_family": fam, "shap_value": round(val_s, 6), "direction": "FAKE" if val_s > 0 else "REAL"}
                        if val_s > 0:
                            fake_ev.append(item)
                        else:
                            real_ev.append(item)

                    fake_ev.sort(key=lambda x: x["shap_value"], reverse=True)
                    real_ev.sort(key=lambda x: x["shap_value"], reverse=False)

                    explainability_dict = {
                        "available": True,
                        "top_fake_evidence": fake_ev[:top_k_evidence],
                        "top_real_evidence": real_ev[:top_k_evidence],
                    }
                except Exception as ex_tree:
                    explainability_dict = {
                        "available": False,
                        "error": f"SHAP tree explainer unsupported for model {type(model).__name__}: {str(ex_tree)}"
                    }
        except Exception as shap_ex:
            explainability_dict = {
                "available": False,
                "error": f"SHAP explanation failed: {str(shap_ex)}"
            }

        return {
            "status": "success",
            "video": meta,
            "prediction": {
                "label": pred_label,
                "class_id": pred_code,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
            },
            "model": {
                "name": type(model).__name__,
                "model_path": str(mod_p.as_posix()),
                "scaler_path": str(scl_p.as_posix()),
            },
            "explainability": explainability_dict,
            "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
        }

    except Exception as ex:
        return {
            "status": "error",
            "error_type": type(ex).__name__,
            "error_message": str(ex),
            "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
        }


def main():
    parser = argparse.ArgumentParser(description="Explainable Video AI Detection CLI")
    parser.add_argument("video_path", type=str, help="Path to target MP4 video file")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to sample (default: 10)")
    parser.add_argument("--model", type=str, default="data/videos/models/best_model.joblib", help="Path to best_model.joblib")
    parser.add_argument("--scaler", type=str, default="data/videos/models/scaler.joblib", help="Path to scaler.joblib")

    args = parser.parse_args()

    res = predict_video(
        video_path=args.video_path,
        model_path=args.model,
        scaler_path=args.scaler,
        n_frames=args.frames
    )

    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
