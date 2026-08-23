# Phase 11 Completion Report: Unified Video Inference & Explainability API

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Summary of Accomplishments

Phase 11 implements the raw video inference and local explainability pipeline, allowing arbitrary MP4 video streams to be processed directly through the end-to-end traditional visual feature pipeline, persisted scaler, trained model, and SHAP feature attribution engine.

- **Raw Video Inference Script**: [`src/video/predict_video.py`](file:///c:/work/explainable-ai-media-detection/src/video/predict_video.py)
- **API Service Class**: [`src/video/inference_pipeline.py`](file:///c:/work/explainable-ai-media-detection/src/video/inference_pipeline.py) (`VideoInferenceService`)
- **Automated Test Suite**: [`test_predict_video.py`](file:///c:/work/explainable-ai-media-detection/test_predict_video.py)
- **Pipeline Documentation**: [`docs/video_inference.md`](file:///c:/work/explainable-ai-media-detection/docs/video_inference.md)

---

## 2. Files Created / Modified

1. [`src/video/predict_video.py`](file:///c:/work/explainable-ai-media-detection/src/video/predict_video.py) [NEW]
2. [`src/video/inference_pipeline.py`](file:///c:/work/explainable-ai-media-detection/src/video/inference_pipeline.py) [NEW]
3. [`src/video/__init__.py`](file:///c:/work/explainable-ai-media-detection/src/video/__init__.py) [MODIFIED]
4. [`test_predict_video.py`](file:///c:/work/explainable-ai-media-detection/test_predict_video.py) [NEW]
5. [`docs/video_inference.md`](file:///c:/work/explainable-ai-media-detection/docs/video_inference.md) [NEW]
6. [`docs/phase11_completion_report.md`](file:///c:/work/explainable-ai-media-detection/docs/phase11_completion_report.md) [NEW]

---

## 3. Test Execution & Validation Results

- **Executed**: `python test_predict_video.py`
- **Result**: **`RAW VIDEO INFERENCE TEST RESULT: PASSED`**
  - Model file (`best_model.joblib`) verified
  - Scaler file (`scaler.joblib`) verified (`n_features_in_ = 216`)
  - 216 features extracted (0 NaN, 0 Inf)
  - `VideoInferenceService.analyze()` API abstraction verified
  - Graceful error handling for missing/corrupted files verified

---

## 4. Sample Inference & SHAP Output

### Command Executed
```bash
python -m src.video.predict_video Dataset/Video/original/000.mp4
```

### Response
```json
{
  "status": "success",
  "video": {
    "filename": "000.mp4",
    "video_path": "Dataset/Video/original/000.mp4",
    "total_frames_in_stream": 396,
    "frames_analyzed": 10,
    "features_extracted": 216
  },
  "prediction": {
    "label": "FAKE",
    "class_id": 1,
    "fake_probability": 0.9911,
    "real_probability": 0.0089
  },
  "model": {
    "name": "HistGradientBoostingClassifier",
    "model_path": "data/videos/models/best_model.joblib",
    "scaler_path": "data/videos/models/scaler.joblib"
  },
  "explainability": {
    "available": true,
    "top_fake_evidence": [
      {
        "feature": "v_hist_2_mean",
        "feature_family": "Color (HSV)",
        "shap_value": 0.352848,
        "direction": "FAKE"
      },
      {
        "feature": "lbp_hist_6_std",
        "feature_family": "Texture (LBP & Gray)",
        "shap_value": 0.333255,
        "direction": "FAKE"
      }
    ],
    "top_real_evidence": [
      {
        "feature": "v_hist_2_std",
        "feature_family": "Color (HSV)",
        "shap_value": -0.352283,
        "direction": "REAL"
      }
    ]
  },
  "disclaimer": "This system is an explainable machine learning research prototype. Full dataset benchmark validation achieved Balanced Accuracy = 50.00% and ROC-AUC = 53.43%. Predictions and probabilities represent algorithmic decision scores on traditional visual features and should NOT be used as definitive or ground-truth proof of media authenticity."
}
```

---

## 5. Research Limitations & Prototype Status

> [!IMPORTANT]
> **Research Disclaimer**: Full dataset benchmark validation on Stage B/C dataset achieved **Balanced Accuracy = 50.00%** and **ROC-AUC = 53.43%**, with the baseline classifier predicting `FAKE` for most test samples due to 1:6 class imbalance. The output JSON contract explicitly embeds this disclaimer to prevent misleading claims of production accuracy.
