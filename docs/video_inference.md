# Raw Video Inference & Explainability Pipeline Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Pipeline Overview & Architecture

The raw video inference pipeline ([`src/video/predict_video.py`](file:///c:/work/explainable-ai-media-detection/src/video/predict_video.py)) converts an arbitrary input MP4 video file into an explainable classification decision without requiring pre-extracted CSV features.

```text
Raw MP4 Video Stream
       │
       ▼
OpenCV Stream Sampler (Uniform 10-frame extraction)
       │
       ▼
Frame Preprocessing (256 × 256 BGR, Grayscale, HSV)
       │
       ▼
Traditional Visual Feature Extraction (54 features/frame)
       │
       ▼
Temporal Aggregation (Mean, Std, Min, Max → 216 video features)
       │
       ▼
Persisted Scaler (`data/videos/models/scaler.joblib`)
       │
       ▼
Persisted Model (`data/videos/models/best_model.joblib`)
       │
       ▼
Decision Score & Probabilities (REAL vs FAKE)
       │
       ▼
SHAP Feature Attribution (Top 5 FAKE & REAL evidence features)
       │
       ▼
Structured JSON Response
```

---

## 2. Model & Preprocessing Artifacts

- **Trained Model Artifact**: [`data/videos/models/best_model.joblib`](file:///c:/work/explainable-ai-media-detection/data/videos/models/best_model.joblib) (`HistGradientBoostingClassifier`)
- **Fitted Scaler Artifact**: [`data/videos/models/scaler.joblib`](file:///c:/work/explainable-ai-media-detection/data/videos/models/scaler.joblib) (`StandardScaler` fitted **exclusively on training data**, `n_features_in_ = 216`)

---

## 3. CLI Usage

Run prediction on any target MP4 video file:

```bash
python -m src.video.predict_video Dataset/Video/original/000.mp4 --frames 10
```

---

## 4. API Service Abstraction

For REST API and backend integration (e.g. FastAPI / Flask), use `VideoInferenceService` from [`src/video/inference_pipeline.py`](file:///c:/work/explainable-ai-media-detection/src/video/inference_pipeline.py):

```python
from src.video.inference_pipeline import VideoInferenceService

service = VideoInferenceService(
    model_path="data/videos/models/best_model.joblib",
    scaler_path="data/videos/models/scaler.joblib"
)

result = service.analyze("Dataset/Video/original/000.mp4", n_frames=10)
```

---

## 5. Sample JSON Output Contract

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

## 6. Research Prototype Disclaimer & Benchmark Limitations

> [!IMPORTANT]
> **CRITICAL RESEARCH DISCLAIMER**
> 
> - **Current Benchmark Performance**: Full dataset benchmark validation on Stage B/C dataset achieved **Balanced Accuracy = 50.00%** and **ROC-AUC = 53.43%**, with the baseline classifier predicting `FAKE` for most test samples due to 1:6 class imbalance.
> - **Non-Production Prototype**: This system is a research prototype evaluating explainable traditional machine learning. Decision probabilities MUST NOT be presented as ground-truth certainty. The upcoming Phase 12 web application will clearly display benchmark performance limitations to the user.
