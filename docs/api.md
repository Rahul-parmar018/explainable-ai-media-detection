# Media Forensics Lab REST API Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Base URL & Service Overview

- **Base URL**: `http://127.0.0.1:8000`
- **Framework**: FastAPI (Python 3.10+)
- **Interactive OpenAPI Specs**: `http://127.0.0.1:8000/docs`

---

## 2. API Endpoints Specification

### 1. `GET /api/health`
Returns API service status, uptime, and model/scaler artifact availability.

**Response `200 OK`**:
```json
{
  "status": "online",
  "api_name": "Media Forensics Lab API",
  "version": "1.0.0",
  "model_loaded": true,
  "scaler_loaded": true
}
```

---

### 2. `GET /api/model-info`
Returns technical details, benchmark evaluation metrics, and research disclaimers.

**Response `200 OK`**:
```json
{
  "status": "success",
  "model": {
    "name": "HistGradientBoostingClassifier",
    "features_count": 216,
    "frames_analyzed": 10,
    "grouping_strategy": "Connected-Component Actor Graph Grouping",
    "model_path": "c:/work/explainable-ai-media-detection/data/videos/models/best_model.joblib",
    "scaler_path": "c:/work/explainable-ai-media-detection/data/videos/models/scaler.joblib"
  },
  "benchmark_metrics": {
    "dataset_size": "7,000 videos (Stage B/C)",
    "test_accuracy": 0.8333,
    "test_balanced_accuracy": 0.5000,
    "test_precision": 0.8333,
    "test_recall": 1.0000,
    "test_f1": 0.9091,
    "test_roc_auc": 0.5343,
    "test_pr_auc": 0.8517
  },
  "disclaimer": "This system is an explainable machine learning research prototype. Full dataset benchmark validation achieved Balanced Accuracy = 50.00% and ROC-AUC = 53.43%. Predictions and probabilities represent algorithmic decision scores on traditional visual features and should NOT be used as definitive or ground-truth proof of media authenticity."
}
```

---

### 3. `POST /api/analyze`
Accepts an MP4 video file upload, executes frame sampling, $256 \times 256$ preprocessing, 54 traditional visual feature extraction per frame, 216-feature temporal aggregation, model inference via persisted scaler, SHAP feature attributions, and returns structured JSON.

**Query Parameters**:
- `n_frames` (optional `int`, default: `10`): Number of frames to sample.

**Request Body**: `multipart/form-data`
- `file`: MP4, MOV, AVI, or WEBM video file (max 100 MB).

**Response `200 OK`**:
```json
{
  "status": "success",
  "video": {
    "filename": "000.mp4",
    "video_path": "c:/work/explainable-ai-media-detection/temp_uploads/uuid_000.mp4",
    "total_frames_in_stream": 396,
    "frames_analyzed": 10,
    "features_extracted": 216,
    "upload_size_bytes": 10485760
  },
  "prediction": {
    "label": "FAKE",
    "class_id": 1,
    "fake_probability": 0.9911,
    "real_probability": 0.0089
  },
  "explainability": {
    "available": true,
    "top_fake_evidence": [
      {
        "feature": "v_hist_2_mean",
        "feature_family": "Color (HSV)",
        "shap_value": 0.352848,
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
  "feature_families": [
    {"family": "Color (HSV)", "contribution": 50.04, "feature_count": 120},
    {"family": "Texture (LBP & Gray)", "contribution": 29.72, "feature_count": 48},
    {"family": "Texture (GLCM)", "contribution": 9.13, "feature_count": 20},
    {"family": "Frequency (DCT)", "contribution": 5.67, "feature_count": 16},
    {"family": "Edge (Canny)", "contribution": 5.45, "feature_count": 12}
  ],
  "model": {
    "name": "HistGradientBoostingClassifier",
    "model_path": "c:/work/explainable-ai-media-detection/data/videos/models/best_model.joblib",
    "scaler_path": "c:/work/explainable-ai-media-detection/data/videos/models/scaler.joblib"
  },
  "disclaimer": "This system is an explainable machine learning research prototype. Full dataset benchmark validation achieved Balanced Accuracy = 50.00% and ROC-AUC = 53.43%. Predictions and probabilities represent algorithmic decision scores on traditional visual features and should NOT be used as definitive or ground-truth proof of media authenticity."
}
```

---

## 3. Error Handling & Security

- **`400 Bad Request`**: Unsupported file extensions or missing upload payload.
- **`413 Payload Too Large`**: Upload file exceeds 100 MB size limit.
- **`422 Unprocessable Entity`**: Unreadable video stream or feature extraction error.
- **`500 Internal Server Error`**: Unexpected pipeline runtime error.
- **Temporary Upload Management**: Temporary upload streams are written to `temp_uploads/` using `uuid4` safe paths and automatically removed in the `finally` block upon completion.
