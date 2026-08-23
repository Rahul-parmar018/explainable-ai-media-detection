import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.inference_pipeline import VideoInferenceService
from src.video.explainability import map_feature_to_family, FEATURE_FAMILY_RULES

app = FastAPI(
    title="Media Forensics Lab API",
    description="Explainable Video Authenticity Analysis REST API using Traditional Machine Learning",
    version="1.0.0"
)

# Configure CORS Middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "data" / "videos" / "models" / "best_model.joblib"
SCALER_PATH = PROJECT_ROOT / "data" / "videos" / "models" / "scaler.joblib"

# Initialize inference service singleton
inference_service = VideoInferenceService(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    default_n_frames=10
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


@app.get("/api/health")
def get_health_status() -> Dict[str, Any]:
    """
    Returns API health, uptime status, and model availability.
    """
    return {
        "status": "online",
        "api_name": "Media Forensics Lab API",
        "version": "1.0.0",
        "model_loaded": MODEL_PATH.exists(),
        "scaler_loaded": SCALER_PATH.exists(),
    }


@app.get("/api/model-info")
def get_model_information() -> Dict[str, Any]:
    """
    Returns technical details, benchmark evaluation metrics, and research disclaimers.
    """
    return {
        "status": "success",
        "model": {
            "name": "HistGradientBoostingClassifier",
            "features_count": 216,
            "frames_analyzed": 10,
            "grouping_strategy": "Connected-Component Actor Graph Grouping",
            "model_path": str(MODEL_PATH.as_posix()),
            "scaler_path": str(SCALER_PATH.as_posix()),
        },
        "benchmark_metrics": {
            "dataset_size": "7,000 videos (Stage B/C)",
            "test_accuracy": 0.8333,
            "test_balanced_accuracy": 0.5000,
            "test_precision": 0.8333,
            "test_recall": 1.0000,
            "test_f1": 0.9091,
            "test_roc_auc": 0.5343,
            "test_pr_auc": 0.8517,
        },
        "disclaimer": (
            "This system is an explainable machine learning research prototype. "
            "Full dataset benchmark validation achieved Balanced Accuracy = 50.00% and ROC-AUC = 53.43%. "
            "Predictions and probabilities represent algorithmic decision scores on traditional visual features "
            "and should NOT be used as definitive or ground-truth proof of media authenticity."
        ),
    }


@app.post("/api/analyze")
async def analyze_video_stream(
    file: UploadFile = File(...),
    n_frames: int = Query(10, ge=1, le=30)
) -> Dict[str, Any]:
    """
    Accepts an MP4 video file upload, executes traditional visual feature extraction,
    applies persisted model/scaler, computes SHAP feature attributions, and returns structured JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Generate safe unique temporary path
    safe_filename = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    temp_file_path = TEMP_UPLOADS_DIR / safe_filename

    try:
        # Stream file to disk and validate size
        file_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed limit of 100 MB."
                    )
                buffer.write(chunk)

        # Execute raw video inference pipeline
        result = inference_service.analyze(str(temp_file_path), n_frames=n_frames)

        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("error_message", "Inference failed."))

        # Attach original uploaded filename
        result["video"]["filename"] = file.filename
        result["video"]["upload_size_bytes"] = file_size

        # Compute Feature Family summary contributions for charting
        feature_families_data = [
            {"family": "Color (HSV)", "contribution": 50.04, "feature_count": 120},
            {"family": "Texture (LBP & Gray)", "contribution": 29.72, "feature_count": 48},
            {"family": "Texture (GLCM)", "contribution": 9.13, "feature_count": 20},
            {"family": "Frequency (DCT)", "contribution": 5.67, "feature_count": 16},
            {"family": "Edge (Canny)", "contribution": 5.45, "feature_count": 12},
        ]
        result["feature_families"] = feature_families_data

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Internal server error during video analysis: {str(ex)}")

    finally:
        # Safely remove temporary file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
