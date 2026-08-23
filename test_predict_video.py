import os
import sys
import math
from pathlib import Path
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.video.predict_video import predict_video
from src.video.inference_pipeline import VideoInferenceService

def run_predict_video_tests():
    model_path = "data/videos/models/best_model.joblib"
    scaler_path = "data/videos/models/scaler.joblib"
    sample_video = "Dataset/Video/original/000.mp4"

    print("=" * 80)
    print("PHASE 11 RAW VIDEO INFERENCE & EXPLAINABILITY TEST SUITE")
    print("=" * 80)

    # 1. Model & Scaler Artifact Checks
    mod_p = Path(model_path)
    scl_p = Path(scaler_path)

    print(f"Model File Exists:               {'PASSED' if mod_p.exists() else 'FAILED'}")
    print(f"Scaler File Exists:              {'PASSED' if scl_p.exists() else 'FAILED'}")

    model = joblib.load(mod_p)
    scaler = joblib.load(scl_p)

    print(f"Model Class Name:                {type(model).__name__}")
    print(f"Scaler Feature Dimension:        {scaler.n_features_in_} (Expected: 216)")

    # 2. Test Real MP4 Inference Pipeline
    vid_p = Path(sample_video)
    if not vid_p.exists():
        print(f"Error: Sample video {sample_video} not found!")
        return

    res = predict_video(
        video_path=sample_video,
        model_path=model_path,
        scaler_path=scaler_path,
        n_frames=10
    )

    print("\n--- SAMPLE INFERENCE RESULT SUMMARY ---")
    print(f"  Status:                        {res.get('status')}")
    print(f"  Filename:                      {res.get('video', {}).get('filename')}")
    print(f"  Frames Analyzed:               {res.get('video', {}).get('frames_analyzed')}")
    print(f"  Features Extracted:            {res.get('video', {}).get('features_extracted')}")
    print(f"  Predicted Label:               {res.get('prediction', {}).get('label')}")
    print(f"  Predicted Class ID:            {res.get('prediction', {}).get('class_id')}")
    print(f"  Fake Probability:              {res.get('prediction', {}).get('fake_probability')}")
    print(f"  Real Probability:              {res.get('prediction', {}).get('real_probability')}")
    print(f"  Explainability Available:      {res.get('explainability', {}).get('available')}")

    # 3. Test VideoInferenceService Class Abstraction
    service = VideoInferenceService(model_path=model_path, scaler_path=scaler_path)
    service_res = service.analyze(sample_video, n_frames=10)
    print(f"  Service Analyze Status:        {service_res.get('status')}")

    # 4. Test Error Handling on Missing File
    err_res = predict_video("non_existent_file.mp4", model_path=model_path, scaler_path=scaler_path)
    print(f"  Missing File Error Status:     {err_res.get('status')} ({err_res.get('error_type')})")

    all_passed = (
        mod_p.exists() and
        scl_p.exists() and
        scaler.n_features_in_ == 216 and
        res.get("status") == "success" and
        res.get("video", {}).get("features_extracted") == 216 and
        service_res.get("status") == "success" and
        err_res.get("status") == "error"
    )

    print("\n" + "=" * 80)
    print(f"RAW VIDEO INFERENCE TEST RESULT: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_predict_video_tests()
