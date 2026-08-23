import os
import sys
import io
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app, TEMP_UPLOADS_DIR

client = TestClient(app)

def run_api_tests():
    print("=" * 80)
    print("FASTAPI BACKEND ENDPOINTS VALIDATION SUITE")
    print("=" * 80)

    # 1. Test /api/health
    res_health = client.get("/api/health")
    print(f"Health Status Code:              {res_health.status_code} (Expected: 200)")
    json_health = res_health.json()
    print(f"API Online Status:               {json_health.get('status')}")
    print(f"Model Artifacts Loaded:          {json_health.get('model_loaded')}")

    # 2. Test /api/model-info
    res_info = client.get("/api/model-info")
    print(f"\nModel Info Status Code:          {res_info.status_code} (Expected: 200)")
    json_info = res_info.json()
    print(f"Model Name:                      {json_info.get('model', {}).get('name')}")
    print(f"Features Count:                  {json_info.get('model', {}).get('features_count')}")
    print(f"Benchmark Balanced Accuracy:     {json_info.get('benchmark_metrics', {}).get('test_balanced_accuracy')}")

    # 3. Test invalid file extension upload
    fake_txt = io.BytesIO(b"dummy text content")
    res_invalid = client.post("/api/analyze", files={"file": ("test.txt", fake_txt, "text/plain")})
    print(f"\nInvalid Extension Status Code:   {res_invalid.status_code} (Expected: 400)")

    # 4. Test real MP4 file upload inference
    sample_mp4 = PROJECT_ROOT / "Dataset" / "Video" / "original" / "000.mp4"
    if not sample_mp4.exists():
        print(f"Warning: Sample video {sample_mp4} not found!")
        return

    with open(sample_mp4, "rb") as f:
        res_upload = client.post("/api/analyze", files={"file": ("000.mp4", f, "video/mp4")})

    print(f"\nReal MP4 Upload Status Code:     {res_upload.status_code} (Expected: 200)")
    json_up = res_upload.json()
    print(f"Inference Status:                {json_up.get('status')}")
    print(f"Predicted Label:                 {json_up.get('prediction', {}).get('label')}")
    print(f"Fake Probability:                {json_up.get('prediction', {}).get('fake_probability')}")
    print(f"Explainability Available:        {json_up.get('explainability', {}).get('available')}")
    print(f"Top FAKE Evidence Count:         {len(json_up.get('explainability', {}).get('top_fake_evidence', []))}")

    # 5. Check temporary file cleanup
    temp_files = list(TEMP_UPLOADS_DIR.glob("*"))
    print(f"\nResidual Temp Files in Uploads: {len(temp_files)} (Expected: 0)")

    all_passed = (
        res_health.status_code == 200 and
        res_info.status_code == 200 and
        res_invalid.status_code == 400 and
        res_upload.status_code == 200 and
        json_up.get("status") == "success" and
        len(temp_files) == 0
    )

    print("\n" + "=" * 80)
    print(f"FASTAPI ENDPOINTS TEST RESULT:   {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_api_tests()
