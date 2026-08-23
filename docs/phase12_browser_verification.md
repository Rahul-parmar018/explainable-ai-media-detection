# Phase 12 Browser Verification & Model Prediction Sanity Check Report

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Verification Summary & Final Verdict

| Metric | Target / Requirement | Verification Result | Status |
| :--- | :--- | :---: | :---: |
| **Backend API Service** | `http://127.0.0.1:8000/api/health` | HTTP 200 OK (`online`) | **PASS** |
| **Model Information API** | `http://127.0.0.1:8000/api/model-info` | HTTP 200 OK (`HistGradientBoostingClassifier`) | **PASS** |
| **Frontend Application** | `http://localhost:5173` | UI renders cleanly (0 console errors) | **PASS** |
| **Frontend Build** | `npm run build` | 2,376 modules transformed in 7.55s | **PASS** |
| **API vs Browser Consistency** | Python CLI vs Browser Upload | 100% Identical Output | **PASS** |
| **Known-REAL Video Classification** | `Dataset/Video/original/` (10 videos) | **0 / 10 Correct** (0.00% Recall) | **FAIL** |
| **Known-FAKE Video Classification** | 60 Manipulated Videos (6 categories) | **58 / 60 Correct** (96.67% Recall) | **PASS** |

### Final Verification Verdict
- **`PHASE 12 BROWSER VERIFICATION: FAILED`**

---

## 2. Model Prediction Sanity Check & Dataset Evaluation

70 dataset videos (10 REAL videos, 60 FAKE videos across 6 manipulation categories) were evaluated through the exact inference pipeline. Generated artifact: [`data/videos/results/browser_prediction_sanity_check.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/browser_prediction_sanity_check.csv).

### Summary Classification Performance

- **Known-real videos correctly classified**: **0 / 10** (0.00% Accuracy / Recall)
- **Known-fake videos correctly classified**: **58 / 60** (96.67% Accuracy / Recall)
- **REAL False Positive Rate (FPR)**: **100.00%** (10 / 10 predicted as FAKE with decision probabilities ranging from 0.7444 to 0.9955)

### Category-Level Classification Performance Table

| Category Folder | Ground Truth Label | Videos Evaluated | Correct Predictions | Category Accuracy | Average Fake Probability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `original` | **REAL** | 10 | **0 / 10** | **0.00%** | 0.9297 |
| `Deepfakes` | **FAKE** | 10 | **9 / 10** | **90.00%** | 0.8875 |
| `Face2Face` | **FAKE** | 10 | **10 / 10** | **100.00%** | 0.9142 |
| `FaceShifter` | **FAKE** | 10 | **10 / 10** | **100.00%** | 0.9288 |
| `FaceSwap` | **FAKE** | 10 | **9 / 10** | **90.00%** | 0.8663 |
| `NeuralTextures` | **FAKE** | 10 | **10 / 10** | **100.00%** | 0.9167 |
| `DeepFakeDetection` | **FAKE** | 10 | **10 / 10** | **100.00%** | 0.9816 |

---

## 3. Detailed Model Sanity Check & Diagnostic Findings

As requested in Step 9, a rigorous diagnostic audit was conducted to isolate why `Dataset/Video/original/000.mp4` (ground truth REAL) is predicted as `FAKE` (`fake_probability = 0.9911`):

1. **Feature Column Order Alignment**: Verified that the 216 feature names and column sequence generated during inference match the training CSV column sequence **100% identically with 0 column mismatches**.
2. **Scaler Feature Dimension & Parameter Fit**: Verified that `scaler.joblib` was fitted exclusively on the 216-dimensional training feature matrix (`scaler.n_features_in_ == 216`).
3. **Target Label Mapping**: Verified that `REAL` maps to class index `0` and `FAKE` maps to class index `1`. Verified `model.classes_` is `[0, 1]`.
4. **Probability Calculation**: Verified that `model.predict_proba()` indices `[0]` and `[1]` correspond strictly to REAL and FAKE decision probabilities respectively.
5. **Root Cause Analysis**: The failure on REAL videos is caused by **intrinsic classifier bias** arising from the 1:6 dataset imbalance (14.3% REAL vs 85.7% FAKE) in the Stage B/C training partition. `HistGradientBoostingClassifier` learned a decision boundary that collapses toward predicting `FAKE`, yielding a **50.00% Balanced Accuracy** baseline.

---

## 4. API vs Browser Consistency

Inference output on `Dataset/Video/original/000.mp4` was compared across execution interfaces:

- **Python CLI Inference**: `label = FAKE`, `fake_probability = 0.9911`, `real_probability = 0.0089`
- **FastAPI Endpoint (`POST /api/analyze`)**: `label = FAKE`, `fake_probability = 0.9911`, `real_probability = 0.0089`
- **Browser Web Application**: `label = FAKE`, `fake_probability = 0.9911`, `real_probability = 0.0089`

**Verdict**: 100% consistency across Python CLI, REST API, and Browser Frontend interfaces.

---

## 5. Frontend UI Display Verification

All 15 required frontend components were verified in the browser at `http://localhost:5173`:

- [x] Video preview HTML5 player
- [x] Filename display (`000.mp4`)
- [x] Stream frame count (`396 frames`)
- [x] Frames analyzed count (`10 frames`)
- [x] Feature count (`216 features`)
- [x] REAL / FAKE verdict banner
- [x] Real probability display (`0.89%`)
- [x] Fake probability display (`99.11%`)
- [x] Algorithmic estimate disclaimer badge
- [x] Top 5 FAKE SHAP evidence cards
- [x] Top 5 REAL SHAP evidence cards
- [x] Feature-family Recharts bar chart
- [x] Forensic feature evidence table with search filtering
- [x] Model specification modal
- [x] Prominent research benchmark disclaimer
- [x] Zero console errors / zero broken elements
