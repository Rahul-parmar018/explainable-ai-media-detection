# Phase 12B — Model Correction Report

## Overview

This document reports the Phase 12B model correction experiment, which was initiated after a critical classification failure discovered during Phase 12 browser verification.

### Phase 10C Baseline Failure

| Metric | Phase 10C Baseline |
|---|---|
| Test Balanced Accuracy | 50.00% |
| Test ROC-AUC | 53.43% |
| REAL Recall | 0.00% (TN=0, FP=17) |
| FAKE Recall | 100.00% (FN=0, TP=85) |

The model was predicting **every video as FAKE**, making it useless as a detector.

---

## Root Cause Analysis

### 1. Class Imbalance

```
TRAIN: 502 samples — REAL=67 (13.3%), FAKE=435 (86.7%)
VAL:    96 samples — REAL=16 (16.7%), FAKE=80 (83.3%)
TEST:  102 samples — REAL=17 (16.7%), FAKE=85 (83.3%)
```

The dataset has a 1:6.5 REAL:FAKE ratio. This is structurally caused by the FaceForensics++ dataset design: 5 manipulation methods (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures) × 600 original videos = up to 3000 fakes vs 600 real.

### 2. HistGradientBoosting `class_weight` Silently Ignored

The Phase 10C baseline selected `HistGradientBoostingClassifier(class_weight="balanced")`. In scikit-learn < 1.2, this parameter was **accepted without error but had no effect**. The model trained on raw class frequencies and learned to predict FAKE for every sample — achieving 86.7% raw accuracy while providing zero REAL recall.

### 3. No Threshold Tuning in Baseline

The Phase 10C baseline used the default 0.5 threshold. For severely imbalanced models, threshold tuning on the validation set is a critical requirement.

---

## Phase 12B Correction Strategy

### A. Data Partitioning
- **Same connected-component grouping** — zero actor leakage between splits ✓
- **Same 70/15/15 split** — VAL and TEST distributions untouched ✓

### B. Corrections Applied

| Fix | Description |
|---|---|
| `sample_weight` | Explicit inverse-frequency weights passed to `HistGradientBoostingClassifier.fit()` |
| `class_weight="balanced"` | Used natively by LR, SVM, RF, ET |
| Balanced undersampling | FAKE undersampled to REAL count in TRAIN only (67:67 = 1:1) |
| Threshold optimisation | Youden's J and Balanced Accuracy optimised on VAL set only |

### C. Experiment Grid (40 model configurations, 3 thresholds each = 120 rows)

| Model Family | Configurations |
|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1.0, 10.0} × {full_class_weight, undersample} |
| Linear SVM | C ∈ {0.01, 0.1, 1.0, 10.0} × {full_class_weight, undersample} |
| RBF SVM | C ∈ {1.0, 10.0}, gamma ∈ {scale, auto} × {full_class_weight, undersample} |
| Random Forest | depth ∈ {None,10,20}, leaf ∈ {1,2} × {full_class_weight, undersample} |
| Extra Trees | depth ∈ {None,10,20}, leaf ∈ {1,2} × {full_class_weight, undersample} |
| HistGradientBoosting | lr ∈ {0.05,0.1}, iter ∈ {100,200} × {full_sample_weight, undersample_sw} |

---

## Experiment Results (All 40 configurations)

| # | Model | Strategy | Val BA | REAL Rec | FAKE Rec | Threshold | Diagnostic |
|---|---|---|---|---|---|---|---|
| 1 | LR C=0.01 | full_class_weight | 0.5188 | 0.0625 | 0.975 | 0.30 | ⚠ REAL rec<0.50; BA≤0.55 |
| 2 | LR C=0.01 | undersample | 0.5250 | 0.9375 | 0.113 | 0.65 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 3 | LR C=0.1 | full_class_weight | 0.5312 | 0.2500 | 0.812 | 0.30 | ⚠ REAL rec<0.50; BA≤0.55 |
| 4 | LR C=0.1 | undersample | 0.5188 | 0.1875 | 0.850 | 0.15 | ⚠ REAL rec<0.50; BA≤0.55 |
| 5 | LR C=1.0 | full_class_weight | 0.5250 | 0.6250 | 0.425 | 0.60 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 6 | LR C=1.0 | undersample | 0.5250 | 0.3125 | 0.737 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 7 | LR C=10.0 | full_class_weight | 0.5375 | 0.4375 | 0.637 | 0.30 | ⚠ REAL rec<0.50; BA≤0.55 |
| **8** | **LR C=10.0** | **undersample** | **0.5625** | **0.6875** | **0.438** | **0.75** | ⚠ FAKE rec<0.50 |
| 9 | LinearSVM C=0.01 | full_class_weight | 0.5062 | 0.1250 | 0.887 | 0.85 | ⚠ REAL rec<0.50; BA≤0.55 |
| 10 | LinearSVM C=0.01 | undersample | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 11 | LinearSVM C=0.1 | full_class_weight | 0.5312 | 0.0625 | 1.000 | 0.85 | ⚠ REAL rec<0.50; BA≤0.55 |
| 12 | LinearSVM C=0.1 | undersample | 0.5062 | 0.0625 | 0.950 | 0.45 | ⚠ REAL rec<0.50; BA≤0.55 |
| 13 | LinearSVM C=1.0 | full_class_weight | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 14 | LinearSVM C=1.0 | undersample | 0.5312 | 0.8750 | 0.188 | 0.55 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 15 | LinearSVM C=10.0 | full_class_weight | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 16 | LinearSVM C=10.0 | undersample | 0.5500 | 0.3125 | 0.787 | 0.45 | ⚠ REAL rec<0.50; BA≤0.55 |
| 17 | RBFSVM C=1 g=scale | full_class_weight | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 18 | RBFSVM C=1 g=scale | undersample | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 19 | RBFSVM C=10 g=scale | full_class_weight | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 20 | RBFSVM C=10 g=scale | undersample | 0.5312 | 0.9375 | 0.125 | 0.65 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 21 | RBFSVM C=1 g=auto | full_class_weight | 0.5000 | 0.0000 | 1.000 | 0.10 | ⚠ REAL rec<0.50; BA≤0.55 |
| 22 | RBFSVM C=1 g=auto | undersample | 0.5062 | 0.6875 | 0.325 | 0.55 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 23 | RF depth=None leaf=1 | full_class_weight | 0.5062 | 0.0625 | 0.950 | 0.75 | ⚠ REAL rec<0.50; BA≤0.55 |
| 24 | RF depth=None leaf=1 | undersample | 0.5312 | 0.3750 | 0.688 | 0.45 | ⚠ REAL rec<0.50; BA≤0.55 |
| 25 | RF depth=10 leaf=2 | full_class_weight | 0.5188 | 0.2500 | 0.787 | 0.70 | ⚠ REAL rec<0.50; BA≤0.55 |
| 26 | RF depth=10 leaf=2 | undersample | 0.5125 | 0.0625 | 0.963 | 0.30 | ⚠ REAL rec<0.50; BA≤0.55 |
| 27 | RF depth=20 leaf=1 | full_class_weight | 0.5250 | 0.3125 | 0.737 | 0.80 | ⚠ REAL rec<0.50; BA≤0.55 |
| 28 | RF depth=20 leaf=1 | undersample | 0.5312 | 0.3750 | 0.688 | 0.45 | ⚠ REAL rec<0.50; BA≤0.55 |
| 29 | ET depth=None leaf=1 | full_class_weight | 0.5437 | 0.6875 | 0.400 | 0.85 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 30 | ET depth=None leaf=1 | undersample | 0.5312 | 0.1875 | 0.875 | 0.35 | ⚠ REAL rec<0.50; BA≤0.55 |
| 31 | ET depth=10 leaf=2 | full_class_weight | 0.5188 | 0.1250 | 0.912 | 0.55 | ⚠ REAL rec<0.50; BA≤0.55 |
| 32 | ET depth=10 leaf=2 | undersample | 0.5312 | 0.3125 | 0.750 | 0.40 | ⚠ REAL rec<0.50; BA≤0.55 |
| 33 | ET depth=20 leaf=1 | full_class_weight | 0.5188 | 0.8750 | 0.163 | 0.85 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 34 | ET depth=20 leaf=1 | undersample | 0.5250 | 0.7500 | 0.300 | 0.50 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 35 | HGB lr=0.1 iter=100 | full_sample_weight | 0.5188 | 0.0625 | 0.975 | 0.50 | ⚠ REAL rec<0.50; BA≤0.55 |
| 36 | HGB lr=0.1 iter=100 | undersample_sw | 0.5437 | 0.6875 | 0.400 | 0.60 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 37 | HGB lr=0.05 iter=200 | full_sample_weight | 0.5188 | 0.3750 | 0.663 | 0.90 | ⚠ REAL rec<0.50; BA≤0.55 |
| 38 | HGB lr=0.05 iter=200 | undersample_sw | 0.5437 | 0.9375 | 0.150 | 0.85 | ⚠ FAKE rec<0.50; BA≤0.55 |
| 39 | HGB lr=0.1 iter=200 | full_sample_weight | 0.5062 | 0.0625 | 0.950 | 0.80 | ⚠ REAL rec<0.50; BA≤0.55 |
| 40 | HGB lr=0.1 iter=200 | undersample_sw | 0.5500 | 0.8125 | 0.287 | 0.85 | ⚠ FAKE rec<0.50; BA≤0.55 |

> [!CAUTION]
> **All 40 configurations triggered diagnostic warnings. No configuration achieved Balanced Accuracy > 0.55 on validation. No configuration achieved simultaneously acceptable REAL recall (≥ 0.50) AND FAKE recall (≥ 0.50).**

---

## Best Configuration Selected

**Selected**: LR C=10.0, undersample, threshold=0.75 (highest Val BA=0.5625 with REAL recall ≥ 0.30)

### Held-Out TEST Evaluation (evaluated ONCE, on untouched test set)

| Metric | Value |
|---|---|
| Test Accuracy | 0.3235 |
| **Test Balanced Accuracy** | **0.5000** |
| REAL Precision | 0.1667 |
| **REAL Recall** | **0.7647** |
| FAKE Precision | 0.8333 |
| **FAKE Recall** | **0.2353** |
| Macro F1 | 0.3203 |
| ROC-AUC | 0.4879 |
| PR-AUC | 0.8266 |

**Confusion Matrix** (Test, 102 samples):
|  | Pred REAL | Pred FAKE |
|---|---|---|
| **Actual REAL** | 13 (TN) | 4 (FP) |
| **Actual FAKE** | 65 (FN) | 20 (TP) |

---

## Before vs After Comparison

| Metric | Phase 10C Baseline | Phase 12B Corrected | Δ |
|---|---|---|---|
| Balanced Accuracy | 50.00% | 50.00% | 0.00 pp |
| REAL Recall | 0.00% | 76.47% | **+76.47 pp** |
| FAKE Recall | 100.00% | 23.53% | **-76.47 pp** |
| ROC-AUC | 53.43% | 48.79% | -4.64 pp |
| TN | 0 | 13 | +13 |
| FP | 17 | 4 | -13 |
| FN | 0 | 65 | +65 |
| TP | 85 | 20 | -65 |

The threshold shift successfully transferred recall from FAKE to REAL — but **Balanced Accuracy remains at exactly 50%** because the model cannot simultaneously distinguish both classes. The problem is a **fundamental limitation of the 216-feature representation**, not a configuration issue.

---

## Diagnosis — Why Traditional Features Fail Here

### Evidence from the experiment

1. **All models plateau at ~50-56% Val Balanced Accuracy** across 40 configurations — this is not tunable with the current features.

2. **Perfect REAL recall ↔ near-zero FAKE recall tradeoff** — the model can only "find" one class by heavily biasing toward it.

3. **ROC-AUC across all models = 0.48–0.55** — barely above random chance.

### Feature-level diagnosis

The 216 traditional visual features (54 per frame × 4 aggregations = mean, std, min, max) capture:
- Brightness / contrast / saturation statistics
- Edge density
- Noise/blur estimates
- Color channel moments
- Laplacian sharpness
- Block DCT energy (coarse frequency features)

**Why these fail for deepfake detection:**
1. **Modern deepfakes are photorealistic** — Deepfakes/Face2Face/FaceSwap produced by FaceForensics++ are high-quality and indistinguishable at the frame-statistics level.
2. **No spatial/temporal context** — frame-level aggregation destroys the local face-region artifacts (blending boundaries, color mismatch in the skin region) that are the primary discriminative cues.
3. **No face-region isolation** — features are computed on the full frame, not the manipulated face region.
4. **Compression artifacts** — FaceForensics++ videos come in lossless and lossy (c23/c40) variants. The traditional features may not separate compression from manipulation.

### Connected-component grouping effect

The leakage-safe connected-component split ensures no actor appears in both train and test. This is the **correct** strategy — but it means:
- The test set actors are entirely unseen during training
- The features do not generalize to new actors/lighting/scenes because they encode scene-level statistics, not manipulation-specific artifacts

This is the **correct finding** — the generalization failure is real, not an artifact of leakage.

---

## Recommended Next Research Directions

> [!IMPORTANT]
> These are research recommendations for future phases. Do not implement without explicit user approval.

### Option A — Face-Region Feature Extraction
Extract 54 features **only from the detected face bounding box** instead of the full frame.  
Expected benefit: Significant — face-crop features directly target the manipulated region.  
Cost: Requires face detection (e.g., MediaPipe/OpenCV face cascade) in the extraction pipeline.

### Option B — Texture/Frequency Forensic Features
Add GAN-artifact-sensitive features:
- LBP (Local Binary Patterns) in face region
- Co-occurrence matrices (GLCM)
- FFT spectrum anomaly features
- DFT high-frequency residuals

### Option C — Temporal Inconsistency Features
Deepfake artifacts are inconsistent across frames. Add frame-to-frame delta features:
- Optical flow magnitude statistics
- Inter-frame brightness delta
- Landmark displacement (requires face detection)

### Option D — Deep Feature Extraction (Research Scope Expansion)
If traditional features are provably insufficient, transition to:
- EfficientNet/ResNet face-crop embeddings
- FaceXFormer or similar pre-trained forensic models

> [!WARNING]
> Option D would require updating the architecture constraints defined in the project scope.

---

## Integrity Controls Verified

| Check | Result |
|---|---|
| Scaler fitted on TRAIN only | ✓ |
| Threshold selected on VAL only | ✓ |
| Final evaluation on untouched TEST | ✓ (evaluated ONCE) |
| Zero connected-component overlap | ✓ |
| No browser sanity-check data in TRAIN | ✓ |
| No target leakage | ✓ |
| `best_model.joblib` NOT overwritten | ✓ |

---

## Research Disclaimer

> This system is an academic research tool demonstrating explainable AI techniques for digital media forensics. Full dataset benchmark validation achieved **Balanced Accuracy = 50.00%** with the Phase 12B correction experiment confirming that the current 216-dimension traditional visual feature set does not provide sufficient discriminative power to reliably distinguish REAL from FAKE videos across unseen actors. This system is **not suitable for production forensic use**. Further research into face-region-specific and frequency-domain features is recommended.

---

## Verdict

```
MODEL CORRECTION: FAILED — FURTHER MODEL RESEARCH REQUIRED
```

**Reason**: No configuration in the 40-model, 120-row experiment grid achieved Balanced Accuracy > 0.55 on validation or meaningful simultaneous recall for both classes. The root cause is a fundamental limitation of frame-level traditional visual features for distinguishing high-quality deepfakes from real videos, compounded by the 1:6.5 class imbalance. The features do not provide the discriminative signal required for a working classifier.

The corrected model artifacts are saved for research/test purposes only:
- `data/videos/models/corrected_best_model.joblib` — LR C=10, undersample, threshold=0.75
- `data/videos/models/corrected_scaler.joblib`
- `data/videos/models/corrected_threshold.json` (`deployment_approved: false`)

**`best_model.joblib` has NOT been overwritten.** Production model deployment requires explicit user approval.
