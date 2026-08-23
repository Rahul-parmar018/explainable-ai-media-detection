# Phase 13A — Face-Region Forensic Feature Pipeline

## Motivation

Phase 12B established that the existing 216 whole-frame traditional visual features
(brightness, contrast, HSV stats, LBP, GLCM, DCT, edge density) provide insufficient
discriminative power for deepfake detection on the FaceForensics++ dataset.
The key hypothesis for Phase 13A is:

> **"Traditional forensic features extracted from the manipulated facial region should
> provide stronger deepfake discrimination than whole-frame statistics."**

The rationale: deepfake manipulation occurs at the face level. Blending boundaries,
chroma mismatches, noise residual differences, and compression block artifacts are
localized to the face crop. Whole-frame statistics dilute these signals with
scene-level background content.

---

## Face Detection Method

### Primary: MediaPipe FaceDetection (model_selection=0)
- **Library**: MediaPipe 0.10.9 (already installed)
- **Model**: Short-range detection (≤2m), optimised for close-up video frames
- **Confidence threshold**: 0.5
- **Multi-face strategy**: Select largest bounding box by area (deterministic)

### Fallback: OpenCV Haar Cascade
- `haarcascade_frontalface_default.xml` (bundled with OpenCV 4.8.1)
- Used only when MediaPipe returns no detections
- Same largest-face selection strategy

### Crop Strategy
1. Detect primary face bounding box (raw pixels)
2. Expand bounding box by **20% on each side** to include facial boundary context
3. Clamp to image bounds (handles partial/edge faces)
4. Crop + resize to **128×128** (deterministic, consistent across all videos)
5. Convert to BGR, Grayscale, and HSV representations

### Resilience
- No face detected → return `detected=False`, skip frame gracefully
- Multiple faces → largest area selected
- Empty/invalid frames → immediate `detected=False` return
- Minimum **30% of frames** must have detected faces for a video to produce features
- Never crashes the entire video because one frame has no face

---

## Feature Families

### A. HSV Color (30 features)
```
face_h_mean, face_h_std       — Hue mean and std
face_s_mean, face_s_std       — Saturation mean and std
face_v_mean, face_v_std       — Value mean and std
face_h_hist_{0..7}            — 8-bin hue histogram
face_s_hist_{0..7}            — 8-bin saturation histogram
face_v_hist_{0..7}            — 8-bin value histogram
```

### B. Gray + LBP Texture (12 features)
```
face_gray_mean, face_gray_std — Grayscale statistics
face_lbp_hist_{0..9}          — Uniform LBP histogram (P=8, R=1)
```

### C. GLCM Texture (8 features)
```
face_glcm_contrast_d1         — GLCM contrast at distance 1
face_glcm_homogeneity_d1      — GLCM homogeneity at distance 1
face_glcm_energy_d1           — GLCM energy at distance 1
face_glcm_correlation_d1      — GLCM correlation at distance 1
face_glcm_contrast_d2         (distance 2 variants)
face_glcm_homogeneity_d2
face_glcm_energy_d2
face_glcm_correlation_d2
```
Multi-distance GLCM captures both fine and coarse texture co-occurrence.

### D. Frequency / DCT (5 features)
```
face_dct_low_energy           — Low-frequency energy (DC + low AC)
face_dct_mid_energy           — Mid-frequency energy
face_dct_high_energy          — High-frequency energy
face_dct_high_low_ratio       — High/low frequency ratio
face_dct_dc                   — DC component energy
```
Thresholds are face-size-aware (relative to 128×128 crop).

### E. Edge and Sharpness (7 features)
```
face_edge_density             — Canny edge pixel fraction
face_edge_mean, face_edge_std — Canny edge image statistics
face_laplacian_var            — Laplacian variance (focus/blur measure)
face_laplacian_mean           — Mean absolute Laplacian
face_sobel_mean, face_sobel_std — Sobel gradient magnitude statistics
```
Laplacian variance is a classic sharpness/blur measure sensitive to deepfake
upsampling artifacts.

### F. Forensic Features (10 features — new)
```
face_noise_mean               — Mean absolute noise residual (median filter diff)
face_noise_std                — Noise residual standard deviation
face_noise_energy             — Noise residual energy
face_chroma_cr_mean           — YCrCb Cr (red chroma) mean
face_chroma_cr_std            — YCrCb Cr standard deviation
face_chroma_cb_mean           — YCrCb Cb (blue chroma) mean
face_chroma_cb_std            — YCrCb Cb standard deviation
face_block_artifact           — 8×8 DCT block boundary discontinuity
face_hf_power_ratio           — FFT high-frequency to low-frequency power ratio
face_chroma_spread            — Coefficient of variation of channel stds
```

**Why forensic features matter:**
- **Noise residual**: GAN-generated faces have characteristically different noise patterns from camera sensor noise
- **Chroma channels**: Deepfakes often have chroma inconsistencies at blending boundaries
- **Block artifact**: Compression block structure differs between authentic and reconstructed faces
- **HF power ratio**: High-frequency Fourier content reveals upsampling and blending artifacts

---

## Per-Frame Summary

| Family | Count |
|---|---|
| A. HSV Color | 30 |
| B. Gray + LBP | 12 |
| C. GLCM | 8 |
| D. DCT | 5 |
| E. Edge | 7 |
| F. Forensic | 10 |
| **Total per frame** | **72** |

---

## Temporal Aggregation

For each video: 10 frames uniformly sampled → detect face → extract 72 features.

Aggregate over detected-face frames using:
- Mean
- Standard deviation
- Minimum
- Maximum

**72 × 4 = 288 video-level features** per video.

The aggregated std features capture temporal inconsistency — a deepfake's face blending
quality may vary across frames, producing higher temporal variance in forensic features.

---

## Dataset Generation

- **Source**: Exactly the same 700 Stage B videos (100 per category)
- **Labels**: Preserved exactly (REAL=0, FAKE=1)
- **Metadata**: video_path, video_name, category, source_id, label
- **Added**: frames_sampled, faces_detected, face_detection_rate
- **Output**: `data/videos/features/video_face_features.csv`
- **Resumable**: Re-running skips already-processed paths

Minimum detection rate threshold: **30%** (at least 3 frames must have a detected face).
Videos below this threshold are recorded as errors, not included in the features CSV.

---

## Leakage Prevention

The **connected-component actor grouping** strategy from Phase 10C is reused exactly:
- All video pairs sharing an actor (source or target) are grouped into the same component
- Split at component level: 70% Train / 15% Validation / 15% Test
- **Zero connected-component overlap** verified between all three partitions
- StandardScaler fitted **only** on the training partition

---

## Model Configurations

Same families as Phase 12B correction experiment, applied to the new 288 features:

| Model | Strategies |
|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10} × {full_class_weight, undersample} |
| Linear SVM | C ∈ {0.01, 0.1, 1, 10} × {full_class_weight, undersample} |
| RBF SVM | C ∈ {1, 10}, gamma=scale × {full_class_weight, undersample} |
| Random Forest | depth ∈ {None, 10, 20} × {full_class_weight, undersample} |
| Extra Trees | depth ∈ {None, 10, 20} × {full_class_weight, undersample} |
| HistGradientBoosting | lr ∈ {0.1, 0.05}, iter ∈ {100, 200} × {full_sample_weight, undersample_sw} |

**Imbalance handling:**
- `class_weight="balanced"` for all models that support it natively (LR, SVM, RF, ET)
- Explicit `sample_weight` for HistGradientBoosting (sklearn < 1.2 workaround)
- Balanced undersampling (1:1 REAL:FAKE) as alternative strategy

**Threshold optimisation:**
- Sweep 0.10 → 0.90 on validation set only
- Select threshold maximising validation Balanced Accuracy
- Never touch the test set during selection

---

## Diagnostic Rules (same as Phase 12B)

A model is flagged with WARNING if:
- REAL recall < 0.50, OR
- FAKE recall < 0.50, OR
- Balanced Accuracy ≤ 0.55

A model that collapses to one class is disqualified from selection.

---

## Face Detection Audit Results

*(Populated after running `face_detection_audit.py`)*

| Category | n_videos | avg_detection_rate | avg_face_area_ratio |
|---|---|---|---|
| original | 10 | **100.0%** | 8.12% |
| Deepfakes | 10 | **100.0%** | 5.81% |
| Face2Face | 10 | **100.0%** | 7.31% |
| FaceShifter | 10 | **100.0%** | 7.43% |
| FaceSwap | 10 | **100.0%** | 6.38% |
| NeuralTextures | 10 | **100.0%** | 6.97% |
| DeepFakeDetection | 10 | **99.0%** | 3.18% |
| **Overall** | **70** | **99.86%** | — |

> [!NOTE]
> Detection quality is excellent. All categories achieved ≥99% face detection rate across
> 70 videos × 10 frames = 700 frames tested. Proceeding to full 700-video extraction.

Full audit: [`face_detection_audit.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/face_detection_audit.csv)

---

## Validation Results

*(Populated after running `face_model.py`)*

Full comparison: [`face_model_comparison.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/face_model_comparison.csv)

---

## Test Results

*(Populated after running `face_model.py`)*

Full results: [`face_test_results.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/face_test_results.csv)

---

## Whole-Frame vs Face-Region Comparison

*(Populated after running `face_model.py`)*

| Method | Test Bal Acc | REAL Recall | FAKE Recall | ROC-AUC |
|---|---|---|---|---|
| Whole Frame 216 features | 50.00% | 0.00% | 100.00% | 53.43% |
| Face Region 288 features | *TBD* | *TBD* | *TBD* | *TBD* |

Full comparison: [`face_vs_fullframe_comparison.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/face_vs_fullframe_comparison.csv)

---

## Artifacts

| Artifact | Path | Description |
|---|---|---|
| Face feature CSV | `data/videos/features/video_face_features.csv` | 700 videos × 288 features |
| Best model | `data/videos/models/face_best_model.joblib` | Best face-region classifier |
| Scaler | `data/videos/models/face_scaler.joblib` | Fitted on train only (288-feature) |
| Threshold | `data/videos/models/face_threshold.json` | Validation-optimised threshold |

> [!CAUTION]
> The original production model (`best_model.joblib`) has NOT been modified.
> `deployment_approved: false` in `face_threshold.json`.

---

## Limitations

1. **MediaPipe short-range detection** may miss faces in profile shots or extreme camera angles
2. **Minimum detection rate** of 30% may still allow noisy samples to enter the feature set
3. **Static 10-frame sampling** ignores temporal ordering — frame delta features require ordered sampling
4. **128×128 crop** may lose fine-grained texture at the blending boundary
5. **FaceForensics++ video compression** (c23/c40) may suppress the high-frequency forensic signals

---

## Research Disclaimer

> This system is an academic research tool. The Phase 12B whole-frame baseline achieved
> Balanced Accuracy = 50.00% with REAL recall = 0.00%. Phase 13A investigates whether
> face-region forensic features overcome this limitation. Results are not guaranteed to
> show improvement and findings are reported transparently regardless of outcome.

---

## Verdict

*(Populated after full evaluation)*

```
PHASE 13A: PASSED — FACE FEATURES IMPROVED DETECTION
```
or
```
PHASE 13A: FAILED — FACE FEATURES DID NOT PROVIDE SUFFICIENT SIGNAL
```
