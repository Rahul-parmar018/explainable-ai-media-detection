# Video Processing Pipeline Documentation

## Overview

The video processing pipeline converts input MP4 video streams into structured, explainable traditional visual feature representations for binary video classification (REAL vs FAKE).

---

## Pipeline Architecture

```text
Input Video (.mp4)
      │
      ▼
1. OpenCV Video Reader
   - Metadata extraction (FPS, frame count, width, height, duration)
   - Stream validation & resource cleanup
      │
      ▼
2. Uniform Frame Sampling
   - 10 uniformly spaced frame indices generated via NumPy linspace
   - In-memory frame extraction
      │
      ▼
3. Frame Preprocessing
   - Resizing to standardized 256 × 256 resolution
   - Color space conversions (BGR, Grayscale, HSV)
      │
      ▼
4. Traditional Visual Feature Extraction (54 features/frame)
   - Color (HSV statistics & 8-bin histograms)
   - Texture (Grayscale statistics & 10-bin uniform LBP histogram)
   - Edge (Canny edge density, mean, std)
   - GLCM (Contrast, dissimilarity, homogeneity, energy, correlation)
   - DCT (Low, mid, high frequency energy & high/low ratio)
      │
      ▼
5. Temporal Feature Aggregation (216 features/video)
   - Converts 10 frame feature vectors (10 × 54 matrix) into 1 video feature vector
   - Computes 4 summary statistics per feature: Mean, Std, Min, Max
```

---

## Why Video-Level Feature Aggregation Is Required

Single frame feature vectors capture isolated spatial snapshots of a video. However, video classification models operate at the video level (predicting whether an entire video instance is REAL or FAKE). 

Temporal aggregation via 4 summary statistics ($\text{Mean}, \text{Std}, \text{Min}, \text{Max}$) compresses the 10 sampled frame feature vectors into a single static 216-dimensional feature vector ($54 \times 4 = 216$). This transformation:
1. Standardizes temporal video input into fixed tabular feature columns suitable for traditional ML classifiers (Random Forest, SVM, Logistic Regression).
2. Captures both central visual tendencies ($\text{Mean}$) and temporal volatility/consistency ($\text{Std}, \text{Min}, \text{Max}$) caused by deepfake temporal flickering or synthesis artifacts across frames.
