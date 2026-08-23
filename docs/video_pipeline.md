# Video Processing Pipeline Documentation

## Overview

The video processing pipeline converts input MP4 video streams into structured, explainable traditional visual feature representations.

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
```
