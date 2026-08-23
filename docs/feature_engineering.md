# Feature Engineering Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## Overview

This document describes the design rationale, parameters, and research objectives for the traditional visual feature set extracted from video frames. Every feature family is chosen for its mathematical explainability and diagnostic value in distinguishing pristine (REAL) from manipulated or synthetic (FAKE) media.

---

## Feature Families & Rationale

| Feature Family | Image Space | Feature Count | Target Artifact / Research Purpose |
| :--- | :--- | :---: | :--- |
| **Color (HSV)** | HSV | 30 | Captures global color distribution, saturation anomalies, and brightness distortions common in generative video models. |
| **Texture (LBP)** | Grayscale | 12 | Detects micro-texture irregularities, pixel local structure, and unnatural facial surface smoothing using uniform Local Binary Patterns. |
| **Edge (Canny)** | Grayscale | 3 | Quantifies structural sharpness and boundary density via Canny edge detection. Deepfake blending often introduces edge blurring or unnatural sharp boundaries. |
| **Texture (GLCM)** | Grayscale | 5 | Second-order statistical texture properties (contrast, dissimilarity, homogeneity, energy, correlation) measuring spatial pixel relationship irregularities. |
| **Frequency (DCT)** | Grayscale | 4 | 2D Discrete Cosine Transform energy distribution across low, mid, and high frequency bands; detects high-frequency compression and synthesis artifacts. |

**Total Features per Frame**: **54**

---

## Key Parameter Choices

1. **Color (HSV)**:
   - Resized $256 \times 256$ frames converted to HSV.
   - 8-bin normalized histograms for Hue ($[0, 180]$), Saturation ($[0, 256]$), and Value ($[0, 256]$).

2. **LBP (Local Binary Pattern)**:
   - Uniform LBP configuration: $P = 8$ neighbors, radius $R = 1$.
   - Yields 10 uniform pattern histogram bins + grayscale mean and standard deviation.

3. **Canny Edge Detection**:
   - Hysteresis thresholds: $T_{\text{low}} = 100$, $T_{\text{high}} = 200$.
   - Computes `edge_density` ($\frac{\text{edge pixels}}{\text{total pixels}}$), `edge_mean`, and `edge_std`.

4. **GLCM (Gray-Level Co-occurrence Matrix)**:
   - Quantized to 64 gray levels ($256 / 4$) for computational efficiency.
   - Distance: $d = 1$ pixel.
   - Directional angles: $\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$ ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
   - Standardized output: Mean across angles for `contrast`, `dissimilarity`, `homogeneity`, `energy`, and `correlation`.

5. **DCT (2D Discrete Cosine Transform)**:
   - Applied via 2D DCT (`cv2.dct`) on float32 grayscale frames.
   - Frequency bands partitioned by spectral coordinate radius $r = u + v$:
     - Low frequency: $r < 32$
     - Mid frequency: $32 \le r < 128$
     - High frequency: $r \ge 128$
   - Computes mean spectral energy per band and `dct_high_low_ratio`.

---

## Temporal Feature Aggregation Strategy

To summarize temporal frame sequences into a single 216-dimensional video feature vector ($54 \text{ frame features} \times 4 \text{ statistics} = 216$), 4 statistical measures are computed across sampled frames:

1. **Mean ($\mu$)**: Captures central tendency of visual characteristics across the video.
2. **Standard Deviation ($\sigma$)**: Measures temporal volatility, flickering, and frame-to-frame synthesis inconsistency.
3. **Minimum ($\text{Min}$)**: Identifies extreme low bound artifacts (e.g. temporary loss of edge detail or extreme color drops).
4. **Maximum ($\text{Max}$)**: Identifies extreme high bound artifacts (e.g. transient high-frequency noise spikes or boundary artifacts).
