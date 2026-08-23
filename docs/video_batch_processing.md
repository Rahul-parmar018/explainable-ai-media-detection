# Video Batch Processing Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Purpose

This document outlines the controlled batch feature extraction strategy for generating tabular video-level datasets from raw video files (`Dataset/Video/`). The pipeline processes video streams sequentially without accumulating raw video frames in memory, extracting 216 explainable traditional visual features per video.

---

## 2. Stage B Dataset Size & Selection Rationale

Rather than executing an expensive extraction run across all 7,000 videos, the development pipeline utilizes **Stage B** (700 videos):

- **Composition**:
  - `original`: 100 REAL videos
  - `Deepfakes`: 100 FAKE videos
  - `Face2Face`: 100 FAKE videos
  - `FaceShifter`: 100 FAKE videos
  - `FaceSwap`: 100 FAKE videos
  - `NeuralTextures`: 100 FAKE videos
  - `DeepFakeDetection`: 100 FAKE videos
- **Total Stage B Size**: **700 Videos** (100 REAL, 600 FAKE)

### Why Start with 700 Videos Instead of 7,000?
1. **Computational & Iterative Efficiency**: Processing 700 videos (7,000 frames) takes ~3.5 minutes, enabling rapid experimentation with ML scaling, feature selection, and hyperparameter optimization before launching the full 7,000-video extraction run.
2. **Balanced Representation**: Contains equal 100-video samples across all 7 dataset categories while preserving the 1:6 REAL-to-FAKE binary ratio.

---

## 3. Source-Aware Selection & Leakage Prevention

To prevent subject/scene data leakage across categories:
1. 100 Original Source Video IDs (`000`–`999`) were selected using a fixed random seed (`random_state=42`).
2. For each selected Source ID (e.g. `006`), the original video (`original/006.mp4`) and all 5 corresponding manipulated variants (`Deepfakes/006_...`, `Face2Face/006_...`, `FaceShifter/006_...`, `FaceSwap/006_...`, `NeuralTextures/006_...`) were grouped together.
3. This ensures zero cross-category subject leakage within the Stage B dataset.

---

## 4. End-to-End Feature Extraction Pipeline

```text
Raw MP4 Video Stream
       │
       ▼
OpenCV Reader (Metadata check)
       │
       ▼
Uniform Frame Sampling (10 frames via NumPy linspace)
       │
       ▼
Frame Preprocessing (256 × 256 BGR, Grayscale, HSV)
       │
       ▼
Traditional Feature Extraction (54 features/frame)
  - Color (HSV statistics & 8-bin histograms): 30
  - Texture (Grayscale & 10-bin uniform LBP): 12
  - Edge (Canny density, mean, std): 3
  - GLCM (Contrast, dissimilarity, homogeneity, energy, correlation): 5
  - DCT (Low, mid, high frequency energy & ratio): 4
       │
       ▼
Temporal Aggregation (Mean, Std, Min, Max across 10 frames)
       │
       ▼
216-Dimensional Video Feature Vector
       │
       ▼
Output CSV (`data/videos/features/video_features_stage_b.csv`)
```

---

## 5. Output CSV Structure & Paths

- **Features CSV Path**: [`data/videos/features/video_features_stage_b.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/features/video_features_stage_b.csv)
- **Error Log CSV Path**: [`data/videos/features/video_processing_errors.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/features/video_processing_errors.csv)

### CSV Column Specification (224 total columns)
1. **Metadata Columns (8 columns)**: `video_path`, `category`, `label`, `source_id`, `frame_count`, `fps`, `width`, `height`.
2. **Aggregated Feature Columns (216 columns)**: `{feature_name}_mean`, `{feature_name}_std`, `{feature_name}_min`, `{feature_name}_max` for all 54 frame features.

---

## 6. Error Handling & Reproducibility

- **Fault Tolerance**: Videos with stream opening or frame extraction errors are logged to `video_processing_errors.csv` without halting the batch loop.
- **Memory Safety**: `cv2.VideoCapture` resources are explicitly released after processing each video. No video frames are persisted to RAM or disk.
- **Reproducibility**: Source selection uses Python `random.Random(42)` for deterministic sample replication.

---

## 7. Validation Results
- **Total Processed Rows**: 700 / 700
- **Category Counts**: 100 per category across all 7 folders
- **Binary Labels**: 100 REAL, 600 FAKE
- **Duplicate Video Paths**: 0
- **NaN / Inf Values**: 0
- **Processing Failures**: 0
