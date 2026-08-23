# Full 7,000-Video Dataset Feature Extraction & Resumable Batch Processing

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Overview & Dataset Scope

Phase 10A executes the complete batch feature extraction across the entire **7,000-video benchmark dataset** located in `Dataset/Video/`. The pipeline converts raw MP4 video streams into a tabular dataset containing **216 explainable traditional visual features** per video without saving intermediate video frames to disk or exhausting system memory.

### Complete Dataset Distribution

| Category / Folder Name | Manipulation Type / Method | Video Count | Binary Classification Label |
| :--- | :--- | :---: | :---: |
| `original` | Pristine Source Videos | 1,000 | **REAL** |
| `Deepfakes` | Deepfake Face Swap (Autoencoder) | 1,000 | **FAKE** |
| `Face2Face` | Facial Expression Reenactment | 1,000 | **FAKE** |
| `FaceShifter` | High-Fidelity Face Swapping | 1,000 | **FAKE** |
| `FaceSwap` | Graphics-Based Face Swap | 1,000 | **FAKE** |
| `NeuralTextures` | Neural Rendering Reenactment | 1,000 | **FAKE** |
| `DeepFakeDetection` | Actor-Based Deepfake Benchmarks | 1,000 | **FAKE** |
| **Total Benchmark Dataset** | **Complete Dataset Coverage** | **7,000 Videos** | **1,000 REAL / 6,000 FAKE** |

---

## 2. Resumable Processing & Fault-Tolerance Mechanism

To ensure reliability during long-running extraction across 7,000 videos:

1. **State Inspection**: Upon initialization, `load_existing_processed_paths()` inspects [`data/videos/features/video_features_full.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/features/video_features_full.csv).
2. **Path Deduplication**: All completed `video_path` records are loaded into an in-memory hash set (`already_processed`).
3. **Incremental Execution**: Discovered videos whose paths exist in `already_processed` are skipped instantly. Newly processed feature rows are flushed to disk every 10 videos.
4. **Interruption Recovery**: If processing is interrupted (e.g. process termination or reboot), re-executing `run_full_batch_processing()` resumes immediately from the exact video where it left off, avoiding duplicate computation.

---

## 3. End-to-End Extraction Pipeline

```text
Raw Video Stream (7,000 MP4 Files)
       │
       ▼
OpenCV Stream Reader (Metadata check)
       │
       ▼
Uniform 10-Frame Sampling (NumPy linspace)
       │
       ▼
Frame Preprocessing (256 × 256 BGR, Grayscale, HSV)
       │
       ▼
Traditional Visual Feature Extraction (54 features/frame)
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
Resumable CSV Output (`data/videos/features/video_features_full.csv`)
```

---

## 4. Output CSV Schema & Column Specifications

- **Main Features CSV**: [`data/videos/features/video_features_full.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/features/video_features_full.csv) (224 total columns)
- **Error Log CSV**: [`data/videos/features/full_video_processing_errors.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/features/full_video_processing_errors.csv)

### Schema Column Breakdown
1. **Metadata Columns (8 columns)**:
   - `video_path`: Relative file path to the video file
   - `category`: Dataset category folder name (`original`, `Deepfakes`, etc.)
   - `label`: Binary target label (`REAL` or `FAKE`)
   - `source_id`: Source video identifier (`000`–`999` or actor pair ID)
   - `frame_count`: Total frame count in the video stream
   - `fps`: Video frames per second
   - `width`: Original stream resolution width
   - `height`: Original stream resolution height
2. **Feature Columns (216 columns)**:
   - `{feature_name}_mean`, `{feature_name}_std`, `{feature_name}_min`, `{feature_name}_max` for all 54 visual features.

---

## 5. Automated Validation & Integrity Rules

The validation suite in [`test_full_batch_processor.py`](file:///c:/work/explainable-ai-media-detection/test_full_batch_processor.py) enforces:
- **Dataset Coverage**: Exactly 7,000 total rows.
- **Category Counts**: Exactly 1,000 rows per folder across all 7 categories.
- **Binary Labels**: Exactly 1,000 REAL and 6,000 FAKE.
- **Feature Dimension**: Exactly 216 numeric feature columns.
- **Numeric Integrity**: 0 NaN values and 0 Inf values.
- **Deduplication**: 0 duplicate `video_path` entries.
- **Error Logging**: 0 unhandled video stream failures.
