# Video Dataset Split & Data Leakage Prevention Strategy

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Dataset Composition & Distribution

The video benchmark dataset (`Dataset/Video/`) comprises 7,000 total MP4 video streams distributed across 7 distinct categories:

| Folder Name | Category / Manipulation Method | Video Count | Binary Classification Label |
| :--- | :--- | :---: | :---: |
| `original` | Pristine Source Videos | 1,000 | **REAL** |
| `Deepfakes` | Deepfake Face Swap (Autoencoder) | 1,000 | **FAKE** |
| `Face2Face` | Facial Expression Reenactment | 1,000 | **FAKE** |
| `FaceShifter` | High-Fidelity Face Swapping | 1,000 | **FAKE** |
| `FaceSwap` | Graphics-Based Face Swap | 1,000 | **FAKE** |
| `NeuralTextures` | Neural Rendering Reenactment | 1,000 | **FAKE** |
| `DeepFakeDetection` | Actor-Based Deepfake Benchmarks | 1,000 | **FAKE** |

- **Total Pristine (REAL) Videos**: 1,000 (14.3%)
- **Total Manipulated (FAKE) Videos**: 6,000 (85.7%)
- **Total Dataset Size**: **7,000 Videos** (~16.69 GB)

---

## 2. Metadata Structure & Underlying Relationships

Inspection of `Dataset/Video/csv/*.csv` metadata reveals explicit relationships across folders:

```text
original/000.mp4           ──────► Source Video ID: 000 (REAL)
                             │
                             ├─► Deepfakes/000_003.mp4         (FAKE)
                             ├─► Face2Face/000_003.mp4         (FAKE)
                             ├─► FaceShifter/000_003.mp4       (FAKE)
                             ├─► FaceSwap/000_003.mp4          (FAKE)
                             └─► NeuralTextures/000_003.mp4    (FAKE)
```

1. **Source Video ID Pairing**:
   - `original/000.mp4` corresponds to primary subject ID `000`.
   - The manipulated categories (`Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`) use a `{source_id}_{target_id}.mp4` naming convention.
   - For every original video (e.g. `000.mp4`), there exist up to 5 manipulated versions across different manipulation techniques sharing the identical original actor, background, lighting, and camera properties.

2. **DeepFakeDetection (DFD)**:
   - DFD videos (`01_02__...`) represent controlled actor pairs in specific scenes. Multiple clips originate from identical actor interactions (`01_02`).

---

## 3. Data Leakage Risks & Research Prevention Strategy

### Data Leakage Hazard
If videos are split using a simple naive random split (e.g. `train_test_split(random_state=42)`):
- `original/000.mp4` might be assigned to **Train**.
- `Deepfakes/000_003.mp4` or `Face2Face/000_003.mp4` might be assigned to **Test**.

In this scenario, the machine learning classifier will learn facial features, hair, background room scenery, and clothing of subject `000` during training. During testing, it will recognize the familiar subject background rather than generalizable synthetic artifacts. This causes severe **subject/scene data leakage**, yielding artificially inflated test metrics that fail to generalize to novel videos.

### Recommended Group-Based Split Strategy
To eliminate data leakage, the split **MUST** be performed at the **Source Video ID Group Level**:
- **Rule**: All videos (original + all 5 manipulated variants) associated with a given Source Video ID (e.g. IDs `000` through `699`) MUST be assigned exclusively to the same split.

### Recommended Split Percentages
- **Train Set (70%)**: Source Video IDs `000` – `699` (~4,900 videos)
- **Validation Set (15%)**: Source Video IDs `700` – `849` (~1,050 videos)
- **Test Set (15%)**: Source Video IDs `850` – `999` (~1,050 videos)

---

## 4. Class Imbalance Mitigation Strategy

The dataset exhibits a 1:6 class imbalance (1,000 REAL vs. 6,000 FAKE):

1. **Training Phase Mitigation**:
   - Apply **Class Weighting** (`class_weight='balanced'`) in traditional ML models (Random Forest, SVM, Logistic Regression). This weights loss penalty inversely proportional to class frequencies, forcing models to treat REAL and FAKE classes with equal priority without discarding data.
2. **Experimental Downsampling Benchmark**:
   - For controlled ablation studies, construct a 1:1 balanced subset (1,000 REAL vs. 1,000 FAKE by selecting 166 videos per FAKE category).
3. **Evaluation Metric Protocol**:
   - **Do NOT rely solely on raw accuracy** (a naive model predicting all FAKE achieves 85.7% accuracy).
   - Evaluate using **Balanced Accuracy**, **Precision**, **Recall (Sensitivity)**, **F1-Score**, **ROC-AUC**, and **Confusion Matrix**.

---

## 5. Multi-Stage Development Strategy

To ensure computational efficiency and avoid premature execution overhead, feature extraction and model development will proceed in 3 controlled stages:

| Stage | Scope / Purpose | Video Count | Composition |
| :--- | :--- | :---: | :--- |
| **Stage A** | **Pipeline & Code Validation** | 28 | 4 sample videos per category (Already verified in Phase 1–5). |
| **Stage B** | **Feature Selection & ML Experiments** | 700 | 100 REAL + 100 per FAKE category (Balanced 1:6 ratio, fast iteration). |
| **Stage C** | **Full Benchmark & Final Evaluation** | 7,000 | Complete dataset (70% Train, 15% Val, 15% Test group-split). |
