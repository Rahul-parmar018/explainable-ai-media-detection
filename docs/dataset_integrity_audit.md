# Full Video Dataset Integrity & Data Leakage Audit Report

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Audit Purpose & Scope

Before executing final machine learning benchmark training on the full video dataset, a comprehensive integrity and data leakage audit was conducted. The audit evaluates feature column dimensions, missing or corrupted values, duplicate video stream entries, and underlying actor pair relationships across dataset folders.

---

## 2. Summary of Integrity Checks

| Audit Metric | Target Requirement | Audit Result | Status |
| :--- | :---: | :---: | :---: |
| **Feature Column Count** | Exactly 216 Features | 216 Features | **PASS** |
| **NaN Values Count** | 0 NaN | 0 NaN | **PASS** |
| **Inf Values Count** | 0 Inf | 0 Inf | **PASS** |
| **Duplicate Video Paths** | 0 Duplicates | 0 Duplicates | **PASS** |
| **Duplicate Feature Vectors** | 0 Duplicates | 0 Duplicates | **PASS** |
| **Target Label Integrity** | `REAL` & `FAKE` only | `REAL` & `FAKE` mapped | **PASS** |
| **Category Coverage** | All 7 categories present | 7 Categories verified | **PASS** |

---

## 3. Underlying Source ID Structure & Leakage Investigation

Investigation of filenames across manipulated categories (`Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`) and `DeepFakeDetection` reveals explicit multi-actor relationships:

```text
Pristine Video:        original/000.mp4            ──► Primary Subject ID: 000
Pristine Target:       original/003.mp4            ──► Primary Subject ID: 003
Manipulated Video:     Deepfakes/000_003.mp4       ──► Source Actor: 000 | Target Actor: 003
DeepFakeDetection:     DeepFakeDetection/01_02__.. ──► Actor Pair: (01, 02)
```

### Potential Data Leakage Risk
If videos are grouped *only* by the primary `source_id` (e.g. `000`):
- `original/000.mp4` and `Deepfakes/000_003.mp4` are assigned to **Train**.
- `original/003.mp4` and `Deepfakes/003_000.mp4` are assigned to **Test**.

Because `Deepfakes/000_003.mp4` in Train contains the facial background and features of actor `003`, placing `original/003.mp4` in Test creates **actor/scene leakage**, artificially inflating test evaluation metrics.

### Leak-Free Connected Component Grouping Strategy
To eliminate 100% of subject leakage:
1. An **Undirected Actor Graph** $G=(V, E)$ is constructed where nodes $V$ represent individual actor IDs (e.g., `000`, `003`, `01`, `02`) and edges $E$ represent co-occurrence in manipulated or reenacted video clips (`000_003.mp4`).
2. **Connected Components** of $G$ are extracted via Breadth-First Search (BFS).
3. All videos sharing any connected actor ID are assigned the identical `connected_component_id` (e.g., `COMP_001`).

Assigning partition splits (Train 70%, Validation 15%, Test 15%) at the **Connected Component Level** guarantees **0 cross-partition actor leakage**.

---

## 4. Source Group Summary Artifacts

- **Group Summary CSV**: [`data/videos/results/source_group_summary.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/source_group_summary.csv)
  - Documents `source_id`, `video_count`, `categories_present`, `labels_present`, and `connected_component_id` for every unique video group.

---

## 5. Audit Verdict & Recommendation

> [!IMPORTANT]
> **FINAL AUDIT VERDICT: PASS**
> 
> - **Integrity**: 0 NaN, 0 Inf, 0 duplicate paths, 216 features confirmed.
> - **Grouping Strategy**: Connected component actor grouping (`build_connected_component_groups`) confirmed as the leak-free grouping key for Phase 10B benchmark model training.
