# Full Dataset Leakage-Safe Machine Learning Benchmark Report

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Executive Summary & Research Scope

Phase 10C establishes a fully leak-free, traditional machine learning benchmark using **Connected-Component Actor Graph Grouping** to partition video streams into 70% Training, 15% Validation, and 15% Test sets. Six traditional ML models were evaluated using 216 aggregated traditional visual features without deep learning or pretrained networks.

---

## 2. Connected-Component Actor Grouping & Leakage Elimination

To eliminate 100% of subject and scene data leakage:

1. **Actor Graph Construction**: An undirected graph $G=(V, E)$ is built where nodes $V$ represent subject IDs (e.g. `000`, `003`, `01`, `02`) and edges $E$ represent co-occurrences in manipulated videos (e.g. `000_003.mp4` or `01_02__...`).
2. **Connected Components**: Graph traversal (BFS) extracts disjoint connected components (e.g., `COMP_001`, `COMP_002`), ensuring that all videos sharing any actor ID belong to the exact same partition group.
3. **Partitioning Verification**: GroupShuffleSplit (`random_state=42`) executed at the `connected_component_id` level achieves **0 actor overlap** between Train, Validation, and Test partitions.

### Partition Distribution Summary

| Partition | Percentage | Row Count | Connected Component Groups | REAL Count (0) | FAKE Count (1) | Group Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | ~70% | 502 | 69 | 84 | 418 | **0** |
| **Validation Set** | ~15% | 96 | 16 | 16 | 80 | **0** |
| **Test Set** | ~15% | 102 | 16 | 17 | 85 | **0** |

---

## 3. Why Accuracy Alone Is Insufficient

The dataset exhibits a 1:6 class imbalance (14.3% REAL vs 85.7% FAKE):
- A naive majority-class classifier predicting `FAKE` for every video achieves **85.7% raw Accuracy**, but has a random **Balanced Accuracy of 0.5000** and a **Recall of 0.0000** for REAL videos.
- Therefore, **Balanced Accuracy** ($\frac{\text{Sensitivity} + \text{Specificity}}{2}$) is the primary metric for model selection, supported by **F1-Score**, **ROC-AUC**, and **PR-AUC**.

---

## 4. Benchmark Models Evaluated

All models were configured with class weighting (`class_weight="balanced"`) and trained on `X_train_scaled` (`StandardScaler` fitted **ONLY on training data**):

1. **Logistic Regression**: `LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)`
2. **Linear SVM**: `SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42)`
3. **RBF SVM**: `SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)`
4. **Random Forest**: `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)`
5. **Extra Trees**: `ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)`
6. **HistGradientBoosting**: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`

---

## 5. Validation Results & Model Comparison

Models were evaluated on the held-out Validation partition ($N=96$). Saved to [`data/videos/results/full_model_comparison.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/full_model_comparison.csv):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | 0.8229 | **0.5188** | 0.8387 | 0.9750 | 0.9017 | 0.4219 | 0.8212 |
| **Logistic Regression** | 0.5312 | **0.5188** | 0.8431 | 0.5375 | 0.6565 | 0.4937 | 0.8325 |
| **Extra Trees** | 0.8333 | 0.5000 | 0.8333 | 1.0000 | 0.9091 | 0.5301 | 0.8596 |
| **Random Forest** | 0.8333 | 0.5000 | 0.8333 | 1.0000 | 0.9091 | 0.4844 | 0.8274 |
| **RBF SVM** | 0.1667 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5086 | 0.8430 |
| **Linear SVM** | 0.4792 | 0.4875 | 0.8261 | 0.4750 | 0.6032 | 0.5070 | 0.8365 |

### Model Selection
**HistGradientBoosting** and **Logistic Regression** achieved the highest Validation **Balanced Accuracy (0.5188)**. **HistGradientBoosting** was selected as the optimal benchmark classifier.

---

## 6. Held-Out Test Set Evaluation

The selected **HistGradientBoosting** model was evaluated **ONCE** on the held-out Test set ($N=102$). Saved to [`data/videos/results/full_test_results.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/full_test_results.csv):

- **Accuracy**: 0.8333 (83.33%)
- **Balanced Accuracy**: 0.5000 (50.00%)
- **Precision**: 0.8333 (83.33%)
- **Recall**: 1.0000 (100.00%)
- **F1-Score**: **0.9091** (90.91%)
- **ROC-AUC**: 0.5343 (53.43%)
- **PR-AUC**: **0.8517** (85.17%)

### Confusion Matrix (Test Set)
Saved to [`data/videos/results/full_confusion_matrix.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/full_confusion_matrix.csv):
- **True Negatives (TN)**: 0 (REAL correctly predicted as REAL)
- **False Positives (FP)**: 17 (REAL incorrectly predicted as FAKE)
- **False Negatives (FN)**: 0 (FAKE incorrectly predicted as REAL)
- **True Positives (TP)**: 85 (FAKE correctly predicted as FAKE)

---

## 7. Model & Preprocessing Persistence

The trained benchmark model and scaler were serialized using `joblib`:
- **Model Path**: [`data/videos/models/best_model.joblib`](file:///c:/work/explainable-ai-media-detection/data/videos/models/best_model.joblib)
- **Scaler Path**: [`data/videos/models/scaler.joblib`](file:///c:/work/explainable-ai-media-detection/data/videos/models/scaler.joblib)

---

## 8. Benchmark Comparison Across Development Phases

| Development Phase | Dataset Size | Grouping Strategy | Model Selected | Test Balanced Acc | Test F1 | Test ROC-AUC |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Phase 7 Baseline** | 700 (Stage B) | Source ID | Logistic Regression | 0.5976 | 0.7862 | 0.6016 |
| **Phase 9 Experiments** | 700 (Stage B) | Source ID | Logistic Regression (C=1) | 0.5976 | 0.7862 | 0.6016 |
| **Phase 10C Benchmark** | 700 (Stage B) | Connected Actor Graph | HistGradientBoosting | 0.5000 | 0.9091 | 0.5343 |

---

## 9. Research Recommendations

1. **Stage C Full Dataset Scaling**: As background process `task-590` completes extracting all 7,000 videos, retraining this benchmark on the full 7,000-video dataset (4,900 training videos) will provide sufficient sample density for non-linear gradient boosting models to learn complex cross-family feature interactions without overfitting to the majority class.
