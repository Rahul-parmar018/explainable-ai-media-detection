# Traditional Machine Learning Video Baseline Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Overview & Research Scope

This document details the baseline traditional machine learning pipeline established for video-level deepfake and synthetic video detection. In strict accordance with project guidelines, **no deep learning, neural networks, or pretrained transformers were used**. All models are trained exclusively on the 216 explainable, handcrafted traditional visual features extracted from video streams.

---

## 2. Dataset & Feature Representation

- **Dataset**: Stage B Video Benchmark Dataset (`data/videos/features/video_features_stage_b.csv`).
- **Total Samples**: 700 videos (100 REAL, 600 FAKE).
- **Target Encoding**: `REAL` = 0, `FAKE` = 1.
- **Feature Vector**: 216 aggregated traditional visual features per video ($54 \text{ frame features} \times 4 \text{ temporal statistics [Mean, Std, Min, Max]} = 216$).
  - HSV Color (30 features $\times 4 = 120$)
  - LBP Texture & Grayscale (12 features $\times 4 = 48$)
  - Canny Edge Density (3 features $\times 4 = 12$)
  - GLCM Texture (5 features $\times 4 = 20$)
  - DCT Frequency Energy (4 features $\times 4 = 16$)

---

## 3. Data Leakage Prevention & Source-Aware Splitting

To eliminate subject/scene data leakage:
- **Grouping Variable**: `source_id` (derived from original video IDs `000`–`999` and actor pairs).
- **Split Strategy**: `GroupShuffleSplit` with `random_state=42`. All manipulated variants derived from a specific source ID are kept in the same split partition as the original video.
- **Partitioning Breakdown**:

| Partition | Percentage | Sample Count | Unique Source IDs | REAL Count | FAKE Count | Source ID Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | ~70% | 488 | 129 | 71 | 417 | **0** |
| **Validation Set** | ~15% | 118 | 28 | 17 | 101 | **0** |
| **Test Set** | ~15% | 94 | 28 | 12 | 82 | **0** |

---

## 4. Preprocessing & Feature Scaling

- **Scaler**: `StandardScaler()` from `scikit-learn`.
- **Leakage Prevention**: To prevent data leakage, `StandardScaler.fit()` was executed **EXCLUSIVELY on the Training set (`X_train`)**.
- **Transformation**: `scaler.transform()` was applied to `X_train`, `X_val`, and `X_test` independently using the training distribution parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).

---

## 5. Traditional Machine Learning Baseline Models

All models were configured with class weighting (`class_weight="balanced"`) to handle the 1:6 REAL-to-FAKE class imbalance:

1. **Logistic Regression**:
   - `LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)`
2. **Random Forest Classifier**:
   - `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)`
3. **Support Vector Classifier (SVM)**:
   - `SVC(class_weight="balanced", probability=True, random_state=42)`

---

## 6. Validation Results & Model Comparison

Models were evaluated on the held-out Validation set ($N=118$) to compare performance:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5339 | **0.5810** | 0.8966 | 0.5149 | 0.6541 | 0.5702 |
| **Support Vector Machine (SVM)** | 0.3136 | 0.5745 | 0.9545 | 0.2079 | 0.3415 | 0.4211 |
| **Random Forest** | 0.8559 | 0.5000 | 0.8559 | 1.0000 | 0.9224 | 0.5740 |

### Why Balanced Accuracy Is the Primary Metric
Standard Accuracy can be misleading on imbalanced datasets. For instance, Random Forest predicted the majority class `FAKE` for all validation samples, achieving a high raw Accuracy of 85.59% but a random Balanced Accuracy of **0.5000**. **Logistic Regression** achieved the highest **Balanced Accuracy (0.5810)** on validation data and was selected as the primary baseline model.

---

## 7. Held-Out Test Set Evaluation

The selected **Logistic Regression** model was evaluated **ONCE** on the held-out Test set ($N=94$):

- **Accuracy**: 0.6702 (67.02%)
- **Balanced Accuracy**: **0.5976** (59.76%)
- **Precision**: 0.9048 (90.48%)
- **Recall**: 0.6951 (69.51%)
- **F1-Score**: 0.7862 (78.62%)
- **ROC-AUC**: 0.6016 (60.16%)

### Confusion Matrix (Test Set)
- **True Negatives (TN)**: 6 (REAL correctly predicted as REAL)
- **False Positives (FP)**: 6 (REAL incorrectly predicted as FAKE)
- **False Negatives (FN)**: 25 (FAKE incorrectly predicted as REAL)
- **True Positives (TP)**: 57 (FAKE correctly predicted as FAKE)

---

## 8. Explainability Artifacts Created

Result files generated under [`data/videos/results/`](file:///c:/work/explainable-ai-media-detection/data/videos/results/):
1. [`model_comparison.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/model_comparison.csv): Validation performance comparison table.
2. [`test_results.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/test_results.csv): Final test set evaluation metrics and confusion matrix.
3. [`feature_importance.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/feature_importance.csv): Feature importance rankings and Logistic Regression coefficients for post-hoc explainability analysis.

---

## 9. Limitations & Research Next Steps

- **Dataset Scale**: Stage B dataset ($N=700$) provides a fast baseline. Scaling up to Stage C ($N=7,000$) will improve model stability and feature significance.
- **Feature Selection**: High dimensionality (216 features for 488 training samples) benefits from feature selection or regularization (e.g. L1/Lasso or SHAP analysis) to eliminate redundant temporal features.
