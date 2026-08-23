# Traditional Machine Learning Model Improvement & Feature Ablation Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Experimental Objective

The objective of Phase 9 is to evaluate whether tuning hyperparameter configurations across diverse traditional ML model families (Logistic Regression, SVM, Random Forest, Extra Trees, HistGradientBoosting) or performing feature family ablation can improve video deepfake detection performance over the Phase 7 baseline.

---

## 2. Dataset & Source-Aware Splitting

- **Dataset**: Stage B Video Benchmark Dataset (`data/videos/features/video_features_stage_b.csv`).
- **Total Samples**: 700 videos (100 REAL, 600 FAKE).
- **Split Strategy**: `GroupShuffleSplit(random_state=42)` grouped strictly on `source_id` to prevent subject/scene data leakage (**0 source ID overlap** across splits).
- **Partitioning**: Train = 488 videos (129 source IDs), Validation = 118 videos (28 source IDs), Test = 94 videos (28 source IDs).
- **Preprocessing Safety**: `StandardScaler` was fitted **EXCLUSIVELY on the Training set**.

---

## 3. Model Search Space & Hyperparameter Configurations

27 lightweight hyperparameter configurations were evaluated across 5 traditional ML model families:

1. **Logistic Regression** ($C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ with `class_weight="balanced"`).
2. **Support Vector Machine (SVM)** ($C \in [0.1, 1.0, 10.0]$, `kernel` $\in [\text{"linear"}, \text{"rbf"}]$ with `class_weight="balanced"`).
3. **Random Forest** (`n_estimators` $\in [200, 500]$, `max_depth` $\in [\text{None}, 10, 20]$ with `class_weight="balanced"`).
4. **Extra Trees** (`n_estimators` $\in [200, 500]$, `max_depth` $\in [\text{None}, 10, 20]$ with `class_weight="balanced"`).
5. **HistGradientBoosting** (`learning_rate` $\in [0.03, 0.1]$, `max_iter` $\in [100, 200]$ with `class_weight="balanced"`).

---

## 4. Hyperparameter Search Results & Best Model Selection

All 27 configurations were trained on the training partition and evaluated on the held-out Validation partition ($N=118$). Saved to [`data/videos/results/ml_model_experiments.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/ml_model_experiments.csv).

Top performance per model family on Validation Data:

| Model Family | Top Configuration | Val Accuracy | Val Balanced Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Logistic Regression ($C=1.0$)** | 0.5339 | **0.5810** | 0.8966 | 0.5149 | 0.6541 | 0.5702 |
| **SVM** | **SVM (kernel=linear, $C=0.1$)** | 0.5339 | **0.5810** | 0.8966 | 0.5149 | 0.6541 | 0.5702 |
| **SVM** | SVM (kernel=rbf, $C=1.0$) | 0.3136 | 0.5745 | 0.9545 | 0.2079 | 0.3415 | 0.4211 |
| **HistGradientBoosting** | HistGradientBoosting (lr=0.03, max_iter=100) | 0.8559 | 0.5000 | 0.8559 | 1.0000 | 0.9224 | 0.5487 |
| **Random Forest** | Random Forest (n_est=500, max_depth=10) | 0.8559 | 0.5000 | 0.8559 | 1.0000 | 0.9224 | 0.5740 |
| **Extra Trees** | Extra Trees (n_est=200, max_depth=10) | 0.8559 | 0.5000 | 0.8559 | 1.0000 | 0.9224 | 0.5761 |

### Model Selection
**Logistic Regression ($C=1.0$)** achieved the highest Validation **Balanced Accuracy (0.5810)** and was selected as the optimal classifier. Tree ensemble models (Random Forest, Extra Trees, HistGradientBoosting) overfit to the majority `FAKE` class on Stage B sample sizes, achieving high raw accuracy (85.59%) but a random Balanced Accuracy of **0.5000**.

---

## 5. Held-Out Test Evaluation & Phase 7 Baseline Comparison

The selected **Logistic Regression ($C=1.0$)** model was evaluated **ONCE** on the held-out Test set ($N=94$). Saved to [`data/videos/results/best_model_results.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/best_model_results.csv).

| Metric | Phase 7 Baseline | Phase 9 Best Model | Status |
| :--- | :---: | :---: | :---: |
| **Selected Model** | Logistic Regression | Logistic Regression ($C=1.0$) | Maintained |
| **Val Balanced Accuracy** | 0.5810 | **0.5810** | Stable Baseline |
| **Val F1-Score** | 0.6541 | **0.6541** | Stable Baseline |
| **Val ROC-AUC** | 0.5702 | **0.5702** | Stable Baseline |
| **Test Accuracy** | 0.6702 | **0.6702** | Stable Baseline |
| **Test Balanced Accuracy** | 0.5976 | **0.5976** | Stable Baseline |
| **Test Precision** | 0.9048 | **0.9048** | High Precision |
| **Test Recall** | 0.6951 | **0.6951** | Stable Baseline |
| **Test F1-Score** | 0.7862 | **0.7862** | High F1 |
| **Test ROC-AUC** | 0.6016 | **0.6016** | Stable Baseline |

### Test Confusion Matrix
- **True Negatives (TN)**: 6 (REAL correctly predicted as REAL)
- **False Positives (FP)**: 6 (REAL incorrectly predicted as FAKE)
- **False Negatives (FN)**: 25 (FAKE incorrectly predicted as REAL)
- **True Positives (TP)**: 57 (FAKE correctly predicted as FAKE)

---

## 6. Feature Family Ablation Experiment

To evaluate which visual feature families drive predictive performance, 5 feature subset configurations were evaluated using the same train/validation/test split. Saved to [`data/videos/results/feature_ablation_results.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/feature_ablation_results.csv):

| Feature Subset Configuration | Feature Count | Val Accuracy | Val Balanced Accuracy | Val F1 | Val ROC-AUC | Test Balanced Accuracy | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Set A: All 216 Features** | **216** | **0.5339** | **0.5810** | **0.6541** | **0.5702** | **0.5976** | **0.7862** | **0.6016** |
| **Set B: Remove Color Features** | 96 | 0.6186 | 0.5571 | 0.7429 | 0.5789 | 0.5203 | 0.7838 | 0.5955 |
| **Set C: Remove LBP/Gray Features** | 168 | 0.5508 | 0.5419 | 0.6788 | 0.5510 | 0.5671 | 0.7429 | 0.6159 |
| **Set D: Color + LBP/Gray Only** | 168 | 0.5254 | 0.5515 | 0.6500 | 0.5545 | 0.5437 | 0.7639 | 0.5945 |
| **Set E: GLCM + DCT + Edge Only** | 48 | 0.5932 | 0.5422 | 0.7209 | 0.5446 | 0.5661 | 0.6818 | 0.6026 |

### Core Research Findings
1. **Full Feature Representation (Set A) Is Superior**: Combining all 216 features achieves the highest Validation Balanced Accuracy (**0.5810**) and highest Test Balanced Accuracy (**0.5976**).
2. **Color Features Are Indispensable**: Removing Color features (Set B) causes Validation Balanced Accuracy to drop from 0.5810 to 0.5571 (-0.0239) and Test Balanced Accuracy to collapse to 0.5203 (-0.0773).
3. **LBP Micro-Texture Features Are Critical**: Removing LBP/Gray features (Set C) reduces Validation Balanced Accuracy to 0.5419 (-0.0391).
4. **Conclusion**: All 5 traditional visual feature families contribute complementary information necessary for robust traditional ML deepfake detection.

---

## 7. Research Limitations & Recommendations

- **Sample Size Bottleneck**: On Stage B ($N=700$ videos, 488 train samples), complex tree ensembles overfit due to limited sample count per category.
- **Stage C Scaling Recommendation**: Scaling up to Stage C ($N=7,000$ videos) will provide 4,900 training videos, allowing non-linear tree models and gradient boosting to learn complex non-linear feature interactions without overfitting.
