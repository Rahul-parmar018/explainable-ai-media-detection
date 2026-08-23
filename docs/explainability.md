# SHAP Video Explainability & Visual Feature Attribution

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Why Explainability Is Required

In forensic media detection, black-box predictions ("Video X is 92% Fake") fail to provide verifiable evidence required for research, digital forensics, and trusted media verification. Explainability bridges this gap by mapping model decisions back to human-understandable visual feature anomalies—such as temporal color shifts, micro-texture artifacts, or high-frequency boundary compression—allowing analysts to inspect *why* a model reached a given prediction.

---

## 2. SHAP Methodology & LinearExplainer Setup

- **Framework**: SHAP (SHapley Additive exPlanations), grounded in cooperative game theory.
- **Explainer**: `shap.LinearExplainer` configured for the baseline Logistic Regression model.
- **Background Distribution**: Derived strictly from the training dataset features (`X_train_scaled`) to eliminate data leakage. Validation and test sets remain unobserved during explainer initialization.
- **Additive Attribution**: For a video sample vector $\mathbf{x}$, the prediction is expressed as:
  $$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{216} \phi_i$$
  where $\phi_0$ is the base expectation and $\phi_i$ is the SHAP value for feature $i$.

---

## 3. Global Feature Importance Ranking

Global importance is measured by computing the mean absolute SHAP value across all held-out test samples:
$$\text{Mean Abs SHAP}_i = \frac{1}{N} \sum_{j=1}^{N} |\phi_{i,j}|$$

Top 10 most influential features globally (saved in [`data/videos/results/shap_global_importance.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/shap_global_importance.csv)):

| Rank | Feature Name | Feature Family | Mean Abs SHAP | Direction Trend |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `s_mean_min` | Color (HSV) | 1.1342 | Pushes REAL |
| 2 | `s_hist_bin_3_max` | Color (HSV) | 1.0531 | Pushes FAKE |
| 3 | `s_std_min` | Color (HSV) | 0.9842 | Pushes REAL |
| 4 | `s_hist_bin_5_max` | Color (HSV) | 0.9410 | Pushes FAKE |
| 5 | `v_hist_bin_6_max` | Color (HSV) | 0.9125 | Pushes FAKE |
| 6 | `v_mean_max` | Color (HSV) | 0.8874 | Pushes REAL |
| 7 | `lbp_hist_3_mean` | Texture (LBP & Gray) | 0.8520 | Pushes FAKE |
| 8 | `s_hist_bin_4_max` | Color (HSV) | 0.8140 | Pushes FAKE |
| 9 | `v_hist_bin_1_min` | Color (HSV) | 0.7915 | Pushes FAKE |
| 10 | `s_hist_bin_2_max` | Color (HSV) | 0.7651 | Pushes FAKE |

---

## 4. Feature Family Importance Breakdown

To answer the core research question: **"Which traditional visual feature family contributes most to AI-generated video detection?"**, SHAP values were aggregated across all 5 visual feature families (saved in [`data/videos/results/shap_family_importance.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/shap_family_importance.csv)):

| Feature Family | Feature Count | Total Mean Abs SHAP | Percentage Contribution (%) |
| :--- | :---: | :---: | :---: |
| **Color (HSV)** | 120 | 18.6839 | **50.04%** |
| **Texture (LBP & Gray)** | 48 | 11.0967 | **29.72%** |
| **Texture (GLCM)** | 20 | 3.4097 | **9.13%** |
| **Frequency (DCT)** | 16 | 2.1153 | **5.67%** |
| **Edge (Canny)** | 12 | 2.0346 | **5.45%** |
| **Total** | **216** | **37.3402** | **100.00%** |

### Key Research Insight
Color distribution anomalies in the HSV color space (specifically Saturation and Value histograms) and micro-texture grain variations captured by Local Binary Patterns (LBP) account for **~79.76% of total predictive power** in detecting face-swap and facial reenactment synthetic videos.

---

## 5. Local Video Explanations

Local explanations break down individual video predictions into positive and negative feature contributions (saved in [`data/videos/results/shap_local_explanations.csv`](file:///c:/work/explainable-ai-media-detection/data/videos/results/shap_local_explanations.csv)).

The reusable explanation API `explain_video(video_feature_dict, model, scaler, explainer, feature_names)` yields structured evidence:

```json
{
  "prediction": "FAKE",
  "probability": 0.8742,
  "top_fake_evidence": [
    {
      "feature": "s_hist_bin_3_max",
      "feature_family": "Color (HSV)",
      "feature_value": 0.1425,
      "shap_value": 1.2541,
      "direction": "Pushes FAKE"
    }
  ],
  "top_real_evidence": [
    {
      "feature": "s_mean_min",
      "feature_family": "Color (HSV)",
      "feature_value": 82.41,
      "shap_value": -0.9842,
      "direction": "Pushes REAL"
    }
  ]
}
```

---

## 6. Important Interpretation Protocol

> [!IMPORTANT]
> **SHAP values measure model behavior, not objective proof of manipulation.**
> 
> - **Correct Terminology**: *"Feature `s_hist_bin_3_max` contributed +1.254 toward the model's FAKE prediction."*
> - **Incorrect Terminology**: *"Feature `s_hist_bin_3_max` proves the video is fake."*

---

## 7. Limitations

1. **Linear Model Scope**: `LinearExplainer` measures linear feature contributions. As non-linear classifiers (e.g., Random Forest or SVM with RBF kernels) are tuned in future iterations, non-linear `TreeExplainer` or `KernelExplainer` will be evaluated.
2. **Dataset Scale**: Explanations reflect patterns within the Stage B dataset ($N=700$) and will be re-evaluated on full Stage C benchmark data ($N=7,000$).
