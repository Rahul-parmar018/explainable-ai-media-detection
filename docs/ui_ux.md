# UI/UX Design Specification

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Design Vision & Philosophy

The user interface for this platform is designed as an **editorial-grade, research-focused diagnostic suite**. Rather than acting as a black-box binary classifier, the primary objective of the interface is to answer **WHY** a media item (image or video) was classified as **REAL** or **AI-GENERATED (FAKE)** by surfacing underlying traditional machine learning features and explainability metrics.

### Design Principles
- **Editorial & Scholarly Aesthetic**: Clean typography, high information hierarchy, and data-dense visualization inspired by scientific literature and analytical tools (e.g., Bloomberg Terminal, Observable, Nature data visualizers).
- **Transparency & Explainability First**: Every prediction must be accompanied by explicit visual feature breakdowns (color, texture, edge, frequency, GLCM).
- **Calm & Trustworthy**: Soft neutral palettes (slate, charcoal, off-white, warm zinc) with restrained semantic color accents (emerald green for REAL, warm amber/coral for FAKE).

### anti-Patterns to Explicitly Avoid
- ❌ Dark neon "cyberpunk" AI aesthetic
- ❌ Glowing neon blue/purple borders and excessive drop-shadows
- ❌ Heavy glassmorphism or distracting blur overlays
- ❌ Generic AI robot vectors, brain graphics, or stock tech imagery
- ❌ Massive marketing hero sections with empty buzzwords
- ❌ Overcrowded widget grids that obscure core diagnostic metrics

---

## 2. Core UI Sections & Component Layout

### Section 1: Landing & Media Upload
- **Header**: Minimal title banner with dataset/model selection toggles (Image Dataset vs. Video Dataset models).
- **Drag-and-Drop Zone**: Dual dropzone supporting images (`.jpg`, `.png`) and video streams (`.mp4`, `.avi`, `.mov`). Includes sample media pre-loader for instant demonstration.
- **Pre-Analysis Metadata Box**: Displays media attributes prior to processing (resolution, file size, codec, total frames, video duration).

### Section 2: Image Analysis View
- **Split-View Canvas**:
  - Left Panel: Original image preview with color space toggles (RGB, Grayscale, HSV, Canny Edges).
  - Right Panel: Primary diagnostic summary.

### Section 3: Video Analysis View & Frame Timeline
- **Sampled Frame Strip**: Horizontal interactive thumbnail carousel showing the 10 uniformly sampled frames.
- **Frame Detail Inspector**: Clicking any sampled frame displays its corresponding 256×256 preprocessed BGR, Grayscale, HSV, and Canny representations.
- **Temporal Stability Graph**: Sparkline showing metric stability across sampled frames (e.g., edge density variance or LBP micro-texture volatility).

### Section 4 & 5: Prediction Result & Confidence Metric
- **Classification Status Card**: High-contrast, clean summary pill:
  - `REAL` (Pristine Media) or `AI-GENERATED / FAKE` (Manipulated Media).
- **Probability Breakdown**: Calibrated probability bar chart showing model confidence score (e.g., 94.2% FAKE confidence score).
- **Model Metadata**: Indicates active traditional ML classifier (e.g., Random Forest, SVM, or Gradient Boosting) and feature vector dimension (216-D video / 54-D image).

### Section 6 & 7: Explainability & Feature Contribution Visualization
- **Feature Importance Ranking Bar Chart**: Horizontal bar chart presenting top $K$ influential visual features driving the specific prediction (derived via SHAP / feature weights).
- **Feature Family Group Breakdown**: Radar/Spider chart or grouped bar chart displaying normalized score across the 5 traditional feature families:
  1. **Color (HSV)**: Saturation & Hue distribution anomalies
  2. **Texture (LBP)**: Micro-texture smoothness vs. natural skin variance
  3. **Edge Density (Canny)**: Boundary sharpness and structural noise
  4. **Spatial Co-occurrence (GLCM)**: Homogeneity, contrast, and correlation
  5. **Spectral Frequency (DCT)**: High-frequency compression artifact energy ratio

### Section 8: Video Frame Analysis Grid
- **Frame-by-Frame Comparative Heatmap**: Grid showing frame-level feature trajectories over the 10 sampled frame indices to visually highlight temporal flickering or blending inconsistencies.

### Section 9: Research & Methodology Section
- **Dataset Summary Modal/Drawer**: Overview of benchmark datasets (`Dataset/Image/` and `Dataset/Video/` FaceForensics++ benchmark).
- **Feature Definition Glossary**: Accessible drawer explaining the mathematical definition and physical intuition behind each traditional visual feature (e.g., explaining why high `dct_high_low_ratio` indicates synthetic frequency noise).

---

## 3. Color Palette & Typography Specifications

| Element | Color Code / Utility | Purpose |
| :--- | :--- | :--- |
| **Background (Primary)** | `#FAFAFA` / `#FFFFFF` | Clean off-white canvas |
| **Background (Secondary)** | `#F4F4F5` (Zinc 100) | Card backgrounds & sidebars |
| **Text (Primary)** | `#18181B` (Zinc 900) | High-legibility headlines & body |
| **Text (Muted)** | `#71717A` (Zinc 500) | Labels, units, and timestamps |
| **Status: REAL** | `#059669` (Emerald 600) | Authentic media confirmation |
| **Status: FAKE** | `#DC2626` (Red 600) | Synthetic/deepfake detection alert |
| **Accent / Focus** | `#2563EB` (Blue 600) | Active controls & selected tabs |
| **Chart Series 1..5** | Palette: Slate, Teal, Amber, Indigo, Rose | Grouped visual feature families |

### Typography
- **Primary Interface Font**: `Inter`, `Roboto`, or `-apple-system` (Clean sans-serif for UI labels and data tables).
- **Monospace Code / Metrics Font**: `JetBrains Mono` or `Fira Code` (Used for exact float values, feature names, frame indices, and file paths).
