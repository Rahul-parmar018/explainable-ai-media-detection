# Media Forensics Lab: Web Application Interface Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## 1. Overview & Visual Aesthetic

The Phase 12 Web Application Interface provides a forensic laboratory workspace for explainable video authenticity analysis. Designed around an **editorial and data-journalism aesthetic**, the application avoids generic AI website clichés (such as neon blue glow, black gradients, glowing cards, or generic robot imagery) in favor of high-contrast data visualization, restrained neutral slate/zinc color palettes, and clear typographic hierarchy.

---

## 2. Frontend Application Architecture

Built with **React + Vite** and styled with modular CSS & **Recharts**:

- **Header Workspace**: Displays application title ("Media Forensics Lab"), status indicator ("Research Prototype"), API health monitor, and Model Specification modal launcher.
- **Video Input Workspace**: Drag-and-drop MP4 dropzone with file format validation, 100 MB file size validation, upload progress indicator, and HTML5 video player stream preview.
- **Verdict & Decision Score Panel**: Displays REAL / FAKE decision banner, decision score progress gauge, probability breakdowns, and explicit research disclaimers ("Algorithmic Estimate" / "Model Decision Score").
- **Explainability SHAP Evidence Panel**: Displays top 5 positive evidence features pushing FAKE and top 5 negative evidence features pushing REAL, complete with visual feature family badges (`Color (HSV)`, `Texture (LBP & Gray)`, `Texture (GLCM)`, `Frequency (DCT)`, `Edge (Canny)`).
- **Feature Family Contribution Chart**: Recharts bar chart visualizing the percentage contribution of each visual feature family.
- **Forensic Feature Table**: Searchable interactive table listing extracted traditional visual features, raw values, SHAP contributions, and directional trends.
- **Model Specification Modal**: Provides full transparency on the `HistGradientBoostingClassifier`, 216 feature dimensions, 10 sampled frames, connected-component actor grouping strategy, and held-out test benchmark metrics.

---

## 3. Running the Web Application Locally

### Step 1: Start FastAPI Backend Service
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Backend API interactive documentation will be available at: `http://127.0.0.1:8000/docs`

### Step 2: Start React Frontend Development Server
```bash
cd frontend
npm run dev
```
Frontend web application will be accessible at: `http://localhost:5173`

---

## 4. Research Prototype Disclaimer & Limitations

> [!IMPORTANT]
> **RESEARCH BENCHMARK DISCLAIMER**
> 
> - Full dataset benchmark validation achieved **Balanced Accuracy = 50.00%** and **ROC-AUC = 53.43%** on the Stage B/C dataset ($N=7,000$ videos), with the baseline classifier predicting `FAKE` for most samples due to 1:6 class imbalance.
> - The web application prominently communicates that decision probabilities represent **algorithmic decision estimates on traditional visual features** and must not be treated as ground-truth certainty.
