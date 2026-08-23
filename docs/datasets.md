# Dataset Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## Overview

This project aims to detect AI-generated images and deepfake videos using traditional machine learning algorithms alongside explainable AI (XAI) techniques. 

To support both image and video detection, the project utilizes two benchmark datasets:

1. **Image Dataset**: [CIFAKE — Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
2. **Video Dataset**: [FaceForensics++](https://github.com/ondyari/FaceForensics) (Official Repository: https://github.com/ondyari/FaceForensics)

---

## Dataset Storage & Version Control Policy

> **IMPORTANT**: Actual image and video datasets are **NOT** committed to GitHub or stored in this repository.

### Why datasets are omitted from Git:
- **File Size**: Raw image datasets and high-resolution deepfake video benchmarks (such as FaceForensics++) exceed standard version control storage limits.
- **Terms of Service & Licensing**: FaceForensics++ and other research datasets require explicit access requests or agreement to dataset terms of use before downloading.
- **Clean Repository Maintenance**: Keeping data files out of version control ensures fast cloning and clean repository state.

All raw, processed, extracted frame, and feature data directories are ignored via `.gitignore`. Empty folder structures are maintained in Git using `.gitkeep` files.

---

## Download & Placement Instructions

### 1. Image Dataset (CIFAKE)
Download the CIFAKE dataset from Kaggle or the official provider.
Place the downloaded images into the local project directory as follows:
- Real images $\rightarrow$ `data/images/raw/real/`
- AI-generated synthetic images $\rightarrow$ `data/images/raw/ai_generated/`

### 2. Video Dataset (FaceForensics++)
Follow the download instructions in the [official FaceForensics++ repository](https://github.com/ondyari/FaceForensics). 
> *Note: The full FaceForensics++ dataset is NOT pre-downloaded in this repository. Users must request download access from the official repository maintainers.*

Place downloaded video files into the local project directory as follows:
- Pristine/Original real videos $\rightarrow$ `data/videos/raw/real/`
- Manipulated/Deepfake videos $\rightarrow$ `data/videos/raw/fake/`

---

## Expected Folder Structure

```text
explainable-ai-media-detection/
│
├── data/
│   ├── images/
│   │   ├── raw/
│   │   │   ├── real/            # Raw real images (e.g., CIFAKE real)
│   │   │   └── ai_generated/    # Raw AI-generated synthetic images (e.g., CIFAKE fake)
│   │   │
│   │   ├── processed/
│   │   │   ├── real/            # Preprocessed real images (resized, normalized, etc.)
│   │   │   └── ai_generated/    # Preprocessed AI-generated images
│   │   │
│   │   └── features/            # Extracted image features (CSV/NPY/Parquet)
│   │
│   ├── videos/
│   │   ├── raw/
│   │   │   ├── real/            # Raw original videos (e.g., FaceForensics++ original)
│   │   │   └── fake/            # Raw manipulated videos (e.g., FaceForensics++ deepfakes)
│   │   │
│   │   ├── frames/
│   │   │   ├── real/            # Extracted frames from real videos
│   │   │   └── fake/            # Extracted frames from fake videos
│   │   │
│   │   ├── processed/
│   │   │   ├── real/            # Preprocessed video frames
│   │   │   └── fake/            # Preprocessed video frames
│   │   │
│   │   └── features/            # Extracted video/frame features
│   │
│   └── metadata/                # Dataset annotations, labels, split lists
```

---

## Future Processing Workflow

1. **Image Preprocessing**:
   - Images in `data/images/raw/` will later be processed using **OpenCV** (e.g., resizing, color space conversions, noise filtering, texture analysis) and stored in `data/images/processed/`.

2. **Video Frame Extraction & Preprocessing**:
   - Videos in `data/videos/raw/` will later be processed using **OpenCV** to extract representative frames into `data/videos/frames/real/` and `data/videos/frames/fake/`.
   - Extracted frames will be preprocessed (e.g., face crop/alignment, quality assessment) and saved to `data/videos/processed/`.

3. **Feature Extraction**:
   - Traditional handcrafted features (e.g., LBP, GLCM, FFT/frequency domain metrics, HOG, color histograms) will be extracted from processed images/frames and stored in `data/images/features/` and `data/videos/features/`.
