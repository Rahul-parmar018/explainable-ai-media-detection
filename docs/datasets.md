# Dataset Documentation

## Project Title
**Explainable Detection of AI-Generated Images and Videos Using Traditional Machine Learning**

---

## Overview

This project aims to detect AI-generated images and deepfake videos using traditional machine learning algorithms alongside explainable AI (XAI) techniques.

The project relies strictly on the following local datasets:

- **Image Dataset**: `Dataset/Image/`
- **Video Dataset**: `Dataset/Video/`

---

## 1. Image Dataset (`Dataset/Image/`)

```text
Dataset/Image/
├── Train/
│   ├── Fake/
│   └── Real/
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

- **Classes**: `Fake` and `Real`
- **Splits**: `Train`, `Validation`, `Test`
- **Total Images**: 190,335

---

## 2. Video Dataset (`Dataset/Video/`)

```text
Dataset/Video/
├── csv/                   # 10 CSV metadata files (labels, resolution, frame counts, codec)
├── DeepFakeDetection/     # 1,000 .mp4 videos (FAKE)
├── Deepfakes/             # 1,000 .mp4 videos (FAKE)
├── Face2Face/             # 1,000 .mp4 videos (FAKE)
├── FaceShifter/           # 1,000 .mp4 videos (FAKE)
├── FaceSwap/              # 1,000 .mp4 videos (FAKE)
├── NeuralTextures/        # 1,000 .mp4 videos (FAKE)
└── original/              # 1,000 .mp4 videos (REAL)
```

### Classification Mapping (REAL vs FAKE)

| Folder | Video Count | Format / Extensions | Binary Classification Label |
| :--- | :---: | :---: | :---: |
| `original` | 1,000 | `.mp4` | **REAL** |
| `Deepfakes` | 1,000 | `.mp4` | **FAKE** |
| `Face2Face` | 1,000 | `.mp4` | **FAKE** |
| `FaceShifter` | 1,000 | `.mp4` | **FAKE** |
| `FaceSwap` | 1,000 | `.mp4` | **FAKE** |
| `NeuralTextures` | 1,000 | `.mp4` | **FAKE** |
| `DeepFakeDetection` | 1,000 | `.mp4` | **FAKE** |

- **Total Original (REAL) Videos**: 1,000
- **Total Manipulated (FAKE) Videos**: 6,000
- **Total Videos**: 7,000
- **Total Storage Size**: ~16.69 GB (17,919,806,673 bytes)

---

## Dataset Storage & Version Control Policy

> **IMPORTANT**: Actual image and video media files are **NOT** committed to GitHub or remote version control repositories to manage repository size and maintain clean project hygiene.

---
