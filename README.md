# Lung Cancer Detection using 3D CNN and Radiomics (Research Tool)

This repository provides a **research/educational** web application for lung cancer risk modeling from 3D CT scans using a modular pipeline that combines:
- 3D CNN-derived features (PyTorch)
- Radiomic features (PyRadiomics, with a safe fallback)
- Feature fusion and binary risk classification

> ⚠️ **Important:** This project is for research and education only. It does **not** provide a clinical diagnosis.

## Features
- Flask-based upload and prediction dashboard
- Support for `.npy`, `.nii`, and `.nii.gz` 3D CT volumes
- CT preprocessing and normalization
- 3D CNN feature extraction
- Radiomics extraction (or statistical fallback)
- Feature fusion and classification with probability/confidence
- Evaluation utilities for:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── routes.py
│   └── services/
│       └── pipeline_service.py
├── ml/
│   ├── pipeline.py
│   ├── models/cnn3d.py
│   ├── preprocessing/ct_preprocessing.py
│   ├── radiomics/radiomics_extractor.py
│   ├── fusion/feature_fusion.py
│   ├── classification/classifier.py
│   └── evaluation/metrics.py
├── training/train.py
├── evaluation/evaluate.py
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── scripts/example_predict.py
├── uploads/
├── config/settings.yaml
├── tests/test_app.py
├── requirements.txt
└── run.py
```

## Expected Dataset Structure (LIDC-IDRI-style)

Use a public CT dataset (e.g., LIDC-IDRI) organized in a split-oriented layout:

```text
data/
├── train/
│   ├── benign/
│   │   ├── case_0001.npy
│   │   └── ...
│   └── malignant/
│       ├── case_0101.npy
│       └── ...
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

Each file should represent one pre-segmented 3D lung volume.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run Web App

```bash
python run.py
```

Open `http://127.0.0.1:5000` and upload a CT volume file.

## Run Tests

```bash
pytest tests/test_app.py -q
```

## Example CLI Prediction

```bash
python scripts/example_predict.py /absolute/path/to/sample.npy
```

## Reproducibility Notes
- Pipeline seed is fixed in `ml/pipeline.py`.
- Configuration defaults are versioned in `config/settings.yaml`.
- Evaluation metrics are implemented in `ml/evaluation/metrics.py`.
