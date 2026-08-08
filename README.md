# Lung Cancer Detection — 3D U-Net Demo (Educational)

**Purpose:** A compact, runnable demo project that shows how to preprocess a folder of PNG slices into a 3D volume, run a lightweight 3D U-Net segmentation model, postprocess detected nodules, and serve a simple FastAPI endpoint with a React demo UI.

**Important:** This is a research/demo tool **only** — not for clinical use.

## What's included
- `preprocess.py` : convert PNG stack -> resampled .npz volume
- `utils.py` : helper functions (resampling, normalization, connected components, diameter)
- `train_unet.py` : lightweight training script (for small local experiments)
- `inference.py` : run model on .npz -> masks + nodule info + T-stage suggestion
- `backend/app.py` : FastAPI server (endpoint `/predict` accepts .npz or .zip of PNGs)
- `frontend/App.jsx` : single-file React component (drop into a Vite/CRA app)
- `sample/` : a tiny synthetic example `.npz` to test inference without training
- `requirements.txt`

## Quick run (local test using the included synthetic sample)

1. Create and activate Python environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run inference on synthetic sample to see pipeline output:
   ```bash
   python inference.py --volume sample/synthetic_volume.npz --model sample/unet_dummy_ts.pt
   ```
   This will print a JSON result with detected nodules (the included model is a tiny TorchScript model that produces a synthetic mask for demo).

3. Run the FastAPI backend for demo:
   ```bash
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```
   Then open the React demo (see `frontend/App.jsx`) or call `/predict` with `curl`/Postman.

## If you want to actually train
- `train_unet.py` includes a minimal training loop and saving. For real training, prepare `.npz` volumes and paired `_mask.npz` files, and adjust patch size / augmentations / GPU settings.

## Notes
- The included TorchScript model is intentionally tiny to allow quick CPU runs for testing the pipeline.
- For a production/research-quality system, replace with full LIDC/LUNA preprocessed datasets, heavier models, multi-scale inference, extensive augmentations, and clinician validation.

---
If you'd like, I can now:
- Expand the dataset loader to create 64x64x64 patches with on-the-fly augmentation.
- Provide a script that converts DICOM -> PNG stack (with HU preservation).
- Build and upload a full trained checkpoint (if you provide data or allow me to use public data).

# 🫁 Lung Cancer Detection Using 3D CNN and Radiomics

A deep learning and medical image analysis project for **lung cancer detection from 3D CT scans** using a hybrid approach that combines **3D Convolutional Neural Networks (3D CNNs)** with **radiomics features**.

The system is designed to learn meaningful spatial features from volumetric CT scans using a 3D CNN while simultaneously extracting handcrafted quantitative features using radiomics. These features can then be combined to build a hybrid classification model.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a clinical diagnostic tool and must not be used to make medical decisions.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Objectives](#-objectives)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Project Workflow](#-project-workflow)
* [Technology Stack](#-technology-stack)
* [Dataset](#-dataset)
* [Dataset Structure](#-dataset-structure)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Data Preprocessing](#-data-preprocessing)
* [3D CNN Model](#-3d-cnn-model)
* [Radiomics Feature Extraction](#-radiomics-feature-extraction)
* [Feature Fusion](#-feature-fusion)
* [Model Training](#-model-training)
* [Model Evaluation](#-model-evaluation)
* [Web Application](#-web-application)
* [Running the Project](#-running-the-project)
* [Example Workflow](#-example-workflow)
* [Performance Metrics](#-performance-metrics)
* [Future Improvements](#-future-improvements)
* [Research Applications](#-research-applications)
* [Limitations](#-limitations)
* [Contributing](#-contributing)
* [License](#-license)
* [Disclaimer](#-disclaimer)

---

# 🔬 Overview

Lung cancer is one of the most significant causes of cancer-related mortality worldwide. Early identification of suspicious lung nodules from CT scans can support further clinical investigation.

This project explores an automated approach for analyzing **three-dimensional chest CT scans**.

Instead of relying only on conventional machine-learning features or a standard 2D image classifier, the project combines two complementary approaches:

### 1. 3D CNN

A 3D Convolutional Neural Network processes volumetric CT data and learns spatial patterns directly from the scan.

### 2. Radiomics

Radiomics converts medical images into a large number of quantitative features describing characteristics such as:

* Shape
* Intensity
* Texture
* Distribution
* Statistical properties

### 3. Hybrid Feature Fusion

The learned deep features and handcrafted radiomics features can be combined and supplied to a final classification model.

```text
                 3D CT Scan
                     │
                     ▼
             Preprocessing
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
      3D CNN              Radiomics
          │                     │
          ▼                     ▼
   Deep Features        Radiomic Features
          │                     │
          └──────────┬──────────┘
                     ▼
              Feature Fusion
                     │
                     ▼
             Classification
                     │
                     ▼
              Prediction
```

---

# 🎯 Objectives

The main objectives of this project are:

* Process volumetric lung CT scans.
* Normalize and prepare CT data for deep learning.
* Extract three-dimensional spatial features using a 3D CNN.
* Extract quantitative radiomics features from regions of interest.
* Combine deep-learning and radiomics features.
* Train a classification model.
* Evaluate the model using multiple performance metrics.
* Provide a simple web interface for research experimentation.
* Create a reproducible framework for future medical-imaging research.

---

# 🚀 Key Features

### Medical Image Processing

* 3D CT scan loading
* HU-based preprocessing
* Intensity normalization
* Resampling
* Volume resizing/cropping
* Region-of-interest processing

### Deep Learning

* 3D CNN architecture
* GPU/CPU training support
* Training and validation pipelines
* Model checkpointing
* Configurable hyperparameters

### Radiomics

* Automated radiomics feature extraction
* First-order statistical features
* Shape-based features
* Texture features
* Feature normalization
* Feature selection

### Hybrid Model

* Deep feature extraction
* Radiomics feature extraction
* Feature concatenation
* Classification

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* ROC curve

### Web Interface

* CT scan upload
* Prediction interface
* Model result display
* Probability/confidence display where supported
* Responsive frontend

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      CT Scan        │
                    │   DICOM / NIfTI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Preprocessing      │
                    │                     │
                    │ • Resampling        │
                    │ • Normalization     │
                    │ • Cropping          │
                    │ • ROI preparation   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │     3D CNN       │   │    Radiomics     │
          │                  │   │                  │
          │ Conv3D           │   │ Shape            │
          │ Pooling          │   │ First Order      │
          │ BatchNorm        │   │ Texture          │
          │ Dropout          │   │ Statistical      │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   ▼                      ▼
             Deep Features        Radiomics Features
                   │                      │
                   └──────────┬───────────┘
                              ▼
                     ┌─────────────────┐
                     │ Feature Fusion  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Classifier     │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Prediction    │
                     └─────────────────┘
```

---

# 🔄 Project Workflow

The complete pipeline consists of the following stages:

```text
1. Dataset Collection
        ↓
2. CT Scan Loading
        ↓
3. Preprocessing
        ↓
4. Lung/Nodule ROI Preparation
        ↓
5. 3D CNN Feature Extraction
        ↓
6. Radiomics Feature Extraction
        ↓
7. Feature Cleaning & Normalization
        ↓
8. Feature Fusion
        ↓
9. Model Training
        ↓
10. Model Evaluation
        ↓
11. Prediction
        ↓
12. Web Application
```

---

# 🛠️ Technology Stack

## Programming Language

* Python 3.10+

## Deep Learning

* PyTorch
* Torchvision where applicable

## Medical Image Processing

* SimpleITK
* NumPy
* pydicom where required

## Radiomics

* PyRadiomics

## Machine Learning

* Scikit-learn
* Pandas

## Backend

* Flask

## Frontend

* HTML5
* CSS3
* JavaScript

## Visualization

* Matplotlib
* Seaborn where appropriate for research visualization

---

# 📊 Dataset

The project is designed for use with publicly available lung CT datasets such as:

## LIDC-IDRI

**LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** is a widely used public dataset for lung CT research.

It contains thoracic CT scans together with annotations produced by radiologists.

The dataset can be used for research involving:

* Lung nodule detection
* Nodule characterization
* CT image analysis
* Radiomics
* Deep learning
* Computer-aided detection research

The dataset should be obtained from an appropriate official/public repository and used according to its applicable terms and conditions.

---

# 📁 Dataset Structure

The exact structure depends on how the dataset is downloaded and converted.

A recommended processed structure is:

```text
data/
│
├── images/
│   ├── patient_001/
│   │   └── scan.nii.gz
│   │
│   ├── patient_002/
│   │   └── scan.nii.gz
│   │
│   └── ...
│
├── masks/
│   ├── patient_001/
│   │   └── mask.nii.gz
│   │
│   ├── patient_002/
│   │   └── mask.nii.gz
│   │
│   └── ...
│
└── labels.csv
```

Example `labels.csv`:

```csv
patient_id,label
patient_001,0
patient_002,1
patient_003,0
patient_004,1
```

Where the label definition must be clearly documented according to the specific dataset preparation protocol.

---

# 📂 Project Structure

A recommended repository structure is:

```text
lung-cancer-3dcnn-radiomics/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels.csv
│
├── models/
│   ├── cnn3d.py
│   ├── fusion_model.py
│   └── checkpoints/
│
├── preprocessing/
│   ├── __init__.py
│   ├── ct_preprocessing.py
│   ├── resampling.py
│   └── dataset.py
│
├── radiomics/
│   ├── __init__.py
│   ├── extract.py
│   └── feature_selection.py
│
├── training/
│   ├── train_cnn.py
│   ├── train_fusion.py
│   └── evaluate.py
│
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── script.js
│
├── uploads/
│
└── results/
    ├── figures/
    ├── metrics/
    └── predictions/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-3dcnn-radiomics.git
```

Move into the project directory:

```bash
cd lung-cancer-3dcnn-radiomics
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
flask
numpy
pandas
scikit-learn
torch
torchvision
SimpleITK
pydicom
pyradiomics
matplotlib
joblib
```

The exact versions should be pinned in `requirements.txt` after the environment has been tested.

---

# 🧩 Configuration

Project configuration can be stored in `config.py`.

Example:

```python
IMAGE_SIZE = (64, 64, 64)

BATCH_SIZE = 4

LEARNING_RATE = 0.001

EPOCHS = 50

NUM_CLASSES = 2

MODEL_PATH = "models/checkpoints/best_model.pth"

DATA_PATH = "data/processed"
```

These values should be adjusted according to the available hardware and dataset.

---

# 🧹 Data Preprocessing

Medical CT data cannot normally be passed directly into a neural network.

The preprocessing pipeline may include:

### 1. Loading

Read CT images from DICOM or converted medical-image formats such as NIfTI.

### 2. Resampling

CT scans may have different voxel spacings.

Resampling creates a more consistent spatial representation.

### 3. Intensity Processing

CT intensity values can be represented using Hounsfield Units (HU).

A lung-focused intensity window can be applied before normalization.

### 4. Normalization

The processed intensities are normalized into a range appropriate for model training.

### 5. ROI Extraction

Relevant lung/nodule regions can be extracted using annotations or segmentation masks.

### 6. Volume Standardization

The resulting 3D region can be resized or cropped to a fixed input shape such as:

```text
64 × 64 × 64
```

The exact size should be selected based on the experimental design and available GPU memory.

---

# 🧠 3D CNN

The 3D CNN receives a volumetric CT region instead of a single 2D image.

A simplified architecture can be represented as:

```text
Input
  │
  ▼
Conv3D
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
MaxPool3D
  │
  ▼
Conv3D
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
MaxPool3D
  │
  ▼
Conv3D
  │
  ▼
Global Pooling
  │
  ▼
Dense Layer
  │
  ▼
Deep Feature Vector
```

A simplified PyTorch model:

```python
import torch
import torch.nn as nn


class CNN3D(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        features = self.features(x)

        features = features.flatten(1)

        output = self.classifier(features)

        return output
```

The architecture should be tuned experimentally rather than assuming that this example is optimal.

---

# 🧬 Radiomics Feature Extraction

Radiomics extracts quantitative measurements from a defined image region.

Typical categories include:

### Shape Features

Examples:

* Volume
* Surface area
* Sphericity
* Compactness

### First-Order Features

Examples:

* Mean intensity
* Median
* Variance
* Skewness
* Kurtosis

### Texture Features

Examples:

* GLCM
* GLRLM
* GLSZM
* NGTDM

A simplified extraction workflow:

```python
from radiomics import featureextractor


def extract_radiomics(image_path, mask_path):

    extractor = featureextractor.RadiomicsFeatureExtractor()

    features = extractor.execute(
        image_path,
        mask_path
    )

    result = {}

    for key, value in features.items():

        if isinstance(value, (int, float)):

            result[key] = value

    return result
```

For reproducible research, the PyRadiomics configuration should be documented and preferably stored in a configuration file.

---

# 🔗 Feature Fusion

The hybrid approach combines:

```text
3D CNN Features
       +
Radiomics Features
       ↓
Combined Feature Vector
       ↓
Classifier
```

For example:

```python
import torch
import torch.nn as nn


class FusionModel(nn.Module):

    def __init__(
        self,
        cnn_features,
        radiomics_features,
        num_classes=2
    ):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                cnn_features + radiomics_features,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, cnn_features, radiomics_features):

        combined = torch.cat(
            [cnn_features, radiomics_features],
            dim=1
        )

        return self.classifier(combined)
```

---

# 🏋️ Model Training

The training process should include:

* Training dataset
* Validation dataset
* Independent test dataset
* Loss function
* Optimizer
* Learning-rate scheduling where appropriate
* Model checkpointing
* Early stopping where appropriate

Example:

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Training loop:

```python
for epoch in range(epochs):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

    print(
        f"Epoch: {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )
```

For medical imaging experiments, patient-level splitting is important to reduce the risk of data leakage.

---

# 📈 Model Evaluation

The model should be evaluated on data that was not used during training.

Important metrics include:

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall / Sensitivity

Measures how many positive cases were correctly identified.

### F1-score

Provides a balance between precision and recall.

### ROC-AUC

Measures classification performance across different decision thresholds.

### Confusion Matrix

Shows:

```text
                 Predicted
              Negative Positive

Actual Negative    TN       FP

Actual Positive    FN       TP
```

Example evaluation code:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_probability
)
```

---

# 🌐 Web Application

The project includes an optional Flask-based web interface.

The frontend consists of:

* HTML
* CSS
* JavaScript

The backend handles:

* File upload
* CT preprocessing
* Model loading
* Prediction
* Result generation

Example architecture:

```text
Browser
   │
   ▼
HTML/CSS/JavaScript
   │
   ▼
Flask API
   │
   ▼
Preprocessing
   │
   ▼
3D CNN + Radiomics
   │
   ▼
Classifier
   │
   ▼
Prediction
```

---

# ▶️ Running the Project

After installing the dependencies and preparing the dataset:

### Train the model

```bash
python training/train_cnn.py
```

### Extract radiomics

```bash
python radiomics/extract.py
```

### Train the fusion model

```bash
python training/train_fusion.py
```

### Evaluate

```bash
python training/evaluate.py
```

### Start the web application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 🖥️ Example Web Workflow

```text
Open Web Application
        ↓
Upload CT Scan / Prepared Volume
        ↓
Validate Input
        ↓
Preprocess Volume
        ↓
Extract 3D CNN Features
        ↓
Extract Radiomics Features
        ↓
Fuse Features
        ↓
Run Classifier
        ↓
Display Research Prediction
```

---

# 📊 Experimental Comparison

A useful research experiment is to compare three approaches:

| Model     | Input          | Purpose                      |
| --------- | -------------- | ---------------------------- |
| 3D CNN    | CT volume      | Deep feature learning        |
| Radiomics | ROI + CT       | Handcrafted feature analysis |
| Hybrid    | CT + radiomics | Feature fusion               |

This allows the project to investigate whether combining deep-learning and radiomics features provides an advantage over using either approach independently.

---

# 📉 Results

After training, store experimental results in:

```text
results/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curve.png
│
├── metrics/
│   └── evaluation.csv
│
└── predictions/
    └── predictions.csv
```

Do not report fabricated accuracy or performance values. Results should be generated from actual experiments and should include the dataset split and evaluation methodology.

---

# 🔮 Future Improvements

Possible extensions include:

* Transfer learning for 3D medical-image models
* Advanced 3D segmentation
* Attention mechanisms
* Transformer-based architectures
* Explainable AI
* Grad-CAM or related visualization methods
* Automated lung segmentation
* Automated nodule detection
* Feature-selection algorithms
* Cross-validation
* External dataset validation
* Model calibration
* Docker deployment
* Cloud deployment
* GPU optimization
* Research dashboard with interactive metrics

---

# 🔬 Research Applications

This project can be used as a foundation for research in:

* Medical image analysis
* Computer-aided lung nodule analysis
* Deep learning
* Radiomics
* Feature fusion
* 3D image classification
* Explainable AI
* Computer-aided diagnosis research

---

# ⚠️ Limitations

This project has several important limitations:

1. CT datasets can vary substantially in acquisition parameters.
2. Radiomics features can be sensitive to preprocessing and segmentation.
3. Model performance may depend strongly on dataset composition.
4. A small or biased dataset can lead to poor generalization.
5. Medical-image datasets can contain class imbalance.
6. Results from one dataset may not generalize to other hospitals or scanners.
7. A model's numerical performance does not establish clinical effectiveness.

For research-quality evaluation, patient-level splitting and independent external validation should be considered.

---

# 🔐 Data Privacy

Do not upload private or identifiable patient data to this repository.

Never commit:

```text
*.dcm
*.nii
*.nii.gz
patient records
personal information
clinical reports
```

unless the data is explicitly permitted for redistribution.

Add appropriate data directories to `.gitignore` where necessary.

---

# 🤝 Contributing

Contributions are welcome.

### 1. Fork the repository

```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-3dcnn-radiomics.git
```

### 2. Create a branch

```bash
git checkout -b feature/new-feature
```

### 3. Make your changes

### 4. Commit

```bash
git add .
git commit -m "Add new feature"
```

### 5. Push

```bash
git push origin feature/new-feature
```

### 6. Open a Pull Request

Please include a clear description of:

* What was changed
* Why it was changed
* How it was tested
* Any limitations

---

# 📜 License

This project can be released under an appropriate open-source license such as the MIT License.

If you use external datasets or third-party code, follow their respective licenses and usage requirements.

---

# 👨‍💻 Author

**Your Name**

GitHub: `https://github.com/YOUR_USERNAME`

Project: **Lung Cancer Detection Using 3D CNN and Radiomics**

---

# ⚕️ Disclaimer

This software is developed strictly for **educational and research purposes**.

It has not been validated for clinical use and should not be used to diagnose, treat, or make medical decisions about lung cancer or any other disease.

Any prediction generated by this system must not be interpreted as a medical diagnosis. Clinical decisions should always be made by qualified healthcare professionals using appropriate clinical examination, medical imaging, laboratory testing, and other relevant evidence.

---

## ⭐ Acknowledgements

This project builds upon research and open-source technologies in:

* Deep learning
* 3D medical image analysis
* Radiomics
* Computer vision
* Medical imaging

Special acknowledgement should be given to the providers of any public dataset, medical-imaging libraries, and open-source frameworks used in the implementation.

---

## 📌 Project Status

**🚧 Active Development**

Current development areas:

* [ ] Dataset preparation
* [ ] CT preprocessing
* [ ] Lung/nodule segmentation
* [ ] 3D CNN implementation
* [ ] Radiomics extraction
* [ ] Feature fusion
* [ ] Model training
* [ ] Model evaluation
* [ ] Web interface
* [ ] Explainability
* [ ] External validation

---

### ⭐ If you find this project useful

Consider giving the repository a ⭐ and contributing improvements to the project.
