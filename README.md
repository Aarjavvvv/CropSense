# 🌿 CropSense — AI Crop Disease Detection

> Upload a leaf photo. Get an instant diagnosis — crop type, disease name, confidence score, and a Grad-CAM heatmap showing *where* the model looked.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **EfficientNetB0** | Fine-tuned on PlantVillage — 2-phase transfer learning |
| 🌾 **38 Disease Classes** | 14 crop species · 17 disease types + healthy variants |
| 🔍 **Grad-CAM** | Visual explainability — highlights the diseased leaf region |
| 📊 **Top-5 Predictions** | Confidence scores for top 5 diagnoses |
| 🎛️ **Gradio Web UI** | Clean editorial interface, launch with one command |
| ⚡ **Fast Inference** | ~200ms per image on CPU after model load |

---

## 🖥️ Interface

> Upload any leaf image and get a full diagnosis in under a second.

![CropSense UI](assets/ui_empty.png)

---

## 📸 Demo

### Example 1 — Tomato Early Blight

| Input | Grad-CAM Heatmap |
|---|---|
| ![Input](assets/demo1_input.jpg) | ![GradCAM](assets/demo1_gradcam.png) |

**Diagnosis:**
- **Crop:** Tomato
- **Status:** ⚠️ Disease Detected
- **Condition:** Early Blight
- **Confidence:** 97.3%

---

### Example 2 — Healthy Pepper Leaf

| Input | Output |
|---|---|
| ![Input](assets/demo2_input.jpg) | ![Output](assets/demo2_output.png) |

**Diagnosis:**
- **Crop:** Pepper
- **Status:** ✅ Healthy
- **Confidence:** 94.1%

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/CropSense.git
cd CropSense
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download PlantVillage from Kaggle:
```
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```
Extract so the structure is:
```
data/
└── plantvillage/
    ├── Apple__Apple_scab/
    ├── Tomato__Early_blight/
    ├── Potato__healthy/
    └── ...
```

### 4. Train
```bash
jupyter notebook CropSense_Train.ipynb
```
Runs 2-phase training: head-only (5 epochs) → full fine-tune (5 epochs).

### 5. Run Web App
```bash
python app.py
```
Opens at **http://localhost:7860**

---

## 🏗️ Architecture & Training Strategy

```
Input Image (224×224)
        │
        ▼
┌───────────────────────┐
│  EfficientNetB0       │  ← Pretrained on ImageNet
│  Feature Extractor    │     (5.3M parameters)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Custom Classifier    │  ← Dropout → Linear(512) → ReLU
│  Head                 │     → Dropout → Linear(38)
└──────────┬────────────┘
           │
           ▼
    38-class Softmax

Training:
  Phase 1 — Freeze base, train head only (lr=1e-3, 5 epochs)
  Phase 2 — Unfreeze all, fine-tune end-to-end (lr=1e-5, 5 epochs)
  Optimizer: AdamW + CosineAnnealingLR
  Augmentation: flip, rotate, color jitter
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~95%+ |
| Macro F1 Score | ~94% |
| Inference Speed | ~200ms / image (CPU) |
| Model Size | ~20MB |

*Results vary slightly based on random seed and hardware.*

---

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap showing which pixels the model used to make its prediction — critical for trust in medical/agricultural AI.

```python
from pytorch_grad_cam import GradCAM

target_layer = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layer)
```

---

## 📂 Project Structure

```
CropSense/
├── app.py                  # Gradio web application
├── CropSense_Train.ipynb   # Full training + evaluation notebook
├── requirements.txt
├── class_names.json        # Generated after training
├── cropsense_efficientnet.pth  # Saved model weights
├── assets/                 # Screenshots and demo images
└── README.md
```

---

## 🔧 Tech Stack

- `torch` + `torchvision` — Model training and inference
- `grad-cam` — Explainability heatmaps
- `gradio` — Web UI
- `scikit-learn` — Evaluation metrics
- `matplotlib` + `seaborn` — Training curves, confusion matrix

---

## 📄 License

MIT License
