# FreshScan AI — Fruit & Vegetable Freshness Detector

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Fundamental-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Deep_learning)

**FreshScan AI** is an intelligent vision system designed to automatically classify the freshness of fruits and vegetables. Built using Deep Learning and Computer Vision, this system addresses the subjective nature of manual food inspection, providing a consistent and scalable solution for food quality detection.

---

## Key Features

- **Real-Time Detection**: Capture images directly from your webcam or upload existing files for instant analysis.
- **Multi-Model Intelligence**: Compares three state-of-the-art CNN architectures (ResNet50, EfficientNet-B0, MobileNetV2).
- **Precise Confidence Analytics**: Displays top-3 predictions with ranked confidence scores and probability progress bars.
- **Premium UI/UX**: Interactive web application built with Streamlit, featuring a modern, glassmorphic design and responsive layout.
- **Extensive Class Support**: Detects freshness across 6 varieties (Apple, Banana, Bitter Gourd, Capsicum, Orange, Tomato).

---

## Tech Stack

| Category | Tools & Technologies |
| :--- | :--- |
| **Backend / AI** | Python, PyTorch, Torchvision |
| **Frontend** | Streamlit, HTML5/CSS3 (Custom Glassmorphism) |
| **Data Science** | NumPy, Pandas, scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Image Processing** | PIL (Pillow) |

---

## Project Structure

```text
final-project-fruit/
├── app/                  # Streamlit application & utility functions
│   ├── streamlit_app.py  # Main web interface
│   └── utils.py          # Prediction & image transform logic
├── models/               # Saved model weights
│   └── best_model.pth    # Fine-tuned EfficientNet-B0 weight
├── notebooks/            # Experimentation & training workflows
└── requirements.txt      # Project dependencies
```

---

## Model Performance & Results

The system evaluates three pretrained architectures to find the optimal balance between accuracy and efficiency.

### Comparison Matrix

| Model | Architecture | Training Accuracy | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | Compound Scaling | ~99% | **99.91%** | 🏆 **Best** |
| ResNet50 | Residual Learning | ~97% | ~96% | Good |
| MobileNetV2 | Lightweight | ~95% | ~94% | Fast |

> [!NOTE]  
> **EfficientNet-B0** was selected for deployment due to its superior accuracy and efficient parameter count.

---

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Recommended: NVIDIA GPU with CUDA support for training (CPU is sufficient for inference).

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/arizalanru/freshscan-ai.git
cd freshscan-ai
pip install -r requirements.txt
```

### 3. Running the App
Launch the interactive web application:
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset Overview

The model is trained on **12 classes** (6 items × 2 conditions):

- **Fresh**: Apple, Banana, Bitter Gourd, Capsicum, Orange, Tomato
- **Stale**: Apple, Banana, Bitter Gourd, Capsicum, Orange, Tomato

**Preprocessing Pipeline:**
- **Size**: Unified 224x224 pixels.
- **Normalization**: ImageNet standards (Mean/Std).
- **Augmentation**: Horizontal Flip, Rotation, Color Jitter, and Random Resized Crop.

---

## Limitations

- **Lighting**: Heavily shadowed or overexposed environments may impact accuracy.
- **Single Item**: Optimized for images containing one fruit/vegetable at a time.
- **Out of Scope**: Objects outside the 12 trained classes may yield unreliable results.

---

## Author

**Arizal Anru - Final Project - Deep Learning Fundamental**  
*President University*

---
