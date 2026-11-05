# Computer Vision Assignment 2: SOTA Action & Pose Classification + Streamlit Demo

**Student Name:** Inayatulla  
**University:** Astana IT University  
**Course:** Computer Vision (Assignment 2 Continuation)  
**Instructor:** Nurym Kaiyrkhan  

---

## Project Overview
This repository contains the complete end-to-end Computer Vision pipeline for fitness action and posture classification (`pushup`, `squat`, `standing`). Building upon Assignment 1 (Baseline CNN & ResNet18), **Assignment 2** upgrades the pipeline with:
1. **SOTA Backbone:** Integrated **ConvNeXt-Tiny** with fine-tuning.
2. **Stress-Testing & ArtAug:** Evaluated models against severe conditions (motion blur, extreme camera tilt, occlusion).
3. **Visual Explainability (XAI):** Integrated **Grad-CAM heatmaps** and structural edge ROI silhouette masks.
4. **Interactive Demo:** A fully functioning **Streamlit Web Application** supporting single-image uploads, webcam live streams, and temporal multi-frame video consistency checks.

---

## Repository Structure
```text
cv_exercise_project/
│
├── app.py                  # Streamlit Interactive Web Application
├── train_a2_sota.py        # SOTA training, benchmarking & stress-testing script
├── slice_videos.py         # Raw video frame extraction utility
├── split_data.py           # Stratified train/val/test data splitter (SEED=42)
├── requirements.txt        # Python package dependencies
├── exercise_classification.ipynb # A1 Jupyter notebook with outputs
└── weights/                # Saved PyTorch model checkpoints (.pth)
```

---

##  Getting Started & Local Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AnauMynau/cv_exercise_sota_a2.git
cd cv_exercise_project

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Training & Benchmarks (Optional)
To retrain models and evaluate clean vs. stress-test performance:
```bash
python train_a2_sota.py
```

### 3. Launch the Interactive Web Demo
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

##  Quantitative Summary (Clean vs. Stress-Test)

| Model Architecture | Clean Test Accuracy | Clean Macro-F1 | Stress-Test F1 (Blur/Tilt) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: |
| **A1 Baseline CNN** | 100.0% | 1.0000 | 0.8099 | **13.78 ms** |
| **A1 ResNet18** | 100.0% | 1.0000 | 0.9201 | 27.23 ms |
| **A2 SOTA ConvNeXt-Tiny** | **100.0%** | **1.0000** | **1.0000** | 63.11 ms |

---

##  Ethics & Data Privacy
All training footage was recorded with explicit personal consent. Augmented and perturbation stress-test outputs are explicitly watermarked within the application interface to ensure transparent AI usage.