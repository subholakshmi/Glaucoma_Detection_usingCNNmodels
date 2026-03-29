# 🧠 Glaucoma Detection using Hybrid CNN (VGG16 + DenseNet121)

A deep learning-based system for **automated glaucoma detection** from retinal fundus images, enhanced with **Grad-CAM explainability** and deployed using **Streamlit**.

---

## 📌 Overview

Glaucoma is a leading cause of blindness worldwide. Early detection is critical but challenging.

This project presents a **hybrid deep learning model** combining:
- VGG16 (feature extraction)
- DenseNet121 (deep feature propagation)

to accurately classify fundus images into:
- ✅ Normal
- ⚠️ Glaucoma

---

## 🚀 Features

- 🔍 Hybrid CNN architecture (VGG16 + DenseNet121)
- 🧠 High-performance image classification
- 🔥 Grad-CAM visualization (model explainability)
- 🌐 Streamlit web app for real-time prediction
- 📊 Evaluation using ROC-AUC, accuracy, etc.

---

## 🖼️ Sample Output

- Upload a fundus image
- Get prediction (Normal / Glaucoma)
- View confidence score
- Visualize **Grad-CAM heatmap**

---

## 🏗️ Model Architecture

- Input size: **256 × 256**
- Backbone:
  - VGG16
  - DenseNet121
- Feature fusion + fully connected layers
- Output:
  - Binary classification

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

## ⚙️ How to Run

### 1️⃣ Clone repository
```bash
git clone https://github.com/subholakshmi/glaucoma-detection.git
cd glaucoma-detection

