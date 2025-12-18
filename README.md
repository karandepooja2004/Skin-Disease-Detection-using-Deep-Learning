# Skin Cancer Detection Using Deep Learning

This project implements a **Skin Cancer Detection System** using a **Convolutional Neural Network (CNN)**.  
It classifies skin lesion images as **Cancer (Melanoma)** or **Healthy** and provides a **GUI-based application** for easy image upload and prediction.

---

## 📌 Project Features

- Deep Learning based image classification using **CNN**
- Uses the **HAM10000 skin lesion dataset**
- Binary classification:
  - **Healthy**
  - **Cancer (Melanoma)**
- User-friendly **Tkinter GUI**
- Trained model saved and reused for predictions

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Tkinter (GUI)
- PIL (Image Processing)

---

---

## 📊 Dataset Information

- **Dataset Name:** HAM10000 – Human Against Machine with 10000 training images
- **Images:** Skin lesion images
- **Metadata File:** `HAM10000_metadata.csv`
- **Label Used:**  
  - `mel` → Cancer (Melanoma)  
  - Others → Healthy

> Download Dataset from here : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## ⚙️ Model Architecture

- Conv2D (32 filters, ReLU)
- MaxPooling2D
- Conv2D (64 filters, ReLU)
- MaxPooling2D
- Flatten
- Dense (128 neurons, ReLU)
- Dropout (0.5)
- Dense (2 neurons, Softmax)

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
    pip install tensorflow opencv-python numpy pandas scikit-learn pillow

### 2️⃣ Train the Model
    python run SkinCancer.py

- This will train the CNN and save the model as:
    skin_cancer_model.h5

### 2️⃣ Run GUI Application
    streamlit run GUI.py
    
- Click Upload Skin Image

- Select a skin lesion image

- The prediction result will be displayed on screen
