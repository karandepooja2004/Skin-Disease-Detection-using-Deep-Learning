# gui_app.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('skin_cancer_model.h5')

# Function to predict
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return "Cancer" if class_idx == 1 else "Healthy"

# Upload and predict function
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
        result = predict_image(file_path)
        label_result.config(text=f"Prediction: {result}", font=("Arial", 16, "bold"))

# GUI
root = tk.Tk()
root.title("Skin Cancer Detection GUI")

upload_btn = tk.Button(root, text="Upload Skin Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack(pady=20)

root.mainloop()
