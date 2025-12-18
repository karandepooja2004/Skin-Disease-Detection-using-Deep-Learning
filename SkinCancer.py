import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Load Dataset
def load_images_labels(image_folders, csv_file, image_size=(64, 64)):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_name = row['image_id'] + '.jpg'
        label = row['dx']  # 'dx' is the disease type
        
        # Define label: 0 = Healthy, 1 = Cancer (for simplicity, here non-melanoma = healthy)
        if label == 'mel':  # 'mel' = melanoma (cancer)
            label_numeric = 1
        else:
            label_numeric = 0
        
        found = False
        for folder in image_folders:
            img_path = os.path.join(folder, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label_numeric)
                found = True
                break
        if not found:
            print(f"Warning: Image {img_name} not found in folders.")
    
    images = np.array(images) / 255.0
    labels = to_categorical(np.array(labels), num_classes=2)
    return images, labels

# Paths
image_folders = [
    r'D:\AIDS_TY\Sem VI\Deep Learning\DL_MP\archive (3)\HAM10000_images_part_1',
    r'D:\AIDS_TY\Sem VI\Deep Learning\DL_MP\archive (3)\HAM10000_images_part_2'
]
csv_file = 'D:\\AIDS_TY\\Sem VI\\Deep Learning\\DL_MP\\archive (3)\\HAM10000_metadata.csv'

# Load data
X, y = load_images_labels(image_folders, csv_file)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save('skin_cancer_model.h5')

print("Model trained and saved as skin_cancer_model.h5")