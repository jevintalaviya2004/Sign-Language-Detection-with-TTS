import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------- CONFIG ----------
DATASET_PATH = 'Frames'
IMG_SIZE = (64, 64)
EPOCHS = 30
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'best_model.h5'
LABEL_MAP_PATH = 'label_map.json'

# ---------- LOAD & PREPROCESS ----------
def load_images(folder, img_size=(64, 64)):
    data, labels = [], []
    for label in sorted(os.listdir(folder)):
        path = os.path.join(folder, label)
        if not os.path.isdir(path): continue
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_arr = cv2.imread(img_path)
                if img_arr is None:
                    continue
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                img_arr = cv2.resize(img_arr, img_size)
                data.append(img_arr)
                labels.append(label)
    return np.array(data), np.array(labels)

print("[INFO] Loading data...")
X, y = load_images(DATASET_PATH, IMG_SIZE)
if len(X) == 0:
    raise Exception("No data found.")

X = X / 255.0
X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)

classes = sorted(set(y))
label_map = {label: idx for idx, label in enumerate(classes)}
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)

y_encoded = np.array([label_map[label] for label in y])
y_cat = to_categorical(y_encoded, num_classes=len(classes))

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ---------- AUGMENTATION ----------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# ---------- MODEL ----------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------- CALLBACK ----------
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

# ---------- TRAIN ----------
print("[INFO] Training...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print(f"[INFO] Best model saved to: {MODEL_SAVE_PATH}")
