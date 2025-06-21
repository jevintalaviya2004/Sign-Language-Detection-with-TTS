import cv2
import numpy as np
import pyttsx3
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model and label map
model = load_model('best_model.h5')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
label_map = {str(k): v for k, v in label_map.items()}
rev_map = {v: k for k, v in label_map.items()}

# Init TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
IMG_SIZE = (64, 64)
prev_prediction = None

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box from landmarks
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_vals) * w) - 20
            ymin = int(min(y_vals) * h) - 20
            xmax = int(max(x_vals) * w) + 20
            ymax = int(max(y_vals) * h) + 20
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # Draw hand landmarks and box
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Extract and preprocess ROI
            roi = frame[ymin:ymax, xmin:xmax]
            if roi.size == 0: continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMG_SIZE)
            norm = resized / 255.0
            input_img = norm.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

            # Predict
            pred = model.predict(input_img, verbose=0)
            label_idx = np.argmax(pred)
            label = rev_map[label_idx]

            if label != prev_prediction:
                speak(label)
                prev_prediction = label

            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
