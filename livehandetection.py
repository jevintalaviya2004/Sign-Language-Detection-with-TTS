import cv2
import numpy as np
import pyttsx3
import json
import mediapipe as mp
import threading
from tensorflow.keras.models import load_model
from tkinter import Tk, Label, Button, Checkbutton, IntVar, Listbox, Scrollbar, END, RIGHT, Y, Frame
from PIL import Image, ImageTk

# Constants
IMG_SIZE = (64, 64)
running = False
prev_prediction = None

# Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# MediaPipe Hand Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Load model and label map
model = load_model('best_model.h5')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
label_map = {str(k): v for k, v in label_map.items()}
rev_map = {v: k for k, v in label_map.items()}


# GUI Application Class
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language Detection")

        self.tts_enabled = IntVar(value=1)

        self.label = Label(window)
        self.label.pack()

        self.tts_check = Checkbutton(window, text="Enable TTS", variable=self.tts_enabled)
        self.tts_check.pack()

        self.btn_start = Button(window, text="Start Detection", command=self.start_detection)
        self.btn_start.pack(pady=5)

        self.btn_stop = Button(window, text="Stop Detection", command=self.stop_detection)
        self.btn_stop.pack(pady=5)

        # History Log
        frame = Frame(window)
        frame.pack()
        self.history_list = Listbox(frame, width=30, height=10)
        scrollbar = Scrollbar(frame, command=self.history_list.yview)
        self.history_list.config(yscrollcommand=scrollbar.set)
        self.history_list.pack(side="left")
        scrollbar.pack(side=RIGHT, fill=Y)

        self.cap = None
        self.thread = None

    def speak(self, text):
        if self.tts_enabled.get():
            engine.say(text)
            engine.runAndWait()

    def start_detection(self):
        global running, prev_prediction
        if not running:
            running = True
            prev_prediction = None
            self.history_list.delete(0, END)
            self.cap = cv2.VideoCapture(0)
            self.thread = threading.Thread(target=self.process)
            self.thread.start()

    def stop_detection(self):
        global running
        running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process(self):
        global prev_prediction
        while running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            label = ""
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    xmin = int(min(x_vals) * w) - 20
                    ymin = int(min(y_vals) * h) - 20
                    xmax = int(max(x_vals) * w) + 20
                    ymax = int(max(y_vals) * h) + 20
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w, xmax), min(h, ymax)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    roi = frame[ymin:ymax, xmin:xmax]
                    if roi.size == 0:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, IMG_SIZE)
                    norm = resized / 255.0
                    input_img = norm.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

                    pred = model.predict(input_img, verbose=0)
                    label_idx = np.argmax(pred)
                    label = rev_map[label_idx]

                    if label != prev_prediction:
                        self.history_list.insert(END, label)
                        self.speak(label)
                        prev_prediction = label

                    cv2.putText(frame, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display Frame in GUI
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.label.configure(image='')


# Run Application
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
