import os
import cv2
import time
import uuid
import mediapipe as mp

# === CONFIGURATION ===
labels = ['Hello', 'Yes']
num_images = 20
base_path = 'Frames2'
crop_size = (224, 224)
padding_scale = 0.4

# === SETUP MEDIA PIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === CAPTURE LOOP ===
for label in labels:
    print(f"\n[INFO] Collecting for '{label}' — starting in 5 seconds...")
    time.sleep(5)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            # Use only the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label.lower()  # 'left' or 'right'

            # Bounding box with padding
            x = [lm.x for lm in hand_landmarks.landmark]
            y = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x) * w)
            x_max = int(max(x) * w)
            y_min = int(min(y) * h)
            y_max = int(max(y) * h)

            box_width = x_max - x_min
            box_height = y_max - y_min
            pad_x = int(box_width * padding_scale)
            pad_y = int(box_height * padding_scale)

            xmin = max(x_min - pad_x, 0)
            xmax = min(x_max + pad_x, w)
            ymin = max(y_min - pad_y, 0)
            ymax = min(y_max + pad_y, h)

            # Crop and validate
            cropped = frame[ymin:ymax, xmin:xmax]
            if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                print("[!] Skipped empty crop.")
                continue

            resized = cv2.resize(cropped, crop_size)

            # Save
            output_dir = os.path.join(base_path, label, handedness)
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{label}_{handedness}_{str(uuid.uuid4())}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)
            print(f"[✓] Saved: {output_dir}/{filename}")
            count += 1
            time.sleep(1)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Show window
        cv2.imshow('Hand Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

hands.close()
print("\n✅ All labels processed.")
