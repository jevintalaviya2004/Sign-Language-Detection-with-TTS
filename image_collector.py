import os
import cv2
import time
import uuid

path = "Frames"

labels = ['nice','my', 'name', 'is',]

num_images = 20

for label in labels:
    img_path = os.path.join(path, label)
    os.makedirs(img_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Collecting Images for {label}")
    time.sleep(5)

    for image_num in range(num_images):
        success, frame = cap.read()
        image_name = os.path.join(f"{label}_{str(uuid.uuid4())}.jpg")
        cv2.imwrite(os.path.join(img_path, image_name), frame)
        cv2.imshow('frame', frame)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
