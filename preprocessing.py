import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Define dataset path and image size
dataset_path = 'asl_dataset'  # Update with the path to your dataset folder
image_size = (64, 64)  # Resize images to 64x64

# Initialize lists to store image data and labels
images = []
labels = []

# Define label map (you can extend this to match your folder structure)
label_map = {str(i): i for i in range(10)}  # For 0-9
label_map.update({chr(i): i-65+10 for i in range(65, 91)})  # For A-Z

# Custom label handling (you can manually map other labels here)
label_map["Hello"] = 36  # For 'Hello' label (add more if needed)

# Loop over each folder in the dataset
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    # Skip non-directories
    if not os.path.isdir(folder_path):
        continue

    # Get the label (folder name) and convert to corresponding number
    label = label_map.get(folder_name)

    # Loop through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Resize the image to the target size
            img_resized = cv2.resize(img, image_size)

            # Normalize the image pixel values to [0, 1]
            img_normalized = img_resized / 255.0

            # Get label from the corresponding txt file (e.g., "A_1.txt")
            label_txt_file = filename.split('.')[0] + '.txt'
            label_txt_path = os.path.join(folder_path, label_txt_file)

            if os.path.exists(label_txt_path):
                with open(label_txt_path, 'r') as file:
                    label_str = file.read().strip()
                    if label_str in label_map:
                        label = label_map[label_str]
                    else:
                        print(f"Warning: {label_str} not found in label_map")

            # Append the image and label to their respective lists
            images.append(img_normalized)
            labels.append(label)

# Convert the list of images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Optionally, convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(label_map))

# Print dataset info
print(f"Loaded {len(images)} images with {len(label_map)} labels.")
